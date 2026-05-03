/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "qwen3OmniTTSRuntime.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;
using namespace talker_constants;

namespace
{
uint32_t halfToFloatBits(uint16_t h)
{
    uint32_t const sign = (h >> 15) & 1;
    uint32_t const exp = (h >> 10) & 0x1F;
    uint32_t const mant = h & 0x3FF;
    if (exp == 0x1F)
    {
        return (sign << 31) | (0xFF << 23) | (mant << 13);
    }
    if (exp == 0)
    {
        return sign << 31;
    }
    return (sign << 31) | ((exp + 112) << 23) | (mant << 13);
}

std::vector<float> copyHalfTensorToHostFloat(rt::Tensor const& tensor, int64_t elements, cudaStream_t stream)
{
    std::vector<uint16_t> raw(static_cast<size_t>(elements));
    std::vector<float> out(static_cast<size_t>(elements));
    CUDA_CHECK(cudaMemcpyAsync(raw.data(), tensor.rawPointer(), raw.size() * sizeof(uint16_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int64_t i = 0; i < elements; ++i)
    {
        uint32_t const bits = halfToFloatBits(raw[static_cast<size_t>(i)]);
        std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(float));
    }
    return out;
}

int32_t sampleLogitsCPU(rt::Tensor const& deviceLogits, int32_t vocabSize, int32_t topK, float temperature,
    bool suppressEos, int32_t eosId, float eosBias, bool suppressRange, std::mt19937& rng, cudaStream_t stream)
{
    std::vector<double> logits(static_cast<size_t>(vocabSize));
    std::vector<float> host(static_cast<size_t>(vocabSize));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), deviceLogits.rawPointer(), host.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t i = 0; i < vocabSize; ++i)
    {
        logits[i] = static_cast<double>(host[i]);
    }

    if (suppressEos && eosId >= 0 && eosId < vocabSize)
    {
        logits[eosId] = -1e30;
    }
    if (suppressRange)
    {
        int32_t const suppressStart = vocabSize - 1024;
        for (int32_t i = suppressStart; i < vocabSize; ++i)
        {
            if (i != eosId)
            {
                logits[i] = -1e30;
            }
        }
    }
    if (eosBias != 0.0f && !suppressEos && eosId >= 0 && eosId < vocabSize)
    {
        logits[eosId] += static_cast<double>(eosBias);
    }
    if (temperature > 1e-6f)
    {
        for (auto& v : logits)
        {
            v /= static_cast<double>(temperature);
        }
    }
    if (topK > 0 && topK < vocabSize)
    {
        std::vector<double> sorted(logits);
        std::nth_element(sorted.begin(), sorted.end() - topK, sorted.end());
        double const threshold = sorted[sorted.size() - topK];
        for (auto& v : logits)
        {
            if (v < threshold)
            {
                v = -1e30;
            }
        }
    }

    double const maxVal = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (auto& v : logits)
    {
        v = std::exp(v - maxVal);
        sum += v;
    }
    for (auto& v : logits)
    {
        v /= sum;
    }

    std::discrete_distribution<int32_t> dist(logits.begin(), logits.end());
    return dist(rng);
}

bool detectPrimaryRepetition(std::vector<int32_t> const& history, int32_t minRepeat)
{
    int32_t const n = static_cast<int32_t>(history.size());
    if (n < minRepeat)
    {
        return false;
    }
    int32_t const last = history.back();
    int32_t count = 0;
    for (int32_t i = n - 1; i >= 0 && history[i] == last; --i)
    {
        ++count;
    }
    if (count >= minRepeat)
    {
        return true;
    }
    if (n >= minRepeat * 2)
    {
        int32_t const a = history[n - 2];
        int32_t const b = history[n - 1];
        int32_t pairs = 0;
        for (int32_t i = n - 2; i >= 1; i -= 2)
        {
            if (history[i - 1] == a && history[i] == b)
            {
                ++pairs;
            }
            else
            {
                break;
            }
        }
        if (pairs >= minRepeat)
        {
            return true;
        }
    }
    return false;
}

// Helper: Extract MLP weights from safetensors (eliminates code duplication)
bool extractMLPWeightsFromTensors(std::vector<rt::Tensor>& tensors, rt::Tensor& fc1Weight, rt::Tensor& fc1Bias,
    rt::Tensor& fc2Weight, rt::Tensor& fc2Bias, std::string const& projectionName)
{
    constexpr int32_t kExpectedTensorCount = 4;
    check::check(
        tensors.size() == kExpectedTensorCount, projectionName + ".safetensors should contain exactly 4 tensors");

    bool foundFC1Weight = false, foundFC1Bias = false, foundFC2Weight = false, foundFC2Bias = false;

    for (auto& tensor : tensors)
    {
        std::string const& name = tensor.getName();
        if (name.find("fc1.weight") != std::string::npos)
        {
            fc1Weight = std::move(tensor);
            foundFC1Weight = true;
        }
        else if (name.find("fc1.bias") != std::string::npos)
        {
            fc1Bias = std::move(tensor);
            foundFC1Bias = true;
        }
        else if (name.find("fc2.weight") != std::string::npos)
        {
            fc2Weight = std::move(tensor);
            foundFC2Weight = true;
        }
        else if (name.find("fc2.bias") != std::string::npos)
        {
            fc2Bias = std::move(tensor);
            foundFC2Bias = true;
        }
    }

    if (!foundFC1Weight || !foundFC1Bias || !foundFC2Weight || !foundFC2Bias)
    {
        LOG_ERROR("Failed to find all required tensors in %s.safetensors", projectionName.c_str());
        return false;
    }

    return true;
}
} // anonymous namespace

class Qwen3OmniTTSRuntime::SpecialCodePredictorEngine
{
public:
    SpecialCodePredictorEngine(std::filesystem::path const& enginePath, std::filesystem::path const& embedPath,
        int32_t hiddenSize, int32_t codebookSize, int32_t numLayers, int32_t numHeads, int32_t headDim,
        int32_t numGroups, cudaStream_t stream)
        : mHiddenSize(hiddenSize)
        , mCodebookSize(codebookSize)
        , mNumLayers(numLayers)
        , mNumHeads(numHeads)
        , mHeadDim(headDim)
        , mNumGroups(numGroups)
        , mStream(stream)
        , mRng(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()))
    {
        loadEmbeddings(embedPath);

        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error("Failed to open special CodePredictor engine: " + enginePath.string());
        }
        std::vector<char> engineBlob((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
        mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
        mEngine.reset(mRuntime->deserializeCudaEngine(engineBlob.data(), engineBlob.size()));
        if (!mEngine)
        {
            throw std::runtime_error("Failed to deserialize special CodePredictor engine: " + enginePath.string());
        }
        if (mEngine->getNbOptimizationProfiles() >= 2)
        {
            mPrefillContext.reset(mEngine->createExecutionContext());
            mDecodeContext.reset(mEngine->createExecutionContext());
            mPrefillContext->setOptimizationProfileAsync(0, mStream);
            mDecodeContext->setOptimizationProfileAsync(1, mStream);
        }
        else
        {
            mPrefillContext.reset(mEngine->createExecutionContext());
            mDecodeContext.reset(mEngine->createExecutionContext());
        }

        mLogitsName = hasTensor("logits_all") ? "logits_all" : "logits";
        mHasGenStep = hasTensor("gen_step");
        mHasPastLength = hasTensor("past_length");
        mKVElementSize = trtElementSize(mEngine->getTensorDataType("new_past_key_0"));
        mLogitsElementSize = trtElementSize(mEngine->getTensorDataType(mLogitsName.c_str()));
        mLogitsAreBf16 = mEngine->getTensorDataType(mLogitsName.c_str()) == nvinfer1::DataType::kBF16;

        mPastKeyNames.reserve(mNumLayers);
        mPastValueNames.reserve(mNumLayers);
        mNewPastKeyNames.reserve(mNumLayers);
        mNewPastValueNames.reserve(mNumLayers);
        for (int32_t i = 0; i < mNumLayers; ++i)
        {
            mPastKeyNames.push_back("past_key_" + std::to_string(i));
            mPastValueNames.push_back("past_value_" + std::to_string(i));
            mNewPastKeyNames.push_back("new_past_key_" + std::to_string(i));
            mNewPastValueNames.push_back("new_past_value_" + std::to_string(i));
        }

        CUDA_CHECK(cudaMalloc(&mDeviceEmbeds, static_cast<size_t>(2 * mHiddenSize) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mDeviceCachePosition, static_cast<size_t>(32) * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&mDeviceGenStep, sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&mDevicePastLength, sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&mDeviceDummyKV, 16));
        CUDA_CHECK(cudaMalloc(&mDeviceLogits, static_cast<size_t>(mNumGroups) * mCodebookSize * mLogitsElementSize));

        std::vector<int64_t> cachePositions(32);
        for (int32_t i = 0; i < static_cast<int32_t>(cachePositions.size()); ++i)
        {
            cachePositions[i] = i;
        }
        CUDA_CHECK(cudaMemcpyAsync(mDeviceCachePosition, cachePositions.data(),
            cachePositions.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

        size_t const kvBytes = static_cast<size_t>(mNumHeads) * 32 * mHeadDim * mKVElementSize;
        mKVA.resize(2 * mNumLayers, nullptr);
        mKVB.resize(2 * mNumLayers, nullptr);
        for (int32_t i = 0; i < 2 * mNumLayers; ++i)
        {
            CUDA_CHECK(cudaMalloc(&mKVA[i], kvBytes));
            CUDA_CHECK(cudaMalloc(&mKVB[i], kvBytes));
        }
        LOG_INFO("Special Qwen3-TTS CodePredictor enabled: %s", enginePath.string().c_str());
    }

    ~SpecialCodePredictorEngine()
    {
        for (void* p : mKVA)
        {
            if (p)
            {
                cudaFree(p);
            }
        }
        for (void* p : mKVB)
        {
            if (p)
            {
                cudaFree(p);
            }
        }
        if (mDeviceEmbeds)
            cudaFree(mDeviceEmbeds);
        if (mDeviceCachePosition)
            cudaFree(mDeviceCachePosition);
        if (mDeviceGenStep)
            cudaFree(mDeviceGenStep);
        if (mDevicePastLength)
            cudaFree(mDevicePastLength);
        if (mDeviceDummyKV)
            cudaFree(mDeviceDummyKV);
        if (mDeviceLogits)
            cudaFree(mDeviceLogits);
    }

    bool generate(std::vector<float> const& hidden, std::vector<float> const& primaryEmbedding, int32_t activeGroups,
        int32_t topK, float temperature, std::vector<int32_t>& residualCodes)
    {
        mRng.seed(42);
        residualCodes.assign(static_cast<size_t>(mNumGroups), 0);
        zeroKV();

        size_t const bytes = static_cast<size_t>(mHiddenSize) * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(mDeviceEmbeds, hidden.data(), bytes, cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mDeviceEmbeds) + bytes, primaryEmbedding.data(), bytes,
            cudaMemcpyHostToDevice, mStream));

        nvinfer1::IExecutionContext* prefill = mPrefillContext.get();
        setScalar(mDeviceGenStep, 0);
        setScalar(mDevicePastLength, 0);
        prefill->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 2, mHiddenSize});
        prefill->setTensorAddress("inputs_embeds", mDeviceEmbeds);
        prefill->setInputShape("cache_position", nvinfer1::Dims{1, {2}});
        prefill->setTensorAddress("cache_position", mDeviceCachePosition);
        if (mHasGenStep)
        {
            prefill->setTensorAddress("gen_step", mDeviceGenStep);
        }
        if (mHasPastLength)
        {
            prefill->setTensorAddress("past_length", mDevicePastLength);
        }
        for (int32_t i = 0; i < mNumLayers; ++i)
        {
            prefill->setInputShape(mPastKeyNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, 0, mHeadDim});
            prefill->setInputShape(mPastValueNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, 0, mHeadDim});
            prefill->setTensorAddress(mPastKeyNames[i].c_str(), mDeviceDummyKV);
            prefill->setTensorAddress(mPastValueNames[i].c_str(), mDeviceDummyKV);
            prefill->setTensorAddress(mNewPastKeyNames[i].c_str(), mKVB[2 * i]);
            prefill->setTensorAddress(mNewPastValueNames[i].c_str(), mKVB[2 * i + 1]);
        }
        prefill->setTensorAddress(mLogitsName.c_str(), mDeviceLogits);
        if (!prefill->enqueueV3(mStream))
        {
            LOG_ERROR("Special CodePredictor prefill failed");
            return false;
        }
        residualCodes[0] = sampleDeviceLogits(0, topK, temperature);

        nvinfer1::IExecutionContext* decode = mDecodeContext.get();
        decode->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, mHiddenSize});
        decode->setTensorAddress("inputs_embeds", mDeviceEmbeds);
        decode->setInputShape("cache_position", nvinfer1::Dims{1, {1}});
        decode->setTensorAddress("cache_position", mDeviceCachePosition);
        decode->setTensorAddress(mLogitsName.c_str(), mDeviceLogits);
        if (mHasGenStep)
        {
            decode->setTensorAddress("gen_step", mDeviceGenStep);
        }
        if (mHasPastLength)
        {
            decode->setTensorAddress("past_length", mDevicePastLength);
        }

        std::vector<void*>* read = &mKVB;
        std::vector<void*>* write = &mKVA;
        int32_t const groupsToGenerate = std::min(activeGroups, mNumGroups);
        for (int32_t j = 1; j < groupsToGenerate; ++j)
        {
            float const* embed = embedding(j - 1, residualCodes[j - 1]);
            CUDA_CHECK(cudaMemcpyAsync(mDeviceEmbeds, embed, bytes, cudaMemcpyHostToDevice, mStream));
            int64_t const actualPast = j + 1;
            setScalar(mDeviceGenStep, j);
            setScalar(mDevicePastLength, actualPast);
            CUDA_CHECK(cudaMemcpyAsync(mDeviceCachePosition, &actualPast, sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
            for (int32_t i = 0; i < mNumLayers; ++i)
            {
                decode->setInputShape(mPastKeyNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, actualPast, mHeadDim});
                decode->setInputShape(mPastValueNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, actualPast, mHeadDim});
                decode->setTensorAddress(mPastKeyNames[i].c_str(), (*read)[2 * i]);
                decode->setTensorAddress(mPastValueNames[i].c_str(), (*read)[2 * i + 1]);
                decode->setTensorAddress(mNewPastKeyNames[i].c_str(), (*write)[2 * i]);
                decode->setTensorAddress(mNewPastValueNames[i].c_str(), (*write)[2 * i + 1]);
            }
            if (!decode->enqueueV3(mStream))
            {
                LOG_ERROR("Special CodePredictor decode failed at group %d", j);
                return false;
            }
            residualCodes[j] = sampleDeviceLogits(j, topK, temperature);
            std::swap(read, write);
        }
        return true;
    }

private:
    struct EngineDeleter
    {
        template <typename T>
        void operator()(T* ptr) const
        {
            delete ptr;
        }
    };

    static size_t trtElementSize(nvinfer1::DataType dtype)
    {
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:
        case nvinfer1::DataType::kBF16: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        default: return 4;
        }
    }

    bool hasTensor(char const* name) const
    {
        for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
        {
            if (std::string(mEngine->getIOTensorName(i)) == name)
            {
                return true;
            }
        }
        return false;
    }

    void loadEmbeddings(std::filesystem::path const& embedPath)
    {
        std::ifstream file(embedPath, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("Failed to open CP embedding table: " + embedPath.string());
        }
        size_t const bytes = static_cast<size_t>(file.tellg());
        size_t const expected = static_cast<size_t>(mNumGroups) * mCodebookSize * mHiddenSize * sizeof(float);
        if (bytes != expected)
        {
            throw std::runtime_error("Unexpected CP embedding table size: " + std::to_string(bytes)
                + ", expected " + std::to_string(expected));
        }
        file.seekg(0);
        mEmbeddings.resize(expected / sizeof(float));
        file.read(reinterpret_cast<char*>(mEmbeddings.data()), static_cast<std::streamsize>(bytes));
    }

    float const* embedding(int32_t group, int32_t token) const
    {
        group = std::clamp(group, 0, mNumGroups - 1);
        token = std::clamp(token, 0, mCodebookSize - 1);
        size_t const offset = (static_cast<size_t>(group) * mCodebookSize + token) * mHiddenSize;
        return mEmbeddings.data() + offset;
    }

    void zeroKV()
    {
        size_t const kvBytes = static_cast<size_t>(mNumHeads) * 32 * mHeadDim * mKVElementSize;
        for (void* p : mKVA)
        {
            CUDA_CHECK(cudaMemsetAsync(p, 0, kvBytes, mStream));
        }
        for (void* p : mKVB)
        {
            CUDA_CHECK(cudaMemsetAsync(p, 0, kvBytes, mStream));
        }
    }

    void setScalar(void* devicePtr, int64_t value)
    {
        CUDA_CHECK(cudaMemcpyAsync(devicePtr, &value, sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    }

    int32_t sampleDeviceLogits(int32_t group, int32_t topK, float temperature)
    {
        std::vector<float> logits(static_cast<size_t>(mCodebookSize));
        size_t const offset = mLogitsName == "logits_all" ? static_cast<size_t>(group) * mCodebookSize : 0;
        if (mLogitsElementSize == sizeof(float))
        {
            CUDA_CHECK(cudaMemcpyAsync(logits.data(), static_cast<char*>(mDeviceLogits) + offset * sizeof(float),
                logits.size() * sizeof(float), cudaMemcpyDeviceToHost, mStream));
        }
        else
        {
            std::vector<uint16_t> raw(logits.size());
            CUDA_CHECK(cudaMemcpyAsync(raw.data(), static_cast<char*>(mDeviceLogits) + offset * mLogitsElementSize,
                raw.size() * mLogitsElementSize, cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            for (size_t i = 0; i < raw.size(); ++i)
            {
                uint32_t bits = mLogitsAreBf16 ? (static_cast<uint32_t>(raw[i]) << 16) : halfToFloatBits(raw[i]);
                std::memcpy(&logits[i], &bits, sizeof(float));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        int32_t const k = std::clamp(topK, 1, mCodebookSize);
        std::vector<std::pair<float, int32_t>> vals(static_cast<size_t>(mCodebookSize));
        for (int32_t i = 0; i < mCodebookSize; ++i)
        {
            vals[i] = {logits[i], i};
        }
        std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
            [](auto const& a, auto const& b) { return a.first > b.first; });
        double const maxVal = vals[0].first;
        std::vector<double> probs(static_cast<size_t>(k));
        double sum = 0.0;
        float const temp = temperature > 1e-6f ? temperature : 0.9f;
        for (int32_t i = 0; i < k; ++i)
        {
            probs[i] = std::exp((static_cast<double>(vals[i].first) - maxVal) / temp);
            sum += probs[i];
        }
        for (auto& p : probs)
        {
            p /= sum;
        }
        std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
        return vals[dist(mRng)].second;
    }
    int32_t mHiddenSize{};
    int32_t mCodebookSize{};
    int32_t mNumLayers{};
    int32_t mNumHeads{};
    int32_t mHeadDim{};
    int32_t mNumGroups{};
    cudaStream_t mStream{};
    std::mt19937 mRng;
    std::vector<float> mEmbeddings;
    std::unique_ptr<nvinfer1::IRuntime, EngineDeleter> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, EngineDeleter> mPrefillContext;
    std::unique_ptr<nvinfer1::IExecutionContext, EngineDeleter> mDecodeContext;
    std::string mLogitsName;
    bool mHasGenStep{false};
    bool mHasPastLength{false};
    bool mLogitsAreBf16{false};
    size_t mKVElementSize{4};
    size_t mLogitsElementSize{4};
    void* mDeviceEmbeds{nullptr};
    void* mDeviceCachePosition{nullptr};
    void* mDeviceGenStep{nullptr};
    void* mDevicePastLength{nullptr};
    void* mDeviceDummyKV{nullptr};
    void* mDeviceLogits{nullptr};
    std::vector<void*> mKVA;
    std::vector<void*> mKVB;
    std::vector<std::string> mPastKeyNames;
    std::vector<std::string> mPastValueNames;
    std::vector<std::string> mNewPastKeyNames;
    std::vector<std::string> mNewPastValueNames;
};

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream)
    : mStream(stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::init", nvtx_colors::YELLOW);
    LOG_INFO("Initializing Qwen3-Omni Talker runner");
    LOG_INFO("  Talker: %s", talkerEngineDir.c_str());
    LOG_INFO("  CodePredictor: %s", codePredictorEngineDir.c_str());
    mUseSpecialCodePredictor = std::getenv("QWEN3_TTS_CP_ENGINE") && std::getenv("QWEN3_TTS_CP_EMBED_FP32");

    // Load tokenizer
    std::filesystem::path const tokenizerPath = tokenizerDir.empty()
        ? std::filesystem::path(talkerEngineDir).parent_path()
        : std::filesystem::path(tokenizerDir);
    LOG_INFO("  Tokenizer: %s", tokenizerPath.string().c_str());
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!mTokenizer->loadFromHF(tokenizerPath))
    {
        throw std::runtime_error("Failed to load tokenizer from: " + tokenizerPath.string());
    }

    if (!validateAndFillConfig(talkerEngineDir))
    {
        throw std::runtime_error("Failed to validate and fill config");
    }

    if (!initializeEngineRunners(talkerEngineDir, codePredictorEngineDir))
    {
        throw std::runtime_error("Failed to initialize engine runners");
    }

    // Setup shared execution context memory for Talker and CodePredictor engines.
    // LLMEngineRunner uses kUSER_MANAGED allocation and requires setContextMemory() before execution.
    {
        int64_t const talkerCtxSize = mTalkerLLMRunner->getRequiredContextMemorySize();
        int64_t const cpCtxSize = mCodePredictorRunner ? mCodePredictorRunner->getRequiredContextMemorySize() : 0;
        int64_t const sharedCtxSize = std::max(talkerCtxSize, cpCtxSize);
        LOG_INFO("Setup shared execution context memory: %zu bytes (talker: %zu, code_predictor: %zu)",
            static_cast<size_t>(sharedCtxSize), static_cast<size_t>(talkerCtxSize), static_cast<size_t>(cpCtxSize));
        mSharedExecContextMemory = rt::Tensor({sharedCtxSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
            "Qwen3OmniTTSRuntime::mSharedExecContextMemory");
        if (!mTalkerLLMRunner->setContextMemory(mSharedExecContextMemory))
        {
            throw std::runtime_error("Failed to set context memory for Talker LLM engine");
        }
        if (mCodePredictorRunner && !mCodePredictorRunner->setContextMemory(mSharedExecContextMemory))
        {
            throw std::runtime_error("Failed to set context memory for CodePredictor engine");
        }
    }

    if (!loadCodePredictorWeights(codePredictorEngineDir))
    {
        throw std::runtime_error("Failed to load CodePredictor weights");
    }

    char const* specialCPEnginePath = std::getenv("QWEN3_TTS_CP_ENGINE");
    char const* specialCPEmbedPath = std::getenv("QWEN3_TTS_CP_EMBED_FP32");
    if (specialCPEnginePath && specialCPEmbedPath)
    {
        mSpecialCodePredictorEngine = std::make_unique<SpecialCodePredictorEngine>(specialCPEnginePath,
            specialCPEmbedPath, mTalkerConfig.codePredictorHiddenSize, mTalkerConfig.codebookSize,
            mCodePredictorConfig.numDecoderLayers, mCodePredictorConfig.numKVHeads,
            mCodePredictorConfig.headDim, talker_constants::kNumRvqLayers, stream);
    }

#ifdef CUTE_DSL_GEMM_ENABLED
    if (!CuteDslGemmRunner::loadKernelModule())
    {
        throw std::runtime_error("Failed to load CuTe DSL GEMM kernel module");
    }
#endif

    if (!allocateBuffer())
    {
        throw std::runtime_error("Failed to allocate buffers");
    }

    if (!loadTalkerWeights(talkerEngineDir, stream))
    {
        throw std::runtime_error("Failed to load Talker weights");
    }

    initializeTTSEmbeddings(stream);

    LOG_INFO("Qwen3-Omni TTS runtime initialized successfully");
}

Qwen3OmniTTSRuntime::~Qwen3OmniTTSRuntime()
{
#ifdef CUTE_DSL_GEMM_ENABLED
    CuteDslGemmRunner::unloadKernelModule();
#endif
}

bool Qwen3OmniTTSRuntime::initializeEngineRunners(
    std::string const& talkerEngineDir, std::string const& codePredictorEngineDir)
{
    // Load Talker LLM engine
    std::filesystem::path talkerEnginePath = std::filesystem::path(talkerEngineDir) / "llm.engine";
    std::filesystem::path talkerConfigPath = std::filesystem::path(talkerEngineDir) / "config.json";

    LOG_INFO("Loading Talker LLM engine from: %s", talkerEnginePath.string().c_str());

    try
    {
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mTalkerLLMRunner = std::make_unique<LLMEngineRunner>(talkerEnginePath, talkerConfigPath, emptyLoraMap, mStream);
        mTalkerLLMConfig = mTalkerLLMRunner->getEngineConfig();

        LOG_INFO("Talker LLM engine loaded: vocabSize=%d, hiddenSize=%d", mTalkerLLMConfig.vocabSize,
            mTalkerLLMConfig.hiddenSize);
        auto talkerKVType = mTalkerLLMRunner->getCacheManager().getKVCacheManager().getConfig().kvCacheType;
        LOG_INFO("Talker KV cache dtype: %s",
            talkerKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                      : (talkerKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load Talker LLM engine: %s", e.what());
        return false;
    }

    if (mUseSpecialCodePredictor)
    {
        LOG_INFO("Special CodePredictor requested; loading CodePredictor config only.");
        return loadCodePredictorConfig(codePredictorEngineDir);
    }

    // Load CodePredictor engine from separate directory
    std::filesystem::path codePredictorEnginePath = std::filesystem::path(codePredictorEngineDir) / "llm.engine";
    std::filesystem::path codePredictorConfigPath = std::filesystem::path(codePredictorEngineDir) / "config.json";

    LOG_INFO("Loading CodePredictor engine from: %s", codePredictorEnginePath.string().c_str());

    try
    {
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mCodePredictorRunner = std::make_unique<LLMEngineRunner>(
            codePredictorEnginePath, codePredictorConfigPath, emptyLoraMap, mStream);

        // NOTE: CodePredictor ONNX now outputs FP32 logits directly (lm_head + cast in ONNX),
        // so standard logits shape validation applies.

        mCodePredictorConfig = mCodePredictorRunner->getEngineConfig();

        // Now read CodePredictor dimensions from loaded config
        mTalkerConfig.codePredictorHiddenSize = mCodePredictorConfig.hiddenSize;
        // NOTE: config.vocab_size == hidden_size (for engine compatibility since output is last_hidden)
        //       Real codebook_size is inferred from lm_head weight shape later

        LOG_INFO("CodePredictor engine loaded: vocabSize=%d, hiddenSize=%d, numLayers=%d",
            mCodePredictorConfig.vocabSize, mCodePredictorConfig.hiddenSize, mCodePredictorConfig.numDecoderLayers);
        auto cpKVType = mCodePredictorRunner->getCacheManager().getKVCacheManager().getConfig().kvCacheType;
        LOG_INFO("CodePredictor KV cache dtype: %s",
            cpKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                  : (cpKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load CodePredictor engine: %s", e.what());
        return false;
    }

    return true;
}

bool Qwen3OmniTTSRuntime::loadCodePredictorConfig(std::string const& codePredictorEngineDir)
{
    std::filesystem::path codePredictorConfigPath = std::filesystem::path(codePredictorEngineDir) / "config.json";
    std::ifstream configStream(codePredictorConfigPath);
    if (!configStream.is_open())
    {
        LOG_ERROR("Failed to open CodePredictor config: %s", codePredictorConfigPath.string().c_str());
        return false;
    }
    Json configJson;
    try
    {
        configJson = Json::parse(configStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse CodePredictor config: %s", e.what());
        return false;
    }

    mCodePredictorConfig.numDecoderLayers = configJson.value("num_hidden_layers", 5);
    mCodePredictorConfig.numKVHeads = configJson.value("num_key_value_heads", 8);
    mCodePredictorConfig.headDim = configJson.value("head_dim", 128);
    mCodePredictorConfig.hiddenSize = configJson.value("hidden_size", 1024);
    mCodePredictorConfig.vocabSize = configJson.value("vocab_size", 2048);
    mCodePredictorConfig.maxSupportedInputLength = configJson.value("max_input_len", 16);
    mCodePredictorConfig.maxKVCacheCapacity = configJson.value("max_kv_cache_capacity", 128);
    if (configJson.contains("builder_config"))
    {
        auto const& builderConfig = configJson["builder_config"];
        mCodePredictorConfig.maxSupportedInputLength
            = builderConfig.value("max_input_len", mCodePredictorConfig.maxSupportedInputLength);
        mCodePredictorConfig.maxKVCacheCapacity
            = builderConfig.value("max_kv_cache_capacity", mCodePredictorConfig.maxKVCacheCapacity);
    }
    mTalkerConfig.codePredictorHiddenSize = mCodePredictorConfig.hiddenSize;
    LOG_INFO("CodePredictor config loaded without generic engine: vocabSize=%d, hiddenSize=%d, numLayers=%d",
        mCodePredictorConfig.vocabSize, mCodePredictorConfig.hiddenSize, mCodePredictorConfig.numDecoderLayers);
    return true;
}

bool Qwen3OmniTTSRuntime::validateAndFillConfig(std::string const& talkerEngineDir)
{
    // Load config.json from talker directory
    std::filesystem::path configPath = std::filesystem::path(talkerEngineDir) / "config.json";
    LOG_INFO("Loading Talker config from: %s", configPath.string().c_str());

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        return false;
    }

    Json configJson;
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config: %s", e.what());
        return false;
    }

    // Model dimensions
    mTalkerConfig.thinkerHiddenSize = configJson.value("thinker_hidden_size", 2048);
    mTalkerConfig.talkerHiddenSize = configJson["hidden_size"].get<int32_t>();
    mTalkerConfig.talkerVocabSize = configJson["vocab_size"].get<int32_t>();

    // Runtime parameters
    mTalkerConfig.maxSeqLen = configJson.value("max_position_embeddings", 8192);

    // Validate dimensions with reasonable limits
    constexpr int32_t kMaxReasonableVocabSize = 200000;
    constexpr int32_t kMaxReasonableHiddenSize = 16384;
    constexpr int32_t kMaxReasonableSeqLen = 131072;

    check::check(mTalkerConfig.talkerVocabSize > 0 && mTalkerConfig.talkerVocabSize < kMaxReasonableVocabSize,
        "Invalid talker vocab size: " + std::to_string(mTalkerConfig.talkerVocabSize));
    check::check(mTalkerConfig.thinkerHiddenSize > 0 && mTalkerConfig.thinkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid thinker hidden size: " + std::to_string(mTalkerConfig.thinkerHiddenSize));
    check::check(mTalkerConfig.talkerHiddenSize > 0 && mTalkerConfig.talkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid talker hidden size: " + std::to_string(mTalkerConfig.talkerHiddenSize));
    check::check(mTalkerConfig.maxSeqLen > 0 && mTalkerConfig.maxSeqLen < kMaxReasonableSeqLen,
        "Invalid max sequence length: " + std::to_string(mTalkerConfig.maxSeqLen));

    // TTS special tokens (from thinker vocab)
    mTalkerConfig.ttsPadTokenId = configJson.value("tts_pad_token_id", 151671);
    mTalkerConfig.ttsBosTokenId = configJson.value("tts_bos_token_id", 151672);
    mTalkerConfig.ttsEosTokenId = configJson.value("tts_eos_token_id", 151673);

    // Codec special tokens (from talker vocab)
    mTalkerConfig.codecNothinkId = configJson["codec_nothink_id"].get<int32_t>();
    mTalkerConfig.codecThinkId = configJson.value("codec_think_id", mTalkerConfig.codecNothinkId);
    mTalkerConfig.codecThinkBosId = configJson["codec_think_bos_id"].get<int32_t>();
    mTalkerConfig.codecThinkEosId = configJson["codec_think_eos_id"].get<int32_t>();
    mTalkerConfig.codecPadId = configJson["codec_pad_id"].get<int32_t>();
    mTalkerConfig.codecBosId = configJson["codec_bos_id"].get<int32_t>();
    // Support both codec_eos_token_id (original) and codec_eos_id (legacy) for backward compatibility
    if (configJson.contains("codec_eos_token_id"))
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_token_id"].get<int32_t>();
    }
    else
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_id"].get<int32_t>();
    }

    // Speaker ID configuration
    mTalkerConfig.defaultSpeakerId = configJson.value("default_speaker_id", 2301);

    // Parse codec_language_id
    if (configJson.contains("codec_language_id") && configJson["codec_language_id"].is_object())
    {
        for (auto const& [lang_name, lang_id] : configJson["codec_language_id"].items())
        {
            mTalkerConfig.languageIdMap[lang_name] = lang_id.get<int32_t>();
        }
        LOG_INFO("Loaded %zu language IDs", mTalkerConfig.languageIdMap.size());
    }

    // Load speaker ID mapping if available
    if (configJson.contains("speaker_id") && configJson["speaker_id"].is_object())
    {
        for (auto const& [speaker_name, speaker_id] : configJson["speaker_id"].items())
        {
            mSpeakerIdMap[speaker_name] = speaker_id.get<int32_t>();
        }
        LOG_INFO("Loaded %zu speaker IDs from config", mSpeakerIdMap.size());

        // Log available speakers
        if (!mSpeakerIdMap.empty())
        {
            std::string speakerList;
            for (auto const& [name, id] : mSpeakerIdMap)
            {
                if (!speakerList.empty())
                {
                    speakerList += ", ";
                }
                speakerList += name + ":" + std::to_string(id);
            }
            LOG_DEBUG("Available speakers: %s", speakerList.c_str());
        }
    }

    LOG_INFO("Talker config: vocabSize=%d, hiddenSize=%d, thinkerHiddenSize=%d, defaultSpeaker=%d",
        mTalkerConfig.talkerVocabSize, mTalkerConfig.talkerHiddenSize, mTalkerConfig.thinkerHiddenSize,
        mTalkerConfig.defaultSpeakerId);
    LOG_DEBUG("TTS tokens: pad=%d, bos=%d, eos=%d", mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId,
        mTalkerConfig.ttsEosTokenId);
    LOG_DEBUG("Codec tokens: skipThink=%d, thinkBos=%d, thinkEos=%d, pad=%d, bos=%d, eos=%d",
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, mTalkerConfig.codecEosId);

    return true;
}

bool Qwen3OmniTTSRuntime::loadCodePredictorWeights(std::string const& codePredictorEngineDir)
{
    LOG_INFO("Loading %d CodePredictor lm_head weights", kNumRvqLayers);
    mCodePredictorLmHeadWeights.resize(kNumRvqLayers);
    {
        std::filesystem::path const lmHeadPath = std::filesystem::path(codePredictorEngineDir) / "lm_heads.safetensors";
        std::vector<rt::Tensor> allLmHeadTensors;
        if (!safetensors::loadSafetensors(lmHeadPath, allLmHeadTensors, mStream))
        {
            LOG_ERROR("Failed to load lm_heads.safetensors from: %s", lmHeadPath.string().c_str());
            return false;
        }
        for (int32_t i = 0; i < kNumRvqLayers; ++i)
        {
            std::string const weightKey = "lm_head_" + std::to_string(i) + ".weight";
            auto it = std::find_if(allLmHeadTensors.begin(), allLmHeadTensors.end(),
                [&weightKey](rt::Tensor const& t) { return t.getName() == weightKey; });
            if (it == allLmHeadTensors.end())
            {
                LOG_ERROR("Missing key '%s' in lm_heads.safetensors", weightKey.c_str());
                return false;
            }
            if (it->getShape().getNumDims() != 2)
            {
                LOG_ERROR("%s should be 2D [vocabSize, hiddenSize]", weightKey.c_str());
                return false;
            }
            LOG_DEBUG("Loaded %s [%d, %d]", weightKey.c_str(), it->getShape()[0], it->getShape()[1]);
            mCodePredictorLmHeadWeights[i] = std::move(*it);
        }
    }

    // Set codebookSize from lm_head weight shape [vocab_size, hidden_size]
    mTalkerConfig.codebookSize = static_cast<int32_t>(mCodePredictorLmHeadWeights[0].getShape()[0]);
    LOG_INFO("Loaded %d CodePredictor lm_head weights, codebookSize=%d", kNumRvqLayers, mTalkerConfig.codebookSize);

    // Load small_to_mtp_projection: projects Talker hidden (2048) → CodePredictor input (1024)
    {
        std::filesystem::path const projPath
            = std::filesystem::path(codePredictorEngineDir) / "small_to_mtp_projection.safetensors";
        std::vector<rt::Tensor> projTensors;
        if (!safetensors::loadSafetensors(projPath, projTensors, mStream))
        {
            LOG_ERROR("Failed to load small_to_mtp_projection from: %s", projPath.string().c_str());
            return false;
        }
        bool foundWeight = false, foundBias = false;
        for (auto& t : projTensors)
        {
            if (t.getName() == "weight")
            {
                mSmallToMtpWeight = std::move(t);
                foundWeight = true;
            }
            else if (t.getName() == "bias")
            {
                mSmallToMtpBias = std::move(t);
                foundBias = true;
            }
        }
        if (!foundWeight || !foundBias)
        {
            LOG_ERROR("Missing 'weight' or 'bias' in small_to_mtp_projection.safetensors");
            return false;
        }
        LOG_INFO("Loaded small_to_mtp_projection: weight=%ldx%ld, bias=%ld", mSmallToMtpWeight.getShape()[0],
            mSmallToMtpWeight.getShape()[1], mSmallToMtpBias.getShape()[0]);
    }

    // Load CodePredictor embedding tables (all 15 in codec_embeddings.safetensors)
    LOG_INFO("Loading %d CodePredictor embedding tables", kNumRvqLayers);
    mCodePredictorEmbeddingTables.resize(kNumRvqLayers);
    {
        std::filesystem::path const embedPath
            = std::filesystem::path(codePredictorEngineDir) / "codec_embeddings.safetensors";
        std::vector<rt::Tensor> allEmbedTensors;
        if (!safetensors::loadSafetensors(embedPath, allEmbedTensors, mStream))
        {
            LOG_ERROR("Failed to load codec_embeddings.safetensors from: %s", embedPath.string().c_str());
            return false;
        }
        for (int32_t i = 0; i < kNumRvqLayers; ++i)
        {
            std::string const key = "embedding_" + std::to_string(i);
            auto it = std::find_if(allEmbedTensors.begin(), allEmbedTensors.end(),
                [&key](rt::Tensor const& t) { return t.getName() == key; });
            if (it == allEmbedTensors.end())
            {
                LOG_ERROR("Missing key '%s' in codec_embeddings.safetensors", key.c_str());
                return false;
            }
            if (it->getShape().getNumDims() != 2)
            {
                LOG_ERROR("%s should be 2D [codebookSize, hiddenSize]", key.c_str());
                return false;
            }
            mCodePredictorEmbeddingTables[i] = std::move(*it);
        }
    }
    LOG_INFO("Loaded %d CodePredictor embedding tables", kNumRvqLayers);

    return true;
}

bool Qwen3OmniTTSRuntime::allocateBuffer()
{
    LOG_INFO("Allocating Qwen3-Omni TTS Runtime inference workspace buffers...");

    int64_t const maxSeqLen = mTalkerConfig.maxSeqLen;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;

    try
    {
        // Text embedding output and token ID upload buffers
        mThinkerEmbedBuffer
            = rt::Tensor({maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        mGpuTokenIdsBuffer = rt::Tensor({1, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        // MLP workspace for text_projection
        mMLPWorkspace = rt::Tensor({maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Projected buffer (MLP output)
        mProjectedBuffer = rt::Tensor({maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Final talker input embeddings
        mTalkerInputEmbeds = rt::Tensor({maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Talker LLM workspace
        mTalkerLogits
            = rt::Tensor({1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        mTalkerSelectedIndices = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        mHostSelectedTokenIds = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mHostTalkerContextLength = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

        // CodePredictor workspace
        mCodePredictorLogits
            = rt::Tensor({1, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        // Per-lm_head logits buffers for CUDA graph capture: each graph needs a distinct output
        // address so that LLMEngineRunner's decodingKey differentiates the 15 captured graphs.
        mCodePredictorLogitsPerHead.resize(kNumRvqLayers);
        for (int32_t i = 0; i < kNumRvqLayers; ++i)
        {
            mCodePredictorLogitsPerHead[i]
                = rt::Tensor({1, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        }

        mCodePredictorPrefillInput = rt::Tensor(
            {1, 2, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        mCodePredictorCodecIds = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        // mCodePredictorCodecEmbed: projected (1024-dim) embed fed to CodePredictor engine / CUDA graph
        mCodePredictorCodecEmbed = rt::Tensor(
            {1, 1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        // mRawCodecEmbed: raw (2048-dim) embed from Talker/codec embedding tables, before small_to_mtp_projection
        mRawCodecEmbed
            = rt::Tensor({1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        // mSmallToMtpProjectedHidden: projected (1024-dim) talker hidden state, for prefill input slot 0
        mSmallToMtpProjectedHidden
            = rt::Tensor({1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        // mResidualEmbedBuffer: residual output in Talker space (2048-dim), feeds back to Talker decoder
        mResidualEmbedBuffer
            = rt::Tensor({1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        mCodePredictorSelectedIndices = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        mHostSelectedCodeIds = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mHostCodePredictorContextLength = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

        // Talker decoding buffers (avoid temporary tensor creation)
        mTalkerDecodingIds = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerDecodingIds");
        mTalkerDecodingEmbed = rt::Tensor({1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerDecodingEmbed");

        // KVCache reset helper (avoid temporary tensor creation in handleAudioGeneration)
        mHostReuseKVCacheLengths
            = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostReuseKVCacheLengths");

        // Sampling workspace (calculate max workspace size, same as LLM pattern)
        // Use conservative sampling parameters to reserve max possible workspace
        int32_t const defaultTopK{0}; // TopK=0 means no top-K filtering (max workspace)
        float const defaultTopP{0.9F};
        trt_edgellm::SamplingParams samplingParams(1, mTalkerConfig.talkerVocabSize, 1.0f, defaultTopK, defaultTopP);
        int64_t const samplingWorkspaceSize
            = trt_edgellm::getTopKtopPSamplingWorkspaceSize(1, mTalkerConfig.talkerVocabSize, samplingParams);
        mSamplingWorkspace = rt::Tensor(
            {samplingWorkspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "mSamplingWorkspace");
        mSeenCodecTokensBuf = rt::Tensor({mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "mSeenCodecTokensBuf");

        // Hidden states buffers for generation loop
        mTalkerHiddenStatesBuffer = rt::Tensor({1, maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerHiddenStatesBuffer");
        // CodePredictor uses seqLen=16 at most (not maxSeqLen), so allocate smaller buffer
        mCodePredictorHiddenStatesBuffer = rt::Tensor({1, 16, mTalkerConfig.codePredictorHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodePredictorHiddenStatesBuffer");

        // Talker last hidden state (extracted from mTalkerHiddenStatesBuffer)
        mTalkerLastHidden
            = rt::Tensor({1, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mTalkerLastHidden");

        // Residual computation buffers: stored in Talker space (2048-dim) for residual connection
        mCodecHiddensBuffer = rt::Tensor({1, 16, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodecHiddensBuffer");

        LOG_INFO("Talker buffers allocated successfully");
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate Talker buffers: %s", e.what());
        return false;
    }
}

bool Qwen3OmniTTSRuntime::loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::loadTalkerWeights", nvtx_colors::YELLOW);

    // Load text_projection weights
    std::filesystem::path const textProjPath = std::filesystem::path(weightsDir) / "text_projection.safetensors";
    std::vector<rt::Tensor> textTensors;
    if (!safetensors::loadSafetensors(textProjPath, textTensors, stream))
    {
        LOG_ERROR("Failed to load text_projection from: %s", textProjPath.string().c_str());
        return false;
    }
    if (!extractMLPWeightsFromTensors(
            textTensors, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, "text_projection"))
    {
        return false;
    }

    // Load text embedding table (thinker vocab, for standalone TTS and TTS special token projection)
    std::filesystem::path const textEmbedPath = std::filesystem::path(weightsDir) / "text_embedding.safetensors";
    std::vector<rt::Tensor> textEmbedTensors;
    if (!safetensors::loadSafetensors(textEmbedPath, textEmbedTensors, stream))
    {
        LOG_ERROR("Failed to load text_embedding.safetensors from: %s", textEmbedPath.string().c_str());
        return false;
    }
    check::check(!textEmbedTensors.empty(), "text_embedding.safetensors is empty");
    check::check(
        textEmbedTensors[0].getShape().getNumDims() == 2, "text_embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTextEmbeddingTable = std::move(textEmbedTensors[0]);
    LOG_INFO("Text embedding table loaded: [%lld, %lld]", mTextEmbeddingTable.getShape()[0],
        mTextEmbeddingTable.getShape()[1]);

    // Load Talker embedding table
    std::filesystem::path const talkerEmbedPath = std::filesystem::path(weightsDir) / "embedding.safetensors";
    std::vector<rt::Tensor> talkerEmbedTensors;
    if (!safetensors::loadSafetensors(talkerEmbedPath, talkerEmbedTensors, stream))
    {
        LOG_ERROR("Failed to load Talker embedding from: %s", talkerEmbedPath.string().c_str());
        return false;
    }
    check::check(talkerEmbedTensors.size() == 1, "Talker embedding.safetensors should contain exactly one tensor");
    check::check(talkerEmbedTensors[0].getShape().getNumDims() == 2,
        "Talker embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTalkerEmbeddingTable = std::move(talkerEmbedTensors[0]);
    LOG_INFO("Talker embedding table loaded: [%lld, %lld]", mTalkerEmbeddingTable.getShape()[0],
        mTalkerEmbeddingTable.getShape()[1]);

    LOG_INFO("Talker weights loaded successfully");
    return true;
}

void Qwen3OmniTTSRuntime::initializeTTSEmbeddings(cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::initializeTTSEmbeddings", nvtx_colors::YELLOW);

    auto const shape = mTextEmbeddingTable.getShape();
    if (shape.getNumDims() != 2)
    {
        throw std::runtime_error("Text embedding table must be 2D, got " + std::to_string(shape.getNumDims()) + "D");
    }

    int64_t const vocabSize = shape[0];
    int64_t const thinkerHiddenSize = shape[1];

    if (mTalkerConfig.ttsPadTokenId >= vocabSize || mTalkerConfig.ttsBosTokenId >= vocabSize
        || mTalkerConfig.ttsEosTokenId >= vocabSize)
    {
        throw std::runtime_error("TTS token IDs out of vocab range: pad=" + std::to_string(mTalkerConfig.ttsPadTokenId)
            + ", bos=" + std::to_string(mTalkerConfig.ttsBosTokenId)
            + ", eos=" + std::to_string(mTalkerConfig.ttsEosTokenId) + ", vocabSize=" + std::to_string(vocabSize));
    }

    constexpr int32_t kNumTtsTokens = 3;
    std::vector<int32_t> const hostTtsIds
        = {mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId, mTalkerConfig.ttsEosTokenId};

    rt::Tensor ttsIds({1, kNumTtsTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor ttsRaw({1, kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor ttsProjected(
        {kNumTtsTokens, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor workspace({kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpyAsync(
        ttsIds.rawPointer(), hostTtsIds.data(), kNumTtsTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::embeddingLookup(ttsIds, mTextEmbeddingTable, std::nullopt, ttsRaw, stream);
    // Reshape from [1, 3, hidden] to [3, hidden] for MLP (expects 2D input)
    check::check(ttsRaw.reshape({kNumTtsTokens, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(
        ttsRaw, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, ttsProjected, workspace, stream);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    mTtsPadEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mTtsBosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mTtsEosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    __half* const projectedPtr = static_cast<__half*>(ttsProjected.rawPointer());
    size_t const embedSize = hiddenSize * sizeof(__half);

    CUDA_CHECK(cudaMemcpyAsync(
        mTtsPadEmbed.rawPointer(), projectedPtr + 0 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsBosEmbed.rawPointer(), projectedPtr + 1 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsEosEmbed.rawPointer(), projectedPtr + 2 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));

    LOG_INFO("TTS embeddings initialized");
}

bool Qwen3OmniTTSRuntime::projectToTalkerInput(
    rt::Tensor const& thinkerEmbed, int32_t langId, rt::Tensor& output, int64_t& outputSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = thinkerEmbed.getShape()[0];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;

    // N = raw text body tokens after the fixed assistant role prefix.
    // The official Qwen3-TTS prefill uses role=[151644, 77091, 198] + raw text,
    // not a full chat-template sequence.
    int64_t const N = seqLen - kAssistantPrefixLen;
    if (N <= 0)
    {
        LOG_ERROR("projectToTalkerInput: no text body tokens after assistant prefix");
        return false;
    }
    // Fixed 9-row prefill matching old working runner (body[0] only, rest in trailing)
    outputSeqLen = 9;

    // Store trailing text length for residual addend (body[1:] + tts_eos)
    mTrailingTextLen = static_cast<int32_t>(N);

    // Project all tokens via text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(thinkerEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, mProjectedBuffer,
        mMLPWorkspace, stream);

    // Fused kernel: build complete non-streaming prefill buffer
    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    kernel::invokeAssistantPreamble(mProjectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
        mTalkerConfig.codecThinkId, mTalkerConfig.codecThinkBosId,
        langId, mTalkerConfig.codecThinkEosId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId,
        output, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerPrefillStep(
    rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerPrefillStep", nvtx_colors::PURPLE);

    // Reset Talker KV cache for new sequence
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    reuseData[0] = 0; // No KV cache reuse
    auto& talkerCacheManager = mTalkerLLMRunner->getCacheManager();
    talkerCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    auto& talkerKVManager = talkerCacheManager.getKVCacheManager();
    for (int32_t i = 0; i < talkerKVManager.numLayers(); ++i)
    {
        rt::Tensor& layerKV = talkerKVManager.getCombinedKVCache(i);
        CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
    }

    auto inputShape = inputEmbeds.getShape();
    if (inputShape.getNumDims() != 3)
    {
        LOG_ERROR("executeTalkerPrefillStep: Input must be 3D [batchSize, seqLen, hiddenSize], got %dD",
            inputShape.getNumDims());
        return false;
    }

    int64_t const batchSize = inputEmbeds.getTRTDims().d[0];
    int64_t const seqLen = inputEmbeds.getTRTDims().d[1];

    if (batchSize != 1)
    {
        LOG_ERROR("executeTalkerPrefillStep: Only batchSize=1 supported, got %ld", batchSize);
        return false;
    }

    // Prepare context length (CPU tensor)
    int32_t* hostContextLength = mHostTalkerContextLength.dataPointer<int32_t>();
    hostContextLength[0] = static_cast<int32_t>(seqLen);

    // Execute prefill
    rt::OptionalInputTensors emptyDeepstack{};
    return mTalkerLLMRunner->executePrefillStep(inputEmbeds, mHostTalkerContextLength, emptyDeepstack, outputLogits,
        rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream);
}

bool Qwen3OmniTTSRuntime::executeCodePredictorPrefillStep(rt::Tensor const& codecTokenEmbeds, int32_t generationStep,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorPrefillStep", nvtx_colors::ORANGE);

    // Reset CodePredictor KV cache for new frame (each frame is independent)
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    reuseData[0] = 0; // No KV cache reuse
    auto& cpCacheManager = mCodePredictorRunner->getCacheManager();
    cpCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    auto& cpKVManager = cpCacheManager.getKVCacheManager();
    for (int32_t i = 0; i < cpKVManager.numLayers(); ++i)
    {
        rt::Tensor& layerKV = cpKVManager.getCombinedKVCache(i);
        CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
    }

    int32_t* const hostContextLength = mHostCodePredictorContextLength.dataPointer<int32_t>();
    hostContextLength[0] = kCodePredictorPrefillSeqLen;

    int32_t const lmHeadIdx = std::min(generationStep, kNumRvqLayers - 1);
    if (!mCodePredictorRunner->setLMHeadWeights("lm_head_weight", mCodePredictorLmHeadWeights[lmHeadIdx]))
    {
        LOG_ERROR("Failed to bind lm_head_weight[%d]", lmHeadIdx);
        return false;
    }

    // Execute prefill - engine outputs logits directly (with lm_head applied)
    rt::OptionalInputTensors emptyDeepstack{};
    if (!mCodePredictorRunner->executePrefillStep(codecTokenEmbeds, mHostCodePredictorContextLength, emptyDeepstack,
            outputLogits, rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
    {
        LOG_ERROR("CodePredictor prefill step failed");
        return false;
    }

    return true;
}

bool Qwen3OmniTTSRuntime::executeCodePredictorDecodingStep(int32_t tokenId, int32_t embeddingTableIndex,
    int32_t generationStep, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorDecodingStep", nvtx_colors::ORANGE);

    CUDA_CHECK(cudaMemcpyAsync(
        mCodePredictorCodecIds.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int32_t const embedIdx = std::min(embeddingTableIndex, kNumRvqLayers - 1);
    // Lookup into mRawCodecEmbed (talkerHiddenSize=2048) — codec embedding tables are in Talker's space
    check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(
        mCodePredictorCodecIds, mCodePredictorEmbeddingTables[embedIdx], std::nullopt, mRawCodecEmbed, stream);

    // Save raw (2048-dim) embedding to mCodecHiddensBuffer for residual connection
    // Position mapping: generationStep 1->pos 1, 2->pos 2, ..., 14->pos 14
    if (generationStep >= 1 && generationStep <= 14)
    {
        int64_t const H = mTalkerConfig.talkerHiddenSize;
        __half* dst = static_cast<__half*>(mCodecHiddensBuffer.rawPointer()) + generationStep * H;
        CUDA_CHECK(
            cudaMemcpyAsync(dst, mRawCodecEmbed.rawPointer(), H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    // Project/copy mRawCodecEmbed to CodePredictor hidden size.
    check::check(mRawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecEmbed.rawPointer(), mRawCodecEmbed.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(mRawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias, mCodePredictorCodecEmbed, stream);
    }

    int32_t const lmHeadIdx = std::min(generationStep, kNumRvqLayers - 1);

    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");

    if (mCodePredictorGraphsCaptured)
    {
        // Graph path: lm_head_weight addresses were bound during capture and remain unchanged,
        // so setLMHeadWeights is unnecessary. Each graph is keyed by its per-head output buffer.
        if (!mCodePredictorRunner->executeVanillaDecodingStep(mCodePredictorCodecEmbed,
                mCodePredictorLogitsPerHead[lmHeadIdx], rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
        {
            LOG_ERROR("CodePredictor decoding step failed (graph path)");
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(outputLogits.rawPointer(), mCodePredictorLogitsPerHead[lmHeadIdx].rawPointer(),
            outputLogits.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        // Non-graph path: must bind lm_head_weight before each enqueueV3
        if (!mCodePredictorRunner->setLMHeadWeights("lm_head_weight", mCodePredictorLmHeadWeights[lmHeadIdx]))
        {
            LOG_ERROR("Failed to bind lm_head_weight[%d]", lmHeadIdx);
            return false;
        }
        if (!mCodePredictorRunner->executeVanillaDecodingStep(
                mCodePredictorCodecEmbed, outputLogits, rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
        {
            LOG_ERROR("CodePredictor decoding step failed");
            return false;
        }
    }

    return true;
}

// ========== CUDA Graph Capture ==========

bool Qwen3OmniTTSRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    std::string const emptyLoraWeightsName = "";

    // Talker: same pattern as Thinker (LLMInferenceRuntime::captureDecodingCUDAGraph)
    check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    bool captureStatus = mTalkerLLMRunner->captureVanillaDecodingCudaGraph(mResidualEmbedBuffer, mTalkerLogits,
        emptyLoraWeightsName, stream, rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)});

    // CodePredictor: 15 graphs, one per lm_head_weight.
    // Each graph uses a distinct output logits buffer so that the decodingKey (which includes
    // the output address) naturally produces a unique key per graph.
    if (mUseSpecialCodePredictor)
    {
        mCodePredictorGraphsCaptured = false;
        LOG_INFO("Skipping generic CodePredictor CUDA graph capture; special Qwen3-TTS CP engine is enabled.");
        return captureStatus;
    }

    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
        "Tensor reshape failed");
    rt::OptionalOutputTensor cpHiddenOpt{std::ref(mCodePredictorHiddenStatesBuffer)};

    for (int32_t i = 0; i < kNumRvqLayers; ++i)
    {
        if (!mCodePredictorRunner->setLMHeadWeights("lm_head_weight", mCodePredictorLmHeadWeights[i]))
        {
            LOG_ERROR("Failed to bind lm_head_weight[%d] for CUDA graph capture", i);
            captureStatus = false;
            continue;
        }
        captureStatus &= mCodePredictorRunner->captureVanillaDecodingCudaGraph(
            mCodePredictorCodecEmbed, mCodePredictorLogitsPerHead[i], emptyLoraWeightsName, stream, cpHiddenOpt);
    }

    mCodePredictorGraphsCaptured = captureStatus;

    if (captureStatus)
    {
        LOG_INFO("Successfully captured decoding CUDA graphs for Talker and all CodePredictor lm_heads.");
    }
    else
    {
        LOG_WARNING("Failed to capture some decoding CUDA graphs. Will use fallback engine execution.");
    }

    return captureStatus;
}

// ========== Audio Generation API ==========

bool Qwen3OmniTTSRuntime::prepareTalkerInput(std::vector<int32_t> const& textTokenIds,
    TalkerGenerationRequest const& request, int64_t& outSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    if (seqLen == 0)
    {
        LOG_ERROR("prepareTalkerInput: empty token ID list");
        return false;
    }
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    check::check(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data(), seqLen * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    check::check(mThinkerEmbedBuffer.reshape({1, seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, mThinkerEmbedBuffer, stream);
    check::check(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

    // Determine language ID for codec embedding (default: Chinese 2055)
    int32_t langId = 2055;
    if (!request.language.empty())
    {
        auto it = mTalkerConfig.languageIdMap.find(request.language);
        if (it != mTalkerConfig.languageIdMap.end())
        {
            langId = it->second;
        }
    }

    // MLP projection: thinker embed → talker input embeds (9-row prefill)
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!projectToTalkerInput(mThinkerEmbedBuffer, langId, mTalkerInputEmbeds, outSeqLen, stream))
    {
        LOG_ERROR("MLP projection failed");
        return false;
    }

    // Reshape buffers to 3D [1, seqLen, H] for Talker LLM input
    check::check(mTalkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(
        mTalkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGeneration(
    TalkerGenerationRequest const& request, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGeneration", nvtx_colors::PURPLE);
    LOG_INFO("Starting audio generation for request with %zu messages", request.messages.size());

    // Clear response
    response.rvqCodes.clear();
    response.numFrames = 0;
    response.success = false;

    // Talker/CodePredictor sampling: use dedicated parameters (not shared with Thinker).
    // PyTorch defaults: do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05
    float const talkerTemperature = (request.talkerTemperature > 0) ? request.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (request.talkerTopK > 0) ? request.talkerTopK : 50;
    float const talkerTopP = (request.talkerTopP > 0) ? request.talkerTopP : 1.0f;
    float const repetitionPenalty = request.repetitionPenalty;

    SamplingParams talkerSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);

    // Suppress special codec tokens [vocabSize-1024, vocabSize) except codec_eos, and apply
    // repetition penalty (PyTorch default: 1.05) to previously generated tokens.
    int32_t const suppressStart = mTalkerConfig.talkerVocabSize - 1024;
    int32_t const suppressEnd = mTalkerConfig.talkerVocabSize;
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // Repetition-penalty state: seenTokenSet deduplicates so each unique token is penalised
    // exactly once. trackSeenToken appends new tokens to mSeenCodecTokensBuf; adjustTalkerLogits
    // suppresses special tokens and applies the penalty before each sampling step.
    int32_t numSeenTokens = 0;
    std::unordered_set<int32_t> seenTokenSet;
    auto adjustTalkerLogits = [&](cudaStream_t s) {
        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, suppressStart, suppressEnd, codecEosId,
            numSeenTokens, repetitionPenalty, s);
    };
    auto trackSeenToken = [&](int32_t token, cudaStream_t s) {
        if (seenTokenSet.insert(token).second) // only append if token is new
        {
            CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
                mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, s));
            ++numSeenTokens;
        }
    };

    // Prepare host reuse lengths tensor (all zeros for new sequence)

    // Build the official Qwen3-TTS text sequence:
    //   [<|im_start|>, assistant, "\n"] + raw text tokens
    // EdgeLLM's generic chat-template path formats the user message directly, which
    // does not match the Talker prefill layout expected by Qwen3-TTS.
    std::string rawText;
    for (auto const& message : request.messages)
    {
        for (auto const& content : message.contents)
        {
            if (content.type.empty() || content.type == "text")
            {
                rawText += content.content;
            }
        }
    }
    if (rawText.empty())
    {
        LOG_ERROR("TTS request has no text content");
        return false;
    }

    std::vector<int32_t> textTokenIds{151644, 77091, 198};
    std::vector<int32_t> bodyTokenIds = mTokenizer->encode(rawText);
    textTokenIds.insert(textTokenIds.end(), bodyTokenIds.begin(), bodyTokenIds.end());

    TalkerGenerationRequest effectiveRequest = request;
    if (effectiveRequest.language.empty())
    {
        effectiveRequest.language = "english";
        for (char const ch : rawText)
        {
            unsigned char const c = static_cast<unsigned char>(ch);
            if (c >= 0x80)
            {
                effectiveRequest.language = "chinese";
                break;
            }
        }
    }

    {
        std::ostringstream oss;
        for (size_t i = 0; i < bodyTokenIds.size(); ++i)
        {
            if (i)
            {
                oss << ",";
            }
            oss << bodyTokenIds[i];
        }
        LOG_INFO("TTS debug rawText='%s' bodyTokenIds=[%s] effectiveLanguage=%s", rawText.c_str(),
            oss.str().c_str(), effectiveRequest.language.c_str());
    }

    std::vector<std::vector<int32_t>> rvqCodes;
    std::vector<int32_t> primaryHistory;
    std::mt19937 talkerRng(42);

    // Prepare Talker input: validate hidden states, project via MLP, reshape buffers
    int64_t seqLen = 0;
    if (!prepareTalkerInput(textTokenIds, effectiveRequest, seqLen, stream))
    {
        LOG_ERROR("Input preparation failed");
        return false;
    }

    // Talker Prefill - engine outputs FP32 logits directly
    if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
    {
        LOG_ERROR("Talker prefill failed");
        return false;
    }

    {
        std::vector<float> hostLogits(static_cast<size_t>(mTalkerConfig.talkerVocabSize));
        CUDA_CHECK(cudaMemcpyAsync(hostLogits.data(), mTalkerLogits.rawPointer(),
            hostLogits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const maxIt = std::max_element(hostLogits.begin(), hostLogits.end());
        int32_t const argmax = static_cast<int32_t>(std::distance(hostLogits.begin(), maxIt));
        LOG_INFO("TTS debug pre-adjust logits: argmax=%d max=%f eos=%f seqLen=%ld", argmax, *maxIt,
            hostLogits[mTalkerConfig.codecEosId], seqLen);
    }

    // Suppress special tokens and apply repetition penalty, then sample first codec token
    adjustTalkerLogits(stream);
    {
        std::vector<float> hostLogits(static_cast<size_t>(mTalkerConfig.talkerVocabSize));
        CUDA_CHECK(cudaMemcpyAsync(hostLogits.data(), mTalkerLogits.rawPointer(),
            hostLogits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const maxIt = std::max_element(hostLogits.begin(), hostLogits.end());
        int32_t const argmax = static_cast<int32_t>(std::distance(hostLogits.begin(), maxIt));
        LOG_INFO("TTS debug adjusted logits: argmax=%d max=%f eos=%f", argmax, *maxIt,
            hostLogits[mTalkerConfig.codecEosId]);
    }
    int32_t codecToken = sampleLogitsCPU(mTalkerLogits, mTalkerConfig.talkerVocabSize, talkerTopK,
        talkerTemperature, true, mTalkerConfig.codecEosId, 0.0f, true, talkerRng, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        mTalkerSelectedIndices.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    trackSeenToken(codecToken, stream);
    primaryHistory.push_back(codecToken);
    LOG_INFO("First codec token (from prefill): %d (eos=%d)", codecToken, mTalkerConfig.codecEosId);

    // Clamp maxAudioLength to avoid Talker KV cache overflow.
    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t const safeMaxFrames = std::max(1, talkerKVCapacity - static_cast<int32_t>(seqLen));
    int32_t const textBasedMax = std::max(50, mTrailingTextLen * 10);
    int32_t const requestedMaxAudio = std::min(request.maxAudioLength, textBasedMax);
    int32_t const effectiveMaxAudio = std::min(requestedMaxAudio, safeMaxFrames);
    if (effectiveMaxAudio < request.maxAudioLength)
    {
        LOG_WARNING("Clamped maxAudioLength from %d to %d (prefill=%lld, KV capacity=%d)", request.maxAudioLength,
            effectiveMaxAudio, seqLen, talkerKVCapacity);
    }

    // Main generation loop
    int32_t numFrames = 0;
    std::vector<int32_t> frameCodes;

    {
        TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);

        while (codecToken != mTalkerConfig.codecEosId && numFrames < effectiveMaxAudio)
        {
            // Extract Talker hidden state (use pre-allocated buffer)
            if (!extractTalkerLastHidden(mTalkerHiddenStatesBuffer, mTalkerLastHidden, stream))
            {
                LOG_ERROR("Failed to extract Talker hidden state at frame %d", numFrames);
                break;
            }

            // Clear codes for this frame
            frameCodes.clear();

            // CodePredictor generation for this frame (16 codes)
            // Hidden states are written directly to mCodecHiddensBuffer
            {
                TIME_STAGE(metrics::StageNames::kCODE_PREDICTOR, stream);
                if (!runCodePredictorGenerationForFrame(
                        codecToken, mTalkerLastHidden, predictorSamplingParams, frameCodes, stream))
                {
                    LOG_ERROR("CodePredictor generation failed at frame %d", numFrames);
                    break;
                }
            }

            // Store RVQ codes for this frame
            rvqCodes.push_back(frameCodes);

            // Compute residual connection using pre-allocated buffer
            // Non-streaming: always add tts_pad_embed as addend
            if (!computeResidualConnection(frameCodes, mResidualEmbedBuffer, numFrames, stream))
            {
                LOG_ERROR("Residual connection failed at frame %d", numFrames);
                break;
            }

            // Talker decoding step with residual embedding as input
            // PyTorch: inputs["inputs_embeds"] = codec_hiddens.sum(1, keepdim=True)
            check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

            // CRITICAL: Reshape hidden states buffer for decoding output (seqLen=1)
            // Otherwise extractTalkerLastHidden reads stale prefill data
            check::check(
                mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

            // Call Talker engine with decoding step (uses KV cache from prefill)
            if (!mTalkerLLMRunner->executeVanillaDecodingStep(mResidualEmbedBuffer, mTalkerLogits,
                    rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)}, stream))
            {
                LOG_ERROR("Talker decoding step failed at frame %d", numFrames);
                break;
            }

            // Suppress special tokens and apply repetition penalty, then sample next codec token
            adjustTalkerLogits(stream);
            int32_t const sampleStep = numFrames + 1;
            int32_t const biasOnset = mTrailingTextLen * 3;
            int32_t const stepsPastOnset = sampleStep - biasOnset;
            float eosBias = 0.0f;
            if (stepsPastOnset >= 0)
            {
                eosBias = std::min(25.0f, 5.0f + static_cast<float>(stepsPastOnset) * 0.5f);
            }
            codecToken = sampleLogitsCPU(mTalkerLogits, mTalkerConfig.talkerVocabSize, talkerTopK, talkerTemperature,
                sampleStep < 2, mTalkerConfig.codecEosId, eosBias, true, talkerRng, stream);
            CUDA_CHECK(cudaMemcpyAsync(
                mTalkerSelectedIndices.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

            trackSeenToken(codecToken, stream);
            primaryHistory.push_back(codecToken);
            numFrames++;
            if (detectPrimaryRepetition(primaryHistory, 5))
            {
                LOG_INFO("Force EOS at frame %d due to primary-code repetition", numFrames);
                codecToken = mTalkerConfig.codecEosId;
            }
        }

    } // end TIME_STAGE talker_generation

    bool const hitEos = (codecToken == mTalkerConfig.codecEosId);
    LOG_INFO(
        "Generated %d audio frames (exit: %s, last_code=%d)", numFrames, hitEos ? "EOS" : "maxAudioLength", codecToken);

    response.rvqCodes = std::move(rvqCodes);
    response.numFrames = numFrames;

    mMultimodalMetrics.recordRun(0, 0, 1, numFrames);

    response.success = true;
    return true;
}

bool Qwen3OmniTTSRuntime::runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
    SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runCodePredictorGenerationForFrame", nvtx_colors::ORANGE);

    // Original model logic:
    // - codes: code_0 (from Talker) + code_1 to code_15 (from CodePredictor) = 16 codes
    // - hidden_states: written directly to mCodecHiddensBuffer

    static thread_local std::mt19937 predictorRng(42);

    outputCodes.clear();
    outputCodes.reserve(16); // code_0 to code_15

    // code_0 comes from Talker
    outputCodes.push_back(codecToken);

    int64_t const hiddenSize = mTalkerConfig.codePredictorHiddenSize;
    constexpr int32_t kActiveGroups = talker_constants::kActiveCodePredictorGroups;
    static_assert(kActiveGroups > 0 && kActiveGroups <= kNumRvqLayers);

    // ========== Prefill: generate code_1 ==========
    // Input: concat([proj(talker_hidden), proj(embed(code_0))]) -> [1, 2, codePredictorHiddenSize]
    // NOTE: code_0 embedding uses TALKER's codec_embedding (2048-dim), projected to 1024 via small_to_mtp_projection
    // PyTorch: last_id_hidden = self.small_to_mtp_projection(self.get_input_embeddings()(input_ids))

    // Step 1: Lookup code_0 from Talker's embedding table into mRawCodecEmbed (2048-dim)
    CUDA_CHECK(cudaMemcpyAsync(
        mCodePredictorCodecIds.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mCodePredictorCodecIds, mTalkerEmbeddingTable, std::nullopt, mRawCodecEmbed, stream);

    // Step 2: Project/copy talkerHiddenState to CodePredictor hidden size.
    // Official qwen_tts uses small_to_mtp_projection only when the Talker and
    // CodePredictor hidden sizes differ; Qwen3-TTS 0.6B uses 1024 for both.
    if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(mSmallToMtpProjectedHidden.rawPointer(), talkerHiddenState.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(
            talkerHiddenState, mSmallToMtpWeight, mSmallToMtpBias, mSmallToMtpProjectedHidden, stream);
    }

    // Step 3: Project/copy mRawCodecEmbed to CodePredictor hidden size.
    check::check(mRawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecEmbed.rawPointer(), mRawCodecEmbed.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(mRawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias, mCodePredictorCodecEmbed, stream);
    }

    if (mSpecialCodePredictorEngine)
    {
        std::vector<float> hiddenHost
            = copyHalfTensorToHostFloat(mSmallToMtpProjectedHidden, mTalkerConfig.codePredictorHiddenSize, stream);
        std::vector<float> primaryEmbeddingHost
            = copyHalfTensorToHostFloat(mCodePredictorCodecEmbed, mTalkerConfig.codePredictorHiddenSize, stream);
        std::vector<int32_t> residualCodes;
        if (!mSpecialCodePredictorEngine->generate(hiddenHost, primaryEmbeddingHost, kActiveGroups,
                samplingParams.topK, samplingParams.temperature, residualCodes))
        {
            return false;
        }

        check::check(mCodecHiddensBuffer.reshape({1, 16, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemsetAsync(mCodecHiddensBuffer.rawPointer(), 0, mCodecHiddensBuffer.getMemoryCapacity(), stream));

        int32_t const groupsToMaterialize = std::min<int32_t>(kActiveGroups, static_cast<int32_t>(residualCodes.size()));
        for (int32_t group = 0; group < groupsToMaterialize; ++group)
        {
            int32_t const residualCode = residualCodes[group];
            outputCodes.push_back(residualCode);
            CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecIds.rawPointer(), &residualCode, sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));
            check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
            kernel::embeddingLookup(
                mCodePredictorCodecIds, mCodePredictorEmbeddingTables[group], std::nullopt, mRawCodecEmbed, stream);
            __half* dst = static_cast<__half*>(mCodecHiddensBuffer.rawPointer())
                + static_cast<int64_t>(group + 1) * mTalkerConfig.talkerHiddenSize;
            CUDA_CHECK(cudaMemcpyAsync(dst, mRawCodecEmbed.rawPointer(),
                static_cast<size_t>(mTalkerConfig.talkerHiddenSize) * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        while (static_cast<int32_t>(outputCodes.size()) < 16)
        {
            outputCodes.push_back(0);
        }
        return true;
    }

    // Step 4: Concat projected tensors into mCodePredictorPrefillInput [1, 2, codePredictorHiddenSize]
    CUDA_CHECK(cudaMemcpyAsync(mCodePredictorPrefillInput.rawPointer(), mSmallToMtpProjectedHidden.rawPointer(),
        hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + hiddenSize,
        mCodePredictorCodecEmbed.rawPointer(), hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));

    check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 2, mTalkerConfig.codePredictorHiddenSize}),
        "Tensor reshape failed");

    // NOTE: CodePredictor ONNX outputs FP32 logits directly (lm_head + cast in ONNX)
    // generationStep=0 corresponds to code_1 (using lm_head_0)
    if (!executeCodePredictorPrefillStep(
            mCodePredictorPrefillInput, 0, mCodePredictorLogits, mCodePredictorHiddenStatesBuffer, stream))
    {
        return false;
    }

    // Sample code_1
    int32_t code = sampleLogitsCPU(mCodePredictorLogits, mTalkerConfig.codebookSize, samplingParams.topK,
        samplingParams.temperature, false, -1, 0.0f, false, predictorRng, stream);
    outputCodes.push_back(code); // code_1

    // ========== Write embedding lookups to mCodecHiddensBuffer for residual connection ==========
    // mCodecHiddensBuffer layout: [1, 16, H]
    // Position 0:  embed(code_0)  using Talker's embedding   - filled in computeResidualConnection
    // Position 1-14: codec_embedding[step-1](code_step)      - filled here (input embeddings, NOT engine hidden states)
    // Position 15: embed(code_15) using CodePredictor embed[-1] - filled in computeResidualConnection
    //
    // PyTorch original (modeling_qwen3_omni.py:3419):
    //   mid_residual_hiddens = [hid[0] for hid in predictor_result.hidden_states[1:]]
    //   hid[0] = inputs_embeds for each decode step = codec_embedding[step-1](code_step)
    //   NOT the transformer output (last layer hidden states)

    // mCodecHiddensBuffer stores raw (2048-dim) codec embeddings for the residual connection
    check::check(mCodecHiddensBuffer.reshape({1, 16, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(mCodecHiddensBuffer.rawPointer(), 0, mCodecHiddensBuffer.getMemoryCapacity(), stream));

    // ========== Decoding loop: generate active residual codes only ==========
    for (int step = 2; step <= kActiveGroups; ++step)
    {
        check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
            "Tensor reshape failed");

        rt::OptionalInputTensor prevHiddenOpt{std::ref(mCodePredictorHiddenStatesBuffer)};

        // PyTorch generation_steps logic:
        //   - generation_steps=1: embed(code_1) with codec_embedding[0], output with lm_head[1] -> code_2
        //   - generation_steps=2: embed(code_2) with codec_embedding[1], output with lm_head[2] -> code_3
        //   - ...
        //   - generation_steps=14: embed(code_14) with codec_embedding[13], output with lm_head[14] -> code_15
        // TRT step 2-15 corresponds to PyTorch generation_steps 1-14
        int32_t const embeddingIdx = step - 2; // step=2->embed[0], step=3->embed[1], ..., step=15->embed[13]
        int32_t const lmHeadIdx = step - 1;    // step=2->lm_head[1], step=3->lm_head[2], ..., step=15->lm_head[14]

        if (!executeCodePredictorDecodingStep(
                code, embeddingIdx, lmHeadIdx, mCodePredictorLogits, mCodePredictorHiddenStatesBuffer, stream))
        {
            return false;
        }

        // Embedding is now saved inside executeCodePredictorDecodingStep before engine execution

        code = sampleLogitsCPU(mCodePredictorLogits, mTalkerConfig.codebookSize, samplingParams.topK,
            samplingParams.temperature, false, -1, 0.0f, false, predictorRng, stream);
        outputCodes.push_back(code); // code_2 to code_kActiveGroups
    }

    // The final active code embedding is normally materialized as the input
    // to the next CP decode step. When inactive groups are zero-filled, perform
    // just that lookup/copy without running another predictor step.
    CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecIds.rawPointer(), &code, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mCodePredictorCodecIds, mCodePredictorEmbeddingTables[kActiveGroups - 1], std::nullopt,
        mRawCodecEmbed, stream);
    {
        int64_t const H = mTalkerConfig.talkerHiddenSize;
        __half* dst = static_cast<__half*>(mCodecHiddensBuffer.rawPointer()) + static_cast<int64_t>(kActiveGroups) * H;
        CUDA_CHECK(cudaMemcpyAsync(dst, mRawCodecEmbed.rawPointer(), H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    while (static_cast<int32_t>(outputCodes.size()) < 16)
    {
        outputCodes.push_back(0);
    }

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnection(
    std::vector<int32_t> const& codes, rt::Tensor& outputResidual, int32_t frameIdx, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnection", nvtx_colors::BLUE);

    // Old working runner: codec_sum + trailing_text[step] for early frames, then tts_pad
    // mCodecHiddensBuffer positions 1-14 are pre-filled by runCodePredictorGenerationForFrame.

    check::check(codes.size() == 16, "Expected 16 codes (code_0 from Talker + code_1-15 from CodePredictor)");

    // Residual output is in Talker's space (2048-dim) since it feeds back to the Talker decoder
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    check::check(outputResidual.reshape({1, 1, hiddenSize}), "Tensor reshape failed");

    // Select addend: trailing text for early frames, tts_pad for later
    __half const* addend;
    if (frameIdx < mTrailingTextLen)
    {
        if (frameIdx == mTrailingTextLen - 1)
        {
            // Last trailing entry: tts_eos
            addend = mTtsEosEmbed.dataPointer<__half>();
        }
        else
        {
            // body[1 + frameIdx] from projected buffer (text starts at index 3)
            int32_t const textIdx = kAssistantPrefixLen + 1 + frameIdx;
            addend = mProjectedBuffer.dataPointer<__half>() + static_cast<int64_t>(textIdx) * hiddenSize;
        }
    }
    else
    {
        addend = mTtsPadEmbed.dataPointer<__half>();
    }

    kernel::invokeResidualConnection(mCodecHiddensBuffer, mTalkerEmbeddingTable, mCodePredictorEmbeddingTables[14],
        codes[0], codes[15], talker_constants::kActiveCodePredictorGroups, addend, outputResidual, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::extractTalkerLastHidden(
    rt::Tensor const& talkerHiddenStates, rt::Tensor& outputLastHidden, cudaStream_t stream)
{
    // talkerHiddenStates is expected to be the full hidden states buffer from LLMEngineRunner
    // which typically has shape [numLayers+1][batchSize, seqLen, hiddenSize]
    // We need to extract the last layer's last token's hidden state

    // For now, assume talkerHiddenStates is already prepared as [numLayers, batchSize, seqLen, hiddenSize]
    // or as a single tensor [batchSize, seqLen, hiddenSize] for the last layer

    // Get dimensions
    auto const& shape = talkerHiddenStates.getShape();
    int32_t const numDims = shape.getNumDims();

    if (numDims != 3)
    {
        LOG_ERROR("extractTalkerLastHidden: Expected 3D tensor [batchSize, seqLen, hiddenSize], got %dD", numDims);
        return false;
    }

    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];
    int64_t const hiddenSize = shape[2];

    if (batchSize != 1)
    {
        LOG_ERROR("extractTalkerLastHidden: Only batchSize=1 supported, got %ld", batchSize);
        return false;
    }

    // Extract last token: [0, seqLen-1, :]
    // Source offset: (batchSize=0) * seqLen * hiddenSize + (seqLen-1) * hiddenSize
    size_t const lastTokenOffset = (seqLen - 1) * hiddenSize * sizeof(__half);
    size_t const copySize = hiddenSize * sizeof(__half);

    // Ensure output tensor has correct shape [1, hiddenSize]
    if (outputLastHidden.getShape().volume() != hiddenSize)
    {
        check::check(outputLastHidden.reshape({1, hiddenSize}), "Tensor reshape failed");
    }

    // Copy last token's hidden state
    CUDA_CHECK(cudaMemcpyAsync(outputLastHidden.rawPointer(),
        static_cast<char const*>(talkerHiddenStates.rawPointer()) + lastTokenOffset, copySize, cudaMemcpyDeviceToDevice,
        stream));

    return true;
}

int32_t Qwen3OmniTTSRuntime::getSpeakerIdByName(std::string const& speakerName) const
{
    auto it = mSpeakerIdMap.find(speakerName);
    if (it != mSpeakerIdMap.end())
    {
        return it->second;
    }

    LOG_WARNING(
        "Speaker '%s' not found, using default speaker ID %d", speakerName.c_str(), mTalkerConfig.defaultSpeakerId);
    return mTalkerConfig.defaultSpeakerId;
}

} // namespace rt
} // namespace trt_edgellm
