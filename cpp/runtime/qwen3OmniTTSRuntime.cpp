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
#include "kernels/qwen3TtsCpKernels/qwen3TtsCpKernels.h"
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
#include <condition_variable>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;
using namespace talker_constants;

namespace
{
std::filesystem::path getDumpPath(std::string const& name)
{
    char const* dumpDir = std::getenv("QWEN3_TTS_DUMP_DIR");
    if (!dumpDir || !*dumpDir)
    {
        return {};
    }
    std::filesystem::create_directories(dumpDir);
    std::string prefix = "official";
    if (char const* prefixEnv = std::getenv("QWEN3_TTS_DUMP_PREFIX"))
    {
        if (*prefixEnv)
        {
            prefix = prefixEnv;
        }
    }
    return std::filesystem::path(dumpDir) / (prefix + "_" + name);
}

template <typename T>
void dumpVector(std::string const& name, std::vector<T> const& values)
{
    auto path = getDumpPath(name);
    if (path.empty())
    {
        return;
    }
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<char const*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void dumpFloats(std::string const& name, float const* values, size_t count)
{
    auto path = getDumpPath(name);
    if (path.empty())
    {
        return;
    }
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<char const*>(values), static_cast<std::streamsize>(count * sizeof(float)));
}

uint32_t makeQwen3TTSSamplingSeed(uint32_t salt)
{
    if (char const* seedEnv = std::getenv("QWEN3_TTS_SEED"))
    {
        try
        {
            return static_cast<uint32_t>(std::stoul(seedEnv)) ^ salt;
        }
        catch (std::exception const&)
        {
            LOG_WARNING("Ignoring invalid QWEN3_TTS_SEED=%s", seedEnv);
        }
    }

    std::random_device rd;
    uint64_t const now = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uint64_t const mixed = now ^ (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(salt);
    return static_cast<uint32_t>(mixed ^ (mixed >> 32));
}

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

size_t dataTypeSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16: return sizeof(uint16_t);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
    case nvinfer1::DataType::kINT64: return sizeof(int64_t);
    default: throw std::runtime_error("Unsupported TensorRT data type");
    }
}

bool loadLLMConfigOnly(std::filesystem::path const& configPath, LLMEngineRunnerConfig& config)
{
    std::ifstream configStream(configPath);
    if (!configStream.is_open())
    {
        LOG_ERROR("Failed to open LLM config: %s", configPath.string().c_str());
        return false;
    }

    Json configJson;
    try
    {
        configJson = Json::parse(configStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse LLM config %s: %s", configPath.string().c_str(), e.what());
        return false;
    }

    try
    {
        auto const& builderConfig = configJson.value("builder_config", Json::object());
        config = LLMEngineRunnerConfig{};
        config.numDecoderLayers = configJson.value("num_hidden_layers", 0);
        config.numKVHeads = configJson.value("num_key_value_heads", 0);
        config.headDim = configJson.value("head_dim", 0);
        config.rotaryDim = static_cast<int32_t>(config.headDim * configJson.value("partial_rotary_factor", 1.0f));
        config.hiddenSize = configJson.value("hidden_size", 0);
        config.vocabSize = configJson.value("vocab_size", 0);
        config.reducedVocabSize = configJson.value("reduced_vocab_size", 0);
        config.outputVocabSize = config.reducedVocabSize > 0 ? config.reducedVocabSize : config.vocabSize;
        config.numDeepstackFeatures = configJson.value("num_deepstack_features", 0);
        config.audioTokenId = configJson.value("audio_token_id", 0);
        config.imageTokenId = configJson.value("image_token_id", 0);
        config.numLinearAttnLayers = configJson.value("num_linear_attn_layers", 0);
        config.numAttentionLayers = configJson.value("num_attention_layers", config.numDecoderLayers);
        config.recurrentStateNumHeads = configJson.value("recurrent_state_num_heads", 0);
        config.recurrentStateHeadDim = configJson.value("recurrent_state_head_dim", 0);
        config.recurrentStateSize = configJson.value("recurrent_state_size", 0);
        config.convDim = configJson.value("conv_dim", 0);
        config.convKernel = configJson.value("conv_kernel", 0);
        config.maxSupportedBatchSize = builderConfig.value("max_batch_size", 1);
        config.maxSupportedInputLength = builderConfig.value("max_input_len", 1);
        config.maxKVCacheCapacity = builderConfig.value("max_kv_cache_capacity", 1);
        config.maxSupportedLoraRank = builderConfig.value("max_lora_rank", 0);
        config.enableEagleSpecDecode = builderConfig.value("eagle_base", false);
        config.useTrtNativeOps = builderConfig.value("trt_native_ops", false);
        config.ropeConfig = collectRopeConfig(configJson);

        if (config.numDecoderLayers <= 0 || config.numKVHeads <= 0 || config.headDim <= 0 || config.hiddenSize <= 0
            || config.vocabSize <= 0 || config.maxSupportedInputLength <= 0 || config.maxKVCacheCapacity <= 0)
        {
            LOG_ERROR("Invalid LLM config in %s", configPath.string().c_str());
            return false;
        }

        for (int32_t i = 0; i < config.numDecoderLayers; ++i)
        {
            config.layerTypes.push_back(rt::HybridCacheManager::LayerType::kAttention);
            config.kvLayerConfigs.push_back({config.numKVHeads, config.headDim});
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load LLM config %s: %s", configPath.string().c_str(), e.what());
        return false;
    }

    return true;
}

int32_t getQwen3TTSActiveCodePredictorGroups()
{
    int32_t groups = talker_constants::kQwen3TTSActiveCodePredictorGroups;
    if (char const* groupsEnv = std::getenv("QWEN3_TTS_ACTIVE_CP_GROUPS"))
    {
        try
        {
            groups = std::stoi(groupsEnv);
        }
        catch (std::exception const&)
        {
            LOG_WARNING("Ignoring invalid QWEN3_TTS_ACTIVE_CP_GROUPS=%s", groupsEnv);
        }
    }
    return std::clamp(groups, talker_constants::kQwen3TTSMinActiveCodePredictorGroups, talker_constants::kNumRvqLayers);
}

std::vector<float> copyTensorToHostFloat(rt::Tensor const& tensor, int64_t elements, cudaStream_t stream)
{
    std::vector<float> out(static_cast<size_t>(elements));
    auto const dtype = tensor.getDataType();
    if (dtype == nvinfer1::DataType::kFLOAT)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            out.data(), tensor.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return out;
    }
    if (dtype != nvinfer1::DataType::kHALF && dtype != nvinfer1::DataType::kBF16)
    {
        throw std::runtime_error("Unsupported CodePredictor embedding table dtype");
    }

    std::vector<uint16_t> raw(static_cast<size_t>(elements));
    CUDA_CHECK(cudaMemcpyAsync(
        raw.data(), tensor.rawPointer(), raw.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int64_t i = 0; i < elements; ++i)
    {
        uint32_t bits = 0;
        if (dtype == nvinfer1::DataType::kBF16)
        {
            bits = static_cast<uint32_t>(raw[static_cast<size_t>(i)]) << 16;
        }
        else
        {
            bits = halfToFloatBits(raw[static_cast<size_t>(i)]);
        }
        std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(float));
    }
    return out;
}

bool loadFloatBin(std::filesystem::path const& path, size_t expectedElements, std::vector<float>& out)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        return false;
    }
    size_t const bytes = static_cast<size_t>(file.tellg());
    size_t const expectedBytes = expectedElements * sizeof(float);
    if (bytes != expectedBytes)
    {
        LOG_WARNING("Ignoring %s: size=%zu, expected=%zu", path.string().c_str(), bytes, expectedBytes);
        return false;
    }
    file.seekg(0);
    out.resize(expectedElements);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(file);
}

std::vector<float> cpuTextProjection(std::vector<float> const& input, int32_t numTokens, int32_t inputDim,
    int32_t hiddenDim, int32_t outputDim, std::vector<float> const& fc1Weight, std::vector<float> const& fc1Bias,
    std::vector<float> const& fc2Weight, std::vector<float> const& fc2Bias)
{
    std::vector<float> workspace(static_cast<size_t>(numTokens) * hiddenDim);
    std::vector<float> output(static_cast<size_t>(numTokens) * outputDim);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        float const* x = input.data() + static_cast<size_t>(t) * inputDim;
        float* h = workspace.data() + static_cast<size_t>(t) * hiddenDim;
        for (int32_t o = 0; o < hiddenDim; ++o)
        {
            float acc = fc1Bias[static_cast<size_t>(o)];
            float const* w = fc1Weight.data() + static_cast<size_t>(o) * inputDim;
            for (int32_t i = 0; i < inputDim; ++i)
            {
                acc = std::fma(x[i], w[i], acc);
            }
            h[o] = acc / (1.0f + std::exp(-acc));
        }
        float* y = output.data() + static_cast<size_t>(t) * outputDim;
        for (int32_t o = 0; o < outputDim; ++o)
        {
            float acc = fc2Bias[static_cast<size_t>(o)];
            float const* w = fc2Weight.data() + static_cast<size_t>(o) * hiddenDim;
            for (int32_t i = 0; i < hiddenDim; ++i)
            {
                acc = std::fma(h[i], w[i], acc);
            }
            y[o] = acc;
        }
    }
    return output;
}

void addHostRows(float* dst, float const* a, float const* b, int32_t hiddenSize)
{
    for (int32_t i = 0; i < hiddenSize; ++i)
    {
        dst[i] = a[i] + b[i];
    }
}

std::filesystem::path getQwen3TTSCodePredictorEnginePath(std::string const& codePredictorEngineDir)
{
    std::filesystem::path const engineDir(codePredictorEngineDir);
    std::filesystem::path const configPath = engineDir / "config.json";
    std::ifstream configStream(configPath);
    if (!configStream.is_open())
    {
        return {};
    }

    Json configJson;
    try
    {
        configJson = Json::parse(configStream);
    }
    catch (Json::parse_error const&)
    {
        return {};
    }

    bool const isQwen3TTSCodePredictor
        = configJson.value("model", std::string{}) == "qwen3ttstalkercodepredictor"
        || configJson.value("model_type", std::string{}) == "qwen3_tts_code_predictor";
    if (!isQwen3TTSCodePredictor)
    {
        return {};
    }

    std::filesystem::path const explicitEnginePath = engineDir / "qwen3_tts_cp.engine";
    if (std::filesystem::exists(explicitEnginePath))
    {
        return explicitEnginePath;
    }

    return {};
}

int32_t sampleLogitsCPU(rt::Tensor const& deviceLogits, int32_t vocabSize, int32_t topK, float topP, float temperature,
    bool suppressEos, int32_t eosId, float eosBias, bool suppressRange, std::vector<int32_t> const* prevTokens,
    float repetitionPenalty, std::mt19937& rng, cudaStream_t stream)
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

    if (prevTokens != nullptr && repetitionPenalty > 1.0f)
    {
        for (int32_t const token : *prevTokens)
        {
            if (token >= 0 && token < vocabSize)
            {
                double& logit = logits[static_cast<size_t>(token)];
                logit = (logit < 0.0) ? logit * static_cast<double>(repetitionPenalty)
                                      : logit / static_cast<double>(repetitionPenalty);
            }
        }
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
    if (std::getenv("QWEN3_TTS_GREEDY"))
    {
        return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
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
    if (topP > 0.0f && topP < 1.0f)
    {
        std::vector<int32_t> indices(static_cast<size_t>(vocabSize));
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&logits](int32_t a, int32_t b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });

        double cumulative = 0.0;
        int32_t keep = 0;
        for (; keep < vocabSize; ++keep)
        {
            cumulative += logits[static_cast<size_t>(indices[static_cast<size_t>(keep)])];
            if (cumulative >= static_cast<double>(topP))
            {
                ++keep;
                break;
            }
        }
        keep = std::max(1, std::min(keep, vocabSize));
        for (int32_t i = keep; i < vocabSize; ++i)
        {
            logits[static_cast<size_t>(indices[static_cast<size_t>(i)])] = 0.0;
        }
        double filteredSum = 0.0;
        for (double const v : logits)
        {
            filteredSum += v;
        }
        if (filteredSum > 0.0)
        {
            for (auto& v : logits)
            {
                v /= filteredSum;
            }
        }
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

// [Phase 3b-B-1] Read OVS_TTS_WORKER_CONCURRENCY env, clamp to [1,8], log
// once per engine ctor invocation. Both CodePredictor and Talker engines use
// this for their SlotPool capacity. Default 1 (preserves Phase 3b-A
// byte-equivalence). 3b-B-2 will be the first iter that actually sets N>1.
inline int readTtsWorkerConcurrencyEnv(char const* engineTag)
{
    int capacity = 1;
    if (char const* env = std::getenv("OVS_TTS_WORKER_CONCURRENCY"))
    {
        try
        {
            int const parsed = std::stoi(env);
            if (parsed < 1)
            {
                LOG_WARNING(
                    "[%s] OVS_TTS_WORKER_CONCURRENCY=%d below min=1; clamping to 1", engineTag, parsed);
                capacity = 1;
            }
            else if (parsed > 8)
            {
                LOG_WARNING(
                    "[%s] OVS_TTS_WORKER_CONCURRENCY=%d above max=8; clamping to 8", engineTag, parsed);
                capacity = 8;
            }
            else
            {
                capacity = parsed;
            }
        }
        catch (std::exception const& e)
        {
            LOG_WARNING(
                "[%s] OVS_TTS_WORKER_CONCURRENCY=\"%s\" not parseable (%s); defaulting to 1",
                engineTag, env, e.what());
            capacity = 1;
        }
    }
    LOG_INFO("[%s] SlotPool capacity=%d (OVS_TTS_WORKER_CONCURRENCY)", engineTag, capacity);
    return capacity;
}

} // anonymous namespace

class Qwen3OmniTTSRuntime::Qwen3TTSCodePredictorEngine
{
    class DeviceBuffer
    {
    public:
        DeviceBuffer() = default;
        ~DeviceBuffer()
        {
            reset();
        }
        DeviceBuffer(DeviceBuffer const&) = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : mPtr(other.mPtr)
        {
            other.mPtr = nullptr;
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                mPtr = other.mPtr;
                other.mPtr = nullptr;
            }
            return *this;
        }

        void allocate(size_t bytes)
        {
            reset();
            CUDA_CHECK(cudaMalloc(&mPtr, bytes));
        }

        void reset()
        {
            if (mPtr)
            {
                cudaFree(mPtr);
                mPtr = nullptr;
            }
        }

        void* get() const
        {
            return mPtr;
        }

    private:
        void* mPtr{nullptr};
    };

public:
    Qwen3TTSCodePredictorEngine(std::filesystem::path const& enginePath, std::filesystem::path const& embeddingBinPath,
        std::vector<rt::Tensor> const& embeddingTables, int32_t hiddenSize, int32_t codebookSize, int32_t numLayers,
        int32_t numHeads, int32_t headDim, int32_t numGroups, cudaStream_t stream)
        : mHiddenSize(hiddenSize)
        , mCodebookSize(codebookSize)
        , mNumLayers(numLayers)
        , mNumHeads(numHeads)
        , mHeadDim(headDim)
        , mNumGroups(numGroups)
        , mStream(stream)
        , mRng(makeQwen3TTSSamplingSeed(0x5157454E))
    {
        if (!embeddingBinPath.empty() && std::filesystem::exists(embeddingBinPath))
        {
            loadEmbeddings(embeddingBinPath);
        }
        else
        {
            loadEmbeddings(embeddingTables);
        }

        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error("Failed to open Qwen3-TTS CodePredictor engine: " + enginePath.string());
        }
        std::vector<char> engineBlob((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
        mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
        mEngine.reset(mRuntime->deserializeCudaEngine(engineBlob.data(), engineBlob.size()));
        if (!mEngine)
        {
            throw std::runtime_error("Failed to deserialize Qwen3-TTS CodePredictor engine: " + enginePath.string());
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
        mDumpDebug = std::getenv("QWEN3_TTS_DUMP_CP") != nullptr;
        mGreedy = std::getenv("QWEN3_TTS_GREEDY") != nullptr;
        mProfile = std::getenv("QWEN3_TTS_CP_PROFILE") != nullptr;
        bool const requestedDecodeCudaGraph = std::getenv("QWEN3_TTS_CP_DECODE_CUDA_GRAPH") != nullptr
            && std::string(std::getenv("QWEN3_TTS_CP_DECODE_CUDA_GRAPH")) != "0";

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

        mDeviceEmbeds.allocate(static_cast<size_t>(2 * mHiddenSize) * sizeof(float));
        mDeviceCachePosition.allocate(static_cast<size_t>(32) * sizeof(int64_t));
        mDeviceGenStep.allocate(sizeof(int64_t));
        mDevicePastLength.allocate(sizeof(int64_t));
        mDeviceDummyKV.allocate(16);
        mDeviceLogits.allocate(static_cast<size_t>(mNumGroups) * mCodebookSize * mLogitsElementSize);
        mDeviceSelectedTokens.allocate(static_cast<size_t>(mNumGroups) * sizeof(int32_t));
        mDeviceDecodeGenSteps.allocate(static_cast<size_t>(mNumGroups) * sizeof(int64_t));
        mDeviceDecodePastLengths.allocate(static_cast<size_t>(mNumGroups) * sizeof(int64_t));
        std::vector<int64_t> decodeGenSteps(static_cast<size_t>(mNumGroups));
        std::vector<int64_t> decodePastLengths(static_cast<size_t>(mNumGroups));
        for (int32_t i = 0; i < mNumGroups; ++i)
        {
            decodeGenSteps[static_cast<size_t>(i)] = i;
            decodePastLengths[static_cast<size_t>(i)] = i + 1;
        }
        CUDA_CHECK(cudaMemcpyAsync(mDeviceDecodeGenSteps.get(), decodeGenSteps.data(),
            decodeGenSteps.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mDeviceDecodePastLengths.get(), decodePastLengths.data(),
            decodePastLengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        bool const requestedDeviceEmbeddings = std::getenv("QWEN3_TTS_CP_DEVICE_EMBEDDINGS") != nullptr
            && std::string(std::getenv("QWEN3_TTS_CP_DEVICE_EMBEDDINGS")) != "0";
        if (requestedDeviceEmbeddings || requestedDecodeCudaGraph)
        {
            size_t const embeddingBytes = mEmbeddings.size() * sizeof(float);
            mDeviceEmbeddingTable.allocate(embeddingBytes);
            CUDA_CHECK(cudaMemcpyAsync(
                mDeviceEmbeddingTable.get(), mEmbeddings.data(), embeddingBytes, cudaMemcpyHostToDevice, mStream));
            mUseDeviceEmbeddingTable = true;
            LOG_INFO("Qwen3-TTS CP device embedding table enabled: %.1f MiB",
                static_cast<double>(embeddingBytes) / (1024.0 * 1024.0));
        }
        bool const requestedGpuGreedy = std::getenv("QWEN3_TTS_CP_GPU_GREEDY") != nullptr
            && std::string(std::getenv("QWEN3_TTS_CP_GPU_GREEDY")) != "0";
        mUseGpuGreedy = requestedGpuGreedy && mGreedy && mUseDeviceEmbeddingTable && mLogitsElementSize == sizeof(float);
        if (requestedGpuGreedy && !mUseGpuGreedy)
        {
            LOG_WARNING("Qwen3-TTS CP GPU greedy requested but disabled; requires QWEN3_TTS_GREEDY=1, "
                        "QWEN3_TTS_CP_DEVICE_EMBEDDINGS=1, and FP32 logits.");
        }
        if (mUseGpuGreedy)
        {
            LOG_WARNING("Qwen3-TTS CP GPU greedy sampling enabled (experimental; requires codes parity quality gate)");
        }
        bool const requestedGpuSampling = std::getenv("QWEN3_TTS_CP_GPU_SAMPLING") != nullptr
            && std::string(std::getenv("QWEN3_TTS_CP_GPU_SAMPLING")) != "0";
        mUseGpuSampling = requestedGpuSampling && !mGreedy && mUseDeviceEmbeddingTable && mLogitsElementSize == sizeof(float);
        if (requestedGpuSampling && !mUseGpuSampling)
        {
            LOG_WARNING("Qwen3-TTS CP GPU sampling requested but disabled; requires non-greedy sampling, "
                        "QWEN3_TTS_CP_DEVICE_EMBEDDINGS=1, and FP32 logits.");
        }
        if (mUseGpuSampling)
        {
            SamplingParams const maxSamplingParams(1, mCodebookSize, 1.0f, mCodebookSize, 1.0f);
            mGpuSamplingWorkspaceBytes = getTopKtopPSamplingWorkspaceSize(1, mCodebookSize, maxSamplingParams);
            mDeviceSamplingWorkspace.allocate(mGpuSamplingWorkspaceBytes);
            mGpuSamplingSeed = makeQwen3TTSSamplingSeed(0x53414D50);
            LOG_WARNING("Qwen3-TTS CP GPU top-k/top-p sampling enabled "
                        "(experimental; requires TTS/ASR quality gate)");
        }
        mUseDecodeCudaGraph = requestedDecodeCudaGraph && mUseDeviceEmbeddingTable && !mUseGpuGreedy && !mUseGpuSampling;
        if (requestedDecodeCudaGraph && !mUseDecodeCudaGraph)
        {
            LOG_WARNING("Qwen3-TTS CP decode CUDA graph requested but disabled; requires CPU sampling path and "
                        "device embedding table.");
        }
        if (mUseDecodeCudaGraph)
        {
            mDecodeCudaGraphs.resize(static_cast<size_t>(mNumGroups) * 2);
            LOG_WARNING("Qwen3-TTS CP decode CUDA graph enabled (experimental; CPU sampling keeps token quality)");
        }

        mSampleLogits.resize(static_cast<size_t>(mCodebookSize));
        mSampleRaw.resize(static_cast<size_t>(mCodebookSize));
        mSampleVals.resize(static_cast<size_t>(mCodebookSize));
        mSampleProbs.resize(static_cast<size_t>(mCodebookSize));

        std::vector<int64_t> cachePositions(32);
        for (int32_t i = 0; i < static_cast<int32_t>(cachePositions.size()); ++i)
        {
            cachePositions[i] = i;
        }
        CUDA_CHECK(cudaMemcpyAsync(mDeviceCachePosition.get(), cachePositions.data(),
            cachePositions.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

        size_t const kvBytes = static_cast<size_t>(mNumHeads) * 32 * mHeadDim * mKVElementSize;
        mKVA.resize(2 * mNumLayers);
        mKVB.resize(2 * mNumLayers);
        for (int32_t i = 0; i < 2 * mNumLayers; ++i)
        {
            mKVA[i].allocate(kvBytes);
            mKVB[i].allocate(kvBytes);
        }
        LOG_INFO("Qwen3-TTS CodePredictor enabled: %s", enginePath.string().c_str());

        // [Phase 3b-B-4] Eager pre-allocate the slot pool. Without this the
        // FIRST N=2 concurrent request triggers lazy-alloc of slot 2 inside
        // acquirePoolSlot, which calls cudaStreamSynchronize(mStream) and
        // blocks until client 1's primary-stream work drains — a one-time
        // cold-start spike that has been observed adding ~2 s to client 2's
        // TTFA. Pre-allocating here eliminates the spike; releases on first
        // use just hand back already-warm slots from the free list.
        eagerInitSlotPool();
    }

    ~Qwen3TTSCodePredictorEngine()
    {
        destroyDecodeCudaGraphs();
    }

private:
    //! [Phase 3b-B-4] Pre-allocate every slot up to mSlotPoolCapacity, populating
    //! mFreeSlots so first-time acquire() never hits the lazy-init path.
    //! Failed pair-creations leave the pool short — first-use will still try
    //! lazy-init for those, matching prior behavior. Holds mSlotPoolMutex
    //! while populating the vectors; no other thread can race the ctor.
    void eagerInitSlotPool()
    {
        std::lock_guard<std::mutex> lk(mSlotPoolMutex);
        for (int i = 0; i < mSlotPoolCapacity; ++i)
        {
            auto slot = std::make_unique<Qwen3OmniTTSRuntime::CodePredictorSlot>();
            slot->stream = mStream;
            auto ctxPair = createExecutionContextPair();
            if (!ctxPair.first || !ctxPair.second)
            {
                LOG_WARNING("CodePredictorEngine eager pool: pair-init failed at slot %d "
                            "(pool short; subsequent acquire() will retry)", i);
                break;
            }
            slot->prefillCtxOwned = std::move(ctxPair.first);
            slot->decodeCtxOwned = std::move(ctxPair.second);
            allocateSlot(*slot);
            mFreeSlots.push_back(slot.get());
            mAllSlots.push_back(std::move(slot));
        }
        LOG_INFO("Qwen3-TTS CodePredictor SlotPool eager-initialised: %zu/%d slots ready",
                 mFreeSlots.size(), mSlotPoolCapacity);
    }

public:

    //! [Phase 2 must-fix 3] Create a paired (prefill, decode) execution context set
    //! bound to this engine. CP requires both contexts per generate call (prefill on
    //! profile 0 for the 2-token warmup, decode on profile 1 for residual groups),
    //! so a single factory ctx is structurally insufficient. Phase 3 will use one
    //! such pair per request slot. Returns {nullptr, nullptr} on failure.
    std::pair<std::unique_ptr<nvinfer1::IExecutionContext>, std::unique_ptr<nvinfer1::IExecutionContext>>
    createExecutionContextPair()
    {
        std::pair<std::unique_ptr<nvinfer1::IExecutionContext>, std::unique_ptr<nvinfer1::IExecutionContext>> out{
            nullptr, nullptr};
        if (!mEngine)
        {
            return out;
        }
        out.first.reset(mEngine->createExecutionContext());
        out.second.reset(mEngine->createExecutionContext());
        if (out.first && out.second && mEngine->getNbOptimizationProfiles() >= 2)
        {
            out.first->setOptimizationProfileAsync(0, mStream);
            out.second->setOptimizationProfileAsync(1, mStream);
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
        return out;
    }

    //! [Phase 3a Iter1] Allocate all per-request scratch tensors on `slot`,
    //! mirroring this engine's own ctor allocations. Tensors are byte buffers
    //! sized to match the corresponding engine member (mDeviceEmbeds, mKVA[i],
    //! etc.). Initialises `slot->rng` with the same seed policy the engine
    //! uses (makeQwen3TTSSamplingSeed(0x5157454E)). Does NOT bind the
    //! tensors to any execution context — that is Iteration 2's job.
    void allocateSlot(Qwen3OmniTTSRuntime::CodePredictorSlot& slot) const
    {
        size_t const embedBytes = static_cast<size_t>(2 * mHiddenSize) * sizeof(float);
        size_t const logitsBytes = static_cast<size_t>(mNumGroups) * mCodebookSize * mLogitsElementSize;
        size_t const cachePosBytes = static_cast<size_t>(32) * sizeof(int64_t);
        size_t const kvBytes = static_cast<size_t>(mNumHeads) * 32 * mHeadDim * mKVElementSize;

        slot.deviceEmbeds = rt::Tensor({static_cast<int64_t>(embedBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "CodePredictorSlot::deviceEmbeds");
        slot.deviceLogits = rt::Tensor({static_cast<int64_t>(logitsBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "CodePredictorSlot::deviceLogits");
        slot.deviceCachePosition = rt::Tensor({static_cast<int64_t>(cachePosBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "CodePredictorSlot::deviceCachePosition");
        slot.deviceSelectedTokens = rt::Tensor({static_cast<int64_t>(static_cast<size_t>(mNumGroups) * sizeof(int32_t))},
            rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "CodePredictorSlot::deviceSelectedTokens");
        slot.deviceGenStep = rt::Tensor({static_cast<int64_t>(sizeof(int64_t))}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "CodePredictorSlot::deviceGenStep");
        slot.devicePastLength = rt::Tensor({static_cast<int64_t>(sizeof(int64_t))}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "CodePredictorSlot::devicePastLength");
        if (mGpuSamplingWorkspaceBytes > 0)
        {
            slot.deviceSamplingWorkspace = rt::Tensor({static_cast<int64_t>(mGpuSamplingWorkspaceBytes)},
                rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "CodePredictorSlot::deviceSamplingWorkspace");
        }

        slot.kvA.clear();
        slot.kvB.clear();
        slot.kvA.reserve(static_cast<size_t>(2 * mNumLayers));
        slot.kvB.reserve(static_cast<size_t>(2 * mNumLayers));
        for (int32_t i = 0; i < 2 * mNumLayers; ++i)
        {
            slot.kvA.emplace_back(rt::Tensor({static_cast<int64_t>(kvBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "CodePredictorSlot::kvA[" + std::to_string(i) + "]"));
            slot.kvB.emplace_back(rt::Tensor({static_cast<int64_t>(kvBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "CodePredictorSlot::kvB[" + std::to_string(i) + "]"));
        }

        slot.sampleLogits.assign(static_cast<size_t>(mCodebookSize), 0.0f);
        slot.sampleRaw.assign(static_cast<size_t>(mCodebookSize), 0);
        slot.sampleVals.assign(static_cast<size_t>(mCodebookSize), std::pair<float, int32_t>{0.0f, 0});
        slot.sampleProbs.assign(static_cast<size_t>(mCodebookSize), 0.0);

        // Match engine mRng seed policy (ctor + resetSampling use the same salt).
        slot.rng.seed(makeQwen3TTSSamplingSeed(0x5157454E));
        slot.gpuSamplingOffset = 0;
    }

    void resetSampling()
    {
        mRng.seed(makeQwen3TTSSamplingSeed(0x5157454E));
        // [Phase 3b-A audio-parity] Per-request reset boundary: the runtime
        // calls resetSampling() once per TTS request before invoking generate().
        // Phase 3a's slot path uses slot->rng, NOT mRng, so we must re-seed the
        // pooled slot's rng here to keep request boundaries deterministic and
        // byte-identical to the engine-globals path. gpuSamplingOffset is left
        // untouched (engine path also accumulates it across requests).
        // [Phase 3b-B-1] Reseed ALL allocated slots' rngs, not just the
        // first. At N=1 only one slot is ever allocated so this is
        // byte-identical to Phase 3b-A. At N>1 every concurrent
        // request still gets the same deterministic seed boundary.
        std::lock_guard<std::mutex> lk(mSlotPoolMutex);
        for (auto& slot : mAllSlots)
        {
            if (slot)
            {
                slot->rng.seed(makeQwen3TTSSamplingSeed(0x5157454E));
            }
        }
    }

    bool generate(std::vector<float> const& hidden, std::vector<float> const& primaryEmbedding, int32_t activeGroups,
        int32_t topK, float topP, float temperature, std::vector<int32_t>& residualCodes,
        cudaStream_t stream = nullptr, nvinfer1::IExecutionContext* prefillCtxOverride = nullptr,
        nvinfer1::IExecutionContext* decodeCtxOverride = nullptr,
        Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // [Phase 3b-A] Activate the slot path at N=1 — acquire the single
        // pooled slot when the caller didn't supply one, recurse, release.
        if (slot == nullptr)
        {
            auto* pooled = acquirePoolSlot();
            if (pooled == nullptr)
            {
                LOG_ERROR("Qwen3-TTS CP generate: pool slot unavailable");
                return false;
            }
            bool const ok = generate(hidden, primaryEmbedding, activeGroups, topK, topP, temperature, residualCodes,
                stream, prefillCtxOverride, decodeCtxOverride, pooled);
            releasePoolSlot(pooled);
            return ok;
        }
        // Phase 2 must-fix 2: resolve per-invocation stream + dual ctx overrides.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        // Phase 3a Iter3: when slot != nullptr, route per-request mutable state through
        // the slot's owned buffers. At N=1 callers pass slot=nullptr and behavior is
        // byte-identical to Phase 2 / Iter 2.
        void* const deviceEmbedsPtr = (slot != nullptr) ? slot->deviceEmbeds.rawPointer() : mDeviceEmbeds.get();
        residualCodes.assign(static_cast<size_t>(mNumGroups), 0);
        if (!mHasPastLength)
        {
            zeroKV(s, slot);
        }

        size_t const bytes = static_cast<size_t>(mHiddenSize) * sizeof(float);
        auto const inputCopyStart = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(deviceEmbedsPtr, hidden.data(), bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(deviceEmbedsPtr) + bytes, primaryEmbedding.data(), bytes,
            cudaMemcpyHostToDevice, s));
        profileAdd(mProfileInputCopyMs, inputCopyStart);
        if (mProfile)
        {
            ++mProfileHostHiddenFrames;
        }
        return generatePreparedInputs(
            activeGroups, topK, topP, temperature, residualCodes, s, prefillCtxOverride, decodeCtxOverride, slot);
    }

    bool generateDeviceHidden(float const* hiddenDevice, std::vector<float> const& primaryEmbedding,
        int32_t activeGroups, int32_t topK, float topP, float temperature, std::vector<int32_t>& residualCodes,
        cudaStream_t stream = nullptr, nvinfer1::IExecutionContext* prefillCtxOverride = nullptr,
        nvinfer1::IExecutionContext* decodeCtxOverride = nullptr,
        Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // [Phase 3b-A] Activate the slot path at N=1 — see generate() above.
        if (slot == nullptr)
        {
            auto* pooled = acquirePoolSlot();
            if (pooled == nullptr)
            {
                LOG_ERROR("Qwen3-TTS CP generateDeviceHidden: pool slot unavailable");
                return false;
            }
            bool const ok = generateDeviceHidden(hiddenDevice, primaryEmbedding, activeGroups, topK, topP, temperature,
                residualCodes, stream, prefillCtxOverride, decodeCtxOverride, pooled);
            releasePoolSlot(pooled);
            return ok;
        }
        // Phase 2 must-fix 2: resolve per-invocation stream + dual ctx overrides.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        // Phase 3a Iter3: see generate() above for slot semantics.
        void* const deviceEmbedsPtr = (slot != nullptr) ? slot->deviceEmbeds.rawPointer() : mDeviceEmbeds.get();
        residualCodes.assign(static_cast<size_t>(mNumGroups), 0);
        if (!mHasPastLength)
        {
            zeroKV(s, slot);
        }

        size_t const bytes = static_cast<size_t>(mHiddenSize) * sizeof(float);
        auto const inputCopyStart = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(deviceEmbedsPtr, hiddenDevice, bytes, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(deviceEmbedsPtr) + bytes, primaryEmbedding.data(), bytes,
            cudaMemcpyHostToDevice, s));
        profileAdd(mProfileInputCopyMs, inputCopyStart);
        if (mProfile)
        {
            ++mProfileDeviceHiddenFrames;
        }
        return generatePreparedInputs(
            activeGroups, topK, topP, temperature, residualCodes, s, prefillCtxOverride, decodeCtxOverride, slot);
    }

private:
    bool generatePreparedInputs(
        int32_t activeGroups, int32_t topK, float topP, float temperature, std::vector<int32_t>& residualCodes,
        cudaStream_t stream = nullptr, nvinfer1::IExecutionContext* prefillCtxOverride = nullptr,
        nvinfer1::IExecutionContext* decodeCtxOverride = nullptr,
        Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // Phase 2 must-fix 2: resolve per-invocation stream + dual ctx overrides for the
        // entire CP generate frame. Every helper call below forwards `s` so KV memset,
        // logits copy, sampling kernels, and TRT enqueueV3 all share one ordering domain.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        nvinfer1::IExecutionContext* prefill = (prefillCtxOverride != nullptr) ? prefillCtxOverride : mPrefillContext.get();
        nvinfer1::IExecutionContext* decode = (decodeCtxOverride != nullptr) ? decodeCtxOverride : mDecodeContext.get();
        // Phase 3a Iter2: when slot != nullptr, route mutable per-request state through
        // the slot's owned buffers instead of the engine globals. At N=1 callers pass
        // slot=nullptr and behavior is byte-identical to Phase 2. The slot's KV pair
        // (slot->kvA / slot->kvB) is rt::Tensor while the engine globals are
        // DeviceBuffer — we acquire void* via small lambdas (kvReadPtr/kvWritePtr)
        // rather than forking the code paths.
        void* const deviceEmbedsPtr = (slot != nullptr) ? slot->deviceEmbeds.rawPointer() : mDeviceEmbeds.get();
        void* const deviceLogitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        void* const deviceCachePositionPtr
            = (slot != nullptr) ? slot->deviceCachePosition.rawPointer() : mDeviceCachePosition.get();
        void* const deviceGenStepPtr = (slot != nullptr) ? slot->deviceGenStep.rawPointer() : mDeviceGenStep.get();
        void* const devicePastLengthPtr
            = (slot != nullptr) ? slot->devicePastLength.rawPointer() : mDevicePastLength.get();
        void* const deviceSelectedTokensPtr
            = (slot != nullptr) ? slot->deviceSelectedTokens.rawPointer() : mDeviceSelectedTokens.get();
        // KV double-buffer swap policy stays identical to Phase 2 (start with read=B,
        // write=A, swap each group). We track engine-side pointers with the original
        // DeviceBuffer* read/write pair, and slot-side pointers via parallel
        // std::vector<rt::Tensor>* refs. The bindDecodeContext helper accepts both.
        std::vector<DeviceBuffer>* read = &mKVB;
        std::vector<DeviceBuffer>* write = &mKVA;
        std::vector<rt::Tensor>* slotRead = (slot != nullptr) ? &slot->kvB : nullptr;
        std::vector<rt::Tensor>* slotWrite = (slot != nullptr) ? &slot->kvA : nullptr;
        // CUDA graphs were captured against mDecodeContext + mStream. When the caller
        // supplies its own slot context/stream, the captured graph cannot be reused, so
        // we transparently fall back to direct enqueueV3 on the override ctx/stream.
        bool const useDecodeCudaGraph = mUseDecodeCudaGraph && (decodeCtxOverride == nullptr) && (stream == nullptr)
            && (slot == nullptr);

        auto const frameStart = Clock::now();
        size_t const bytes = static_cast<size_t>(mHiddenSize) * sizeof(float);
        int64_t const prefillCachePositions[2] = {0, 1};
        CUDA_CHECK(cudaMemcpyAsync(deviceCachePositionPtr, prefillCachePositions, sizeof(prefillCachePositions),
            cudaMemcpyHostToDevice, s));

        auto const prefillSetupStart = Clock::now();
        setScalar(deviceGenStepPtr, 0, s);
        setScalar(devicePastLengthPtr, 0, s);
        prefill->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 2, mHiddenSize});
        prefill->setTensorAddress("inputs_embeds", deviceEmbedsPtr);
        prefill->setInputShape("cache_position", nvinfer1::Dims{1, {2}});
        prefill->setTensorAddress("cache_position", deviceCachePositionPtr);
        if (mHasGenStep)
        {
            prefill->setTensorAddress("gen_step", deviceGenStepPtr);
        }
        if (mHasPastLength)
        {
            prefill->setTensorAddress("past_length", devicePastLengthPtr);
        }
        for (int32_t i = 0; i < mNumLayers; ++i)
        {
            prefill->setInputShape(mPastKeyNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, 0, mHeadDim});
            prefill->setInputShape(mPastValueNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, 0, mHeadDim});
            prefill->setTensorAddress(mPastKeyNames[i].c_str(), mDeviceDummyKV.get());
            prefill->setTensorAddress(mPastValueNames[i].c_str(), mDeviceDummyKV.get());
            void* const writeK = (slot != nullptr) ? slot->kvB[2 * i].rawPointer() : mKVB[2 * i].get();
            void* const writeV = (slot != nullptr) ? slot->kvB[2 * i + 1].rawPointer() : mKVB[2 * i + 1].get();
            prefill->setTensorAddress(mNewPastKeyNames[i].c_str(), writeK);
            prefill->setTensorAddress(mNewPastValueNames[i].c_str(), writeV);
        }
        prefill->setTensorAddress(mLogitsName.c_str(), deviceLogitsPtr);
        profileAdd(mProfilePrefillSetupMs, prefillSetupStart);
        if (!prefill->enqueueV3(s))
        {
            LOG_ERROR("Qwen3-TTS CodePredictor prefill failed");
            return false;
        }
        if (mUseGpuGreedy)
        {
            sampleDeviceLogitsGreedyToDevice(0, s, slot);
        }
        else if (mUseGpuSampling)
        {
            sampleDeviceLogitsTopKTopPToDevice(0, topK, topP, temperature, s, slot);
        }
        else
        {
            residualCodes[0] = sampleDeviceLogits(0, topK, topP, temperature, s, slot);
        }

        if (!useDecodeCudaGraph)
        {
            decode->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, mHiddenSize});
            decode->setTensorAddress("inputs_embeds", deviceEmbedsPtr);
            decode->setInputShape("cache_position", nvinfer1::Dims{1, {1}});
            decode->setTensorAddress("cache_position", deviceCachePositionPtr);
            decode->setTensorAddress(mLogitsName.c_str(), deviceLogitsPtr);
            if (mHasGenStep)
            {
                decode->setTensorAddress("gen_step", deviceGenStepPtr);
            }
            if (mHasPastLength)
            {
                decode->setTensorAddress("past_length", devicePastLengthPtr);
            }
        }

        int32_t const groupsToGenerate = std::min(activeGroups, mNumGroups);
        for (int32_t j = 1; j < groupsToGenerate; ++j)
        {
            auto const embedStart = Clock::now();
            if (useDecodeCudaGraph)
            {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<int32_t*>(deviceSelectedTokensPtr) + j - 1,
                    &residualCodes[j - 1], sizeof(int32_t), cudaMemcpyHostToDevice, s));
            }
            else if (mUseGpuGreedy || mUseGpuSampling)
            {
                kernel::qwen3TtsCpGatherEmbedding(static_cast<float const*>(mDeviceEmbeddingTable.get()), mCodebookSize,
                    mHiddenSize, j - 1, static_cast<int32_t const*>(deviceSelectedTokensPtr),
                    static_cast<float*>(deviceEmbedsPtr), s);
            }
            else if (mUseDeviceEmbeddingTable)
            {
                CUDA_CHECK(cudaMemcpyAsync(deviceEmbedsPtr, embeddingDevice(j - 1, residualCodes[j - 1]), bytes,
                    cudaMemcpyDeviceToDevice, s));
            }
            else
            {
                CUDA_CHECK(cudaMemcpyAsync(deviceEmbedsPtr, embedding(j - 1, residualCodes[j - 1]), bytes,
                    cudaMemcpyHostToDevice, s));
            }
            profileAdd(mProfileEmbedCopyMs, embedStart);
            if (mProfile)
            {
                ++mProfileDecodeGroups;
            }

            auto const decodeSetupStart = Clock::now();
            int64_t const actualPast = j + 1;
            if (!useDecodeCudaGraph)
            {
                setScalar(deviceGenStepPtr, j, s);
                setScalar(devicePastLengthPtr, actualPast, s);
                CUDA_CHECK(cudaMemcpyAsync(
                    deviceCachePositionPtr, &actualPast, sizeof(int64_t), cudaMemcpyHostToDevice, s));
                bindDecodeContext(decode, j, actualPast, *read, *write, deviceGenStepPtr, devicePastLengthPtr,
                    deviceCachePositionPtr, slot, slotRead, slotWrite);
            }
            profileAdd(mProfileDecodeSetupMs, decodeSetupStart);
            bool decodeOk = false;
            if (useDecodeCudaGraph)
            {
                // Note: launchDecodeCudaGraph is intentionally pinned to engine globals
                // (graph captured against mDecodeContext + mStream + mKVA/mKVB). When
                // slot != nullptr we never reach this branch (useDecodeCudaGraph above
                // forces false), so no slot threading is needed here.
                decodeOk = launchDecodeCudaGraph(j, *read, *write);
                if (!decodeOk)
                {
                    LOG_WARNING("Qwen3-TTS CP decode CUDA graph disabled; falling back to normal decode");
                    mUseDecodeCudaGraph = false;
                    setScalar(deviceGenStepPtr, j, s);
                    setScalar(devicePastLengthPtr, actualPast, s);
                    CUDA_CHECK(cudaMemcpyAsync(
                        deviceCachePositionPtr, &actualPast, sizeof(int64_t), cudaMemcpyHostToDevice, s));
                    bindDecodeContext(decode, j, actualPast, *read, *write, deviceGenStepPtr,
                        devicePastLengthPtr, deviceCachePositionPtr, slot, slotRead, slotWrite);
                    decodeOk = decode->enqueueV3(s);
                }
            }
            else
            {
                decodeOk = decode->enqueueV3(s);
            }
            if (!decodeOk)
            {
                LOG_ERROR("Qwen3-TTS CodePredictor decode failed at group %d", j);
                return false;
            }
            if (mUseGpuGreedy)
            {
                sampleDeviceLogitsGreedyToDevice(j, s, slot);
            }
            else if (mUseGpuSampling)
            {
                sampleDeviceLogitsTopKTopPToDevice(j, topK, topP, temperature, s, slot);
            }
            else
            {
                residualCodes[j] = sampleDeviceLogits(j, topK, topP, temperature, s, slot);
            }
            std::swap(read, write);
            if (slot != nullptr)
            {
                std::swap(slotRead, slotWrite);
            }
        }
        if ((mUseGpuGreedy || mUseGpuSampling) && groupsToGenerate > 0)
        {
            auto const waitStart = Clock::now();
            CUDA_CHECK(cudaMemcpyAsync(residualCodes.data(), deviceSelectedTokensPtr,
                static_cast<size_t>(groupsToGenerate) * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            profileAdd(mProfileSampleWaitMs, waitStart);
        }
        if (mProfile)
        {
            ++mProfileFrames;
            mProfileGroups += groupsToGenerate;
            mProfileFrameTotalMs += elapsedMs(frameStart);
            if (mProfileFrames % 25 == 0)
            {
                logProfile();
            }
        }
        return true;
    }

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

    void loadEmbeddings(std::vector<rt::Tensor> const& embeddingTables)
    {
        if (static_cast<int32_t>(embeddingTables.size()) < mNumGroups)
        {
            throw std::runtime_error("Not enough CodePredictor embedding tables for Qwen3-TTS CP runtime");
        }

        size_t const elementsPerTable = static_cast<size_t>(mCodebookSize) * mHiddenSize;
        mEmbeddings.resize(static_cast<size_t>(mNumGroups) * elementsPerTable);
        for (int32_t i = 0; i < mNumGroups; ++i)
        {
            auto const& table = embeddingTables[i];
            if (table.getShape().getNumDims() != 2 || table.getShape()[0] != mCodebookSize
                || table.getShape()[1] != mHiddenSize)
            {
                throw std::runtime_error("Unexpected CodePredictor embedding table shape for Qwen3-TTS CP runtime");
            }
            auto host = copyTensorToHostFloat(table, static_cast<int64_t>(elementsPerTable), mStream);
            std::copy(host.begin(), host.end(), mEmbeddings.begin() + static_cast<size_t>(i) * elementsPerTable);
        }
    }

    void loadEmbeddings(std::filesystem::path const& embedPath)
    {
        std::ifstream file(embedPath, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("Failed to open Qwen3-TTS CP embedding table: " + embedPath.string());
        }
        size_t const bytes = static_cast<size_t>(file.tellg());
        size_t const expected = static_cast<size_t>(mNumGroups) * mCodebookSize * mHiddenSize * sizeof(float);
        if (bytes != expected)
        {
            throw std::runtime_error("Unexpected Qwen3-TTS CP embedding table size: " + std::to_string(bytes)
                + ", expected " + std::to_string(expected));
        }
        file.seekg(0);
        mEmbeddings.resize(expected / sizeof(float));
        file.read(reinterpret_cast<char*>(mEmbeddings.data()), static_cast<std::streamsize>(bytes));
        LOG_INFO("Loaded Qwen3-TTS CP embedding table: %s", embedPath.string().c_str());
    }

    float const* embedding(int32_t group, int32_t token) const
    {
        group = std::clamp(group, 0, mNumGroups - 1);
        token = std::clamp(token, 0, mCodebookSize - 1);
        size_t const offset = (static_cast<size_t>(group) * mCodebookSize + token) * mHiddenSize;
        return mEmbeddings.data() + offset;
    }

    void const* embeddingDevice(int32_t group, int32_t token) const
    {
        group = std::clamp(group, 0, mNumGroups - 1);
        token = std::clamp(token, 0, mCodebookSize - 1);
        size_t const offset = (static_cast<size_t>(group) * mCodebookSize + token) * mHiddenSize * sizeof(float);
        return static_cast<char const*>(mDeviceEmbeddingTable.get()) + offset;
    }

    void zeroKV(cudaStream_t stream = nullptr, Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // Phase 2 must-fix 2: honor caller-supplied stream.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        size_t const kvBytes = static_cast<size_t>(mNumHeads) * 32 * mHeadDim * mKVElementSize;
        // Phase 3a Iter2: when slot != nullptr, zero the slot's KV double-buffers
        // instead of the engine globals. At N=1 callers pass slot=nullptr and
        // behavior is byte-identical (operates on mKVA/mKVB).
        if (slot != nullptr)
        {
            for (auto& t : slot->kvA)
            {
                CUDA_CHECK(cudaMemsetAsync(t.rawPointer(), 0, kvBytes, s));
            }
            for (auto& t : slot->kvB)
            {
                CUDA_CHECK(cudaMemsetAsync(t.rawPointer(), 0, kvBytes, s));
            }
            return;
        }
        for (auto const& p : mKVA)
        {
            CUDA_CHECK(cudaMemsetAsync(p.get(), 0, kvBytes, s));
        }
        for (auto const& p : mKVB)
        {
            CUDA_CHECK(cudaMemsetAsync(p.get(), 0, kvBytes, s));
        }
    }

    void resetProfiles()
    {
        if (mEngine->getNbOptimizationProfiles() >= 2)
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            mPrefillContext->setOptimizationProfileAsync(0, mStream);
            mDecodeContext->setOptimizationProfileAsync(1, mStream);
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
    }

    void setScalar(void* devicePtr, int64_t value, cudaStream_t stream = nullptr)
    {
        // Phase 2 must-fix 2: honor caller-supplied stream.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        CUDA_CHECK(cudaMemcpyAsync(devicePtr, &value, sizeof(int64_t), cudaMemcpyHostToDevice, s));
    }

    // Phase 2 must-fix 2: take an explicit decode-context pointer so callers can supply
    // a per-slot context. Old single-arg overload retained for CUDA-graph capture path
    // (which is intentionally pinned to mDecodeContext).
    void bindDecodeContext(nvinfer1::IExecutionContext* decode, int32_t group, int64_t actualPast,
        std::vector<DeviceBuffer> const& read, std::vector<DeviceBuffer> const& write, void* genStepPtr,
        void* pastLengthPtr, void* cachePositionPtr, Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr,
        std::vector<rt::Tensor>* slotRead = nullptr, std::vector<rt::Tensor>* slotWrite = nullptr)
    {
        // Phase 3a Iter2: when slot != nullptr, route mutable per-request state
        // (deviceEmbeds, deviceLogits, KV double-buffers) through the slot's
        // owned tensors. The caller supplies slotRead / slotWrite (pointers
        // into slot->kvA / slot->kvB after applying the same swap policy as
        // engine globals) so this function does not need to know which side
        // is currently read vs write. At N=1 callers pass slot=nullptr and
        // behavior is byte-identical to Phase 2.
        void* const embedsPtr = (slot != nullptr) ? slot->deviceEmbeds.rawPointer() : mDeviceEmbeds.get();
        void* const logitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        decode->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, mHiddenSize});
        decode->setTensorAddress("inputs_embeds", embedsPtr);
        decode->setInputShape("cache_position", nvinfer1::Dims{1, {1}});
        decode->setTensorAddress("cache_position", cachePositionPtr);
        decode->setTensorAddress(mLogitsName.c_str(), logitsPtr);
        if (mHasGenStep)
        {
            decode->setTensorAddress("gen_step", genStepPtr);
        }
        if (mHasPastLength)
        {
            decode->setTensorAddress("past_length", pastLengthPtr);
        }
        for (int32_t i = 0; i < mNumLayers; ++i)
        {
            decode->setInputShape(mPastKeyNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, actualPast, mHeadDim});
            decode->setInputShape(mPastValueNames[i].c_str(), nvinfer1::Dims4{1, mNumHeads, actualPast, mHeadDim});
            void* readK = (slot != nullptr && slotRead != nullptr) ? (*slotRead)[2 * i].rawPointer()
                                                                   : read[2 * i].get();
            void* readV = (slot != nullptr && slotRead != nullptr) ? (*slotRead)[2 * i + 1].rawPointer()
                                                                   : read[2 * i + 1].get();
            void* writeK = (slot != nullptr && slotWrite != nullptr) ? (*slotWrite)[2 * i].rawPointer()
                                                                     : write[2 * i].get();
            void* writeV = (slot != nullptr && slotWrite != nullptr) ? (*slotWrite)[2 * i + 1].rawPointer()
                                                                     : write[2 * i + 1].get();
            decode->setTensorAddress(mPastKeyNames[i].c_str(), readK);
            decode->setTensorAddress(mPastValueNames[i].c_str(), readV);
            decode->setTensorAddress(mNewPastKeyNames[i].c_str(), writeK);
            decode->setTensorAddress(mNewPastValueNames[i].c_str(), writeV);
        }
    }

    size_t decodeGraphIndex(int32_t group, std::vector<DeviceBuffer> const& read) const
    {
        bool const readIsA = !mKVA.empty() && read[0].get() == mKVA[0].get();
        return static_cast<size_t>(group) * 2 + (readIsA ? 1U : 0U);
    }

    void destroyDecodeCudaGraphs()
    {
        for (auto& graphPair : mDecodeCudaGraphs)
        {
            if (graphPair.second != nullptr)
            {
                cudaGraphExecDestroy(graphPair.second);
                graphPair.second = nullptr;
            }
            if (graphPair.first != nullptr)
            {
                cudaGraphDestroy(graphPair.first);
                graphPair.first = nullptr;
            }
        }
    }

    bool captureDecodeCudaGraph(int32_t group, std::vector<DeviceBuffer> const& read, std::vector<DeviceBuffer> const& write)
    {
        size_t const graphIdx = decodeGraphIndex(group, read);
        if (graphIdx >= mDecodeCudaGraphs.size())
        {
            return false;
        }
        int64_t const actualPast = group + 1;
        auto* genStepPtr = static_cast<char*>(mDeviceDecodeGenSteps.get()) + static_cast<size_t>(group) * sizeof(int64_t);
        auto* pastLengthPtr
            = static_cast<char*>(mDeviceDecodePastLengths.get()) + static_cast<size_t>(group) * sizeof(int64_t);
        bindDecodeContext(
            mDecodeContext.get(), group, actualPast, read, write, genStepPtr, pastLengthPtr, pastLengthPtr);

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        bool executeStatus = true;
        try
        {
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            kernel::qwen3TtsCpGatherEmbedding(static_cast<float const*>(mDeviceEmbeddingTable.get()), mCodebookSize,
                mHiddenSize, group - 1, static_cast<int32_t const*>(mDeviceSelectedTokens.get()),
                static_cast<float*>(mDeviceEmbeds.get()), mStream);
            executeStatus &= mDecodeContext->enqueueV3(mStream);
            CUDA_CHECK(cudaStreamEndCapture(mStream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        }
        catch (std::exception const& e)
        {
            LOG_WARNING("Failed to capture Qwen3-TTS CP decode CUDA graph group=%d: %s", group, e.what());
            static_cast<void>(cudaGetLastError());
            cudaStreamCaptureStatus streamStatus;
            CUDA_CHECK(cudaStreamIsCapturing(mStream, &streamStatus));
            if (streamStatus != cudaStreamCaptureStatusNone)
            {
                static_cast<void>(cudaStreamEndCapture(mStream, &graph));
                static_cast<void>(cudaGetLastError());
            }
            if (graphExec != nullptr)
            {
                cudaGraphExecDestroy(graphExec);
            }
            if (graph != nullptr)
            {
                cudaGraphDestroy(graph);
            }
            return false;
        }
        if (!executeStatus)
        {
            if (graphExec != nullptr)
            {
                cudaGraphExecDestroy(graphExec);
            }
            if (graph != nullptr)
            {
                cudaGraphDestroy(graph);
            }
            return false;
        }
        mDecodeCudaGraphs[graphIdx] = {graph, graphExec};
        return true;
    }

    bool launchDecodeCudaGraph(int32_t group, std::vector<DeviceBuffer> const& read, std::vector<DeviceBuffer> const& write)
    {
        size_t const graphIdx = decodeGraphIndex(group, read);
        if (graphIdx >= mDecodeCudaGraphs.size())
        {
            return false;
        }
        if (mDecodeCudaGraphs[graphIdx].second == nullptr && !captureDecodeCudaGraph(group, read, write))
        {
            return false;
        }
        CUDA_CHECK(cudaGraphLaunch(mDecodeCudaGraphs[graphIdx].second, mStream));
        return true;
    }

    using Clock = std::chrono::steady_clock;

    static double elapsedMs(Clock::time_point start)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    void profileAdd(double& target, Clock::time_point start)
    {
        if (mProfile)
        {
            target += elapsedMs(start);
        }
    }

    void logProfile() const
    {
        double const frames = static_cast<double>(std::max<int64_t>(1, mProfileFrames));
        double const groups = static_cast<double>(std::max<int64_t>(1, mProfileGroups));
        double const decodeGroups = static_cast<double>(std::max<int64_t>(1, mProfileDecodeGroups));
        LOG_WARNING("Qwen3-TTS CP profile frames=%lld groups=%lld device_hidden=%lld host_hidden=%lld "
                    "frame_ms=%.3f input_copy_ms=%.3f prefill_setup_ms=%.3f "
                    "decode_setup_ms/group=%.3f embed_copy_ms/group=%.3f sample_wait_ms/group=%.3f "
                    "sample_cpu_ms/group=%.3f",
            static_cast<long long>(mProfileFrames), static_cast<long long>(mProfileGroups),
            static_cast<long long>(mProfileDeviceHiddenFrames), static_cast<long long>(mProfileHostHiddenFrames),
            mProfileFrameTotalMs / frames, mProfileInputCopyMs / frames, mProfilePrefillSetupMs / frames,
            mProfileDecodeSetupMs / decodeGroups, mProfileEmbedCopyMs / decodeGroups, mProfileSampleWaitMs / groups,
            mProfileSampleCpuMs / groups);
    }

    int32_t sampleDeviceLogits(int32_t group, int32_t topK, float topP, float temperature, cudaStream_t stream = nullptr,
        Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // Phase 2 must-fix 2: honor caller-supplied stream.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        // Phase 3a Iter2: route mutable per-request sampling state through the slot
        // when supplied. RNG/scratch are per-request so concurrent samplers cannot
        // collide. At N=1 callers pass slot=nullptr and behavior is byte-identical.
        void* const deviceLogitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        auto& sampleLogits = (slot != nullptr) ? slot->sampleLogits : mSampleLogits;
        auto& sampleRaw = (slot != nullptr) ? slot->sampleRaw : mSampleRaw;
        auto& sampleVals = (slot != nullptr) ? slot->sampleVals : mSampleVals;
        auto& sampleProbs = (slot != nullptr) ? slot->sampleProbs : mSampleProbs;
        auto& rng = (slot != nullptr) ? slot->rng : mRng;
        auto const waitStart = Clock::now();
        size_t const offset = mLogitsName == "logits_all" ? static_cast<size_t>(group) * mCodebookSize : 0;
        // Phase 2 must-fix (codex round 3): each branch owns its own sync.
        // Previously a second unconditional cudaStreamSynchronize at the end
        // fired every decode step even though the bf16/half branch already
        // synced at line ~1394 before the host conversion loop. Removing
        // the duplicate halves the per-step host-side wait in the
        // common bf16 case.
        if (mLogitsElementSize == sizeof(float))
        {
            CUDA_CHECK(cudaMemcpyAsync(sampleLogits.data(),
                static_cast<char*>(deviceLogitsPtr) + offset * sizeof(float),
                sampleLogits.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(sampleRaw.data(),
                static_cast<char*>(deviceLogitsPtr) + offset * mLogitsElementSize,
                sampleRaw.size() * mLogitsElementSize, cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            for (size_t i = 0; i < sampleRaw.size(); ++i)
            {
                uint32_t bits = mLogitsAreBf16 ? (static_cast<uint32_t>(sampleRaw[i]) << 16) : halfToFloatBits(sampleRaw[i]);
                std::memcpy(&sampleLogits[i], &bits, sizeof(float));
            }
        }
        profileAdd(mProfileSampleWaitMs, waitStart);

        auto const cpuStart = Clock::now();
        static bool dumpedFirstCpGroup0Logits = false;
        if (mDumpDebug && group == 0 && !dumpedFirstCpGroup0Logits)
        {
            dumpVector("cp_logits_g0_f32.bin", sampleLogits);
            dumpedFirstCpGroup0Logits = true;
        }
        static int32_t dumpedCpSampleCalls = 0;
        if (mDumpDebug && mGreedy && dumpedCpSampleCalls < 32)
        {
            dumpVector("cp_call_" + std::to_string(dumpedCpSampleCalls) + "_g" + std::to_string(group)
                    + "_logits_f32.bin",
                sampleLogits);
            ++dumpedCpSampleCalls;
        }
        int32_t const k = (topK > 0 && topK < mCodebookSize) ? topK : mCodebookSize;
        for (int32_t i = 0; i < mCodebookSize; ++i)
        {
            sampleVals[i] = {sampleLogits[i], i};
        }
        if (mGreedy)
        {
            auto const best = std::max_element(sampleVals.begin(), sampleVals.end(), [](auto const& a, auto const& b) {
                if (a.first == b.first)
                {
                    return a.second > b.second;
                }
                return a.first < b.first;
            });
            return best->second;
        }
        std::partial_sort(sampleVals.begin(), sampleVals.begin() + k, sampleVals.end(),
            [](auto const& a, auto const& b) { return a.first > b.first; });
        double const maxVal = sampleVals[0].first;
        double sum = 0.0;
        float const temp = temperature > 1e-6f ? temperature : 0.9f;
        for (int32_t i = 0; i < k; ++i)
        {
            sampleProbs[i] = std::exp((static_cast<double>(sampleVals[i].first) - maxVal) / temp);
            sum += sampleProbs[i];
        }
        for (int32_t i = 0; i < k; ++i)
        {
            sampleProbs[i] /= sum;
        }
        if (topP > 0.0f && topP < 1.0f)
        {
            double cumulative = 0.0;
            int32_t keep = 0;
            for (; keep < k; ++keep)
            {
                cumulative += sampleProbs[static_cast<size_t>(keep)];
                if (cumulative >= static_cast<double>(topP))
                {
                    ++keep;
                    break;
                }
            }
            keep = std::max(1, std::min(keep, k));
            for (int32_t i = keep; i < k; ++i)
            {
                sampleProbs[static_cast<size_t>(i)] = 0.0;
            }
            double filteredSum = 0.0;
            for (int32_t i = 0; i < k; ++i)
            {
                filteredSum += sampleProbs[static_cast<size_t>(i)];
            }
            if (filteredSum > 0.0)
            {
                for (int32_t i = 0; i < k; ++i)
                {
                    sampleProbs[static_cast<size_t>(i)] /= filteredSum;
                }
            }
        }
        std::discrete_distribution<int32_t> dist(sampleProbs.begin(), sampleProbs.begin() + k);
        int32_t const token = sampleVals[dist(rng)].second;
        profileAdd(mProfileSampleCpuMs, cpuStart);
        return token;
    }

    void sampleDeviceLogitsGreedyToDevice(int32_t group, cudaStream_t stream = nullptr,
        Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // Phase 2 must-fix 2: honor caller-supplied stream.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        // Phase 3a Iter2: route logits/selected-token pointers through slot when supplied.
        void* const deviceLogitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        void* const deviceSelectedPtr
            = (slot != nullptr) ? slot->deviceSelectedTokens.rawPointer() : mDeviceSelectedTokens.get();
        size_t const offset = mLogitsName == "logits_all" ? static_cast<size_t>(group) * mCodebookSize : 0;
        kernel::qwen3TtsCpArgmax(static_cast<float const*>(deviceLogitsPtr) + offset, mCodebookSize, 0,
            static_cast<int32_t*>(deviceSelectedPtr) + group, s);
    }

    void sampleDeviceLogitsTopKTopPToDevice(
        int32_t group, int32_t topK, float topP, float temperature, cudaStream_t stream = nullptr,
        Qwen3OmniTTSRuntime::CodePredictorSlot* slot = nullptr)
    {
        // Phase 2 must-fix 2: honor caller-supplied stream.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        // Phase 3a Iter2: route GPU sampling buffers + Philox counter through slot when supplied.
        void* const deviceLogitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        void* const deviceSelectedPtr
            = (slot != nullptr) ? slot->deviceSelectedTokens.rawPointer() : mDeviceSelectedTokens.get();
        void* const deviceSamplingWorkspacePtr
            = (slot != nullptr) ? slot->deviceSamplingWorkspace.rawPointer() : mDeviceSamplingWorkspace.get();
        uint64_t& gpuSamplingOffset = (slot != nullptr) ? slot->gpuSamplingOffset : mGpuSamplingOffset;
        int32_t const effectiveTopK = topK > 0 ? std::min(topK, mCodebookSize) : mCodebookSize;
        float const effectiveTopP = (topP > 0.0f && topP <= 1.0f) ? topP : 1.0f;
        float const effectiveTemperature = temperature > 1e-6f ? temperature : 0.9f;
        size_t const offset = mLogitsName == "logits_all" ? static_cast<size_t>(group) * mCodebookSize : 0;
        SamplingParams const params(1, mCodebookSize, effectiveTemperature, effectiveTopK, effectiveTopP);
        rt::Tensor logits(static_cast<float*>(deviceLogitsPtr) + offset, {1, mCodebookSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor selected(static_cast<int32_t*>(deviceSelectedPtr) + group, {1, 1},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor workspace(
            deviceSamplingWorkspacePtr, {static_cast<int64_t>(mGpuSamplingWorkspaceBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8);
        topKtopPSamplingFromLogits(logits, selected, params, workspace, s, mGpuSamplingSeed, gpuSamplingOffset++);
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
    std::vector<float> mSampleLogits;
    std::vector<uint16_t> mSampleRaw;
    std::vector<std::pair<float, int32_t>> mSampleVals;
    std::vector<double> mSampleProbs;
    std::unique_ptr<nvinfer1::IRuntime, EngineDeleter> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, EngineDeleter> mPrefillContext;
    std::unique_ptr<nvinfer1::IExecutionContext, EngineDeleter> mDecodeContext;
    std::string mLogitsName;
    bool mHasGenStep{false};
    bool mHasPastLength{false};
    bool mLogitsAreBf16{false};
    bool mUseDeviceEmbeddingTable{false};
    bool mDumpDebug{false};
    bool mGreedy{false};
    bool mUseGpuGreedy{false};
    bool mUseGpuSampling{false};
    bool mUseDecodeCudaGraph{false};
    bool mProfile{false};
    int64_t mProfileFrames{0};
    int64_t mProfileGroups{0};
    int64_t mProfileDecodeGroups{0};
    int64_t mProfileDeviceHiddenFrames{0};
    int64_t mProfileHostHiddenFrames{0};
    double mProfileFrameTotalMs{0.0};
    double mProfileInputCopyMs{0.0};
    double mProfilePrefillSetupMs{0.0};
    double mProfileDecodeSetupMs{0.0};
    double mProfileEmbedCopyMs{0.0};
    double mProfileSampleWaitMs{0.0};
    double mProfileSampleCpuMs{0.0};
    size_t mKVElementSize{4};
    size_t mLogitsElementSize{4};
    DeviceBuffer mDeviceEmbeds;
    DeviceBuffer mDeviceCachePosition;
    DeviceBuffer mDeviceGenStep;
    DeviceBuffer mDevicePastLength;
    DeviceBuffer mDeviceDecodeGenSteps;
    DeviceBuffer mDeviceDecodePastLengths;
    DeviceBuffer mDeviceDummyKV;
    DeviceBuffer mDeviceLogits;
    DeviceBuffer mDeviceSelectedTokens;
    DeviceBuffer mDeviceEmbeddingTable;
    DeviceBuffer mDeviceSamplingWorkspace;
    std::vector<std::pair<cudaGraph_t, cudaGraphExec_t>> mDecodeCudaGraphs;
    size_t mGpuSamplingWorkspaceBytes{0};
    uint64_t mGpuSamplingSeed{0};
    uint64_t mGpuSamplingOffset{0};
    std::vector<DeviceBuffer> mKVA;
    std::vector<DeviceBuffer> mKVB;
    std::vector<std::string> mPastKeyNames;
    std::vector<std::string> mPastValueNames;
    std::vector<std::string> mNewPastKeyNames;
    std::vector<std::string> mNewPastValueNames;

    // ==============================================================
    // [Phase 3b-B-1] Per-engine SlotPool (N-slot counted pool).
    //
    // Capacity N is read from OVS_TTS_WORKER_CONCURRENCY at engine ctor
    // (clamped to [1,8], default 1). At N=1 behavior is byte-identical
    // to Phase 3b-A (single slot, lazy-allocated on first acquire).
    // At N>1 the pool may hand out up to N slots concurrently; slots
    // are lazy-allocated on demand and never released back to system
    // (kept alive in mAllSlots for the engine's lifetime).
    //
    // mAllSlots owns the slot unique_ptrs (kept alive across
    // acquire/release cycles). mFreeSlots is a LIFO free-list of raw
    // pointers into those owned slots. acquirePoolSlot waits on
    // mSlotPoolCv until either a free slot exists OR we can
    // lazy-allocate one more (allocated < capacity).
    //
    // Phase 3b-B-2 plug-in: the worker main loop will start
    // acquiring its own slot per request (instead of relying on the
    // implicit acquire-on-entry path inside generate/prefill/decode),
    // and OVS_TTS_WORKER_CONCURRENCY will finally be set >1.
    // ==============================================================
    mutable std::mutex mSlotPoolMutex;
    std::condition_variable mSlotPoolCv;
    std::vector<std::unique_ptr<Qwen3OmniTTSRuntime::CodePredictorSlot>> mAllSlots;
    std::vector<Qwen3OmniTTSRuntime::CodePredictorSlot*> mFreeSlots;
    int mSlotPoolCapacity{readTtsWorkerConcurrencyEnv("CodePredictorEngine")};

    // Lazily allocate one pooled slot using this engine's own factory
    // helpers (createExecutionContextPair + allocateSlot), matching the
    // logic in Qwen3OmniTTSRuntime::createCodePredictorSlot (cpp ~:4958).
    // Returns nullptr on context-pair failure.
    Qwen3OmniTTSRuntime::CodePredictorSlot* acquirePoolSlot()
    {
        std::unique_lock<std::mutex> lk(mSlotPoolMutex);
        mSlotPoolCv.wait(lk, [this] {
            return !mFreeSlots.empty() || static_cast<int>(mAllSlots.size()) < mSlotPoolCapacity;
        });
        if (mFreeSlots.empty())
        {
            auto slot = std::make_unique<Qwen3OmniTTSRuntime::CodePredictorSlot>();
            slot->stream = mStream;
            auto ctxPair = createExecutionContextPair();
            if (!ctxPair.first || !ctxPair.second)
            {
                LOG_WARNING("CodePredictorEngine pool: failed to allocate execution context pair");
                // Wake any waiter that might now succeed via existing free slots if released
                // concurrently; capacity unchanged so reattempts are fine.
                mSlotPoolCv.notify_one();
                return nullptr;
            }
            slot->prefillCtxOwned = std::move(ctxPair.first);
            slot->decodeCtxOwned = std::move(ctxPair.second);
            allocateSlot(*slot);
            auto* raw = slot.get();
            mAllSlots.push_back(std::move(slot));
            return raw; // hand out directly without putting on free-list
        }
        auto* slot = mFreeSlots.back();
        mFreeSlots.pop_back();
        return slot;
    }

    void releasePoolSlot(Qwen3OmniTTSRuntime::CodePredictorSlot* slot)
    {
        if (slot == nullptr)
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mSlotPoolMutex);
            mFreeSlots.push_back(slot);
        }
        mSlotPoolCv.notify_one();
    }
};

class Qwen3OmniTTSRuntime::Qwen3TTSTalkerEngine
{
    class DeviceBuffer
    {
    public:
        DeviceBuffer() = default;
        ~DeviceBuffer()
        {
            reset();
        }
        DeviceBuffer(DeviceBuffer const&) = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : mPtr(other.mPtr)
            , mBytes(other.mBytes)
        {
            other.mPtr = nullptr;
            other.mBytes = 0;
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                mPtr = other.mPtr;
                mBytes = other.mBytes;
                other.mPtr = nullptr;
                other.mBytes = 0;
            }
            return *this;
        }

        void allocate(size_t bytes)
        {
            reset();
            mBytes = bytes;
            CUDA_CHECK(cudaMalloc(&mPtr, bytes));
        }

        void reset()
        {
            if (mPtr)
            {
                cudaFree(mPtr);
                mPtr = nullptr;
            }
            mBytes = 0;
        }

        void* get() const
        {
            return mPtr;
        }

        size_t bytes() const
        {
            return mBytes;
        }

    private:
        void* mPtr{nullptr};
        size_t mBytes{0};
    };

public:
    Qwen3TTSTalkerEngine(std::filesystem::path const& enginePath, LLMEngineRunnerConfig const& config,
        int32_t maxSeqLen, cudaStream_t stream)
        : mConfig(config)
        , mMaxSeqLen(maxSeqLen)
        , mMaxKVSeqLen(maxSeqLen)
        , mStream(stream)
    {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error("Failed to open Qwen3-TTS Talker engine: " + enginePath.string());
        }
        std::vector<char> engineBlob((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
        mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
        mEngine.reset(mRuntime->deserializeCudaEngine(engineBlob.data(), engineBlob.size()));
        if (!mEngine)
        {
            throw std::runtime_error("Failed to deserialize Qwen3-TTS Talker engine: " + enginePath.string());
        }
        if (!hasTensor("past_key_0") || !hasTensor("new_past_key_0"))
        {
            throw std::runtime_error("Qwen3-TTS direct Talker engine must expose explicit KV tensors");
        }

        mEmbedsName = hasTensor("input_embeds") ? "input_embeds" : "inputs_embeds";
        mHasPositionIds = hasTensor("position_ids");
        mHasAttentionMask = hasTensor("attention_mask");
        mHasLastHidden = hasTensor("last_hidden");
        mLogitsType = mEngine->getTensorDataType("logits");
        mHiddenType = mHasLastHidden ? mEngine->getTensorDataType("last_hidden") : nvinfer1::DataType::kFLOAT;
        mKVElementSize = trtElementSize(mEngine->getTensorDataType("past_key_0"));
        mLogitsElementSize = trtElementSize(mLogitsType);
        mHiddenElementSize = trtElementSize(mHiddenType);
        mInputElementSize = trtElementSize(mEngine->getTensorDataType(mEmbedsName.c_str()));
        if (mEngine->getTensorDataType(mEmbedsName.c_str()) != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error("Qwen3-TTS direct Talker path currently expects FP32 inputs_embeds");
        }

        int32_t const profiles = mEngine->getNbOptimizationProfiles();
        mHasDualProfiles = profiles >= 2;
        if (mHasDualProfiles)
        {
            auto maxShape = mEngine->getProfileShape(mEmbedsName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            if (maxShape.nbDims >= 3 && maxShape.d[1] > 0)
            {
                mMaxSeqLen = std::min(mMaxSeqLen, static_cast<int32_t>(maxShape.d[1]));
            }
            auto kvMaxShape = mEngine->getProfileShape("past_key_0", 1, nvinfer1::OptProfileSelector::kMAX);
            if (kvMaxShape.nbDims >= 4 && kvMaxShape.d[2] > 0)
            {
                mMaxKVSeqLen = std::min(mMaxKVSeqLen, static_cast<int32_t>(kvMaxShape.d[2]));
            }
            mPrefillContext.reset(mEngine->createExecutionContext());
            mDecodeContext.reset(mEngine->createExecutionContext());
            mPrefillContext->setOptimizationProfileAsync(0, mStream);
            mDecodeContext->setOptimizationProfileAsync(1, mStream);
        }
        else
        {
            mPrefillContext.reset(mEngine->createExecutionContext());
            mDecodeContext.reset(mEngine->createExecutionContext());
            auto maxShape = mEngine->getProfileShape(mEmbedsName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            if (maxShape.nbDims >= 3 && maxShape.d[1] > 0)
            {
                mMaxSeqLen = std::min(mMaxSeqLen, static_cast<int32_t>(maxShape.d[1]));
            }
            auto kvMaxShape = mEngine->getProfileShape("past_key_0", 0, nvinfer1::OptProfileSelector::kMAX);
            if (kvMaxShape.nbDims >= 4 && kvMaxShape.d[2] > 0)
            {
                mMaxKVSeqLen = std::min(mMaxKVSeqLen, static_cast<int32_t>(kvMaxShape.d[2]));
            }
        }

        for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
        {
            mPastKeyNames.push_back("past_key_" + std::to_string(i));
            mPastValueNames.push_back("past_value_" + std::to_string(i));
            mNewPastKeyNames.push_back("new_past_key_" + std::to_string(i));
            mNewPastValueNames.push_back("new_past_value_" + std::to_string(i));
        }

        allocateBuffers();
        LOG_INFO("Qwen3-TTS explicit-KV Talker enabled: %s, maxSeq=%d, maxKV=%d",
            enginePath.string().c_str(), mMaxSeqLen, mMaxKVSeqLen);

        // [Phase 3b-B-4] Eager pre-allocate the slot pool — see the parallel
        // change in CodePredictorEngine for rationale. Avoids the first-time
        // cudaStreamSynchronize(mStream) blocker that adds ~2 s to client 2's
        // TTFA when the cold lazy-init races client 1's primary-stream work.
        eagerInitSlotPool();
    }

private:
    //! [Phase 3b-B-4] Pre-allocate every slot up to mSlotPoolCapacity.
    void eagerInitSlotPool()
    {
        std::lock_guard<std::mutex> lk(mSlotPoolMutex);
        for (int i = 0; i < mSlotPoolCapacity; ++i)
        {
            auto slot = std::make_unique<Qwen3OmniTTSRuntime::TalkerSlot>();
            slot->stream = mStream;
            auto ctxPair = createExecutionContextPair();
            if (!ctxPair.first || !ctxPair.second)
            {
                LOG_WARNING("TalkerEngine eager pool: pair-init failed at slot %d "
                            "(pool short; subsequent acquire() will retry)", i);
                break;
            }
            slot->prefillCtxOwned = std::move(ctxPair.first);
            slot->decodeCtxOwned = std::move(ctxPair.second);
            allocateSlot(*slot, mStream);
            mFreeSlots.push_back(slot.get());
            mAllSlots.push_back(std::move(slot));
        }
        LOG_INFO("Qwen3-TTS Talker SlotPool eager-initialised: %zu/%d slots ready",
                 mFreeSlots.size(), mSlotPoolCapacity);
    }

public:

    int32_t maxSeqLen() const
    {
        return mMaxSeqLen;
    }

    int32_t maxKVSeqLen() const
    {
        return mMaxKVSeqLen;
    }

    nvinfer1::DataType hiddenStatesDataType() const
    {
        return nvinfer1::DataType::kFLOAT;
    }

    //! [Phase 2 hook] Create a fresh execution context for this engine.
    //! Returned context shares engine weights but has its own CUDA state. Phase 3
    //! will use one such context per request slot. Returns nullptr on failure.
    std::unique_ptr<nvinfer1::IExecutionContext> createExecutionContext()
    {
        if (!mEngine)
        {
            return nullptr;
        }
        std::unique_ptr<nvinfer1::IExecutionContext> ctx{mEngine->createExecutionContext()};
        if (ctx && mHasDualProfiles)
        {
            // Match the default-context convention. Phase 3 will pick prefill vs decode
            // profile per request based on which slot operation is in flight.
            ctx->setOptimizationProfileAsync(1, mStream);
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
        return ctx;
    }

    //! [Phase 3a Iter6 must-fix-B] Create a paired (prefill, decode) context set
    //! matching the engine ctor convention (prefill→profile 0, decode→profile 1).
    //! Required because calling `createExecutionContext()` twice would assign
    //! profile 1 to both contexts and the slot would silently run prefill with
    //! the wrong TRT optimization profile. Returns {nullptr, nullptr} on failure.
    std::pair<std::unique_ptr<nvinfer1::IExecutionContext>, std::unique_ptr<nvinfer1::IExecutionContext>>
    createExecutionContextPair()
    {
        std::pair<std::unique_ptr<nvinfer1::IExecutionContext>, std::unique_ptr<nvinfer1::IExecutionContext>> out{
            nullptr, nullptr};
        if (!mEngine)
        {
            return out;
        }
        out.first.reset(mEngine->createExecutionContext());
        out.second.reset(mEngine->createExecutionContext());
        if (out.first && out.second && mHasDualProfiles)
        {
            out.first->setOptimizationProfileAsync(0, mStream);
            out.second->setOptimizationProfileAsync(1, mStream);
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
        return out;
    }

    //! [Phase 3a Iter1] Allocate per-request scratch tensors on `slot`,
    //! mirroring this engine's own allocateBuffers(). Tensors are byte buffers
    //! sized identically to the corresponding engine members. Prompt-KV cache
    //! tensors are left empty (the engine lazily allocates them on first
    //! cacheTo() call; the slot mirrors that policy). The attention mask byte
    //! buffer is filled with 1s on `stream` to match engine ctor behavior.
    void allocateSlot(Qwen3OmniTTSRuntime::TalkerSlot& slot, cudaStream_t stream) const
    {
        size_t const embedBytes
            = static_cast<size_t>(mMaxSeqLen) * mConfig.hiddenSize * mInputElementSize;
        size_t const logitsBytes
            = static_cast<size_t>(mMaxSeqLen) * mConfig.vocabSize * mLogitsElementSize;
        size_t const hiddenBytes
            = static_cast<size_t>(mMaxSeqLen) * mConfig.hiddenSize * mHiddenElementSize;
        size_t const kvBytes
            = static_cast<size_t>(mConfig.numKVHeads) * mMaxKVSeqLen * mConfig.headDim * mKVElementSize;
        size_t const maskLength = static_cast<size_t>(std::max(mMaxSeqLen, mMaxKVSeqLen + 1));
        size_t const maskBytes = maskLength * sizeof(int64_t);
        size_t const positionBytes = static_cast<size_t>(mMaxSeqLen) * sizeof(int64_t);

        slot.deviceEmbeds = rt::Tensor({static_cast<int64_t>(embedBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::deviceEmbeds");
        slot.deviceLogits = rt::Tensor({static_cast<int64_t>(logitsBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::deviceLogits");
        slot.deviceHidden = rt::Tensor({static_cast<int64_t>(hiddenBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::deviceHidden");
        slot.devicePositionIds = rt::Tensor({static_cast<int64_t>(positionBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::devicePositionIds");
        slot.deviceAttentionMask = rt::Tensor({static_cast<int64_t>(maskBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::deviceAttentionMask");

        // Mirror engine ctor: initialise mask to all-1 int64 on the slot stream.
        std::vector<int64_t> mask(maskLength, 1);
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        CUDA_CHECK(cudaMemcpyAsync(slot.deviceAttentionMask.rawPointer(), mask.data(),
            mask.size() * sizeof(int64_t), cudaMemcpyHostToDevice, s));

        slot.kvA.clear();
        slot.kvB.clear();
        slot.kvA.reserve(static_cast<size_t>(2 * mConfig.numDecoderLayers));
        slot.kvB.reserve(static_cast<size_t>(2 * mConfig.numDecoderLayers));
        for (int32_t i = 0; i < 2 * mConfig.numDecoderLayers; ++i)
        {
            slot.kvA.emplace_back(rt::Tensor({static_cast<int64_t>(kvBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "TalkerSlot::kvA[" + std::to_string(i) + "]"));
            slot.kvB.emplace_back(rt::Tensor({static_cast<int64_t>(kvBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "TalkerSlot::kvB[" + std::to_string(i) + "]"));
        }

        // Phase 3a Iter6 must-fix-A (codex round 3): the engine lazily allocates
        // mPromptKVs/mPromptLogits/mPromptHidden on the first cache miss, but
        // `ensurePromptBuffers` in the slot path REQUIRES these to be pre-sized
        // with capacity >= worst-case prompt size (cpp:~2381-2391 checks +
        // throws). Pre-allocate at worst case here so the first miss-then-store
        // path succeeds. Worst-case sizes mirror the regular KV (kvBytes), full
        // hidden states at mMaxSeqLen, and a single-token logits buffer.
        size_t const promptLogitsBytesMax = static_cast<size_t>(mConfig.vocabSize) * sizeof(float);
        size_t const promptHiddenBytesMax = static_cast<size_t>(mMaxSeqLen) * mConfig.hiddenSize * sizeof(float);
        slot.promptKVs.clear();
        slot.promptKVs.reserve(static_cast<size_t>(2 * mConfig.numDecoderLayers));
        for (int32_t i = 0; i < 2 * mConfig.numDecoderLayers; ++i)
        {
            slot.promptKVs.emplace_back(rt::Tensor({static_cast<int64_t>(kvBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "TalkerSlot::promptKVs[" + std::to_string(i) + "]"));
        }
        slot.promptLogits = rt::Tensor({static_cast<int64_t>(promptLogitsBytesMax)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::promptLogits");
        slot.promptHidden = rt::Tensor({static_cast<int64_t>(promptHiddenBytesMax)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "TalkerSlot::promptHidden");
        // Active sizes start at 0 so the size-cache check in ensurePromptBuffers
        // (cpp:~2369) sees a mismatch on first call and runs the (capacity-only)
        // resize bookkeeping. promptCacheValid stays false until a real store.
        slot.promptKVBytes = 0;
        slot.promptHiddenBytes = 0;
        slot.promptCacheValid = false;
        slot.promptCacheLen = 0;
        slot.promptCacheKey = 0;
        slot.seqLen = 0;
        slot.parity = 0;
    }

    bool prefill(std::vector<float> const& inputEmbeds, int32_t seqLen, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, cudaStream_t stream = nullptr,
        nvinfer1::IExecutionContext* ctxOverride = nullptr,
        Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // [Phase 3b-A] Activate the slot path at N=1: when the caller did not
        // supply an explicit slot, acquire the single pooled slot, recurse with
        // it, then release. Recursive entries already have slot != nullptr so
        // they do NOT re-enter the pool. ctxOverride/stream from the caller
        // remain authoritative; the pool slot owns its own paired ctxs but the
        // current N=1 path keeps the engine's default ctx/stream unless the
        // caller explicitly overrides — matching Phase 3a Iter6 semantics.
        if (slot == nullptr)
        {
            auto* pooled = acquirePoolSlot();
            if (pooled == nullptr)
            {
                LOG_ERROR("Qwen3-TTS direct Talker prefill: pool slot unavailable");
                return false;
            }
            bool const ok = prefill(inputEmbeds, seqLen, outputLogits, outputHiddenStates, stream, ctxOverride, pooled);
            releasePoolSlot(pooled);
            return ok;
        }
        if (seqLen <= 0)
        {
            LOG_ERROR("Qwen3-TTS direct Talker prefill seqLen out of range: %d (max %d)", seqLen, mMaxSeqLen);
            return false;
        }
        if (seqLen > mMaxSeqLen)
        {
            if (!mHasDualProfiles && mMaxSeqLen == 1)
            {
                LOG_INFO("Qwen3-TTS direct Talker using iterative prefill: seqLen=%d", seqLen);
                size_t const hidden = static_cast<size_t>(mConfig.hiddenSize);
                std::vector<float> step(inputEmbeds.begin(), inputEmbeds.begin() + hidden);
                if (!prefill(step, 1, outputLogits, outputHiddenStates, stream, ctxOverride, slot))
                {
                    return false;
                }
                for (int32_t pos = 1; pos < seqLen; ++pos)
                {
                    auto const begin = inputEmbeds.begin() + static_cast<size_t>(pos) * hidden;
                    step.assign(begin, begin + hidden);
                    if (!decode(step, outputLogits, outputHiddenStates, stream, ctxOverride, slot))
                    {
                        LOG_ERROR("Qwen3-TTS direct Talker iterative prefill failed at token %d/%d", pos + 1, seqLen);
                        return false;
                    }
                }
                return true;
            }
            LOG_ERROR("Qwen3-TTS direct Talker prefill seqLen out of range: %d (max %d)", seqLen, mMaxSeqLen);
            return false;
        }
        // Phase 2 must-fix 1: resolve per-invocation stream BEFORE reset() so that
        // reset's KV-memset runs on the same stream as the prefill enqueue below.
        // Phase 3a Iter5: route per-request scalars/buffers through `slot` when supplied.
        // At slot == nullptr behavior is byte-identical to Phase 2. mDeviceDummyKV
        // remains engine-shared (16-byte dummy used only as a zero-length placeholder).
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        nvinfer1::IExecutionContext* ctx = (ctxOverride != nullptr)
            ? ctxOverride
            : (mHasDualProfiles ? mPrefillContext.get() : mDecodeContext.get());
        int32_t& seqLenRef = (slot != nullptr) ? slot->seqLen : mSeqLen;
        int32_t& parity = (slot != nullptr) ? slot->parity : mParity;
        void* const deviceEmbedsPtr = (slot != nullptr) ? slot->deviceEmbeds.rawPointer() : mDeviceEmbeds.get();
        void* const deviceLogitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        void* const deviceHiddenPtr = (slot != nullptr) ? slot->deviceHidden.rawPointer() : mDeviceHidden.get();
        void* const devicePositionIdsPtr
            = (slot != nullptr) ? slot->devicePositionIds.rawPointer() : mDevicePositionIds.get();
        void* const deviceAttentionMaskPtr
            = (slot != nullptr) ? slot->deviceAttentionMask.rawPointer() : mDeviceAttentionMask.get();
        // Prefill writes into the "B" KV buffer (parity will be set to 1 below).
        auto kvBWritePtr = [&](size_t idx) -> void* {
            return (slot != nullptr) ? slot->kvB[idx].rawPointer() : mKVB[idx].get();
        };

        resetProfiles();
        reset(s, slot);

        CUDA_CHECK(cudaMemcpyAsync(deviceEmbedsPtr, inputEmbeds.data(),
            static_cast<size_t>(seqLen) * mConfig.hiddenSize * sizeof(float), cudaMemcpyHostToDevice, s));

        ctx->setInputShape(mEmbedsName.c_str(), nvinfer1::Dims3{1, seqLen, mConfig.hiddenSize});
        ctx->setTensorAddress(mEmbedsName.c_str(), deviceEmbedsPtr);
        if (mHasAttentionMask)
        {
            ctx->setInputShape("attention_mask", nvinfer1::Dims2{1, seqLen});
            ctx->setTensorAddress("attention_mask", deviceAttentionMaskPtr);
        }
        if (mHasPositionIds)
        {
            fillPositions(seqLen, s, slot);
            ctx->setInputShape("position_ids", nvinfer1::Dims2{1, seqLen});
            ctx->setTensorAddress("position_ids", devicePositionIdsPtr);
        }

        nvinfer1::Dims4 emptyKV{1, mConfig.numKVHeads, 0, mConfig.headDim};
        for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
        {
            ctx->setInputShape(mPastKeyNames[i].c_str(), emptyKV);
            ctx->setInputShape(mPastValueNames[i].c_str(), emptyKV);
            ctx->setTensorAddress(mPastKeyNames[i].c_str(), mDeviceDummyKV.get());
            ctx->setTensorAddress(mPastValueNames[i].c_str(), mDeviceDummyKV.get());
            ctx->setTensorAddress(mNewPastKeyNames[i].c_str(), kvBWritePtr(2 * i));
            ctx->setTensorAddress(mNewPastValueNames[i].c_str(), kvBWritePtr(2 * i + 1));
        }
        ctx->setTensorAddress("logits", deviceLogitsPtr);
        if (mHasLastHidden)
        {
            ctx->setTensorAddress("last_hidden", deviceHiddenPtr);
        }
        if (!ctx->enqueueV3(s))
        {
            LOG_ERROR("Qwen3-TTS direct Talker prefill failed");
            return false;
        }

        std::vector<float> logits(static_cast<size_t>(seqLen) * mConfig.vocabSize);
        copyDeviceToFloat(deviceLogitsPtr, mLogitsType, logits.data(), logits.size(), s);
        CUDA_CHECK(cudaMemcpyAsync(outputLogits.rawPointer(), logits.data() + static_cast<size_t>(seqLen - 1) * mConfig.vocabSize,
            static_cast<size_t>(mConfig.vocabSize) * sizeof(float), cudaMemcpyHostToDevice, s));

        if (mHasLastHidden)
        {
            std::vector<float> hidden(static_cast<size_t>(seqLen) * mConfig.hiddenSize);
            copyDeviceToFloat(deviceHiddenPtr, mHiddenType, hidden.data(), hidden.size(), s);
            check::check(outputHiddenStates.reshape({1, seqLen, mConfig.hiddenSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(outputHiddenStates.rawPointer(), hidden.data(),
                hidden.size() * sizeof(float), cudaMemcpyHostToDevice, s));
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
        seqLenRef = seqLen;
        parity = 1;
        return true;
    }

    bool prefillWithPromptCache(std::vector<float> const& inputEmbeds, int32_t seqLen,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream = nullptr,
        nvinfer1::IExecutionContext* ctxOverride = nullptr,
        Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // [Phase 3b-A] Activate the slot path at N=1 (see prefill above for the
        // rationale). Pool acquire here threads `pooled` through to the nested
        // prefill/restorePromptCache helpers so the entire prompt-cache hit/miss
        // path runs against the slot's promptKVs/promptLogits/promptHidden
        // buffers — not the engine globals.
        if (slot == nullptr)
        {
            auto* pooled = acquirePoolSlot();
            if (pooled == nullptr)
            {
                LOG_ERROR("Qwen3-TTS Talker prefillWithPromptCache: pool slot unavailable");
                return false;
            }
            bool const ok = prefillWithPromptCache(
                inputEmbeds, seqLen, outputLogits, outputHiddenStates, stream, ctxOverride, pooled);
            releasePoolSlot(pooled);
            return ok;
        }
        // Phase 3a Iter6: route prompt-KV cache state through `slot` when supplied.
        // At slot == nullptr the path is byte-identical to Phase 2. Pattern mirrors
        // decode/reset (cpp ~:2127, ~:2281): aliases for the hit/miss decision fields,
        // and forward `slot` (plus stream/ctx) to all helper calls so the cache lives
        // on the same per-request buffers as the slot's KV double-buffer.
        if (seqLen <= 0 || seqLen > mMaxSeqLen)
        {
            return prefill(inputEmbeds, seqLen, outputLogits, outputHiddenStates, stream, ctxOverride, slot);
        }

        bool& promptCacheValid = (slot != nullptr) ? slot->promptCacheValid : mPromptCacheValid;
        uint64_t& promptCacheKey = (slot != nullptr) ? slot->promptCacheKey : mPromptCacheKey;
        int32_t& promptCacheLen = (slot != nullptr) ? slot->promptCacheLen : mPromptCacheLen;

        uint64_t const promptKey = hashPrompt(inputEmbeds, seqLen);
        if (!promptCacheValid || promptCacheKey != promptKey || promptCacheLen != seqLen)
        {
            LOG_INFO("Qwen3-TTS Talker prompt KV cache miss: seqLen=%d", seqLen);
            if (!prefill(inputEmbeds, seqLen, outputLogits, outputHiddenStates, stream, ctxOverride, slot))
            {
                return false;
            }
            storePromptCache(seqLen, promptKey, outputLogits, outputHiddenStates, stream, slot);
        }
        else
        {
            LOG_INFO("Qwen3-TTS Talker prompt KV cache hit: seqLen=%d", seqLen);
            restorePromptCache(outputLogits, outputHiddenStates, stream, slot);
        }
        return true;
    }

    bool decode(std::vector<float> const& inputEmbed, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
        cudaStream_t stream = nullptr, nvinfer1::IExecutionContext* ctxOverride = nullptr,
        Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // [Phase 3b-A] Activate the slot path at N=1 — see prefill().
        if (slot == nullptr)
        {
            auto* pooled = acquirePoolSlot();
            if (pooled == nullptr)
            {
                LOG_ERROR("Qwen3-TTS direct Talker decode: pool slot unavailable");
                return false;
            }
            bool const ok = decode(inputEmbed, outputLogits, outputHiddenStates, stream, ctxOverride, pooled);
            releasePoolSlot(pooled);
            return ok;
        }
        // Phase 3a Iter4: route mutable per-request state through `slot` when supplied.
        // At slot == nullptr the path is byte-identical to Phase 2. Pattern mirrors CP
        // generatePreparedInputs (cpp ~:957): aliases for engine-shared scalars/counters,
        // and `void*` locals (acquired via slot->X.rawPointer() vs mX.get()) for device
        // buffers since slot uses rt::Tensor while engine uses DeviceBuffer.
        int32_t& seqLen = (slot != nullptr) ? slot->seqLen : mSeqLen;
        int32_t& parity = (slot != nullptr) ? slot->parity : mParity;

        if (seqLen <= 0 || seqLen >= mMaxKVSeqLen)
        {
            LOG_ERROR("Qwen3-TTS direct Talker decode seqLen out of range: %d (maxKV %d)", seqLen, mMaxKVSeqLen);
            return false;
        }

        // Phase 2: resolve per-invocation stream and execution context, falling back to defaults.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        nvinfer1::IExecutionContext* ctx = (ctxOverride != nullptr) ? ctxOverride : mDecodeContext.get();
        void* const deviceEmbedsPtr = (slot != nullptr) ? slot->deviceEmbeds.rawPointer() : mDeviceEmbeds.get();
        void* const deviceLogitsPtr = (slot != nullptr) ? slot->deviceLogits.rawPointer() : mDeviceLogits.get();
        void* const deviceHiddenPtr = (slot != nullptr) ? slot->deviceHidden.rawPointer() : mDeviceHidden.get();
        void* const devicePositionIdsPtr
            = (slot != nullptr) ? slot->devicePositionIds.rawPointer() : mDevicePositionIds.get();
        void* const deviceAttentionMaskPtr
            = (slot != nullptr) ? slot->deviceAttentionMask.rawPointer() : mDeviceAttentionMask.get();

        CUDA_CHECK(cudaMemcpyAsync(
            deviceEmbedsPtr, inputEmbed.data(), static_cast<size_t>(mConfig.hiddenSize) * sizeof(float),
            cudaMemcpyHostToDevice, s));

        // KV double-buffer: engine path uses DeviceBuffer vectors, slot path uses rt::Tensor
        // vectors. Resolve a single void* per (layer, K/V) below via the kvReadPtr/kvWritePtr
        // helpers to avoid forking the loop body.
        std::vector<DeviceBuffer>& read = (parity == 0) ? mKVA : mKVB;
        std::vector<DeviceBuffer>& write = (parity == 0) ? mKVB : mKVA;
        std::vector<rt::Tensor>* slotRead = (slot != nullptr) ? ((parity == 0) ? &slot->kvA : &slot->kvB) : nullptr;
        std::vector<rt::Tensor>* slotWrite = (slot != nullptr) ? ((parity == 0) ? &slot->kvB : &slot->kvA) : nullptr;
        auto kvReadPtr = [&](size_t idx) -> void* {
            return (slot != nullptr) ? (*slotRead)[idx].rawPointer() : read[idx].get();
        };
        auto kvWritePtr = [&](size_t idx) -> void* {
            return (slot != nullptr) ? (*slotWrite)[idx].rawPointer() : write[idx].get();
        };

        ctx->setInputShape(mEmbedsName.c_str(), nvinfer1::Dims3{1, 1, mConfig.hiddenSize});
        ctx->setTensorAddress(mEmbedsName.c_str(), deviceEmbedsPtr);
        if (mHasAttentionMask)
        {
            ctx->setInputShape("attention_mask", nvinfer1::Dims2{1, seqLen + 1});
            ctx->setTensorAddress("attention_mask", deviceAttentionMaskPtr);
        }
        if (mHasPositionIds)
        {
            int64_t const position = seqLen;
            CUDA_CHECK(cudaMemcpyAsync(devicePositionIdsPtr, &position, sizeof(int64_t), cudaMemcpyHostToDevice, s));
            ctx->setInputShape("position_ids", nvinfer1::Dims2{1, 1});
            ctx->setTensorAddress("position_ids", devicePositionIdsPtr);
        }
        nvinfer1::Dims4 kvShape{1, mConfig.numKVHeads, seqLen, mConfig.headDim};
        for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
        {
            ctx->setInputShape(mPastKeyNames[i].c_str(), kvShape);
            ctx->setInputShape(mPastValueNames[i].c_str(), kvShape);
            ctx->setTensorAddress(mPastKeyNames[i].c_str(), kvReadPtr(2 * i));
            ctx->setTensorAddress(mPastValueNames[i].c_str(), kvReadPtr(2 * i + 1));
            ctx->setTensorAddress(mNewPastKeyNames[i].c_str(), kvWritePtr(2 * i));
            ctx->setTensorAddress(mNewPastValueNames[i].c_str(), kvWritePtr(2 * i + 1));
        }
        ctx->setTensorAddress("logits", deviceLogitsPtr);
        if (mHasLastHidden)
        {
            ctx->setTensorAddress("last_hidden", deviceHiddenPtr);
        }
        if (!ctx->enqueueV3(s))
        {
            LOG_ERROR("Qwen3-TTS direct Talker decode failed");
            return false;
        }

        std::vector<float> logits(static_cast<size_t>(mConfig.vocabSize));
        copyDeviceToFloat(deviceLogitsPtr, mLogitsType, logits.data(), logits.size(), s);
        CUDA_CHECK(cudaMemcpyAsync(
            outputLogits.rawPointer(), logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice, s));

        if (mHasLastHidden)
        {
            std::vector<float> hidden(static_cast<size_t>(mConfig.hiddenSize));
            copyDeviceToFloat(deviceHiddenPtr, mHiddenType, hidden.data(), hidden.size(), s);
            check::check(outputHiddenStates.reshape({1, 1, mConfig.hiddenSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(outputHiddenStates.rawPointer(), hidden.data(), hidden.size() * sizeof(float),
                cudaMemcpyHostToDevice, s));
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
        ++seqLen;
        parity ^= 1;
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
        return dataTypeSize(dtype);
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

    void allocateBuffers()
    {
        size_t const embedBytes = static_cast<size_t>(mMaxSeqLen) * mConfig.hiddenSize * mInputElementSize;
        size_t const logitsBytes = static_cast<size_t>(mMaxSeqLen) * mConfig.vocabSize * mLogitsElementSize;
        size_t const hiddenBytes = static_cast<size_t>(mMaxSeqLen) * mConfig.hiddenSize * mHiddenElementSize;
        size_t const kvBytes = static_cast<size_t>(mConfig.numKVHeads) * mMaxKVSeqLen * mConfig.headDim * mKVElementSize;
        size_t const maskLength = static_cast<size_t>(std::max(mMaxSeqLen, mMaxKVSeqLen + 1));
        mDeviceEmbeds.allocate(embedBytes);
        mDeviceLogits.allocate(logitsBytes);
        mDeviceHidden.allocate(hiddenBytes);
        mDevicePositionIds.allocate(static_cast<size_t>(mMaxSeqLen) * sizeof(int64_t));
        mDeviceAttentionMask.allocate(maskLength * sizeof(int64_t));
        mDeviceDummyKV.allocate(16);
        std::vector<int64_t> mask(maskLength, 1);
        CUDA_CHECK(cudaMemcpyAsync(mDeviceAttentionMask.get(), mask.data(), mask.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, mStream));

        mKVA.resize(static_cast<size_t>(2 * mConfig.numDecoderLayers));
        mKVB.resize(static_cast<size_t>(2 * mConfig.numDecoderLayers));
        for (int32_t i = 0; i < 2 * mConfig.numDecoderLayers; ++i)
        {
            mKVA[static_cast<size_t>(i)].allocate(kvBytes);
            mKVB[static_cast<size_t>(i)].allocate(kvBytes);
        }
    }

    void reset(cudaStream_t stream = nullptr, Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // Phase 2 must-fix 1: honor caller-supplied stream so reset's KV-memset
        // is ordered with prefill's downstream H2D/enqueue on the same stream.
        // Phase 3a Iter5: route per-request KV double-buffers through `slot` when
        // supplied; slot path uses rt::Tensor (rawPointer()/bytes()) instead of
        // engine DeviceBuffer. At slot == nullptr the path is byte-identical to
        // Phase 2.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        int32_t& seqLen = (slot != nullptr) ? slot->seqLen : mSeqLen;
        int32_t& parity = (slot != nullptr) ? slot->parity : mParity;
        seqLen = 0;
        parity = 0;
        if (slot != nullptr)
        {
            for (auto& p : slot->kvA)
            {
                CUDA_CHECK(cudaMemsetAsync(p.rawPointer(), 0, static_cast<size_t>(p.getMemoryCapacity()), s));
            }
            for (auto& p : slot->kvB)
            {
                CUDA_CHECK(cudaMemsetAsync(p.rawPointer(), 0, static_cast<size_t>(p.getMemoryCapacity()), s));
            }
        }
        else
        {
            for (auto const& p : mKVA)
            {
                CUDA_CHECK(cudaMemsetAsync(p.get(), 0, p.bytes(), s));
            }
            for (auto const& p : mKVB)
            {
                CUDA_CHECK(cudaMemsetAsync(p.get(), 0, p.bytes(), s));
            }
        }
    }

    size_t kvBytesForSeqLen(int32_t seqLen) const
    {
        return static_cast<size_t>(mConfig.numKVHeads) * static_cast<size_t>(seqLen) * mConfig.headDim * mKVElementSize;
    }

    static uint64_t fnv1a(uint64_t hash, void const* data, size_t bytes)
    {
        auto const* ptr = static_cast<uint8_t const*>(data);
        for (size_t i = 0; i < bytes; ++i)
        {
            hash ^= ptr[i];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    uint64_t hashPrompt(std::vector<float> const& inputEmbeds, int32_t seqLen) const
    {
        uint64_t hash = 1469598103934665603ULL;
        hash = fnv1a(hash, &seqLen, sizeof(seqLen));
        hash = fnv1a(hash, &mConfig.hiddenSize, sizeof(mConfig.hiddenSize));
        size_t const promptFloats = static_cast<size_t>(seqLen) * mConfig.hiddenSize;
        return fnv1a(hash, inputEmbeds.data(), promptFloats * sizeof(float));
    }

    void ensurePromptBuffers(int32_t seqLen, Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // Phase 3a Iter6: route prompt-cache buffers through `slot` when supplied.
        // Slot path uses rt::Tensor (allocate via rt::BufferManager when constructed;
        // here we (re)allocate by reshaping the existing host-owned tensors). At
        // slot == nullptr the path is byte-identical to Phase 2.
        size_t const kvBytes = kvBytesForSeqLen(seqLen);
        size_t const logitsBytes = static_cast<size_t>(mConfig.vocabSize) * sizeof(float);
        size_t const hiddenBytes = static_cast<size_t>(seqLen) * mConfig.hiddenSize * sizeof(float);

        size_t& promptKVBytes = (slot != nullptr) ? slot->promptKVBytes : mPromptKVBytes;
        size_t& promptHiddenBytes = (slot != nullptr) ? slot->promptHiddenBytes : mPromptHiddenBytes;
        size_t const numKV = (slot != nullptr) ? slot->kvB.size() : mKVB.size();

        if (slot != nullptr)
        {
            if (promptKVBytes == kvBytes && promptHiddenBytes == hiddenBytes && slot->promptKVs.size() == numKV)
            {
                return;
            }
            promptKVBytes = kvBytes;
            promptHiddenBytes = hiddenBytes;
            // For rt::Tensor we reallocate by reshape if backing storage is sufficient;
            // otherwise the slot owner (createTalkerSlot, Phase 3b) is responsible for
            // having allocated tensors with capacity >= the worst-case prompt size. To
            // keep this Iter byte-equivalent we simply require capacity to suffice and
            // record the active size via the *Bytes fields. (Verified: slot path is
            // dormant in Iter 6 because callers pass slot==nullptr.)
            check::check(slot->promptKVs.size() == numKV,
                "ensurePromptBuffers: slot->promptKVs must be pre-sized to slot->kvB.size()");
            for (size_t i = 0; i < numKV; ++i)
            {
                check::check(static_cast<size_t>(slot->promptKVs[i].getMemoryCapacity()) >= kvBytes,
                    "ensurePromptBuffers: slot->promptKVs[i] capacity too small");
            }
            check::check(static_cast<size_t>(slot->promptLogits.getMemoryCapacity()) >= logitsBytes,
                "ensurePromptBuffers: slot->promptLogits capacity too small");
            check::check(static_cast<size_t>(slot->promptHidden.getMemoryCapacity()) >= hiddenBytes,
                "ensurePromptBuffers: slot->promptHidden capacity too small");
            return;
        }

        if (promptKVBytes == kvBytes && promptHiddenBytes == hiddenBytes && mPromptKVs.size() == mKVB.size())
        {
            return;
        }
        promptKVBytes = kvBytes;
        promptHiddenBytes = hiddenBytes;
        mPromptKVs.clear();
        mPromptKVs.resize(mKVB.size());
        for (auto& buffer : mPromptKVs)
        {
            buffer.allocate(kvBytes);
        }
        mPromptLogits.allocate(logitsBytes);
        mPromptHidden.allocate(hiddenBytes);
    }

    void storePromptCache(int32_t seqLen, uint64_t promptKey, rt::Tensor const& outputLogits,
        rt::Tensor const& outputHiddenStates, cudaStream_t stream = nullptr,
        Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // Phase 3a Iter6: route prompt-cache writes through `slot` when supplied.
        // Stream caveat (per Iter-4 pattern): accept caller stream so the store is
        // ordered with the just-completed prefill on the same stream. At
        // slot == nullptr / stream == nullptr the path is byte-identical to Phase 2.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        ensurePromptBuffers(seqLen, slot);
        size_t const promptKVBytes = (slot != nullptr) ? slot->promptKVBytes : mPromptKVBytes;
        size_t const promptHiddenBytes = (slot != nullptr) ? slot->promptHiddenBytes : mPromptHiddenBytes;
        size_t const numKV = (slot != nullptr) ? slot->kvB.size() : mKVB.size();
        auto srcKVPtr = [&](size_t i) -> void const* {
            return (slot != nullptr) ? slot->kvB[i].rawPointer() : mKVB[i].get();
        };
        auto dstKVPtr = [&](size_t i) -> void* {
            return (slot != nullptr) ? slot->promptKVs[i].rawPointer() : mPromptKVs[i].get();
        };
        void* const promptLogitsPtr = (slot != nullptr) ? slot->promptLogits.rawPointer() : mPromptLogits.get();
        void* const promptHiddenPtr = (slot != nullptr) ? slot->promptHidden.rawPointer() : mPromptHidden.get();

        for (size_t i = 0; i < numKV; ++i)
        {
            CUDA_CHECK(cudaMemcpyAsync(dstKVPtr(i), srcKVPtr(i), promptKVBytes, cudaMemcpyDeviceToDevice, s));
        }
        CUDA_CHECK(cudaMemcpyAsync(promptLogitsPtr, outputLogits.rawPointer(),
            static_cast<size_t>(mConfig.vocabSize) * sizeof(float), cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(
            promptHiddenPtr, outputHiddenStates.rawPointer(), promptHiddenBytes, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        int32_t& promptCacheLen = (slot != nullptr) ? slot->promptCacheLen : mPromptCacheLen;
        uint64_t& promptCacheKey = (slot != nullptr) ? slot->promptCacheKey : mPromptCacheKey;
        bool& promptCacheValid = (slot != nullptr) ? slot->promptCacheValid : mPromptCacheValid;
        promptCacheLen = seqLen;
        promptCacheKey = promptKey;
        promptCacheValid = true;
    }

    void restorePromptCache(rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
        cudaStream_t stream = nullptr, Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // Phase 3a Iter6: route prompt-cache reads through `slot` when supplied.
        // Final-state writes go to slot->{seqLen,parity} so subsequent decode() on
        // the same slot continues from the cached position. At slot == nullptr /
        // stream == nullptr the path is byte-identical to Phase 2.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        bool const promptCacheValid = (slot != nullptr) ? slot->promptCacheValid : mPromptCacheValid;
        check::check(promptCacheValid, "restorePromptCache called without a valid prompt cache");

        size_t const promptKVBytes = (slot != nullptr) ? slot->promptKVBytes : mPromptKVBytes;
        size_t const promptHiddenBytes = (slot != nullptr) ? slot->promptHiddenBytes : mPromptHiddenBytes;
        int32_t const promptCacheLen = (slot != nullptr) ? slot->promptCacheLen : mPromptCacheLen;
        size_t const numKV = (slot != nullptr) ? slot->kvB.size() : mKVB.size();
        auto srcKVPtr = [&](size_t i) -> void const* {
            return (slot != nullptr) ? slot->promptKVs[i].rawPointer() : mPromptKVs[i].get();
        };
        auto dstKVPtr = [&](size_t i) -> void* {
            return (slot != nullptr) ? slot->kvB[i].rawPointer() : mKVB[i].get();
        };
        void const* const promptLogitsPtr = (slot != nullptr) ? slot->promptLogits.rawPointer() : mPromptLogits.get();
        void const* const promptHiddenPtr = (slot != nullptr) ? slot->promptHidden.rawPointer() : mPromptHidden.get();

        for (size_t i = 0; i < numKV; ++i)
        {
            CUDA_CHECK(cudaMemcpyAsync(dstKVPtr(i), srcKVPtr(i), promptKVBytes, cudaMemcpyDeviceToDevice, s));
        }
        CUDA_CHECK(cudaMemcpyAsync(outputLogits.rawPointer(), promptLogitsPtr,
            static_cast<size_t>(mConfig.vocabSize) * sizeof(float), cudaMemcpyDeviceToDevice, s));
        check::check(outputHiddenStates.reshape({1, promptCacheLen, mConfig.hiddenSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(
            outputHiddenStates.rawPointer(), promptHiddenPtr, promptHiddenBytes, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        int32_t& seqLen = (slot != nullptr) ? slot->seqLen : mSeqLen;
        int32_t& parity = (slot != nullptr) ? slot->parity : mParity;
        seqLen = promptCacheLen;
        parity = 1;
    }

    void resetProfiles()
    {
        if (mHasDualProfiles)
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            mPrefillContext->setOptimizationProfileAsync(0, mStream);
            mDecodeContext->setOptimizationProfileAsync(1, mStream);
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
    }

    // Phase 2 must-fix (codex round 3): accept a per-invocation stream so
    // these helpers no longer pin to mStream. Silent regression at N=1
    // (s == mStream via nullptr fallback) but mandatory before Phase 3
    // gives callers non-null overrides — otherwise prefill/decode would
    // enqueue on the slot stream while the position copy stays on
    // mStream, creating a cross-stream data race.
    void fillPositions(int32_t seqLen, cudaStream_t stream = nullptr, Qwen3OmniTTSRuntime::TalkerSlot* slot = nullptr)
    {
        // Phase 3a Iter5: route mDevicePositionIds through `slot` when supplied.
        // At slot == nullptr behavior is byte-identical to Phase 2.
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        void* const devicePositionIdsPtr
            = (slot != nullptr) ? slot->devicePositionIds.rawPointer() : mDevicePositionIds.get();
        std::vector<int64_t> positions(static_cast<size_t>(seqLen));
        std::iota(positions.begin(), positions.end(), 0);
        CUDA_CHECK(cudaMemcpyAsync(
            devicePositionIdsPtr, positions.data(), positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice, s));
    }

    void copyDeviceToFloat(void const* devicePtr, nvinfer1::DataType dtype, float* hostPtr, size_t elements,
        cudaStream_t stream = nullptr)
    {
        cudaStream_t const s = (stream != nullptr) ? stream : mStream;
        if (dtype == nvinfer1::DataType::kFLOAT)
        {
            CUDA_CHECK(cudaMemcpyAsync(hostPtr, devicePtr, elements * sizeof(float), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            return;
        }
        std::vector<uint16_t> raw(elements);
        CUDA_CHECK(cudaMemcpyAsync(raw.data(), devicePtr, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        for (size_t i = 0; i < elements; ++i)
        {
            uint32_t const bits = dtype == nvinfer1::DataType::kBF16 ? (static_cast<uint32_t>(raw[i]) << 16)
                                                                     : halfToFloatBits(raw[i]);
            std::memcpy(hostPtr + i, &bits, sizeof(float));
        }
    }

    LLMEngineRunnerConfig mConfig;
    int32_t mMaxSeqLen{};
    int32_t mMaxKVSeqLen{};
    int32_t mSeqLen{0};
    int32_t mParity{0};
    cudaStream_t mStream{};
    std::unique_ptr<nvinfer1::IRuntime, EngineDeleter> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, EngineDeleter> mPrefillContext;
    std::unique_ptr<nvinfer1::IExecutionContext, EngineDeleter> mDecodeContext;
    std::string mEmbedsName{"inputs_embeds"};
    bool mHasDualProfiles{false};
    bool mHasPositionIds{false};
    bool mHasAttentionMask{false};
    bool mHasLastHidden{false};
    nvinfer1::DataType mLogitsType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mHiddenType{nvinfer1::DataType::kFLOAT};
    size_t mInputElementSize{4};
    size_t mLogitsElementSize{4};
    size_t mHiddenElementSize{4};
    size_t mKVElementSize{4};
    DeviceBuffer mDeviceEmbeds;
    DeviceBuffer mDeviceLogits;
    DeviceBuffer mDeviceHidden;
    DeviceBuffer mDevicePositionIds;
    DeviceBuffer mDeviceAttentionMask;
    DeviceBuffer mDeviceDummyKV;
    std::vector<DeviceBuffer> mKVA;
    std::vector<DeviceBuffer> mKVB;
    std::vector<DeviceBuffer> mPromptKVs;
    DeviceBuffer mPromptLogits;
    DeviceBuffer mPromptHidden;
    size_t mPromptKVBytes{0};
    size_t mPromptHiddenBytes{0};
    bool mPromptCacheValid{false};
    int32_t mPromptCacheLen{0};
    uint64_t mPromptCacheKey{0};
    std::vector<std::string> mPastKeyNames;
    std::vector<std::string> mPastValueNames;
    std::vector<std::string> mNewPastKeyNames;
    std::vector<std::string> mNewPastValueNames;

    // ==============================================================
    // [Phase 3b-B-1] Per-engine SlotPool (N-slot counted pool).
    //
    // See CodePredictorEngine's pool block above for the rationale and
    // semantics. Same shape: mAllSlots owns slot unique_ptrs; mFreeSlots
    // is a LIFO free-list of raw pointers; capacity comes from
    // OVS_TTS_WORKER_CONCURRENCY (clamped [1,8], default 1).
    // ==============================================================
    mutable std::mutex mSlotPoolMutex;
    std::condition_variable mSlotPoolCv;
    std::vector<std::unique_ptr<Qwen3OmniTTSRuntime::TalkerSlot>> mAllSlots;
    std::vector<Qwen3OmniTTSRuntime::TalkerSlot*> mFreeSlots;
    int mSlotPoolCapacity{readTtsWorkerConcurrencyEnv("TalkerEngine")};

    Qwen3OmniTTSRuntime::TalkerSlot* acquirePoolSlot()
    {
        std::unique_lock<std::mutex> lk(mSlotPoolMutex);
        mSlotPoolCv.wait(lk, [this] {
            return !mFreeSlots.empty() || static_cast<int>(mAllSlots.size()) < mSlotPoolCapacity;
        });
        if (mFreeSlots.empty())
        {
            auto slot = std::make_unique<Qwen3OmniTTSRuntime::TalkerSlot>();
            slot->stream = mStream;
            auto ctxPair = createExecutionContextPair();
            if (!ctxPair.first || !ctxPair.second)
            {
                LOG_WARNING("TalkerEngine pool: failed to allocate execution context pair");
                mSlotPoolCv.notify_one();
                return nullptr;
            }
            slot->prefillCtxOwned = std::move(ctxPair.first);
            slot->decodeCtxOwned = std::move(ctxPair.second);
            allocateSlot(*slot, mStream);
            auto* raw = slot.get();
            mAllSlots.push_back(std::move(slot));
            return raw;
        }
        auto* slot = mFreeSlots.back();
        mFreeSlots.pop_back();
        return slot;
    }

    void releasePoolSlot(Qwen3OmniTTSRuntime::TalkerSlot* slot)
    {
        if (slot == nullptr)
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mSlotPoolMutex);
            mFreeSlots.push_back(slot);
        }
        mSlotPoolCv.notify_one();
    }
};

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream)
    : Qwen3OmniTTSRuntime(talkerEngineDir, codePredictorEngineDir, tokenizerDir, stream, RuntimeOptions{})
{
}

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream, RuntimeOptions const& options)
    : mRuntimeOptions(options)
    , mStream(stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::init", nvtx_colors::YELLOW);
    LOG_INFO("Initializing Qwen3-Omni Talker runner");
    LOG_INFO("  Talker: %s", talkerEngineDir.c_str());
    LOG_INFO("  CodePredictor: %s", codePredictorEngineDir.c_str());
    mQwen3TTSCodePredictorEnginePath = getQwen3TTSCodePredictorEnginePath(codePredictorEngineDir);
    switch (mRuntimeOptions.codePredictorBackend)
    {
    case CodePredictorBackend::kAuto:
        mUseQwen3TTSCodePredictorEngine = !mQwen3TTSCodePredictorEnginePath.empty();
        break;
    case CodePredictorBackend::kGeneric: mUseQwen3TTSCodePredictorEngine = false; break;
    case CodePredictorBackend::kQwen3TTSNative:
        if (mQwen3TTSCodePredictorEnginePath.empty())
        {
            throw std::runtime_error("Qwen3-TTS native CodePredictor backend requested, but qwen3_tts_cp.engine was "
                                     "not found in: "
                + codePredictorEngineDir);
        }
        mUseQwen3TTSCodePredictorEngine = true;
        break;
    }
    LOG_INFO("  CodePredictor backend: %s",
        mUseQwen3TTSCodePredictorEngine ? "qwen3_tts_native" : "generic_llm_runner");

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
        int64_t const talkerCtxSize = mTalkerLLMRunner ? mTalkerLLMRunner->getRequiredContextMemorySize() : 0;
        int64_t const cpCtxSize = mCodePredictorRunner ? mCodePredictorRunner->getRequiredContextMemorySize() : 0;
        int64_t const sharedCtxSize = std::max(talkerCtxSize, cpCtxSize);
        LOG_INFO("Setup shared execution context memory: %zu bytes (talker: %zu, code_predictor: %zu)",
            static_cast<size_t>(sharedCtxSize), static_cast<size_t>(talkerCtxSize), static_cast<size_t>(cpCtxSize));
        if (sharedCtxSize > 0)
        {
            mSharedExecContextMemory = rt::Tensor({sharedCtxSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
                "Qwen3OmniTTSRuntime::mSharedExecContextMemory");
            if (mTalkerLLMRunner && !mTalkerLLMRunner->setContextMemory(mSharedExecContextMemory))
            {
                throw std::runtime_error("Failed to set context memory for Talker LLM engine");
            }
            if (mCodePredictorRunner && !mCodePredictorRunner->setContextMemory(mSharedExecContextMemory))
            {
                throw std::runtime_error("Failed to set context memory for CodePredictor engine");
            }
        }
    }

    if (!loadCodePredictorWeights(codePredictorEngineDir))
    {
        throw std::runtime_error("Failed to load CodePredictor weights");
    }

    if (mUseQwen3TTSCodePredictorEngine)
    {
        std::filesystem::path const qwen3TTSCodePredictorEmbeddingPath
            = std::filesystem::path(codePredictorEngineDir) / "cp_embed_fp32.bin";
        mQwen3TTSCodePredictorEngine = std::make_unique<Qwen3TTSCodePredictorEngine>(mQwen3TTSCodePredictorEnginePath,
            qwen3TTSCodePredictorEmbeddingPath, mCodePredictorEmbeddingTables, mTalkerConfig.codePredictorHiddenSize,
            mTalkerConfig.codebookSize, mCodePredictorConfig.numDecoderLayers, mCodePredictorConfig.numKVHeads,
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
    std::filesystem::path talkerEnginePath = std::filesystem::path(talkerEngineDir) / "llm.engine";
    std::filesystem::path talkerConfigPath = std::filesystem::path(talkerEngineDir) / "config.json";

    std::string explicitTalkerPath = mRuntimeOptions.qwen3TtsTalkerEnginePath;
    if (explicitTalkerPath.empty())
    {
        if (char const* envPath = std::getenv("QWEN3_TTS_DIRECT_TALKER_ENGINE"))
        {
            explicitTalkerPath = envPath;
        }
    }
    bool useExplicitTalker = false;
    switch (mRuntimeOptions.talkerBackend)
    {
    case TalkerBackend::kAuto:
        useExplicitTalker = !explicitTalkerPath.empty();
        break;
    case TalkerBackend::kGeneric:
        useExplicitTalker = false;
        break;
    case TalkerBackend::kQwen3TTSExplicitKV:
        if (explicitTalkerPath.empty())
        {
            LOG_ERROR("Qwen3-TTS explicit-KV Talker backend requested, but no direct Talker engine path was provided");
            return false;
        }
        useExplicitTalker = true;
        break;
    }
    LOG_INFO("Talker backend: %s", useExplicitTalker ? "qwen3_tts_explicit_kv" : "generic_llm_runner");

    try
    {
        if (useExplicitTalker)
        {
            LOG_INFO("Loading Talker config only from: %s", talkerConfigPath.string().c_str());
            if (!loadLLMConfigOnly(talkerConfigPath, mTalkerLLMConfig))
            {
                return false;
            }
            LLMEngineRunnerConfig directConfig = mTalkerLLMConfig;
            directConfig.outputVocabSize = directConfig.outputVocabSize > 0 ? directConfig.outputVocabSize : directConfig.vocabSize;
            mQwen3TTSTalkerEngine = std::make_unique<Qwen3TTSTalkerEngine>(
                std::filesystem::path(explicitTalkerPath), directConfig, 200, mStream);
            mTalkerHiddenStatesDataType = mQwen3TTSTalkerEngine->hiddenStatesDataType();
            mResidualEmbedDataType = nvinfer1::DataType::kFLOAT;
            switch (mRuntimeOptions.textProjectionMode)
            {
            case TextProjectionMode::kAuto:
                mUseHostTextProjection = std::getenv("QWEN3_TTS_HOST_TEXT_PROJECTION") != nullptr;
                break;
            case TextProjectionMode::kDevice: mUseHostTextProjection = false; break;
            case TextProjectionMode::kHostFP32: mUseHostTextProjection = true; break;
            }
            if (mUseHostTextProjection)
            {
                mTalkerInputEmbedsDataType = nvinfer1::DataType::kFLOAT;
            }
            mTalkerLLMConfig.maxKVCacheCapacity = mQwen3TTSTalkerEngine->maxKVSeqLen();
            LOG_INFO("Talker execution will use explicit-KV Qwen3-TTS engine override.");
            LOG_INFO("Text projection mode: %s", mUseHostTextProjection ? "host_fp32" : "device");
            LOG_INFO("Qwen3-TTS prompt KV cache: %s", mRuntimeOptions.qwen3TtsPromptKvCache ? "enabled" : "disabled");
        }
        else
        {
            LOG_INFO("Loading Talker LLM engine from: %s", talkerEnginePath.string().c_str());
            std::unordered_map<std::string, std::string> emptyLoraMap;
            mTalkerLLMRunner
                = std::make_unique<LLMEngineRunner>(talkerEnginePath, talkerConfigPath, emptyLoraMap, mStream);
            mTalkerLLMConfig = mTalkerLLMRunner->getEngineConfig();

            LOG_INFO("Talker LLM engine loaded: vocabSize=%d, hiddenSize=%d", mTalkerLLMConfig.vocabSize,
                mTalkerLLMConfig.hiddenSize);
            mTalkerInputEmbedsDataType = mTalkerLLMRunner->getTensorDataType("inputs_embeds");
            LOG_INFO("Talker inputs_embeds dtype: %s",
                mTalkerInputEmbedsDataType == nvinfer1::DataType::kBF16
                    ? "BF16"
                    : (mTalkerInputEmbedsDataType == nvinfer1::DataType::kHALF ? "FP16" : "OTHER"));
            mTalkerHiddenStatesDataType = mTalkerLLMRunner->getTensorDataType("hidden_states");
            LOG_INFO("Talker hidden_states dtype: %s",
                mTalkerHiddenStatesDataType == nvinfer1::DataType::kFLOAT ? "FP32"
                    : (mTalkerHiddenStatesDataType == nvinfer1::DataType::kBF16
                            ? "BF16"
                            : (mTalkerHiddenStatesDataType == nvinfer1::DataType::kHALF ? "FP16" : "OTHER")));
            auto talkerKVType = mTalkerLLMRunner->getCacheManager().getKVCacheManager().getConfig().kvCacheType;
            LOG_INFO("Talker KV cache dtype: %s",
                talkerKVType == nvinfer1::DataType::kBF16
                    ? "BF16"
                    : (talkerKVType == nvinfer1::DataType::kHALF
                            ? "FP16"
                            : (talkerKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN")));
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load Talker LLM engine: %s", e.what());
        return false;
    }

    if (mUseQwen3TTSCodePredictorEngine)
    {
        LOG_INFO("Qwen3-TTS CodePredictor requested; loading CodePredictor config only.");
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
        if (!std::filesystem::exists(projPath))
        {
            if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
            {
                LOG_INFO("small_to_mtp_projection.safetensors not found; using identity projection");
            }
            else
            {
                LOG_ERROR("small_to_mtp_projection.safetensors is required when Talker and CodePredictor hidden sizes differ");
                return false;
            }
        }
        else if (!safetensors::loadSafetensors(projPath, projTensors, mStream))
        {
            LOG_ERROR("Failed to load small_to_mtp_projection from: %s", projPath.string().c_str());
            return false;
        }
        if (!projTensors.empty())
        {
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
    {
        size_t const expectedElements
            = static_cast<size_t>(kNumRvqLayers) * mTalkerConfig.codebookSize * mTalkerConfig.talkerHiddenSize;
        std::filesystem::path const fp32EmbedPath
            = std::filesystem::path(codePredictorEngineDir) / "cp_embed_fp32.bin";
        if (loadFloatBin(fp32EmbedPath, expectedElements, mHostCodePredictorEmbeddingTables))
        {
            LOG_INFO("Loaded FP32 CodePredictor embedding table: %s", fp32EmbedPath.string().c_str());
        }
        else
        {
            mHostCodePredictorEmbeddingTables.clear();
        }
    }

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
        // [Phase B C2/C3/C4] The following 19 scratch tensors plus mTrailingTextLen
        // and mHostProjectedBuffer were previously runtime-global and are now
        // per-request locals inside `handleAudioGeneration` (see TalkerLocal /
        // CodePredictorLocal in the header). Their `m*` declarations are kept so
        // that init-time `captureDecodingCUDAGraph` (legacy path) still compiles;
        // those paths are disabled in production (gated on !mQwen3TTSTalkerEngine
        // and !mUseQwen3TTSCodePredictorEngine).
        //
        // Moved to TalkerLocal:
        //   mThinkerEmbedBuffer, mGpuTokenIdsBuffer, mMLPWorkspace, mProjectedBuffer,
        //   mTalkerInputEmbeds, mSpeakerEmbedding, mTalkerLogits, mTalkerSelectedIndices,
        //   mSeenCodecTokensBuf, mTalkerHiddenStatesBuffer, mTalkerLastHidden,
        //   mResidualEmbedBuffer, mTrailingTextLen, mHostProjectedBuffer
        // Moved to CodePredictorLocal:
        //   mCodePredictorPrefillInput, mCodePredictorCodecIds, mCodePredictorCodecEmbed,
        //   mRawCodecEmbed, mSmallToMtpProjectedHidden

        mHostSelectedTokenIds = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mHostTalkerContextLength = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

        // CodePredictor workspace (legacy CP path; production native engine bypasses these)
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

        mCodePredictorSelectedIndices = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        mHostSelectedCodeIds = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mHostCodePredictorContextLength = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

        // Talker decoding buffers (avoid temporary tensor creation)
        mTalkerDecodingIds = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerDecodingIds");
        mTalkerDecodingEmbed = rt::Tensor(
            {1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, mTalkerInputEmbedsDataType,
            "mTalkerDecodingEmbed");

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

        // CodePredictor uses seqLen=16 at most (not maxSeqLen), so allocate smaller buffer
        mCodePredictorHiddenStatesBuffer = rt::Tensor({1, 16, mTalkerConfig.codePredictorHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodePredictorHiddenStatesBuffer");

        (void)maxSeqLen;
        (void)thinkerHiddenSize;
        (void)talkerHiddenSize;

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
    if (mUseHostTextProjection)
    {
        mHostTextFC1Weight
            = copyTensorToHostFloat(mTextFC1Weight, mTextFC1Weight.getShape().volume(), stream);
        mHostTextFC1Bias = copyTensorToHostFloat(mTextFC1Bias, mTextFC1Bias.getShape().volume(), stream);
        mHostTextFC2Weight
            = copyTensorToHostFloat(mTextFC2Weight, mTextFC2Weight.getShape().volume(), stream);
        mHostTextFC2Bias = copyTensorToHostFloat(mTextFC2Bias, mTextFC2Bias.getShape().volume(), stream);
        LOG_INFO("Cached text_projection weights as host FP32");
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
    // Look up tensors by name. The file may contain just `text_embedding`
    // (FP16/BF16 path) or both `text_embedding` (FP8) plus
    // `text_embedding_scale` (FP32 per-group scales) when produced by
    // scripts/quantize_embedding_safetensors_fp8.py. Indexing by position is
    // unsafe because nlohmann::json sorts keys alphabetically on parse, so
    // tensor[0] could be either depending on naming.
    rt::Tensor* textEmbedPtr = nullptr;
    rt::Tensor* textEmbedScalePtr = nullptr;
    for (auto& t : textEmbedTensors)
    {
        std::string const& n = t.getName();
        if (n == "text_embedding")
        {
            textEmbedPtr = &t;
        }
        else if (n == "text_embedding_scale")
        {
            textEmbedScalePtr = &t;
        }
    }
    if (textEmbedPtr == nullptr && textEmbedTensors.size() == 1)
    {
        // Backward-compatible: legacy single-tensor file with no explicit name match.
        textEmbedPtr = &textEmbedTensors[0];
    }
    check::check(
        textEmbedPtr != nullptr, "text_embedding.safetensors must contain a tensor named `text_embedding`");
    check::check(textEmbedPtr->getShape().getNumDims() == 2,
        "text_embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTextEmbeddingTable = std::move(*textEmbedPtr);
    LOG_INFO("Text embedding table loaded: [%lld, %lld]", mTextEmbeddingTable.getShape()[0],
        mTextEmbeddingTable.getShape()[1]);
    if (textEmbedScalePtr != nullptr)
    {
        check::check(textEmbedScalePtr->getShape().getNumDims() == 2,
            "text_embedding_scale tensor should be 2D [vocabSize, hiddenSize/blockSize]");
        mTextEmbeddingScale = std::move(*textEmbedScalePtr);
        mTextEmbeddingHasScale = true;
        LOG_INFO("Text embedding scale loaded: [%lld, %lld] (FP8 dequant per-group scales)",
            mTextEmbeddingScale.getShape()[0], mTextEmbeddingScale.getShape()[1]);
    }
    if (!loadTextTokenMap(std::filesystem::path(weightsDir)))
    {
        return false;
    }

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
    {
        size_t const expectedElements = static_cast<size_t>(mTalkerEmbeddingTable.getShape()[0])
            * static_cast<size_t>(mTalkerEmbeddingTable.getShape()[1]);
        std::filesystem::path const fp32EmbedPath = std::filesystem::path(weightsDir) / "codec_embed_fp32.bin";
        if (loadFloatBin(fp32EmbedPath, expectedElements, mHostTalkerEmbeddingTable))
        {
            LOG_INFO("Loaded FP32 Talker codec embedding table: %s", fp32EmbedPath.string().c_str());
        }
        else
        {
            mHostTalkerEmbeddingTable
                = copyTensorToHostFloat(mTalkerEmbeddingTable, static_cast<int64_t>(expectedElements), stream);
            LOG_INFO("Cached Talker codec embedding table as host FP32 from TensorRT weights");
        }
    }

    LOG_INFO("Talker weights loaded successfully");
    return true;
}

bool Qwen3OmniTTSRuntime::loadTextTokenMap(std::filesystem::path const& weightsDir)
{
    mUsePrunedTextEmbedding = false;
    mTextTokenIdToPrunedRow.clear();

    char const* pruneEnv = std::getenv("QWEN3_TTS_VOCAB_PRUNED");
    bool const forcePruned = pruneEnv != nullptr
        && (std::string(pruneEnv) == "1" || std::string(pruneEnv) == "true" || std::string(pruneEnv) == "yes");
    bool const disablePruned = pruneEnv != nullptr
        && (std::string(pruneEnv) == "0" || std::string(pruneEnv) == "false" || std::string(pruneEnv) == "no");
    if (disablePruned)
    {
        LOG_INFO("Qwen3-TTS text vocab pruning disabled by QWEN3_TTS_VOCAB_PRUNED=%s", pruneEnv);
        return true;
    }

    std::filesystem::path const tokenMapPath = weightsDir / "token_map.bin";
    if (!std::filesystem::exists(tokenMapPath))
    {
        if (forcePruned)
        {
            LOG_ERROR("QWEN3_TTS_VOCAB_PRUNED=1 but token_map.bin was not found in: %s", weightsDir.string().c_str());
            return false;
        }
        return true;
    }

    auto const textShape = mTextEmbeddingTable.getShape();
    if (textShape.getNumDims() != 2)
    {
        LOG_ERROR("Cannot load token_map.bin before a 2D text embedding table is available");
        return false;
    }
    int64_t const prunedRows = textShape[0];

    std::ifstream file(tokenMapPath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        LOG_ERROR("Failed to open text token map: %s", tokenMapPath.string().c_str());
        return false;
    }
    std::streamsize const bytes = file.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int32_t)) != 0)
    {
        LOG_ERROR("Invalid token_map.bin size: %lld", static_cast<long long>(bytes));
        return false;
    }
    int64_t const entries = bytes / static_cast<std::streamsize>(sizeof(int32_t));
    if (entries != prunedRows)
    {
        LOG_ERROR("token_map.bin entries (%lld) do not match text embedding rows (%lld)", entries, prunedRows);
        return false;
    }

    std::vector<int32_t> prunedRowToOrigToken(static_cast<size_t>(entries));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(prunedRowToOrigToken.data()), bytes);
    if (file.gcount() != bytes)
    {
        LOG_ERROR("Failed to read complete token_map.bin: %s", tokenMapPath.string().c_str());
        return false;
    }

    int32_t maxOrigTokenId = 0;
    for (int32_t origTokenId : prunedRowToOrigToken)
    {
        if (origTokenId < 0)
        {
            LOG_ERROR("token_map.bin contains negative token id: %d", origTokenId);
            return false;
        }
        maxOrigTokenId = std::max(maxOrigTokenId, origTokenId);
    }

    mTextTokenIdToPrunedRow.assign(static_cast<size_t>(maxOrigTokenId) + 1, -1);
    for (int32_t row = 0; row < static_cast<int32_t>(prunedRowToOrigToken.size()); ++row)
    {
        int32_t const origTokenId = prunedRowToOrigToken[static_cast<size_t>(row)];
        int32_t& mappedRow = mTextTokenIdToPrunedRow[static_cast<size_t>(origTokenId)];
        if (mappedRow >= 0)
        {
            LOG_ERROR("token_map.bin contains duplicate original token id: %d", origTokenId);
            return false;
        }
        mappedRow = row;
    }

    mUsePrunedTextEmbedding = true;
    LOG_INFO("Loaded pruned text token map: rows=%lld, max_orig_token_id=%d", entries, maxOrigTokenId);
    return true;
}

int32_t Qwen3OmniTTSRuntime::mapTextTokenId(int32_t tokenId, char const* context) const
{
    if (!mUsePrunedTextEmbedding)
    {
        return tokenId;
    }
    if (tokenId < 0 || static_cast<size_t>(tokenId) >= mTextTokenIdToPrunedRow.size()
        || mTextTokenIdToPrunedRow[static_cast<size_t>(tokenId)] < 0)
    {
        throw std::runtime_error(std::string("Pruned text embedding missing token id ")
            + std::to_string(tokenId) + " while mapping " + (context ? context : "text token"));
    }
    return mTextTokenIdToPrunedRow[static_cast<size_t>(tokenId)];
}

std::vector<int32_t> Qwen3OmniTTSRuntime::mapTextTokenIds(
    std::vector<int32_t> const& tokenIds, char const* context) const
{
    if (!mUsePrunedTextEmbedding)
    {
        return tokenIds;
    }

    std::vector<int32_t> mapped;
    mapped.reserve(tokenIds.size());
    for (int32_t tokenId : tokenIds)
    {
        mapped.push_back(mapTextTokenId(tokenId, context));
    }
    return mapped;
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

    constexpr int32_t kNumTtsTokens = 3;
    std::vector<int32_t> const hostTtsIdsOrig
        = {mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId, mTalkerConfig.ttsEosTokenId};
    std::vector<int32_t> const hostTtsIds = mapTextTokenIds(hostTtsIdsOrig, "TTS special token");

    if (hostTtsIds[0] >= vocabSize || hostTtsIds[1] >= vocabSize || hostTtsIds[2] >= vocabSize)
    {
        throw std::runtime_error("TTS token IDs out of vocab range: pad=" + std::to_string(mTalkerConfig.ttsPadTokenId)
            + ", bos=" + std::to_string(mTalkerConfig.ttsBosTokenId)
            + ", eos=" + std::to_string(mTalkerConfig.ttsEosTokenId) + ", vocabSize=" + std::to_string(vocabSize));
    }

    rt::Tensor ttsIds({1, kNumTtsTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor ttsRaw({1, kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor ttsProjected(
        {kNumTtsTokens, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor workspace({kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpyAsync(
        ttsIds.rawPointer(), hostTtsIds.data(), kNumTtsTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::embeddingLookup(ttsIds, mTextEmbeddingTable,
        mTextEmbeddingHasScale ? rt::OptionalInputTensor{mTextEmbeddingScale} : std::nullopt, ttsRaw, stream);
    // Reshape from [1, 3, hidden] to [3, hidden] for MLP (expects 2D input)
    check::check(ttsRaw.reshape({kNumTtsTokens, thinkerHiddenSize}), "Tensor reshape failed");
    if (mUseHostTextProjection)
    {
        auto hostRaw = copyTensorToHostFloat(ttsRaw, kNumTtsTokens * thinkerHiddenSize, stream);
        auto projected = cpuTextProjection(hostRaw, kNumTtsTokens, static_cast<int32_t>(thinkerHiddenSize),
            static_cast<int32_t>(thinkerHiddenSize), mTalkerConfig.talkerHiddenSize, mHostTextFC1Weight,
            mHostTextFC1Bias, mHostTextFC2Weight, mHostTextFC2Bias);
        int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
        mHostTtsPadEmbed.assign(projected.begin(), projected.begin() + hiddenSize);
        mHostTtsBosEmbed.assign(projected.begin() + hiddenSize, projected.begin() + 2 * hiddenSize);
        mHostTtsEosEmbed.assign(projected.begin() + 2 * hiddenSize, projected.begin() + 3 * hiddenSize);
        mTtsPadEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        mTtsBosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        mTtsEosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        LOG_INFO("TTS embeddings initialized through host FP32 text projection");
        return;
    }
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
    rt::Tensor const& thinkerEmbed, int32_t langId, int32_t speakerId, std::vector<float> const& speakerEmbedding,
    rt::Tensor& output, int64_t& outputSeqLen, TalkerLocal& tlocal, cudaStream_t stream)
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
    if (mUseHostTextProjection)
    {
        return projectToTalkerInputHost(thinkerEmbed, langId, speakerId, speakerEmbedding, output, outputSeqLen,
            tlocal, stream);
    }
    bool const hasSpeakerEmbedding = !speakerEmbedding.empty();
    if (hasSpeakerEmbedding && static_cast<int64_t>(speakerEmbedding.size()) != hiddenSize)
    {
        LOG_ERROR("projectToTalkerInput: speaker embedding size %zu does not match hidden size %ld",
            speakerEmbedding.size(), hiddenSize);
        return false;
    }

    // Fixed 9-row prefill. Row 6 is the single speaker-conditioning slot:
    // canonical codecThinkEos, local speaker token, or external clone embedding.
    outputSeqLen = 9;

    // Store trailing text length for residual addend (body[1:] + tts_eos)
    tlocal.trailingTextLen = static_cast<int32_t>(N);

    // Project all tokens via text_projection MLP
    check::check(tlocal.projectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(tlocal.mlpWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(thinkerEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        tlocal.projectedBuffer, tlocal.mlpWorkspace, stream);
    if (hasSpeakerEmbedding)
    {
        std::vector<__half> speakerHalf(speakerEmbedding.size());
        for (size_t i = 0; i < speakerEmbedding.size(); ++i)
        {
            speakerHalf[i] = __float2half(speakerEmbedding[i]);
        }
        CUDA_CHECK(cudaMemcpyAsync(tlocal.speakerEmbedding.rawPointer(), speakerHalf.data(),
            speakerHalf.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
    }

    // Fused kernel: build complete non-streaming prefill buffer
    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    kernel::invokeAssistantPreamble(tlocal.projectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed,
        mTalkerEmbeddingTable, mTalkerConfig.codecThinkId, mTalkerConfig.codecThinkBosId,
        langId, mTalkerConfig.codecThinkEosId, speakerId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId,
        static_cast<int32_t>(N), tlocal.speakerEmbedding, hasSpeakerEmbedding, output, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::projectToTalkerInputHost(
    rt::Tensor const& thinkerEmbed, int32_t langId, int32_t speakerId, std::vector<float> const& speakerEmbedding,
    rt::Tensor& output, int64_t& outputSeqLen, TalkerLocal& tlocal, cudaStream_t stream)
{
    int64_t const seqLen = thinkerEmbed.getShape()[0];
    int32_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const N = seqLen - kAssistantPrefixLen;
    if (N <= 0)
    {
        LOG_ERROR("projectToTalkerInputHost: no text body tokens after assistant prefix");
        return false;
    }
    if (mHostTalkerEmbeddingTable.empty() || mHostTtsPadEmbed.empty() || mHostTextFC1Weight.empty())
    {
        LOG_ERROR("projectToTalkerInputHost: host projection/embedding tables are not available");
        return false;
    }
    bool const hasSpeakerEmbedding = !speakerEmbedding.empty();
    if (hasSpeakerEmbedding && static_cast<int32_t>(speakerEmbedding.size()) != hiddenSize)
    {
        LOG_ERROR("projectToTalkerInputHost: speaker embedding size %zu does not match hidden size %d",
            speakerEmbedding.size(), hiddenSize);
        return false;
    }
    outputSeqLen = 9;
    tlocal.trailingTextLen = static_cast<int32_t>(N);

    auto hostInput = copyTensorToHostFloat(thinkerEmbed, seqLen * thinkerHiddenSize, stream);
    tlocal.hostProjectedBuffer = cpuTextProjection(hostInput, static_cast<int32_t>(seqLen), thinkerHiddenSize,
        thinkerHiddenSize, hiddenSize, mHostTextFC1Weight, mHostTextFC1Bias, mHostTextFC2Weight, mHostTextFC2Bias);

    std::vector<float> prefill(static_cast<size_t>(outputSeqLen) * hiddenSize);
    std::copy(tlocal.hostProjectedBuffer.begin(), tlocal.hostProjectedBuffer.begin() + 3 * hiddenSize, prefill.begin());
    auto codecRow = [&](int32_t token) {
        return mHostTalkerEmbeddingTable.data() + static_cast<size_t>(token) * hiddenSize;
    };
    addHostRows(prefill.data() + 3 * hiddenSize, mHostTtsPadEmbed.data(), codecRow(mTalkerConfig.codecThinkId),
        hiddenSize);
    addHostRows(prefill.data() + 4 * hiddenSize, mHostTtsPadEmbed.data(), codecRow(mTalkerConfig.codecThinkBosId),
        hiddenSize);
    addHostRows(prefill.data() + 5 * hiddenSize, mHostTtsPadEmbed.data(), codecRow(langId), hiddenSize);
    if (hasSpeakerEmbedding)
    {
        std::copy(speakerEmbedding.begin(), speakerEmbedding.end(), prefill.begin() + 6 * hiddenSize);
    }
    else
    {
        int32_t const speakerTokenId = speakerId >= 0 ? speakerId : mTalkerConfig.codecThinkEosId;
        addHostRows(prefill.data() + 6 * hiddenSize, mHostTtsPadEmbed.data(), codecRow(speakerTokenId), hiddenSize);
    }
    int row = 7;
    addHostRows(prefill.data() + row * hiddenSize, mHostTtsBosEmbed.data(), codecRow(mTalkerConfig.codecPadId),
        hiddenSize);
    ++row;
    addHostRows(prefill.data() + row * hiddenSize, tlocal.hostProjectedBuffer.data() + 3 * hiddenSize,
        codecRow(mTalkerConfig.codecBosId), hiddenSize);
    if (char const* prefillOverride = std::getenv("QWEN3_TTS_PREFILL_EMBEDS_BIN"))
    {
        std::ifstream file(prefillOverride, std::ios::binary);
        if (file)
        {
            std::vector<float> overridePrefill(prefill.size());
            file.read(reinterpret_cast<char*>(overridePrefill.data()),
                static_cast<std::streamsize>(overridePrefill.size() * sizeof(float)));
            if (file.gcount() == static_cast<std::streamsize>(overridePrefill.size() * sizeof(float)))
            {
                prefill = std::move(overridePrefill);
            }
            else
            {
                LOG_WARNING("Ignoring short QWEN3_TTS_PREFILL_EMBEDS_BIN=%s", prefillOverride);
            }
        }
    }
    dumpFloats("prefill_embeds_f32.bin", prefill.data(), prefill.size());

    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(output.getDataType() == nvinfer1::DataType::kFLOAT,
        "Host text projection requires FP32 Talker input embeds");
    CUDA_CHECK(cudaMemcpyAsync(
        output.rawPointer(), prefill.data(), prefill.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerPrefillStep(
    rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerPrefillStep", nvtx_colors::PURPLE);

    if (mQwen3TTSTalkerEngine)
    {
        auto const inputShape = inputEmbeds.getShape();
        if (inputShape.getNumDims() != 3)
        {
            LOG_ERROR("executeTalkerPrefillStep: Input must be 3D [batchSize, seqLen, hiddenSize], got %dD",
                inputShape.getNumDims());
            return false;
        }
        int64_t const batchSize = inputEmbeds.getTRTDims().d[0];
        int64_t const seqLen = inputEmbeds.getTRTDims().d[1];
        int64_t const hiddenSize = inputEmbeds.getTRTDims().d[2];
        if (batchSize != 1 || hiddenSize != mTalkerConfig.talkerHiddenSize)
        {
            LOG_ERROR("executeTalkerPrefillStep: unexpected input shape [%ld, %ld, %ld]", batchSize, seqLen, hiddenSize);
            return false;
        }
        std::vector<float> hostInput = copyTensorToHostFloat(inputEmbeds, seqLen * hiddenSize, stream);
        if (mRuntimeOptions.qwen3TtsPromptKvCache)
        {
            return mQwen3TTSTalkerEngine->prefillWithPromptCache(
                hostInput, static_cast<int32_t>(seqLen), outputLogits, outputHiddenStates);
        }
        return mQwen3TTSTalkerEngine->prefill(hostInput, static_cast<int32_t>(seqLen), outputLogits, outputHiddenStates);
    }

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

bool Qwen3OmniTTSRuntime::executeTalkerDecodingStep(
    rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    if (mQwen3TTSTalkerEngine)
    {
        std::vector<float> hostInput
            = copyTensorToHostFloat(inputEmbeds, static_cast<int64_t>(mTalkerConfig.talkerHiddenSize), stream);
        return mQwen3TTSTalkerEngine->decode(hostInput, outputLogits, outputHiddenStates);
    }

    return mTalkerLLMRunner->executeVanillaDecodingStep(
        inputEmbeds, outputLogits, rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream);
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
    int32_t generationStep, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
    rt::Tensor& codecHiddensBuffer, CodePredictorLocal& cplocal, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorDecodingStep", nvtx_colors::ORANGE);

    CUDA_CHECK(cudaMemcpyAsync(
        cplocal.codePredictorCodecIds.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int32_t const embedIdx = std::min(embeddingTableIndex, kNumRvqLayers - 1);
    // Lookup into rawCodecEmbed (talkerHiddenSize=2048) — codec embedding tables are in Talker's space
    check::check(cplocal.rawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(
        cplocal.codePredictorCodecIds, mCodePredictorEmbeddingTables[embedIdx], std::nullopt, cplocal.rawCodecEmbed, stream);

    // Save raw (2048-dim) embedding to codecHiddensBuffer for residual connection
    // Position mapping: generationStep 1->pos 1, 2->pos 2, ..., 14->pos 14
    if (generationStep >= 1 && generationStep <= 14)
    {
        int64_t const H = mTalkerConfig.talkerHiddenSize;
        __half* dst = static_cast<__half*>(codecHiddensBuffer.rawPointer()) + generationStep * H;
        CUDA_CHECK(
            cudaMemcpyAsync(dst, cplocal.rawCodecEmbed.rawPointer(), H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    // Project/copy rawCodecEmbed to CodePredictor hidden size.
    check::check(cplocal.rawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(cplocal.codePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(cplocal.codePredictorCodecEmbed.rawPointer(), cplocal.rawCodecEmbed.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(cplocal.rawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias,
            cplocal.codePredictorCodecEmbed, stream);
    }

    int32_t const lmHeadIdx = std::min(generationStep, kNumRvqLayers - 1);

    check::check(
        cplocal.codePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");

    if (mCodePredictorGraphsCaptured)
    {
        // Graph path: lm_head_weight addresses were bound during capture and remain unchanged,
        // so setLMHeadWeights is unnecessary. Each graph is keyed by its per-head output buffer.
        if (!mCodePredictorRunner->executeVanillaDecodingStep(cplocal.codePredictorCodecEmbed,
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
                cplocal.codePredictorCodecEmbed, outputLogits, rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
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

    if (mQwen3TTSTalkerEngine)
    {
        LOG_INFO("Skipping generic Talker CUDA graph capture; explicit-KV Qwen3-TTS Talker engine is enabled.");
        return true;
    }

    // Talker: same pattern as Thinker (LLMInferenceRuntime::captureDecodingCUDAGraph)
    check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    bool captureStatus = mTalkerLLMRunner->captureVanillaDecodingCudaGraph(mResidualEmbedBuffer, mTalkerLogits,
        emptyLoraWeightsName, stream, rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)});

    // CodePredictor: 15 graphs, one per lm_head_weight.
    // Each graph uses a distinct output logits buffer so that the decodingKey (which includes
    // the output address) naturally produces a unique key per graph.
    if (mUseQwen3TTSCodePredictorEngine)
    {
        mCodePredictorGraphsCaptured = false;
        LOG_INFO("Skipping generic CodePredictor CUDA graph capture; Qwen3-TTS CodePredictor engine is enabled.");
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
    TalkerGenerationRequest const& request, int64_t& outSeqLen, TalkerLocal& tlocal, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    if (seqLen == 0)
    {
        LOG_ERROR("prepareTalkerInput: empty token ID list");
        return false;
    }
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    std::vector<int32_t> const mappedTextTokenIds = mapTextTokenIds(textTokenIds, "text token");
    check::check(tlocal.gpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(tlocal.gpuTokenIdsBuffer.rawPointer(), mappedTextTokenIds.data(), seqLen * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    check::check(tlocal.thinkerEmbedBuffer.reshape({1, seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(tlocal.gpuTokenIdsBuffer, mTextEmbeddingTable,
        mTextEmbeddingHasScale ? rt::OptionalInputTensor{mTextEmbeddingScale} : std::nullopt,
        tlocal.thinkerEmbedBuffer, stream);
    check::check(tlocal.thinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

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
    bool const hasCloneSpeaker = !request.speakerEmbedding.empty();
    bool const hasLocalSpeaker = request.speakerId >= 0 || !request.speakerName.empty();
    if (hasCloneSpeaker && hasLocalSpeaker)
    {
        LOG_ERROR("prepareTalkerInput: speaker_embedding cannot be combined with speaker or speaker_id");
        return false;
    }
    int32_t speakerId = -1;
    if (!hasCloneSpeaker)
    {
        if (request.speakerId >= 0)
        {
            speakerId = request.speakerId;
        }
        else if (!request.speakerName.empty())
        {
            speakerId = getSpeakerIdByName(request.speakerName);
        }
    }

    // MLP projection: thinker embed → talker input embeds (9-row prefill)
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!projectToTalkerInput(tlocal.thinkerEmbedBuffer, langId, speakerId, request.speakerEmbedding,
            tlocal.talkerInputEmbeds, outSeqLen, tlocal, stream))
    {
        LOG_ERROR("MLP projection failed");
        return false;
    }

    // Reshape buffers to 3D [1, seqLen, H] for Talker LLM input
    check::check(tlocal.talkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(
        tlocal.talkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGeneration(TalkerGenerationRequest const& request,
    TalkerGenerationResponse& response, cudaStream_t stream, FrameCallback const& frameCallback)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGeneration", nvtx_colors::PURPLE);
    LOG_INFO("Starting audio generation for request with %zu messages", request.messages.size());

    // Clear response
    response.rvqCodes.clear();
    response.numFrames = 0;
    response.success = false;

    // [Phase B C1] Per-request codec hiddens buffer for the residual connection.
    // Previously this was a runtime-global member (`mCodecHiddensBuffer`) which
    // races on `cudaMemsetAsync` when more than one request is in flight (the
    // root cause traced in `docs/specs/tts-n2-shared-tensor-audit.md` §3.1).
    // Allocating per-call keeps the lifetime tied to the request so concurrent
    // generations cannot collide. C2 folds this into `TalkerSlot` once full
    // slot threading lands.
    rt::Tensor codecHiddensBuffer({1, 16, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "codecHiddensBuffer");

    // [Phase B C2/C3/C4] Per-request scratch buffers. Allocating these on the
    // stack instead of using runtime-global `m*` members eliminates the
    // cross-request races flagged in tts-n2-shared-tensor-audit.md §1 and
    // permits the worker to drop the global runtime mutex (N=2 concurrency).
    TalkerLocal tlocal;
    CodePredictorLocal cplocal;
    {
        int64_t const maxSeqLen = mTalkerConfig.maxSeqLen;
        int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
        int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;

        tlocal.thinkerEmbedBuffer
            = rt::Tensor({maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        tlocal.gpuTokenIdsBuffer
            = rt::Tensor({1, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        tlocal.mlpWorkspace
            = rt::Tensor({maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        tlocal.projectedBuffer
            = rt::Tensor({maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        tlocal.talkerInputEmbeds
            = rt::Tensor({maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, mTalkerInputEmbedsDataType);
        tlocal.speakerEmbedding
            = rt::Tensor({talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        tlocal.talkerLogits
            = rt::Tensor({1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        tlocal.talkerSelectedIndices
            = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        tlocal.seenCodecTokensBuf = rt::Tensor({mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "seenCodecTokensBuf");
        tlocal.talkerHiddenStatesBuffer = rt::Tensor({1, maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            mTalkerHiddenStatesDataType, "talkerHiddenStatesBuffer");
        tlocal.talkerLastHidden = rt::Tensor(
            {1, talkerHiddenSize}, rt::DeviceType::kGPU, mTalkerHiddenStatesDataType, "talkerLastHidden");
        tlocal.residualEmbedBuffer = rt::Tensor(
            {1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, mResidualEmbedDataType);

        cplocal.codePredictorPrefillInput = rt::Tensor(
            {1, 2, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        cplocal.codePredictorCodecIds = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        cplocal.codePredictorCodecEmbed = rt::Tensor(
            {1, 1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        cplocal.rawCodecEmbed
            = rt::Tensor({1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        cplocal.smallToMtpProjectedHidden
            = rt::Tensor({1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    }

    // Talker/CodePredictor sampling: use dedicated parameters (not shared with Thinker).
    // PyTorch defaults: do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05
    float const talkerTemperature = (request.talkerTemperature > 0) ? request.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (request.talkerTopK > 0) ? request.talkerTopK : 50;
    float const talkerTopP = (request.talkerTopP > 0) ? request.talkerTopP : 1.0f;
    float const predictorTemperature = (request.predictorTemperature > 0) ? request.predictorTemperature : talkerTemperature;
    int32_t const predictorTopK = (request.predictorTopK > 0) ? request.predictorTopK : talkerTopK;
    float const predictorTopP = (request.predictorTopP > 0) ? request.predictorTopP : talkerTopP;
    float const repetitionPenalty = request.repetitionPenalty;

    SamplingParams talkerSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(
        1, mTalkerConfig.codebookSize, predictorTemperature, predictorTopK, predictorTopP);

    // Suppress special codec tokens [vocabSize-1024, vocabSize) except codec_eos, and apply
    // repetition penalty (PyTorch default: 1.05) to previously generated tokens.
    int32_t const suppressStart = mTalkerConfig.talkerVocabSize - 1024;
    int32_t const suppressEnd = mTalkerConfig.talkerVocabSize;
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // Repetition-penalty state follows Transformers semantics: the penalty is applied once per
    // previous occurrence, so repeated tokens receive repeated penalties.
    int32_t numSeenTokens = 0;
    auto adjustTalkerLogits = [&](cudaStream_t s) {
        kernel::invokeTalkerLogitAdjust(tlocal.seenCodecTokensBuf, tlocal.talkerLogits, suppressStart, suppressEnd,
            codecEosId, 0, 1.0f, s);
    };
    auto trackSeenToken = [&](int32_t /*token*/, cudaStream_t s) {
        if (numSeenTokens < static_cast<int32_t>(tlocal.seenCodecTokensBuf.getShape()[0]))
        {
            CUDA_CHECK(cudaMemcpyAsync(tlocal.seenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
                tlocal.talkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, s));
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
        LOG_DEBUG("TTS debug rawText='%s' bodyTokenIds=[%s] effectiveLanguage=%s", rawText.c_str(),
            oss.str().c_str(), effectiveRequest.language.c_str());
    }

    std::vector<std::vector<int32_t>> rvqCodes;
    std::vector<int32_t> primaryHistory;
    std::mt19937 talkerRng(makeQwen3TTSSamplingSeed(0x54545352));
    if (mQwen3TTSCodePredictorEngine)
    {
        mQwen3TTSCodePredictorEngine->resetSampling();
    }

    // Prepare Talker input: validate hidden states, project via MLP, reshape buffers
    int64_t seqLen = 0;
    if (!prepareTalkerInput(textTokenIds, effectiveRequest, seqLen, tlocal, stream))
    {
        LOG_ERROR("Input preparation failed");
        return false;
    }

    // Talker Prefill - engine outputs FP32 logits directly
    if (!executeTalkerPrefillStep(tlocal.talkerInputEmbeds, tlocal.talkerLogits, tlocal.talkerHiddenStatesBuffer, stream))
    {
        LOG_ERROR("Talker prefill failed");
        return false;
    }
    {
        auto prefillLogits = copyTensorToHostFloat(tlocal.talkerLogits, mTalkerConfig.talkerVocabSize, stream);
        dumpVector("talker_prefill_logits_f32.bin", prefillLogits);
        auto prefillHidden = copyTensorToHostFloat(
            tlocal.talkerHiddenStatesBuffer, tlocal.talkerHiddenStatesBuffer.getShape().volume(), stream);
        dumpVector("talker_prefill_hidden_f32.bin", prefillHidden);
    }

    {
        std::vector<float> hostLogits(static_cast<size_t>(mTalkerConfig.talkerVocabSize));
        CUDA_CHECK(cudaMemcpyAsync(hostLogits.data(), tlocal.talkerLogits.rawPointer(),
            hostLogits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const maxIt = std::max_element(hostLogits.begin(), hostLogits.end());
        int32_t const argmax = static_cast<int32_t>(std::distance(hostLogits.begin(), maxIt));
        LOG_DEBUG("TTS debug pre-adjust logits: argmax=%d max=%f eos=%f seqLen=%ld", argmax, *maxIt,
            hostLogits[mTalkerConfig.codecEosId], seqLen);
    }

    // Suppress special tokens and apply repetition penalty, then sample first codec token
    adjustTalkerLogits(stream);
    {
        std::vector<float> hostLogits(static_cast<size_t>(mTalkerConfig.talkerVocabSize));
        CUDA_CHECK(cudaMemcpyAsync(hostLogits.data(), tlocal.talkerLogits.rawPointer(),
            hostLogits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const maxIt = std::max_element(hostLogits.begin(), hostLogits.end());
        int32_t const argmax = static_cast<int32_t>(std::distance(hostLogits.begin(), maxIt));
        LOG_DEBUG("TTS debug adjusted logits: argmax=%d max=%f eos=%f", argmax, *maxIt,
            hostLogits[mTalkerConfig.codecEosId]);
    }
    int32_t codecToken = sampleLogitsCPU(tlocal.talkerLogits, mTalkerConfig.talkerVocabSize, talkerTopK,
        talkerTopP, talkerTemperature, true, mTalkerConfig.codecEosId, 0.0f, true, &primaryHistory, repetitionPenalty,
        talkerRng, stream);
    dumpVector("frame0_primary_i32.bin", std::vector<int32_t>{codecToken});
    CUDA_CHECK(cudaMemcpyAsync(
        tlocal.talkerSelectedIndices.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    trackSeenToken(codecToken, stream);
    primaryHistory.push_back(codecToken);
    LOG_INFO("First codec token (from prefill): %d (eos=%d)", codecToken, mTalkerConfig.codecEosId);

    // Clamp maxAudioLength to avoid Talker KV cache overflow.
    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t const safeMaxFrames = std::max(1, talkerKVCapacity - static_cast<int32_t>(seqLen));
    int32_t const textBasedMax = std::max(50, tlocal.trailingTextLen * 10);
    int32_t const requestedMaxAudio = std::min(request.maxAudioLength, textBasedMax);
    int32_t const effectiveMaxAudio = std::min(requestedMaxAudio, safeMaxFrames);
    int32_t minEosFrames = 0;
    if (char const* minEosEnv = std::getenv("QWEN3_TTS_MIN_EOS_FRAMES"))
    {
        try
        {
            minEosFrames = std::max(0, std::stoi(minEosEnv));
        }
        catch (std::exception const&)
        {
            LOG_WARNING("Ignoring invalid QWEN3_TTS_MIN_EOS_FRAMES=%s", minEosEnv);
        }
    }
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
            if (!extractTalkerLastHidden(tlocal.talkerHiddenStatesBuffer, tlocal.talkerLastHidden, stream))
            {
                LOG_ERROR("Failed to extract Talker hidden state at frame %d", numFrames);
                break;
            }

            // Clear codes for this frame
            frameCodes.clear();

            // CodePredictor generation for this frame (16 codes)
            // Hidden states are written directly into the per-request codecHiddensBuffer
            {
                TIME_STAGE(metrics::StageNames::kCODE_PREDICTOR, stream);
                if (!runCodePredictorGenerationForFrame(codecToken, tlocal.talkerLastHidden, predictorSamplingParams,
                        frameCodes, codecHiddensBuffer, tlocal, cplocal, stream))
                {
                    LOG_ERROR("CodePredictor generation failed at frame %d", numFrames);
                    break;
                }
                if (numFrames < 3)
                {
                    std::ostringstream oss;
                    for (size_t i = 0; i < frameCodes.size(); ++i)
                    {
                        if (i)
                        {
                            oss << ",";
                        }
                        oss << frameCodes[i];
                    }
                    LOG_DEBUG("TTS debug frame=%d rvqCodes=[%s]", numFrames, oss.str().c_str());
                }
            }

            // Store RVQ codes for this frame
            rvqCodes.push_back(frameCodes);
            if (frameCallback)
            {
                frameCallback(rvqCodes.back(), numFrames + 1);
            }

            // Compute residual connection using pre-allocated buffer
            // Non-streaming: always add tts_pad_embed as addend
            if (!computeResidualConnection(frameCodes, tlocal.residualEmbedBuffer, numFrames, codecHiddensBuffer,
                    tlocal, stream))
            {
                LOG_ERROR("Residual connection failed at frame %d", numFrames);
                break;
            }

            // Talker decoding step with residual embedding as input
            // PyTorch: inputs["inputs_embeds"] = codec_hiddens.sum(1, keepdim=True)
            check::check(tlocal.residualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

            // CRITICAL: Reshape hidden states buffer for decoding output (seqLen=1)
            // Otherwise extractTalkerLastHidden reads stale prefill data
            check::check(
                tlocal.talkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

            // Call Talker engine with decoding step (uses KV cache from prefill)
            if (!executeTalkerDecodingStep(tlocal.residualEmbedBuffer, tlocal.talkerLogits, tlocal.talkerHiddenStatesBuffer, stream))
            {
                LOG_ERROR("Talker decoding step failed at frame %d", numFrames);
                break;
            }
            if (numFrames < 12)
            {
                auto decodeLogits = copyTensorToHostFloat(tlocal.talkerLogits, mTalkerConfig.talkerVocabSize, stream);
                dumpVector("talker_decode" + std::to_string(numFrames + 1) + "_logits_f32.bin", decodeLogits);
                auto decodeHidden = copyTensorToHostFloat(
                    tlocal.talkerHiddenStatesBuffer, mTalkerConfig.talkerHiddenSize, stream);
                dumpVector("talker_decode" + std::to_string(numFrames + 1) + "_hidden_f32.bin", decodeHidden);
            }

            // Suppress special tokens and apply repetition penalty, then sample next codec token
            adjustTalkerLogits(stream);
            int32_t const sampleStep = numFrames + 1;
            if (sampleStep <= 3)
            {
                std::vector<float> hostLogits(static_cast<size_t>(mTalkerConfig.talkerVocabSize));
                CUDA_CHECK(cudaMemcpyAsync(hostLogits.data(), tlocal.talkerLogits.rawPointer(),
                    hostLogits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                auto const maxIt = std::max_element(hostLogits.begin(), hostLogits.end());
                int32_t const argmax = static_cast<int32_t>(std::distance(hostLogits.begin(), maxIt));
                LOG_DEBUG("TTS debug step=%d adjusted argmax=%d max=%f eos=%f", sampleStep, argmax, *maxIt,
                    hostLogits[mTalkerConfig.codecEosId]);
            }
            int32_t const biasOnset = tlocal.trailingTextLen * 3;
            int32_t const stepsPastOnset = sampleStep - biasOnset;
            float eosBias = request.codecEosLogitOffset;
            if (stepsPastOnset >= 0 && std::getenv("QWEN3_TTS_DISABLE_AUTO_EOS_BIAS") == nullptr)
            {
                eosBias = std::min(25.0f, 5.0f + static_cast<float>(stepsPastOnset) * 0.5f);
            }
            codecToken = sampleLogitsCPU(tlocal.talkerLogits, mTalkerConfig.talkerVocabSize, talkerTopK, talkerTopP,
                talkerTemperature, sampleStep < 2 || sampleStep < minEosFrames, mTalkerConfig.codecEosId, eosBias,
                true, &primaryHistory, repetitionPenalty, talkerRng, stream);
            CUDA_CHECK(cudaMemcpyAsync(
                tlocal.talkerSelectedIndices.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

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
    {
        std::ostringstream oss;
        for (size_t i = 0; i < primaryHistory.size(); ++i)
        {
            if (i)
            {
                oss << ",";
            }
            oss << primaryHistory[i];
        }
        LOG_DEBUG("TTS primary codec history: [%s]", oss.str().c_str());
    }
    LOG_INFO(
        "Generated %d audio frames (exit: %s, last_code=%d)", numFrames, hitEos ? "EOS" : "maxAudioLength", codecToken);
    if (!rvqCodes.empty())
    {
        std::vector<int32_t> flatCodes;
        flatCodes.reserve(rvqCodes.size() * 16);
        for (auto const& frame : rvqCodes)
        {
            flatCodes.insert(flatCodes.end(), frame.begin(), frame.end());
        }
        dumpVector("all_codes_i32.bin", flatCodes);
    }

    response.rvqCodes = std::move(rvqCodes);
    response.numFrames = numFrames;

    mMultimodalMetrics.recordRun(0, 0, 1, numFrames);

    response.success = true;
    return true;
}

bool Qwen3OmniTTSRuntime::runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
    SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes, rt::Tensor& codecHiddensBuffer,
    TalkerLocal& tlocal, CodePredictorLocal& cplocal, cudaStream_t stream)
{
    (void)tlocal;
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runCodePredictorGenerationForFrame", nvtx_colors::ORANGE);

    // Original model logic:
    // - codes: code_0 (from Talker) + code_1 to code_15 (from CodePredictor) = 16 codes
    // - hidden_states: written directly to codecHiddensBuffer (per-request, passed in)

    static thread_local std::mt19937 predictorRng(makeQwen3TTSSamplingSeed(0x4350524E));

    outputCodes.clear();
    outputCodes.reserve(16); // code_0 to code_15

    // code_0 comes from Talker
    outputCodes.push_back(codecToken);

    int64_t const hiddenSize = mTalkerConfig.codePredictorHiddenSize;
    int32_t const activeGroups = mQwen3TTSCodePredictorEngine
        ? getQwen3TTSActiveCodePredictorGroups()
        : talker_constants::kNumRvqLayers;
    check::check(activeGroups > 0 && activeGroups <= kNumRvqLayers, "Invalid active CodePredictor group count");

    // ========== Prefill: generate code_1 ==========
    // Input: concat([proj(talker_hidden), proj(embed(code_0))]) -> [1, 2, codePredictorHiddenSize]
    // NOTE: code_0 embedding uses TALKER's codec_embedding (2048-dim), projected to 1024 via small_to_mtp_projection
    // PyTorch: last_id_hidden = self.small_to_mtp_projection(self.get_input_embeddings()(input_ids))

    // Step 1: Lookup code_0 from Talker's embedding table into rawCodecEmbed (2048-dim)
    CUDA_CHECK(cudaMemcpyAsync(
        cplocal.codePredictorCodecIds.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check::check(cplocal.rawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(cplocal.codePredictorCodecIds, mTalkerEmbeddingTable, std::nullopt, cplocal.rawCodecEmbed, stream);

    // Step 2: Project/copy talkerHiddenState to CodePredictor hidden size.
    // Official qwen_tts uses small_to_mtp_projection only when the Talker and
    // CodePredictor hidden sizes differ; Qwen3-TTS 0.6B uses 1024 for both.
    bool const canUseTalkerHiddenDirectly
        = mQwen3TTSCodePredictorEngine && mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize;
    if (canUseTalkerHiddenDirectly)
    {
        // Native Qwen3-TTS CP consumes host FP32 inputs, so keep the Talker
        // hidden state in its engine output dtype until the D2H conversion.
    }
    else if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(cplocal.smallToMtpProjectedHidden.rawPointer(), talkerHiddenState.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(
            talkerHiddenState, mSmallToMtpWeight, mSmallToMtpBias, cplocal.smallToMtpProjectedHidden, stream);
    }

    // Step 3: Project/copy rawCodecEmbed to CodePredictor hidden size.
    check::check(cplocal.rawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(cplocal.codePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(cplocal.codePredictorCodecEmbed.rawPointer(), cplocal.rawCodecEmbed.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(cplocal.rawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias,
            cplocal.codePredictorCodecEmbed, stream);
    }

    if (mQwen3TTSCodePredictorEngine)
    {
        static bool dumpedFirstDirectCpFrame = false;
        static int32_t directCpFrameIndex = 0;
        bool const dumpCpInputs = std::getenv("QWEN3_TTS_DUMP_CP") != nullptr;
        bool const shouldDumpDirectCpFrame = dumpCpInputs && !dumpedFirstDirectCpFrame;
        std::vector<int32_t> residualCodes;
        std::vector<float> primaryEmbeddingHost;
        if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize
            && !mHostTalkerEmbeddingTable.empty())
        {
            size_t const offset = static_cast<size_t>(codecToken) * mTalkerConfig.talkerHiddenSize;
            primaryEmbeddingHost.assign(mHostTalkerEmbeddingTable.begin() + static_cast<std::ptrdiff_t>(offset),
                mHostTalkerEmbeddingTable.begin()
                    + static_cast<std::ptrdiff_t>(offset + mTalkerConfig.talkerHiddenSize));
        }
        else
        {
            primaryEmbeddingHost
                = copyTensorToHostFloat(cplocal.codePredictorCodecEmbed, mTalkerConfig.codePredictorHiddenSize, stream);
        }
        bool const canUseDeviceHidden = canUseTalkerHiddenDirectly && talkerHiddenState.getDataType() == nvinfer1::DataType::kFLOAT
            && !dumpCpInputs;
        std::vector<float> hiddenHost;
        if (!canUseDeviceHidden)
        {
            hiddenHost = canUseTalkerHiddenDirectly
                ? copyTensorToHostFloat(talkerHiddenState, mTalkerConfig.codePredictorHiddenSize, stream)
                : copyTensorToHostFloat(cplocal.smallToMtpProjectedHidden, mTalkerConfig.codePredictorHiddenSize, stream);
        }
        if (dumpCpInputs && directCpFrameIndex < 2)
        {
            dumpVector("cp_frame" + std::to_string(directCpFrameIndex) + "_input_hidden_f32.bin", hiddenHost);
            dumpVector("cp_frame" + std::to_string(directCpFrameIndex) + "_input_primary_emb_f32.bin",
                primaryEmbeddingHost);
        }
        if (shouldDumpDirectCpFrame)
        {
            dumpVector("cp_input_hidden_f32.bin", hiddenHost);
            dumpVector("cp_input_primary_emb_f32.bin", primaryEmbeddingHost);
        }
        bool const cpOk = canUseDeviceHidden
            ? mQwen3TTSCodePredictorEngine->generateDeviceHidden(
                static_cast<float const*>(talkerHiddenState.rawPointer()), primaryEmbeddingHost, activeGroups,
                samplingParams.topK, samplingParams.topP, samplingParams.temperature, residualCodes)
            : mQwen3TTSCodePredictorEngine->generate(hiddenHost, primaryEmbeddingHost, activeGroups,
                samplingParams.topK, samplingParams.topP, samplingParams.temperature, residualCodes);
        if (!cpOk)
        {
            return false;
        }

        check::check(codecHiddensBuffer.reshape({1, 16, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemsetAsync(codecHiddensBuffer.rawPointer(), 0, codecHiddensBuffer.getMemoryCapacity(), stream));

        int32_t const groupsToMaterialize = std::min<int32_t>(activeGroups, static_cast<int32_t>(residualCodes.size()));
        for (int32_t group = 0; group < groupsToMaterialize; ++group)
        {
            int32_t const residualCode = residualCodes[group];
            outputCodes.push_back(residualCode);
            CUDA_CHECK(cudaMemcpyAsync(cplocal.codePredictorCodecIds.rawPointer(), &residualCode, sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));
            check::check(cplocal.rawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
            kernel::embeddingLookup(
                cplocal.codePredictorCodecIds, mCodePredictorEmbeddingTables[group], std::nullopt, cplocal.rawCodecEmbed, stream);
            __half* dst = static_cast<__half*>(codecHiddensBuffer.rawPointer())
                + static_cast<int64_t>(group + 1) * mTalkerConfig.talkerHiddenSize;
            CUDA_CHECK(cudaMemcpyAsync(dst, cplocal.rawCodecEmbed.rawPointer(),
                static_cast<size_t>(mTalkerConfig.talkerHiddenSize) * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        while (static_cast<int32_t>(outputCodes.size()) < 16)
        {
            outputCodes.push_back(0);
        }
        if (shouldDumpDirectCpFrame)
        {
            dumpVector("frame0_codes_i32.bin", outputCodes);
            dumpedFirstDirectCpFrame = true;
        }
        ++directCpFrameIndex;
        return true;
    }

    // Step 4: Concat projected tensors into codePredictorPrefillInput [1, 2, codePredictorHiddenSize]
    CUDA_CHECK(cudaMemcpyAsync(cplocal.codePredictorPrefillInput.rawPointer(), cplocal.smallToMtpProjectedHidden.rawPointer(),
        hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(cplocal.codePredictorPrefillInput.rawPointer()) + hiddenSize,
        cplocal.codePredictorCodecEmbed.rawPointer(), hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));

    check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 2, mTalkerConfig.codePredictorHiddenSize}),
        "Tensor reshape failed");

    // NOTE: CodePredictor ONNX outputs FP32 logits directly (lm_head + cast in ONNX)
    // generationStep=0 corresponds to code_1 (using lm_head_0)
    if (!executeCodePredictorPrefillStep(
            cplocal.codePredictorPrefillInput, 0, mCodePredictorLogits, mCodePredictorHiddenStatesBuffer, stream))
    {
        return false;
    }

    // Sample code_1
    int32_t code = sampleLogitsCPU(mCodePredictorLogits, mTalkerConfig.codebookSize, samplingParams.topK,
        samplingParams.topP, samplingParams.temperature, false, -1, 0.0f, false, nullptr, 1.0f, predictorRng, stream);
    outputCodes.push_back(code); // code_1

    // ========== Write embedding lookups to codecHiddensBuffer for residual connection ==========
    // codecHiddensBuffer layout: [1, 16, H]
    // Position 0:  embed(code_0)  using Talker's embedding   - filled in computeResidualConnection
    // Position 1-14: codec_embedding[step-1](code_step)      - filled here (input embeddings, NOT engine hidden states)
    // Position 15: embed(code_15) using CodePredictor embed[-1] - filled in computeResidualConnection
    //
    // PyTorch original (modeling_qwen3_omni.py:3419):
    //   mid_residual_hiddens = [hid[0] for hid in predictor_result.hidden_states[1:]]
    //   hid[0] = inputs_embeds for each decode step = codec_embedding[step-1](code_step)
    //   NOT the transformer output (last layer hidden states)

    // codecHiddensBuffer stores raw (2048-dim) codec embeddings for the residual connection
    check::check(codecHiddensBuffer.reshape({1, 16, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(codecHiddensBuffer.rawPointer(), 0, codecHiddensBuffer.getMemoryCapacity(), stream));

    // ========== Decoding loop: generate active residual codes only ==========
    for (int step = 2; step <= activeGroups; ++step)
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

        if (!executeCodePredictorDecodingStep(code, embeddingIdx, lmHeadIdx, mCodePredictorLogits,
                mCodePredictorHiddenStatesBuffer, codecHiddensBuffer, cplocal, stream))
        {
            return false;
        }

        // Embedding is now saved inside executeCodePredictorDecodingStep before engine execution

        code = sampleLogitsCPU(mCodePredictorLogits, mTalkerConfig.codebookSize, samplingParams.topK,
            samplingParams.topP, samplingParams.temperature, false, -1, 0.0f, false, nullptr, 1.0f, predictorRng,
            stream);
        outputCodes.push_back(code); // code_2 to code_activeGroups
    }

    // The final active code embedding is normally materialized as the input
    // to the next CP decode step. When inactive groups are zero-filled, perform
    // just that lookup/copy without running another predictor step.
    CUDA_CHECK(cudaMemcpyAsync(cplocal.codePredictorCodecIds.rawPointer(), &code, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check::check(cplocal.rawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(cplocal.codePredictorCodecIds, mCodePredictorEmbeddingTables[activeGroups - 1], std::nullopt,
        cplocal.rawCodecEmbed, stream);
    {
        int64_t const H = mTalkerConfig.talkerHiddenSize;
        __half* dst = static_cast<__half*>(codecHiddensBuffer.rawPointer()) + static_cast<int64_t>(activeGroups) * H;
        CUDA_CHECK(cudaMemcpyAsync(dst, cplocal.rawCodecEmbed.rawPointer(), H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    while (static_cast<int32_t>(outputCodes.size()) < 16)
    {
        outputCodes.push_back(0);
    }

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnection(std::vector<int32_t> const& codes, rt::Tensor& outputResidual,
    int32_t frameIdx, rt::Tensor const& codecHiddensBuffer, TalkerLocal const& tlocal, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnection", nvtx_colors::BLUE);

    // Old working runner: codec_sum + trailing_text[step] for early frames, then tts_pad
    // codecHiddensBuffer positions 1-14 are pre-filled by runCodePredictorGenerationForFrame.

    check::check(codes.size() == 16, "Expected 16 codes (code_0 from Talker + code_1-15 from CodePredictor)");

    // Residual output is in Talker's space (2048-dim) since it feeds back to the Talker decoder
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    check::check(outputResidual.reshape({1, 1, hiddenSize}), "Tensor reshape failed");
    if (mQwen3TTSTalkerEngine && outputResidual.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        return computeResidualConnectionHost(codes, outputResidual, frameIdx, tlocal, stream);
    }

    // Select addend: trailing text for early frames, tts_pad for later
    __half const* addend;
    if (frameIdx < tlocal.trailingTextLen)
    {
        if (frameIdx == tlocal.trailingTextLen - 1)
        {
            // Last trailing entry: tts_eos
            addend = mTtsEosEmbed.dataPointer<__half>();
        }
        else
        {
            // body[1 + frameIdx] from projected buffer (text starts at index 3)
            int32_t const textIdx = kAssistantPrefixLen + 1 + frameIdx;
            addend = tlocal.projectedBuffer.dataPointer<__half>() + static_cast<int64_t>(textIdx) * hiddenSize;
        }
    }
    else
    {
        addend = mTtsPadEmbed.dataPointer<__half>();
    }

    kernel::invokeResidualConnection(codecHiddensBuffer, mTalkerEmbeddingTable, mCodePredictorEmbeddingTables[14],
        codes[0], codes[15], addend, outputResidual, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnectionHost(
    std::vector<int32_t> const& codes, rt::Tensor& outputResidual, int32_t frameIdx,
    TalkerLocal const& tlocal, cudaStream_t stream)
{
    int32_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const codebookSize = mTalkerConfig.codebookSize;
    int32_t const activeGroups = mQwen3TTSCodePredictorEngine
        ? getQwen3TTSActiveCodePredictorGroups()
        : talker_constants::kNumRvqLayers;
    size_t const talkerOffset = static_cast<size_t>(codes[0]) * hiddenSize;
    if (mHostTalkerEmbeddingTable.size() < talkerOffset + static_cast<size_t>(hiddenSize))
    {
        LOG_ERROR("computeResidualConnectionHost: Talker embedding table is not available");
        return false;
    }
    if (mHostCodePredictorEmbeddingTables.size()
        < static_cast<size_t>(activeGroups) * codebookSize * hiddenSize)
    {
        LOG_ERROR("computeResidualConnectionHost: FP32 CodePredictor embedding table is not available");
        return false;
    }

    std::vector<float> residual(static_cast<size_t>(hiddenSize));
    std::copy(mHostTalkerEmbeddingTable.begin() + static_cast<std::ptrdiff_t>(talkerOffset),
        mHostTalkerEmbeddingTable.begin() + static_cast<std::ptrdiff_t>(talkerOffset + hiddenSize), residual.begin());

    for (int32_t group = 0; group < activeGroups; ++group)
    {
        int32_t const code = codes[static_cast<size_t>(group + 1)];
        size_t const offset = (static_cast<size_t>(group) * codebookSize + code) * hiddenSize;
        float const* embed = mHostCodePredictorEmbeddingTables.data() + offset;
        for (int32_t i = 0; i < hiddenSize; ++i)
        {
            residual[static_cast<size_t>(i)] += embed[i];
        }
    }

    std::vector<float> addend;
    if (frameIdx < tlocal.trailingTextLen)
    {
        if (frameIdx == tlocal.trailingTextLen - 1)
        {
            addend = mUseHostTextProjection ? mHostTtsEosEmbed : copyTensorToHostFloat(mTtsEosEmbed, hiddenSize, stream);
        }
        else
        {
            int32_t const textIdx = kAssistantPrefixLen + 1 + frameIdx;
            if (mUseHostTextProjection)
            {
                addend.assign(tlocal.hostProjectedBuffer.begin() + static_cast<int64_t>(textIdx) * hiddenSize,
                    tlocal.hostProjectedBuffer.begin() + static_cast<int64_t>(textIdx + 1) * hiddenSize);
            }
            else
            {
                void* rowPtr = static_cast<char*>(tlocal.projectedBuffer.rawPointer())
                    + static_cast<int64_t>(textIdx) * hiddenSize * dataTypeSize(tlocal.projectedBuffer.getDataType());
                rt::Tensor row(rowPtr, {hiddenSize}, rt::DeviceType::kGPU, tlocal.projectedBuffer.getDataType());
                addend = copyTensorToHostFloat(row, hiddenSize, stream);
            }
        }
    }
    else
    {
        addend = mUseHostTextProjection ? mHostTtsPadEmbed : copyTensorToHostFloat(mTtsPadEmbed, hiddenSize, stream);
    }
    if (frameIdx < 12)
    {
        dumpVector("frame" + std::to_string(frameIdx) + "_addend_f32.bin", addend);
    }
    for (int32_t i = 0; i < hiddenSize; ++i)
    {
        residual[static_cast<size_t>(i)] += addend[static_cast<size_t>(i)];
    }
    if (frameIdx < 12)
    {
        if (char const* residualOverrideDir = std::getenv("QWEN3_TTS_RESIDUAL_OVERRIDE_DIR"))
        {
            std::vector<float> overrideResidual;
            auto const path = std::filesystem::path(residualOverrideDir)
                / ("native_frame" + std::to_string(frameIdx) + "_residual_f32.bin");
            if (loadFloatBin(path, residual.size(), overrideResidual))
            {
                residual = std::move(overrideResidual);
            }
        }
        if (frameIdx == 0)
        {
            if (char const* decodeOverride = std::getenv("QWEN3_TTS_DECODE1_EMBEDS_BIN"))
            {
                std::vector<float> overrideResidual;
                if (loadFloatBin(decodeOverride, residual.size(), overrideResidual))
                {
                    residual = std::move(overrideResidual);
                }
            }
        }
        dumpVector("frame" + std::to_string(frameIdx) + "_residual_f32.bin", residual);
    }
    CUDA_CHECK(cudaMemcpyAsync(outputResidual.rawPointer(), residual.data(), residual.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream));
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
    check::check(talkerHiddenStates.getDataType() == outputLastHidden.getDataType(),
        "extractTalkerLastHidden: input/output dtype mismatch");
    size_t const elemSize = dataTypeSize(talkerHiddenStates.getDataType());
    size_t const lastTokenOffset = (seqLen - 1) * hiddenSize * elemSize;
    size_t const copySize = hiddenSize * elemSize;

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

// ========== Phase 2 hooks: per-request execution context factories ==========
//
// These return fresh execution contexts that share engine weights but have
// independent CUDA state. Phase 2 ships them as compile-tested scaffolding only —
// the default per-engine contexts continue to serve every current call site.
// Phase 3 (slot pool) will pair these with per-slot CUDA streams and KV cache
// to enable concurrent N>1 inference.

std::unique_ptr<nvinfer1::IExecutionContext> Qwen3OmniTTSRuntime::createTalkerExecutionContext()
{
    if (!mQwen3TTSTalkerEngine)
    {
        LOG_WARNING("createTalkerExecutionContext: explicit-KV Talker engine not loaded; returning nullptr");
        return nullptr;
    }
    return mQwen3TTSTalkerEngine->createExecutionContext();
}

std::pair<std::unique_ptr<nvinfer1::IExecutionContext>, std::unique_ptr<nvinfer1::IExecutionContext>>
Qwen3OmniTTSRuntime::createCodePredictorExecutionContextPair()
{
    if (!mUseQwen3TTSCodePredictorEngine || !mQwen3TTSCodePredictorEngine)
    {
        LOG_WARNING(
            "createCodePredictorExecutionContextPair: native Qwen3-TTS CP engine not enabled; returning {nullptr,nullptr}");
        return {nullptr, nullptr};
    }
    return mQwen3TTSCodePredictorEngine->createExecutionContextPair();
}

// ========== Phase 3a Iter1: slot factories ==========
//
// Each factory allocates a fully-sized slot (per-request scratch tensors,
// KV double-buffers, execution context(s), RNG) that mirrors the engine's
// own ctor allocations. Iteration 1 only constructs the slots; no engine
// method yet consumes the Slot*, so the default (slot=nullptr) path
// continues to drive every current call site and audio output stays
// byte-identical to Phase 2. Iteration 2 will thread `Slot*` through
// generatePreparedInputs / talker decode loop / CP generate.

std::unique_ptr<Qwen3OmniTTSRuntime::TalkerSlot> Qwen3OmniTTSRuntime::createTalkerSlot(cudaStream_t stream)
{
    if (!mQwen3TTSTalkerEngine)
    {
        LOG_WARNING("createTalkerSlot: explicit-KV Talker engine not loaded; returning nullptr");
        return nullptr;
    }
    auto slot = std::make_unique<TalkerSlot>();
    slot->stream = stream;
    // Phase 3a Iter6 must-fix-B (codex round 3): use the paired factory so
    // prefillCtxOwned gets TRT profile 0 and decodeCtxOwned gets profile 1.
    // Calling createExecutionContext() twice would put BOTH at profile 1 and
    // silently corrupt prefill bindings.
    auto ctxPair = mQwen3TTSTalkerEngine->createExecutionContextPair();
    if (!ctxPair.first || !ctxPair.second)
    {
        LOG_WARNING("createTalkerSlot: failed to create per-slot execution context pair");
        return nullptr;
    }
    slot->prefillCtxOwned = std::move(ctxPair.first);
    slot->decodeCtxOwned = std::move(ctxPair.second);
    mQwen3TTSTalkerEngine->allocateSlot(*slot, stream);
    return slot;
}

std::unique_ptr<Qwen3OmniTTSRuntime::CodePredictorSlot> Qwen3OmniTTSRuntime::createCodePredictorSlot(cudaStream_t stream)
{
    if (!mUseQwen3TTSCodePredictorEngine || !mQwen3TTSCodePredictorEngine)
    {
        LOG_WARNING("createCodePredictorSlot: native Qwen3-TTS CP engine not enabled; returning nullptr");
        return nullptr;
    }
    auto slot = std::make_unique<CodePredictorSlot>();
    slot->stream = stream;
    auto ctxPair = mQwen3TTSCodePredictorEngine->createExecutionContextPair();
    if (!ctxPair.first || !ctxPair.second)
    {
        LOG_WARNING("createCodePredictorSlot: failed to create per-slot execution context pair");
        return nullptr;
    }
    slot->prefillCtxOwned = std::move(ctxPair.first);
    slot->decodeCtxOwned = std::move(ctxPair.second);
    mQwen3TTSCodePredictorEngine->allocateSlot(*slot);
    return slot;
}

} // namespace rt
} // namespace trt_edgellm
