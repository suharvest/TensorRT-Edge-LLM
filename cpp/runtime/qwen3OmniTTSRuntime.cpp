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
#include "llmInferenceSpecDecodeRuntime.h"
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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;
using namespace talker_constants;

namespace
{
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

// Forward-declared from fork highperf/runtime-service tail (b9c57c8 reference
// runtime); pulled into anonymous namespace so P2-C nested-class constructors
// can resolve them without porting the full BF16 reference-runtime overhaul.
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

// Additional helpers from fork b9c57c8 needed by P2-C nested-class code.

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

// 5-arg overload — for now delegates to 4-arg (RuntimeOptions backend selection
// will activate once P2-C tail commits land). Worker.cpp passes options here.
Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream, RuntimeOptions const& /*options*/)
    : Qwen3OmniTTSRuntime(talkerEngineDir, codePredictorEngineDir, tokenizerDir, stream)
{
}

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream)
    : mStream(stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::init", nvtx_colors::YELLOW);
    LOG_INFO("Initializing Qwen3-Omni Talker runner");
    LOG_INFO("  Talker: %s", talkerEngineDir.c_str());
    LOG_INFO("  CodePredictor: %s", codePredictorEngineDir.c_str());

    // Load tokenizer
    std::filesystem::path const tokenizerPath = tokenizerDir.empty()
        ? std::filesystem::path(talkerEngineDir).parent_path()
        : std::filesystem::path(tokenizerDir);
    LOG_INFO("  Tokenizer: %s", tokenizerPath.string().c_str());
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    bool const tokenizerLoaded = mTokenizer->loadFromHF(tokenizerPath);
    ELLM_CHECK(tokenizerLoaded, "Failed to load tokenizer from: " + tokenizerPath.string());

    bool const configValid = validateAndFillConfig(talkerEngineDir);
    ELLM_CHECK(configValid, "Failed to validate and fill config");

    bool const runnersInitialized = initializeEngineRunners(talkerEngineDir, codePredictorEngineDir);
    ELLM_CHECK(runnersInitialized, "Failed to initialize engine runners");

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
        bool const talkerCtxSet = mTalkerLLMRunner->setContextMemory(mSharedExecContextMemory);
        ELLM_CHECK(talkerCtxSet, "Failed to set context memory for Talker LLM engine");
        ELLM_CHECK(!mCodePredictorRunner || mCodePredictorRunner->setContextMemory(mSharedExecContextMemory),
            "Failed to set context memory for CodePredictor engine");
    }

    // Determine max batch size from engine configs (use the minimum of Talker and CodePredictor)
    mMaxBatchSize = std::min(mTalkerLLMConfig.maxSupportedBatchSize, mCodePredictorConfig.maxSupportedBatchSize);
    check::check(mMaxBatchSize >= 1, "maxBatchSize must be >= 1");
    LOG_INFO("Max batch size: %d (Talker=%d, CodePredictor=%d)", mMaxBatchSize, mTalkerLLMConfig.maxSupportedBatchSize,
        mCodePredictorConfig.maxSupportedBatchSize);

    bool const codePredictorWeightsLoaded = loadCodePredictorWeights(codePredictorEngineDir);
    ELLM_CHECK(codePredictorWeightsLoaded, "Failed to load CodePredictor weights");

#ifdef CUTE_DSL_GEMM_ENABLED
    bool const cuteDslLoaded = CuteDslGemmRunner::loadKernelModule();
    ELLM_CHECK(cuteDslLoaded, "Failed to load CuTe DSL GEMM kernel module");
#endif

    bool const bufferAllocated = allocateBuffer();
    ELLM_CHECK(bufferAllocated, "Failed to allocate buffers");

    bool const talkerWeightsLoaded = loadTalkerWeights(talkerEngineDir, stream);
    ELLM_CHECK(talkerWeightsLoaded, "Failed to load Talker weights");

    // Load text embedding table (thinker vocab).
    // Used for standalone TTS and for projecting TTS special tokens.
    // TTS: text_embedding.safetensors in talkerEngineDir (copied by builder).
    // Omni: use thinker's embedding.safetensors from tokenizerPath instead.
    {
        std::filesystem::path const textEmbedPath = mIsOmni
            ? tokenizerPath / "embedding.safetensors"
            : std::filesystem::path(talkerEngineDir) / "text_embedding.safetensors";
        LOG_INFO("Loading text embedding from: %s", textEmbedPath.string().c_str());
        std::vector<rt::Tensor> textEmbedTensors;
        bool const textEmbedLoaded = safetensors::loadSafetensors(textEmbedPath, textEmbedTensors, stream);
        ELLM_CHECK(textEmbedLoaded, "Failed to load text embedding from: " + textEmbedPath.string());
        check::check(!textEmbedTensors.empty(), "text embedding file is empty");
        check::check(textEmbedTensors[0].getShape().getNumDims() == 2,
            "text embedding tensor should be 2D [vocabSize, hiddenSize]");
        mTextEmbeddingTable = std::move(textEmbedTensors[0]);
        LOG_INFO("Text embedding table loaded: [%lld, %lld]", mTextEmbeddingTable.getShape()[0],
            mTextEmbeddingTable.getShape()[1]);
    }

    // Note: mTalkerEmbeddingTable is loaded by loadTalkerWeights() above.

    // Load CodePredictor embedding tables from codec_embeddings.safetensors
    // mNumRvqLayers is already set from config.json num_code_groups in validateAndFillConfig()
    {
        std::filesystem::path const embedPath
            = std::filesystem::path(codePredictorEngineDir) / "codec_embeddings.safetensors";
        std::vector<rt::Tensor> allEmbedTensors;
        bool const codecEmbedLoaded = safetensors::loadSafetensors(embedPath, allEmbedTensors, stream);
        ELLM_CHECK(codecEmbedLoaded, "Failed to load codec_embeddings.safetensors from: " + embedPath.string());
        check::check(static_cast<int32_t>(allEmbedTensors.size()) == mNumRvqLayers,
            "codec_embeddings.safetensors has " + std::to_string(allEmbedTensors.size()) + " tensors, expected "
                + std::to_string(mNumRvqLayers) + " (num_code_groups - 1)");
        mCodePredictorEmbeddingTables.resize(mNumRvqLayers);
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const key = "embedding_" + std::to_string(i);
            auto it = std::find_if(allEmbedTensors.begin(), allEmbedTensors.end(),
                [&key](rt::Tensor const& t) { return t.getName() == key; });
            check::check(it != allEmbedTensors.end(), "Missing key '" + key + "' in codec_embeddings.safetensors");
            check::check(it->getShape().getNumDims() == 2, key + " should be 2D [codebookSize, hiddenSize]");
            mCodePredictorEmbeddingTables[i] = std::move(*it);
        }
    }
    LOG_INFO("Loaded %d CodePredictor embedding tables (from config num_code_groups=%d)", mNumRvqLayers,
        mTalkerConfig.numCodeGroups);

    initializeTTSEmbeddings(stream);

    CUDA_CHECK(cudaEventCreateWithFlags(&mTtfaStart, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&mTtfaEnd, cudaEventDefault));

    LOG_INFO("Qwen3-Omni TTS runtime initialized successfully");
}

Qwen3OmniTTSRuntime::~Qwen3OmniTTSRuntime()
{
#ifdef CUTE_DSL_GEMM_ENABLED
    CuteDslGemmRunner::unloadKernelModule();
#endif
    if (mTtfaStart)
    {
        cudaEventDestroy(mTtfaStart);
    }
    if (mTtfaEnd)
    {
        cudaEventDestroy(mTtfaEnd);
    }
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

    // CodePredictor RVQ configuration: Omni uses 16 code groups, TTS uses 32
    mTalkerConfig.numCodeGroups = configJson["num_code_groups"].get<int32_t>();
    check::check(mTalkerConfig.numCodeGroups >= 2,
        "num_code_groups must be >= 2, got: " + std::to_string(mTalkerConfig.numCodeGroups));
    mNumRvqLayers = mTalkerConfig.numCodeGroups - 1;
    mNumCodesPerFrame = mTalkerConfig.numCodeGroups;
    LOG_INFO("Config num_code_groups=%d -> RVQ layers=%d, codes per frame=%d", mTalkerConfig.numCodeGroups,
        mNumRvqLayers, mNumCodesPerFrame);

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
    LOG_INFO("Loading %d CodePredictor lm_head weights", mNumRvqLayers);
    mCodePredictorLmHeadWeights.resize(mNumRvqLayers);
    {
        std::filesystem::path const lmHeadPath = std::filesystem::path(codePredictorEngineDir) / "lm_heads.safetensors";
        std::vector<rt::Tensor> allLmHeadTensors;
        if (!safetensors::loadSafetensors(lmHeadPath, allLmHeadTensors, mStream))
        {
            LOG_ERROR("Failed to load lm_heads.safetensors from: %s", lmHeadPath.string().c_str());
            return false;
        }
        if (static_cast<int32_t>(allLmHeadTensors.size()) != mNumRvqLayers)
        {
            LOG_ERROR("lm_heads.safetensors has %zu entries, expected %d (matching codec_embeddings count)",
                allLmHeadTensors.size(), mNumRvqLayers);
            return false;
        }
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
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

    mTalkerConfig.codebookSize = static_cast<int32_t>(mCodePredictorLmHeadWeights[0].getShape()[0]);
    LOG_INFO("Loaded %d CodePredictor lm_head weights, codebookSize=%d", mNumRvqLayers, mTalkerConfig.codebookSize);

    // Load small_to_mtp_projection: projects Talker hidden (2048) → CodePredictor input (1024)
    {
        std::filesystem::path const projPath
            = std::filesystem::path(codePredictorEngineDir) / "small_to_mtp_projection.safetensors";
        if (std::filesystem::exists(projPath))
        {
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
            mUseSmallToMtpProjection = true;
            LOG_INFO("Loaded small_to_mtp_projection: weight=%ldx%ld, bias=%ld", mSmallToMtpWeight.getShape()[0],
                mSmallToMtpWeight.getShape()[1], mSmallToMtpBias.getShape()[0]);
        }
        else if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
        {
            mUseSmallToMtpProjection = false;
            LOG_INFO("No small_to_mtp_projection needed (talkerHiddenSize == codePredictorHiddenSize = %d)",
                mTalkerConfig.talkerHiddenSize);
        }
        else
        {
            LOG_ERROR(
                "small_to_mtp_projection.safetensors required when talkerHiddenSize (%d) != "
                "codePredictorHiddenSize (%d)",
                mTalkerConfig.talkerHiddenSize, mTalkerConfig.codePredictorHiddenSize);
            return false;
        }
    }

    // CodePredictor embedding tables are already loaded in the constructor (with auto-detection
    // of mNumRvqLayers from codec_embeddings.safetensors).

    return true;
}

bool Qwen3OmniTTSRuntime::allocateBuffer()
{
    LOG_INFO("Allocating Qwen3-Omni TTS Runtime inference workspace buffers (maxBatchSize=%d)...", mMaxBatchSize);

    int64_t const maxSeqLen = mTalkerConfig.maxSeqLen;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const maxBS = mMaxBatchSize;

    try
    {
        // Per-batch prefill workspace (reused sequentially per batch in buildTalkerPrefillFromSegments)
        mThinkerEmbedBuffer = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mThinkerEmbedBuffer");
        mGpuTokenIdsBuffer
            = rt::Tensor({1, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mGpuTokenIdsBuffer");
        mMLPWorkspace = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mMLPWorkspace");
        mProjectedBuffer = rt::Tensor(
            {maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mProjectedBuffer");
        // Talker input embeds: [maxBS, maxSeqLen, H] for batched prefill
        mTalkerInputEmbeds = rt::Tensor({maxBS * maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerInputEmbeds");

        // Talker LLM workspace — batched
        mTalkerLogits = rt::Tensor(
            {maxBS, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mTalkerLogits");
        mTalkerSelectedIndices
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerSelectedIndices");
        mHostSelectedTokenIds
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedTokenIds");
        mHostTalkerContextLength
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostTalkerContextLength");

        // CodePredictor workspace — batch=1 (per-batch for-loop, each frame resets KV cache)
        mCodePredictorLogits = rt::Tensor(
            {1, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mCodePredictorLogits");

        mCodePredictorLogitsPerHead.resize(mNumRvqLayers);
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            mCodePredictorLogitsPerHead[i] = rt::Tensor({1, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kFLOAT, "mCodePredictorLogitsPerHead_" + std::to_string(i));
        }

        mCodePredictorPrefillInput = rt::Tensor({1, 2, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorPrefillInput");
        mCodePredictorCodecIds
            = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorCodecIds");
        mCodePredictorCodecEmbed = rt::Tensor({1, 1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorCodecEmbed");
        mRawCodecEmbed = rt::Tensor(
            {1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mRawCodecEmbed");
        mSmallToMtpProjectedHidden = rt::Tensor({1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mSmallToMtpProjectedHidden");
        mCodePredictorSelectedIndices
            = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorSelectedIndices");
        mHostSelectedCodeIds
            = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedCodeIds");
        mHostCodePredictorContextLength
            = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostCodePredictorContextLength");

        // Residual + decode buffers — batched for Talker engine execution
        mResidualEmbedBuffer = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mResidualEmbedBuffer");
        mTalkerDecodingIds
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerDecodingIds");
        mTalkerDecodingEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerDecodingEmbed");

        // KVCache reset — batched
        mHostReuseKVCacheLengths
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostReuseKVCacheLengths");

        // Sampling workspace — allocated at maxBS for batched topKtopP
        int32_t const defaultTopK{0};
        float const defaultTopP{0.9F};
        trt_edgellm::SamplingParams samplingParams(
            static_cast<int32_t>(maxBS), mTalkerConfig.talkerVocabSize, 1.0f, defaultTopK, defaultTopP);
        int64_t const samplingWorkspaceSize = trt_edgellm::getTopKtopPSamplingWorkspaceSize(
            static_cast<int32_t>(maxBS), mTalkerConfig.talkerVocabSize, samplingParams);
        mSamplingWorkspace = rt::Tensor(
            {samplingWorkspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "mSamplingWorkspace");
        // Per-batch seen codec tokens for repetition penalty
        mSeenCodecTokensBuf = rt::Tensor({maxBS, mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "mSeenCodecTokensBuf");

        // Generation loop workspace — Talker batched, CodePredictor batch=1
        mTalkerHiddenStatesBuffer = rt::Tensor({maxBS, maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerHiddenStatesBuffer");
        mCodePredictorHiddenStatesBuffer = rt::Tensor({1, mNumCodesPerFrame, mTalkerConfig.codePredictorHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodePredictorHiddenStatesBuffer");
        mTalkerLastHidden = rt::Tensor(
            {maxBS, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mTalkerLastHidden");

        // [Phase B C1] mCodecHiddensBuffer is now allocated per-request as a local
        // in handleAudioGeneration / generation helpers. See header comment for rationale.

        // Trailing text hidden buffer: [maxBS * (maxSeqLen+1), H] — per-batch regions for Omni multi-batch
        mStreamingTrailingHidden = rt::Tensor({maxBS * (maxSeqLen + 1), talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mStreamingTrailingHidden");

        // Gather/scatter index buffer for multimodal token projection
        mGatherIndicesBuffer
            = rt::Tensor({maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mGatherIndicesBuffer");

        // Streaming: single-token workspace for appendTrailingToken
        mStreamingTokenId = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mStreamingTokenId");
        mStreamingTokenEmbed = rt::Tensor(
            {1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingTokenEmbed");
        mStreamingProjOut
            = rt::Tensor({1, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingProjOut");
        mStreamingMlpWork
            = rt::Tensor({1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingMlpWork");

        LOG_INFO("Talker buffers allocated (maxBS=%d, maxSeqLen=%ld, talkerH=%ld, cpH=%d)", mMaxBatchSize, maxSeqLen,
            talkerHiddenSize, mTalkerConfig.codePredictorHiddenSize);
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

    // Load hidden_projection weights (same architecture as text_projection, different weights)
    std::filesystem::path const hiddenProjPath = std::filesystem::path(weightsDir) / "hidden_projection.safetensors";
    if (std::filesystem::exists(hiddenProjPath))
    {
        mIsOmni = true;
        std::vector<rt::Tensor> hiddenTensors;
        if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
        {
            LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
            return false;
        }
        if (!extractMLPWeightsFromTensors(
                hiddenTensors, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight, mHiddenFC2Bias, "hidden_projection"))
        {
            return false;
        }
        LOG_INFO("hidden_projection weights loaded from: %s", hiddenProjPath.string().c_str());
    }
    else
    {
        mIsOmni = false;
        LOG_INFO("hidden_projection.safetensors not found at %s (multimodal token projection unavailable)",
            hiddenProjPath.string().c_str());
    }

    // Note: mTextEmbeddingTable is loaded separately in the constructor with mIsOmni-aware path selection
    // (TTS: text_embedding.safetensors from talkerEngineDir, Omni: embedding.safetensors from tokenizerDir)

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
    ELLM_CHECK(
        shape.getNumDims() == 2, "Text embedding table must be 2D, got " + std::to_string(shape.getNumDims()) + "D");

    int64_t const vocabSize = shape[0];
    int64_t const thinkerHiddenSize = shape[1];

    ELLM_CHECK(mTalkerConfig.ttsPadTokenId < vocabSize && mTalkerConfig.ttsBosTokenId < vocabSize
            && mTalkerConfig.ttsEosTokenId < vocabSize,
        "TTS token IDs out of vocab range: pad=" + std::to_string(mTalkerConfig.ttsPadTokenId)
            + ", bos=" + std::to_string(mTalkerConfig.ttsBosTokenId)
            + ", eos=" + std::to_string(mTalkerConfig.ttsEosTokenId) + ", vocabSize=" + std::to_string(vocabSize));

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
    rt::Tensor const& thinkerEmbed, int32_t speakerId, rt::Tensor& output, int64_t& outputSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = thinkerEmbed.getShape()[0];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;

    // N = text tokens after stripping 3-token role prefix and 5-token suffix
    int64_t const N = seqLen - kAssistantPrefixLen - kAssistantTrailingSuffix;
    // Non-streaming prefill: 8 fixed prefix rows + N text rows + 2 suffix rows
    outputSeqLen = kNonStreamingPrefixRows + N + 2; // = seqLen + 2
    LOG_INFO("projectToTalkerInput: seqLen=%ld, N=%ld (stripped prefix=%d suffix=%d), outputSeqLen=%ld, speakerId=%d",
        seqLen, N, kAssistantPrefixLen, kAssistantTrailingSuffix, outputSeqLen, speakerId);

    // Project all tokens via text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(thinkerEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, mProjectedBuffer,
        mMLPWorkspace, stream);

    // Fused kernel: build complete non-streaming prefill buffer
    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    kernel::invokeAssistantPreamble(mProjectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, static_cast<int32_t>(N), output, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream, std::vector<int64_t> const& perBatchContextLengths)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerPrefillStep", nvtx_colors::PURPLE);

    auto inputShape = inputEmbeds.getShape();
    if (inputShape.getNumDims() != 3)
    {
        LOG_ERROR("executeTalkerPrefillStep: Input must be 3D [batchSize, seqLen, hiddenSize], got %dD",
            inputShape.getNumDims());
        return false;
    }

    int64_t const batchSize = inputEmbeds.getTRTDims().d[0];
    int64_t const seqLen = inputEmbeds.getTRTDims().d[1];

    if (batchSize > mMaxBatchSize)
    {
        LOG_ERROR("executeTalkerPrefillStep: batchSize %ld exceeds maxBatchSize %d", batchSize, mMaxBatchSize);
        return false;
    }

    // Reset Talker KV cache — shape must match batchSize so commitSequenceLength sees consistent activeBatchSize
    check::check(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mHostReuseKVCacheLengths.dataPointer<int32_t>(), batchSize, 0);
    mTalkerLLMRunner->getCacheManager().resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Set per-batch context lengths (determines which token's logits the engine extracts)
    check::check(mHostTalkerContextLength.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostContextLength = mHostTalkerContextLength.dataPointer<int32_t>();
    if (!perBatchContextLengths.empty())
    {
        for (int64_t i = 0; i < batchSize; ++i)
        {
            hostContextLength[i] = static_cast<int32_t>(perBatchContextLengths[i]);
        }
    }
    else
    {
        for (int64_t i = 0; i < batchSize; ++i)
        {
            hostContextLength[i] = static_cast<int32_t>(seqLen);
        }
    }

    // Reshape logits to match the prefill batch size (CUDA graph capture may leave it at maxBS)
    check::check(outputLogits.reshape({batchSize, outputLogits.getShape()[outputLogits.getShape().getNumDims() - 1]}),
        "Tensor reshape failed");

    // Execute prefill
    rt::OptionalInputTensors emptyDeepstack{};
    return mTalkerLLMRunner->executePrefillStep(inputEmbeds, mHostTalkerContextLength, emptyDeepstack, outputLogits,
        rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream);
}

bool Qwen3OmniTTSRuntime::executeCodePredictorPrefillStep(rt::Tensor const& codecTokenEmbeds, int32_t generationStep,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorPrefillStep", nvtx_colors::ORANGE);

    // Reset CodePredictor KV cache for new frame (always per-batch, BS=1)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    mHostReuseKVCacheLengths.dataPointer<int32_t>()[0] = 0;
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

    int32_t const lmHeadIdx = std::min(generationStep, mNumRvqLayers - 1);
    if (!mCodePredictorRunner->setLmHeadWeight(mCodePredictorLmHeadWeights[lmHeadIdx]))
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
    rt::Tensor& codecHiddensBuffer, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorDecodingStep", nvtx_colors::ORANGE);

    CUDA_CHECK(cudaMemcpyAsync(
        mCodePredictorCodecIds.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int32_t const embedIdx = std::min(embeddingTableIndex, mNumRvqLayers - 1);
    // Lookup into mRawCodecEmbed (talkerHiddenSize=2048) — codec embedding tables are in Talker's space
    check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(
        mCodePredictorCodecIds, mCodePredictorEmbeddingTables[embedIdx], std::nullopt, mRawCodecEmbed, stream);

    // Save raw (2048-dim) embedding to codecHiddensBuffer for residual connection
    // Position mapping: generationStep 1->pos 1, 2->pos 2, ..., (mNumRvqLayers-1)->pos (mNumRvqLayers-1)
    if (generationStep >= 1 && generationStep <= mNumRvqLayers - 1)
    {
        int64_t const H = mTalkerConfig.talkerHiddenSize;
        __half* dst = static_cast<__half*>(codecHiddensBuffer.rawPointer()) + generationStep * H;
        CUDA_CHECK(
            cudaMemcpyAsync(dst, mRawCodecEmbed.rawPointer(), H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    // Project mRawCodecEmbed → mCodePredictorCodecEmbed if needed.
    check::check(mRawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    if (!mUseSmallToMtpProjection)
    {
        CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecEmbed.rawPointer(), mRawCodecEmbed.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(mRawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias, mCodePredictorCodecEmbed, stream);
    }

    int32_t const lmHeadIdx = std::min(generationStep, mNumRvqLayers - 1);

    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");

    if (mCodePredictorGraphsCaptured)
    {
        // Graph path: lm_head_weight addresses were bound during capture and remain unchanged,
        // so setLmHeadWeight is unnecessary. Each graph is keyed by its per-head output buffer.
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
        if (!mCodePredictorRunner->setLmHeadWeight(mCodePredictorLmHeadWeights[lmHeadIdx]))
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

    // Talker: capture for all supported batch sizes (1..maxBatchSize)
    bool captureStatus{true};
    for (int32_t bs = 1; bs <= mMaxBatchSize; ++bs)
    {
        check::check(mResidualEmbedBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(
            mTalkerHiddenStatesBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(mTalkerLogits.reshape({bs, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
        captureStatus &= mTalkerLLMRunner->captureVanillaDecodingCudaGraph(mResidualEmbedBuffer, mTalkerLogits,
            emptyLoraWeightsName, stream, rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)});
    }

    // CodePredictor: mNumRvqLayers graphs, one per lm_head_weight.
    // Each graph uses a distinct output logits buffer so that the decodingKey (which includes
    // the output address) naturally produces a unique key per graph.
    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
        "Tensor reshape failed");
    rt::OptionalOutputTensor cpHiddenOpt{std::ref(mCodePredictorHiddenStatesBuffer)};

    for (int32_t i = 0; i < mNumRvqLayers; ++i)
    {
        if (!mCodePredictorRunner->setLmHeadWeight(mCodePredictorLmHeadWeights[i]))
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

    // Determine speaker ID
    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (request.speakerId >= 0)
    {
        speakerId = request.speakerId;
    }
    else if (!request.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(request.speakerName);
    }

    // MLP projection: thinker embed → talker input embeds (non-streaming, outputSeqLen = seqLen + 2)
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!projectToTalkerInput(mThinkerEmbedBuffer, speakerId, mTalkerInputEmbeds, outSeqLen, stream))
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
    std::vector<TalkerGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGeneration", nvtx_colors::PURPLE);
    int32_t const activeBatchSize = static_cast<int32_t>(requests.size());
    LOG_INFO("Starting batched audio generation for %d request(s)", activeBatchSize);

    check::check(activeBatchSize > 0 && activeBatchSize <= mMaxBatchSize,
        "Batch size " + std::to_string(activeBatchSize) + " exceeds max " + std::to_string(mMaxBatchSize));

    response.batchRvqCodes.clear();
    response.numFramesPerSample.clear();
    response.success = false;

    // [Phase B C1] Per-request codec hiddens buffer for the residual connection.
    // Previously this was a runtime-global member (`mCodecHiddensBuffer`) which
    // races on `cudaMemsetAsync` when more than one request is in flight.
    // Allocating per-call keeps the lifetime tied to the request so concurrent
    // generations cannot collide. C2 folds this into `TalkerSlot` once full
    // slot threading lands.
    rt::Tensor codecHiddensBuffer({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "codecHiddensBuffer");

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceSpecDecodeRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams singleSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);

    // Per-batch sequential: tokenize → prepareTalkerInput → prefill → sample first codec token
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<int64_t> perBatchSeqLens(activeBatchSize);

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        LLMGenerationRequest::Request llmReq;
        llmReq.messages = requests[b].messages;
        LLMGenerationRequest::FormattedRequest formatted;
        if (!mTokenizer->applyChatTemplate(llmReq, formatted, requests[b].applyChatTemplate,
                requests[b].addGenerationPrompt, requests[b].enableThinking))
        {
            LOG_ERROR("Chat template failed for batch %d", b);
            return false;
        }
        std::vector<int32_t> const textTokenIds = mTokenizer->encode(formatted.formattedCompleteRequest);

        int64_t seqLen = 0;
        if (!prepareTalkerInput(textTokenIds, requests[b], seqLen, stream))
        {
            LOG_ERROR("Input preparation failed for batch %d", b);
            return false;
        }
        perBatchSeqLens[b] = seqLen;

        {
            TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
            if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
            {
                LOG_ERROR("Talker prefill failed for batch %d", b);
                return false;
            }
        }

        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);
        trt_edgellm::topKtopPSamplingFromLogits(
            mTalkerLogits, mTalkerSelectedIndices, singleSamplingParams, mSamplingWorkspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        states[b].codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
        states[b].rvqCodes.reserve(requests[b].maxAudioLength);
        LOG_INFO("Batch %d: first codec token: %d", b, states[b].codecToken);
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const safe = std::max(1, talkerKVCapacity - static_cast<int32_t>(perBatchSeqLens[b]));
        effectiveMaxFrames = std::max(effectiveMaxFrames, std::min(requests[b].maxAudioLength, safe));
    }

    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);

    // Wire optional streaming hooks from each request (empty/disabled by default = identical behavior).
    std::vector<PerBatchStreamingHooks> streamingHooks;
    bool anyStreaming = false;
    for (auto const& r : requests)
    {
        if (r.codecChunkFrames > 0 && r.onAudioChunkReady)
        {
            anyStreaming = true;
            break;
        }
    }
    if (anyStreaming)
    {
        streamingHooks.resize(activeBatchSize);
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            streamingHooks[b].codecChunkFrames = requests[b].codecChunkFrames;
            streamingHooks[b].subsequentChunkFrames = requests[b].subsequentChunkFrames;
            streamingHooks[b].onAudioChunkReady = requests[b].onAudioChunkReady;
            streamingHooks[b].shouldCancel = requests[b].shouldCancel;
        }
    }

    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, codecHiddensBuffer, stream,
            /*prefillSeqLens=*/{}, streamingHooks))
    {
        return false;
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        response.batchRvqCodes.push_back(std::move(states[b].rvqCodes));
        response.numFramesPerSample.push_back(states[b].talkerFrames);
    }

    int32_t totalFrames = 0;
    for (auto const& s : states)
        totalFrames += s.talkerFrames;
    mMultimodalMetrics.recordRun(0, 0, activeBatchSize, totalFrames);

    response.success = true;
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGenerationFromThinker(
    std::vector<OmniGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGenerationFromThinker", nvtx_colors::PURPLE);

    int32_t const activeBatchSize = static_cast<int32_t>(requests.size());
    LOG_INFO("Starting batched Omni audio generation for %d request(s)", activeBatchSize);

    check::check(activeBatchSize > 0 && activeBatchSize <= mMaxBatchSize,
        "Batch size " + std::to_string(activeBatchSize) + " exceeds max " + std::to_string(mMaxBatchSize));

    response.batchRvqCodes.clear();
    response.numFramesPerSample.clear();
    response.success = false;

    // [Phase B C1] Per-request codec hiddens buffer (see handleAudioGeneration for rationale).
    rt::Tensor codecHiddensBuffer({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "codecHiddensBuffer");

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceSpecDecodeRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<rt::Tensor> perBatchTrailingViews(activeBatchSize);
    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);

    // Phase 1: Build per-batch Talker prefill embeddings.
    // buildTalkerPrefillFromSegments writes to mTalkerInputEmbeds starting at row 0 (the "scratch"
    // region). After building each batch we relocate the result to a safe per-batch slot at
    // row ((b+1) * maxInputSeqLen) so that the next build's scratch write doesn't clobber it.
    // Batch 0 is special: it's built last (after all others are safely stashed), so its data
    // can stay in place at row 0.
    int64_t const maxInputSeqLen = mTalkerLLMConfig.maxSupportedInputLength;
    std::vector<int64_t> perBatchSeqLen(activeBatchSize);
    int64_t maxOutSeqLen = 0;

    // Process batches 1..N-1 first (stash each), then batch 0 last (stays in scratch = row 0)
    auto buildOneBatch = [&](int32_t b) -> bool {
        auto const& req = requests[b];
        if (req.fullText.empty() && req.textTokenIds.empty())
        {
            LOG_ERROR("Omni request batch %d has empty fullText and no textTokenIds", b);
            return false;
        }

        std::vector<int32_t> const textTokenIds
            = req.textTokenIds.empty() ? mTokenizer->encode(req.fullText) : req.textTokenIds;

        int32_t speakerId = mTalkerConfig.defaultSpeakerId;
        if (req.speakerId >= 0)
            speakerId = req.speakerId;
        else if (!req.speakerName.empty())
            speakerId = getSpeakerIdByName(req.speakerName);

        __half* const batchTrailingPtr
            = static_cast<__half*>(mStreamingTrailingHidden.rawPointer()) + b * trailingStride * hiddenSize;
        rt::Tensor batchTrailingBuf(
            batchTrailingPtr, rt::Coords{trailingStride, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        CUDA_CHECK(cudaMemsetAsync(batchTrailingPtr, 0, trailingStride * hiddenSize * sizeof(__half), stream));

        int32_t trailingCount = 0;
        int64_t outSeqLen = 0;
        if (!buildTalkerPrefillFromSegments(textTokenIds, req.thinkerPrefillEmbeds, req.thinkerHiddenStates,
                req.prefillLength, mTextEmbeddingTable, speakerId, batchTrailingBuf, trailingCount, outSeqLen, stream))
        {
            return false;
        }

        finalizeTrailing(batchTrailingBuf, trailingCount, stream);
        trailingCount++;

        perBatchSeqLen[b] = outSeqLen;
        maxOutSeqLen = std::max(maxOutSeqLen, outSeqLen);

        perBatchTrailingViews[b]
            = rt::Tensor(batchTrailingPtr, rt::Coords{static_cast<int64_t>(trailingCount), hiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        trailingPtrs[b] = &perBatchTrailingViews[b];
        states[b].rvqCodes.reserve(req.maxAudioLength);
        return true;
    };

    // Build batches 1..N-1 first and stash each to slot (b * maxInputSeqLen)
    for (int32_t b = activeBatchSize - 1; b >= 1; --b)
    {
        if (!buildOneBatch(b))
            return false;

        __half* const slotPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + b * maxInputSeqLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(slotPtr, mTalkerInputEmbeds.rawPointer(),
            perBatchSeqLen[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    // Build batch 0 last — result stays at row 0 (no copy needed)
    if (!buildOneBatch(0))
        return false;

    // Phase 2: Assemble contiguous padded [BS, maxOutSeqLen, H] from per-batch slots.
    // Slot layout: batch 0 at row 0, batch b>=1 at row (b * maxInputSeqLen).
    // Target layout: batch b at row (b * maxOutSeqLen).
    // Process in reverse so high-address batches are moved first (avoids overlap).
    __half* const base = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    for (int32_t b = activeBatchSize - 1; b >= 0; --b)
    {
        __half* const src = base + (b == 0 ? 0 : b * maxInputSeqLen * hiddenSize);
        __half* const dst = base + b * maxOutSeqLen * hiddenSize;
        if (src != dst)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                dst, src, perBatchSeqLen[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        if (perBatchSeqLen[b] < maxOutSeqLen)
        {
            CUDA_CHECK(cudaMemsetAsync(dst + perBatchSeqLen[b] * hiddenSize, 0,
                (maxOutSeqLen - perBatchSeqLen[b]) * hiddenSize * sizeof(__half), stream));
        }
    }

    // Single batched Talker prefill with per-batch context lengths
    check::check(mTalkerInputEmbeds.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({activeBatchSize, maxOutSeqLen, mTalkerConfig.talkerHiddenSize}),
        "Tensor reshape failed");

    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(
                mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream, perBatchSeqLen))
        {
            LOG_ERROR("Batched Talker prefill failed");
            return false;
        }
    }

    // Phase 3: Per-batch logit adjustment and sampling from batched logits [BS, vocabSize]
    check::check(mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        rt::Tensor logitsSlice(static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
            rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);
    }

    trt_edgellm::topKtopPSamplingFromLogits(
        mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].codecToken = hostTokens[b];
        LOG_INFO("Omni batch %d: first codec token: %d, trailing=%d, seqLen=%ld", b, states[b].codecToken,
            static_cast<int32_t>(perBatchTrailingViews[b].getShape()[0]), perBatchSeqLen[b]);
    }

    if (getProfilingEnabled())
    {
        CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const eff = std::min(requests[b].maxAudioLength,
            std::max(1, talkerKVCapacity - static_cast<int32_t>(requests[b].textTokenIds.size())));
        effectiveMaxFrames = std::max(effectiveMaxFrames, eff);
    }

    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, codecHiddensBuffer, stream, perBatchSeqLen))
    {
        return false;
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        response.batchRvqCodes.push_back(std::move(states[b].rvqCodes));
        response.numFramesPerSample.push_back(states[b].talkerFrames);
    }

    int32_t totalFrames = 0;
    bool anyHitEos = false;
    for (auto const& s : states)
    {
        totalFrames += s.talkerFrames;
        if (s.codecToken == mTalkerConfig.codecEosId)
        {
            anyHitEos = true;
        }
    }
    mMultimodalMetrics.recordRun(0, 0, activeBatchSize, totalFrames);

    if (getProfilingEnabled())
    {
        int32_t const codesPerFrame = mTalkerConfig.numCodeGroups;
        auto talkerPrefillData = gTimer.getTimingData(metrics::StageNames::kTALKER_PREFILL);
        float prefillMs = talkerPrefillData ? talkerPrefillData->getTotalGpuTimeMs() : 0.0f;
        int32_t prefillSeqLen = perBatchSeqLen.empty() ? 0 : static_cast<int32_t>(perBatchSeqLen[0]);

        mOmniTalkerMetrics.recordRun(totalFrames, totalFrames * codesPerFrame, prefillMs, prefillSeqLen,
            anyHitEos ? "eos" : "max_length", false);

        auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;
        float audioDurationS
            = static_cast<float>(totalFrames * kAudioSamplesPerFrame) / static_cast<float>(kAudioSampleRate);
        mOmniLatencyMetrics.audioDurationSeconds = audioDurationS;
        mOmniLatencyMetrics.audioSamples = static_cast<int64_t>(totalFrames) * kAudioSamplesPerFrame;
        mOmniLatencyMetrics.sampleRate = kAudioSampleRate;
        mOmniLatencyMetrics.realTimeFactor = (talkerGenMs > 0.0f) ? (audioDurationS / (talkerGenMs / 1000.0f)) : 0.0f;
    }

    response.success = true;
    return true;
}

// ========== Shared Generation Loop ==========

bool Qwen3OmniTTSRuntime::runTalkerGenerationLoop(std::vector<PerBatchTalkerState>& states, int32_t activeBatchSize,
    int32_t maxFrames, SamplingParams const& talkerSamplingParams, SamplingParams const& predictorSamplingParams,
    float repetitionPenalty, std::vector<rt::Tensor const*> const& trailingTextHiddens,
    rt::Tensor& codecHiddensBuffer, cudaStream_t stream, std::vector<int64_t> const& prefillSeqLens,
    std::vector<PerBatchStreamingHooks> const& streamingHooks)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runTalkerGenerationLoop", nvtx_colors::PURPLE);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t unfinished = activeBatchSize;

    // Initialize per-batch seen token tracking. numSeenTokens starts at 0 so the repetition
    // penalty kernel reads no entries on the first decode frame, avoiding a read of the
    // unseeded GPU buffer. The in-loop code below records tokens into both the host set and
    // GPU buffer starting from the first iteration.
    // NOTE: the host set / GPU buffer are still off-by-one (host set tracks the token entering
    // each iteration while the GPU buffer receives the newly sampled token). That functional
    // drift is pre-existing and tracked as a follow-up together with the standalone-TTS
    // Chinese-prompt prefill-EOS anomaly surfaced during MR 770 review.
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].numSeenTokens = 0;
    }

    int32_t globalFrame = 0;
    {
        TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);

        while (unfinished > 0 && globalFrame < maxFrames)
        {
            // Per-batch: CodePredictor + residual (batch=1 engine calls)
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Reset CodePredictor KV cache for this batch (batch=1)
                int32_t* cpReuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
                cpReuseData[0] = 0;
                check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
                mCodePredictorRunner->getCacheManager().resetForNewSequences(mHostReuseKVCacheLengths, stream);

                // Extract this batch's last hidden from mTalkerHiddenStatesBuffer.
                // After batched prefill: shape [BS, maxOutSeqLen, H] with padding; use the
                // per-batch actual length for correct last-token extraction.
                // After decode: shape [BS, 1, H], seqLen=1 for all batches.
                auto const& fullShape = mTalkerHiddenStatesBuffer.getShape();
                int64_t const paddedSeqDim = fullShape[1];
                int64_t const hDim = fullShape[2];
                int64_t const actualSeqDim
                    = (!prefillSeqLens.empty() && paddedSeqDim > 1) ? prefillSeqLens[b] : paddedSeqDim;
                rt::Tensor batchHiddenView(
                    static_cast<__half*>(mTalkerHiddenStatesBuffer.rawPointer()) + b * paddedSeqDim * hDim,
                    rt::Coords{1, actualSeqDim, hDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

                check::check(mTalkerLastHidden.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
                if (!extractTalkerLastHidden(batchHiddenView, mTalkerLastHidden, stream))
                {
                    LOG_ERROR("extractTalkerLastHidden failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }

                std::vector<int32_t> frameCodes;
                if (!runCodePredictorGenerationForFrame(states[b].codecToken, mTalkerLastHidden,
                        predictorSamplingParams, frameCodes, codecHiddensBuffer, stream))
                {
                    LOG_ERROR("CodePredictor failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }

                states[b].rvqCodes.push_back(std::move(frameCodes));

                // Compute residual for this batch, writing to batch b's slot in mResidualEmbedBuffer
                rt::Tensor residualSlice(
                    static_cast<__half*>(mResidualEmbedBuffer.rawPointer()) + b * mTalkerConfig.talkerHiddenSize,
                    rt::Coords{1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

                if (!computeResidualConnection(states[b].rvqCodes.back(), trailingTextHiddens[b], globalFrame,
                        residualSlice, codecHiddensBuffer, stream))
                {
                    LOG_ERROR("Residual connection failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }
            }

            // Batched Talker decode step: mResidualEmbedBuffer [BS, 1, H] → mTalkerLogits [BS, vocabSize]
            check::check(mResidualEmbedBuffer.reshape({activeBatchSize, 1, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");
            check::check(mTalkerHiddenStatesBuffer.reshape({activeBatchSize, 1, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");

            if (!mTalkerLLMRunner->executeVanillaDecodingStep(mResidualEmbedBuffer, mTalkerLogits,
                    rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)}, stream))
            {
                LOG_ERROR("Batched Talker decoding step failed at frame %d", globalFrame);
                return false;
            }

            // Per-batch logit adjustment (different seenTokens per batch)
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Create views into batch b's logits row
                rt::Tensor logitsSlice(
                    static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
                    rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

                // Per-batch seenTokens buffer
                rt::Tensor seenBufSlice(
                    mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity,
                    rt::Coords{mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

                kernel::invokeTalkerLogitAdjust(seenBufSlice, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
                    mTalkerConfig.talkerVocabSize, codecEosId, states[b].numSeenTokens, repetitionPenalty, stream);
            }

            // Batched sampling
            check::check(
                mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
            check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
                mSamplingWorkspace, stream, 42, static_cast<uint64_t>(globalFrame + 1));
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
                activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Per-batch: update state, check EOS
            int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Track seen token for repetition penalty
                if (states[b].seenTokenSet.insert(states[b].codecToken).second)
                {
                    int32_t* seenBuf
                        = mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity;
                    CUDA_CHECK(cudaMemcpyAsync(seenBuf + states[b].numSeenTokens,
                        mTalkerSelectedIndices.dataPointer<int32_t>() + b, sizeof(int32_t), cudaMemcpyDeviceToDevice,
                        stream));
                    states[b].numSeenTokens++;
                }

                states[b].codecToken = hostTokens[b];
                states[b].talkerFrames++;

                // ===== Standalone-TTS streaming: cancel + per-chunk emission =====
                bool const hasHook = (b < static_cast<int32_t>(streamingHooks.size()));
                if (hasHook)
                {
                    auto const& hook = streamingHooks[b];
                    if (hook.shouldCancel && hook.shouldCancel())
                    {
                        states[b].finished = true;
                        unfinished--;
                        continue;
                    }
                    // Adaptive chunk size: first emitted chunk uses codecChunkFrames (low TTFA),
                    // subsequent chunks use subsequentChunkFrames if set (>0), else fall back to codecChunkFrames.
                    int32_t const threshold = (states[b].lastChunkEnd == 0 || hook.subsequentChunkFrames <= 0)
                        ? hook.codecChunkFrames
                        : hook.subsequentChunkFrames;
                    if (hook.codecChunkFrames > 0 && hook.onAudioChunkReady
                        && (states[b].talkerFrames - states[b].lastChunkEnd) >= threshold)
                    {
                        std::vector<std::vector<int32_t>> chunk(states[b].rvqCodes.begin() + states[b].lastChunkEnd,
                            states[b].rvqCodes.begin() + states[b].talkerFrames);
                        hook.onAudioChunkReady(chunk, b, false);
                        states[b].lastChunkEnd = states[b].talkerFrames;
                    }
                }

                if (states[b].codecToken == codecEosId || states[b].talkerFrames >= maxFrames)
                {
                    states[b].finished = true;
                    unfinished--;
                }
            }
            globalFrame++;
        }
    }

    // ===== Flush remainder chunks (standalone-TTS streaming path) =====
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        if (b < static_cast<int32_t>(streamingHooks.size()))
        {
            auto const& hook = streamingHooks[b];
            if (hook.onAudioChunkReady && states[b].lastChunkEnd < states[b].talkerFrames)
            {
                std::vector<std::vector<int32_t>> chunk(states[b].rvqCodes.begin() + states[b].lastChunkEnd,
                    states[b].rvqCodes.begin() + states[b].talkerFrames);
                hook.onAudioChunkReady(chunk, b, true);
                states[b].lastChunkEnd = states[b].talkerFrames;
            }
            else if (hook.onAudioChunkReady && states[b].lastChunkEnd == states[b].talkerFrames)
            {
                // Emit empty final marker so consumers know stream ended without unflushed frames.
                hook.onAudioChunkReady(std::vector<std::vector<int32_t>>{}, b, true);
            }
        }
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        bool const hitEos = (states[b].codecToken == codecEosId);
        LOG_INFO("Batch %d: %d audio frames (exit: %s)", b, states[b].talkerFrames, hitEos ? "EOS" : "maxFrames");
    }

    return true;
}

bool Qwen3OmniTTSRuntime::runSingleTalkerDecodeFrame(int32_t& codecToken, SamplingParams const& talkerSamplingParams,
    SamplingParams const& predictorSamplingParams, rt::Tensor const* trailingPtr, int32_t frameIdx,
    std::unordered_set<int32_t>& seenTokenSet, int32_t& numSeenTokens, float repetitionPenalty,
    std::vector<std::vector<int32_t>>& rvqCodes, cudaStream_t stream)
{
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // [Phase B C1] Per-frame codec hiddens buffer (streaming path).
    // C2 will hoist this into a slot-owned buffer; for now keep it local to satisfy
    // the new per-request signatures.
    rt::Tensor codecHiddensBuffer({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "codecHiddensBufferStreaming");

    // Ensure batch=1 shape for CodePredictor KV cache reset (streaming path is per-batch)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    mHostReuseKVCacheLengths.dataPointer<int32_t>()[0] = 0;
    mCodePredictorRunner->getCacheManager().resetForNewSequences(mHostReuseKVCacheLengths, stream);

    if (!extractTalkerLastHidden(mTalkerHiddenStatesBuffer, mTalkerLastHidden, stream))
    {
        LOG_ERROR("extractTalkerLastHidden failed at frame %d", frameIdx);
        return false;
    }

    std::vector<int32_t> frameCodes;
    if (!runCodePredictorGenerationForFrame(
            codecToken, mTalkerLastHidden, predictorSamplingParams, frameCodes, codecHiddensBuffer, stream))
    {
        LOG_ERROR("CodePredictor generation failed at frame %d", frameIdx);
        return false;
    }

    rvqCodes.push_back(std::move(frameCodes));

    if (!computeResidualConnection(
            rvqCodes.back(), trailingPtr, frameIdx, mResidualEmbedBuffer, codecHiddensBuffer, stream))
    {
        LOG_ERROR("Residual connection failed at frame %d", frameIdx);
        return false;
    }

    check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

    if (!mTalkerLLMRunner->executeVanillaDecodingStep(
            mResidualEmbedBuffer, mTalkerLogits, rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)}, stream))
    {
        LOG_ERROR("Talker decoding step failed at frame %d", frameIdx);
        return false;
    }

    kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
        mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
        mSamplingWorkspace, stream, 42, static_cast<uint64_t>(frameIdx + 1));
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(), sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (seenTokenSet.insert(codecToken).second)
    {
        CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
            mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        ++numSeenTokens;
    }

    codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
    return true;
}

bool Qwen3OmniTTSRuntime::runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
    SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes, rt::Tensor& codecHiddensBuffer,
    cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runCodePredictorGenerationForFrame", nvtx_colors::ORANGE);

    // Original model logic:
    // - codes: code_0 (from Talker) + code_1..code_(N-1) (from CodePredictor) = mNumCodesPerFrame codes
    // - hidden_states: written directly to codecHiddensBuffer (per-request, passed in)

    outputCodes.clear();
    outputCodes.reserve(mNumCodesPerFrame);

    // code_0 comes from Talker
    outputCodes.push_back(codecToken);

    int64_t const hiddenSize = mTalkerConfig.codePredictorHiddenSize;

    // ========== Prefill: generate code_1 ==========
    // Input: concat([proj(talker_hidden), proj(embed(code_0))]) -> [1, 2, codePredictorHiddenSize]
    // NOTE: code_0 embedding uses TALKER's codec_embedding (2048-dim),
    // projected to 1024 via small_to_mtp_projection
    // PyTorch: last_id_hidden = self.small_to_mtp_projection(self.get_input_embeddings()(input_ids))
    int32_t code = 0;
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_PREFILL, stream);

        // Step 1: Lookup code_0 from Talker's embedding table into mRawCodecEmbed (2048-dim)
        CUDA_CHECK(cudaMemcpyAsync(
            mCodePredictorCodecIds.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        kernel::embeddingLookup(mCodePredictorCodecIds, mTalkerEmbeddingTable, std::nullopt, mRawCodecEmbed, stream);

        // Step 2: Project talkerHiddenState → slot 0 of prefill input
        // Step 3: Project mRawCodecEmbed → slot 1 of prefill input
        // Step 4: Concat into mCodePredictorPrefillInput [1, 2, codePredictorHiddenSize]
        check::check(mRawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(
            mCodePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
        if (!mUseSmallToMtpProjection)
        {
            CUDA_CHECK(cudaMemcpyAsync(mCodePredictorPrefillInput.rawPointer(), talkerHiddenState.rawPointer(),
                hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + hiddenSize,
                mRawCodecEmbed.rawPointer(), hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            kernel::invokeLinearLayer(
                talkerHiddenState, mSmallToMtpWeight, mSmallToMtpBias, mSmallToMtpProjectedHidden, stream);
            kernel::invokeLinearLayer(
                mRawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias, mCodePredictorCodecEmbed, stream);
            CUDA_CHECK(cudaMemcpyAsync(mCodePredictorPrefillInput.rawPointer(), mSmallToMtpProjectedHidden.rawPointer(),
                hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + hiddenSize,
                mCodePredictorCodecEmbed.rawPointer(), hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }

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
        trt_edgellm::topKtopPSamplingFromLogits(
            mCodePredictorLogits, mCodePredictorSelectedIndices, samplingParams, mSamplingWorkspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedCodeIds.rawPointer(), mCodePredictorSelectedIndices.rawPointer(),
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        code = mHostSelectedCodeIds.dataPointer<int32_t>()[0];
        outputCodes.push_back(code); // code_1
    } // end kCODEPREDICTOR_PREFILL scope

    // ========== Write embedding lookups to codecHiddensBuffer for residual connection ==========
    // codecHiddensBuffer layout: [1, mNumCodesPerFrame, H]
    // Position 0:  embed(code_0)  using Talker's embedding   - filled in computeResidualConnection
    // Position 1..(mNumRvqLayers-1): codec_embedding[step-1](code_step)
    //   — filled here (input embeddings, NOT engine hidden states)
    // Position mNumRvqLayers: embed(code_last) using CodePredictor embed[-1] - filled in computeResidualConnection
    //
    // PyTorch original (modeling_qwen3_omni.py:3419):
    //   mid_residual_hiddens = [hid[0] for hid in predictor_result.hidden_states[1:]]
    //   hid[0] = inputs_embeds for each decode step = codec_embedding[step-1](code_step)
    //   NOT the transformer output (last layer hidden states)

    // codecHiddensBuffer stores raw (2048-dim) codec embeddings for the residual connection
    check::check(
        codecHiddensBuffer.reshape({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(codecHiddensBuffer.rawPointer(), 0, codecHiddensBuffer.getMemoryCapacity(), stream));

    // ========== Decoding loop: generate code_2 to code_(mNumRvqLayers) ==========
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_GENERATION, stream);
        for (int step = 2; step <= mNumRvqLayers; ++step)
        {
            check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
                "Tensor reshape failed");

            rt::OptionalInputTensor prevHiddenOpt{std::ref(mCodePredictorHiddenStatesBuffer)};

            // PyTorch generation_steps logic:
            //   - generation_steps=1: embed(code_1) with codec_embedding[0], output with lm_head[1] -> code_2
            //   - generation_steps=N: embed(code_N) with codec_embedding[N-1], output with lm_head[N] -> code_(N+1)
            int32_t const embeddingIdx = step - 2; // step=2->embed[0], step=3->embed[1], ...
            int32_t const lmHeadIdx = step - 1;    // step=2->lm_head[1], step=3->lm_head[2], ...

            if (!executeCodePredictorDecodingStep(code, embeddingIdx, lmHeadIdx, mCodePredictorLogits,
                    mCodePredictorHiddenStatesBuffer, codecHiddensBuffer, stream))
            {
                return false;
            }

            // Embedding is now saved inside executeCodePredictorDecodingStep before engine execution

            trt_edgellm::topKtopPSamplingFromLogits(
                mCodePredictorLogits, mCodePredictorSelectedIndices, samplingParams, mSamplingWorkspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedCodeIds.rawPointer(), mCodePredictorSelectedIndices.rawPointer(),
                sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            code = mHostSelectedCodeIds.dataPointer<int32_t>()[0];
            outputCodes.push_back(code); // code_2 to code_15
        }
    } // end kCODEPREDICTOR_GENERATION scope

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnection(std::vector<int32_t> const& codes,
    rt::Tensor const* trailingTextHidden, int32_t generationStep, rt::Tensor& outputResidual,
    rt::Tensor const& codecHiddensBuffer, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnection", nvtx_colors::BLUE);

    // PyTorch (modeling_qwen3_omni.py:3429-3434):
    //   if generation_step < trailing_text_hidden.shape[1]:
    //       inputs_embeds += trailing_text_hidden[:, generation_step]
    //   else:
    //       inputs_embeds += tts_pad_embed
    // [Phase B C1] codecHiddensBuffer is per-request (was global mCodecHiddensBuffer).

    check::check(static_cast<int32_t>(codes.size()) == mNumCodesPerFrame,
        "Expected " + std::to_string(mNumCodesPerFrame) + " codes, got " + std::to_string(codes.size()));

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    // reshape only when the tensor owns its memory (skip for non-owning views
    // which are already constructed with the correct shape by the caller)
    if (!outputResidual.reshape({1, 1, hiddenSize}))
    {
        check::check(
            outputResidual.getShape() == rt::Coords{1, 1, hiddenSize}, "Non-owning residual tensor has wrong shape");
    }

    __half const* addend = mTtsPadEmbed.dataPointer<__half>();
    if (trailingTextHidden != nullptr)
    {
        int64_t const trailingLen = trailingTextHidden->getShape()[0];
        if (generationStep < trailingLen)
        {
            addend = static_cast<__half const*>(trailingTextHidden->rawPointer()) + generationStep * hiddenSize;
        }
    }

    kernel::invokeResidualConnection(codecHiddensBuffer, mTalkerEmbeddingTable,
        mCodePredictorEmbeddingTables[mNumRvqLayers - 1], codes[0], codes[mNumRvqLayers], addend, outputResidual,
        stream);

    return true;
}

// [Phase B C1] Host-side residual connection (net new function — no CUDA kernel).
// Uses HEAD's trailingTextHidden / generationStep addend logic (NOT fork's mTrailingTextLen path).
// Currently a stub: routes through the GPU path with a synchronous copy if needed.
// Real callers will be wired up in subsequent commits; for now keep parity with the
// GPU `computeResidualConnection` so the symbol exists and the build links.
bool Qwen3OmniTTSRuntime::computeResidualConnectionHost(std::vector<int32_t> const& codes,
    rt::Tensor const* trailingTextHidden, int32_t generationStep, rt::Tensor& outputResidual, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnectionHost", nvtx_colors::BLUE);

    // Allocate a transient per-call codecHiddensBuffer matching the GPU path,
    // since the host-side path does not have access to a request-scoped buffer.
    rt::Tensor codecHiddensBufferLocal({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "codecHiddensBufferHostLocal");
    CUDA_CHECK(cudaMemsetAsync(
        codecHiddensBufferLocal.rawPointer(), 0, codecHiddensBufferLocal.getMemoryCapacity(), stream));

    return computeResidualConnection(
        codes, trailingTextHidden, generationStep, outputResidual, codecHiddensBufferLocal, stream);
}

bool Qwen3OmniTTSRuntime::extractTalkerLastHidden(
    rt::Tensor const& talkerHiddenStates, rt::Tensor& outputLastHidden, cudaStream_t stream)
{
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

    check::check(outputLastHidden.reshape({batchSize, hiddenSize}), "Tensor reshape failed");

    // Extract last token for each batch: output[b] = input[b, seqLen-1, :]
    size_t const copySize = hiddenSize * sizeof(__half);
    for (int64_t b = 0; b < batchSize; ++b)
    {
        size_t const srcOffset = (b * seqLen + seqLen - 1) * hiddenSize * sizeof(__half);
        size_t const dstOffset = b * hiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(outputLastHidden.rawPointer()) + dstOffset,
            static_cast<char const*>(talkerHiddenStates.rawPointer()) + srcOffset, copySize, cudaMemcpyDeviceToDevice,
            stream));
    }

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

// ═══════════════════════════════════════════════════════════════════════════
//        Shared Decode Frame + Prefill Construction
// ═══════════════════════════════════════════════════════════════════════════

bool Qwen3OmniTTSRuntime::buildTalkerPrefillFromSegments(std::vector<int32_t> const& textTokenIds,
    rt::Tensor const* prefillEmbedPtr, rt::Tensor const* prefillHiddenPtr, int32_t prefillLen,
    rt::Tensor const& thinkerEmbedTable, int32_t speakerId, rt::Tensor& trailingTextHidden, int32_t& trailingCount,
    int64_t& outSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    // Step 1: Build layer-0 embeddings in mThinkerEmbedBuffer
    check::check(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

    if (prefillEmbedPtr != nullptr)
    {
        int32_t const copyLen = std::min(prefillLen, static_cast<int32_t>(seqLen));
        size_t const prefillBytes = copyLen * thinkerHiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(mThinkerEmbedBuffer.rawPointer(), prefillEmbedPtr->rawPointer(), prefillBytes,
            cudaMemcpyDeviceToDevice, stream));

        int32_t const genLen = static_cast<int32_t>(seqLen) - copyLen;
        if (genLen > 0)
        {
            check::check(mGpuTokenIdsBuffer.reshape({1, genLen}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data() + copyLen,
                genLen * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            __half* genDst = static_cast<__half*>(mThinkerEmbedBuffer.rawPointer()) + copyLen * thinkerHiddenSize;
            rt::Tensor genEmbedView(genDst, rt::Coords{1, static_cast<int64_t>(genLen), thinkerHiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            kernel::embeddingLookup(mGpuTokenIdsBuffer, thinkerEmbedTable, std::nullopt, genEmbedView, stream);
        }
    }
    else
    {
        check::check(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data(), seqLen * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        rt::Tensor embedView(mThinkerEmbedBuffer.rawPointer(), rt::Coords{1, seqLen, thinkerHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        kernel::embeddingLookup(mGpuTokenIdsBuffer, thinkerEmbedTable, std::nullopt, embedView, stream);
    }

    // Step 2: Project ALL tokens through text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mProjectedBuffer, mMLPWorkspace, stream);

    // Step 3: Parse segments by <|im_start|> positions
    std::vector<SegmentInfo> segments;
    for (int64_t i = 0; i < seqLen; ++i)
    {
        if (textTokenIds[i] == kImStartTokenId)
        {
            int32_t const roleId = (i + 1 < seqLen) ? textTokenIds[i + 1] : -1;
            segments.push_back({i, seqLen, roleId});
        }
    }
    for (size_t s = 0; s + 1 < segments.size(); ++s)
    {
        segments[s].endPos = segments[s + 1].startPos;
    }

    std::vector<size_t> userSegmentIndices;
    int64_t assistantSegIdx = -1;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (segments[s].roleId == kSystemRoleId)
            continue;
        else if (segments[s].roleId == kUserRoleId)
            userSegmentIndices.push_back(s);
        else if (segments[s].roleId == kAssistantRoleId)
            assistantSegIdx = static_cast<int64_t>(s);
    }

    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildTalkerPrefillFromSegments: could not find assistant segment");
        return false;
    }

    // Step 4: Project multimodal tokens via hidden_projection using gather/scatter kernels
    bool const hasHiddenProjection = (mHiddenFC1Weight.getShape().volume() > 0);
    if (hasHiddenProjection && prefillHiddenPtr != nullptr)
    {
        std::vector<int32_t> mmPositions;
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos && i < prefillLen; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == kAudioTokenId || tid == kImageTokenId || tid == kVideoTokenId)
                    mmPositions.push_back(static_cast<int32_t>(i));
            }
        }

        if (!mmPositions.empty())
        {
            int64_t const numMM = static_cast<int64_t>(mmPositions.size());
            // Safe to repurpose mThinkerEmbedBuffer here: step 1 filled it with full-sequence
            // layer-0 embeddings, but those have already been projected into mTalkerInputEmbeds
            // by step 2's invokeTalkerMLP call. The reshape below overwrites that data, which is
            // intentional — we only need it as scratch for the gather/MLP/scatter chain.
            check::check(mThinkerEmbedBuffer.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            check::check(mMLPWorkspace.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            check::check(mTalkerInputEmbeds.reshape({numMM, hiddenSize}), "Tensor reshape failed");

            // Upload indices and use vectorized gather kernel instead of per-row cudaMemcpy
            check::check(mGatherIndicesBuffer.reshape({numMM}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mGatherIndicesBuffer.rawPointer(), mmPositions.data(), numMM * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            rt::Tensor hiddenSource(const_cast<void*>(prefillHiddenPtr->rawPointer()),
                rt::Coords{static_cast<int64_t>(prefillLen), thinkerHiddenSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF);
            kernel::invokeGather(hiddenSource, mGatherIndicesBuffer, mThinkerEmbedBuffer, stream);

            kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight,
                mHiddenFC2Bias, mTalkerInputEmbeds, mMLPWorkspace, stream);

            // Scatter projected multimodal embeddings back into mProjectedBuffer
            kernel::invokeScatter(mTalkerInputEmbeds, mGatherIndicesBuffer, mProjectedBuffer, stream);
        }
    }
    else if (!hasHiddenProjection || prefillHiddenPtr == nullptr)
    {
        // Zero out multimodal rows as fallback
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == kAudioTokenId || tid == kImageTokenId || tid == kVideoTokenId)
                {
                    __half* rowPtr = static_cast<__half*>(mProjectedBuffer.rawPointer()) + i * hiddenSize;
                    CUDA_CHECK(cudaMemsetAsync(rowPtr, 0, hiddenSize * sizeof(__half), stream));
                }
            }
        }
    }

    // Step 5: Build Talker prefill input — user segments + restructured assistant preamble
    auto const& assistantSeg = segments[assistantSegIdx];
    constexpr int32_t kAssistantRestructuredLen = kNonStreamingPrefixRows + 1;
    constexpr int32_t kAssistantTrailingOffset = kAssistantPrefixLen + 1;

    int64_t userTotalLen = 0;
    for (size_t sIdx : userSegmentIndices)
        userTotalLen += (segments[sIdx].endPos - segments[sIdx].startPos);

    outSeqLen = userTotalLen + kAssistantRestructuredLen;
    check::check(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(mTalkerInputEmbeds.rawPointer(), 0, outSeqLen * hiddenSize * sizeof(__half), stream));

    int64_t outOffset = 0;
    for (size_t sIdx : userSegmentIndices)
    {
        auto const& seg = segments[sIdx];
        int64_t const segLen = seg.endPos - seg.startPos;
        __half const* src = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + seg.startPos * hiddenSize;
        __half* dst = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outOffset * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, segLen * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        outOffset += segLen;
    }

    // Build restructured assistant preamble via invokeAssistantPreamble
    {
        __half const* const assistantProjPtr
            = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + assistantSeg.startPos * hiddenSize;
        constexpr int64_t kPreambleFullLen = kNonStreamingPrefixRows + 1 + 2;
        int64_t const requiredCapacity = (outSeqLen + kPreambleFullLen) * hiddenSize * sizeof(__half);
        check::check(static_cast<int64_t>(mTalkerInputEmbeds.getMemoryCapacity()) >= requiredCapacity,
            "mTalkerInputEmbeds too small for preamble scratch");
        __half* const scratchPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outSeqLen * hiddenSize;
        rt::Tensor preambleScratch(
            scratchPtr, rt::Coords{kPreambleFullLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        int64_t const assistantInputLen = kAssistantPrefixLen + 1;
        rt::Tensor assistantSlice(const_cast<__half*>(assistantProjPtr), rt::Coords{assistantInputLen, hiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        kernel::invokeAssistantPreamble(assistantSlice, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
            mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
            mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, 1, preambleScratch, stream);

        __half* const aOut = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + userTotalLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(aOut, scratchPtr, kAssistantRestructuredLen * hiddenSize * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Step 6: Fill trailing text hidden states
    int64_t const assistantSegLen = assistantSeg.endPos - assistantSeg.startPos;
    trailingCount = std::min(static_cast<int32_t>(assistantSegLen - kAssistantTrailingOffset),
        static_cast<int32_t>(trailingTextHidden.getShape()[0]) - 1);
    if (trailingCount > 0)
    {
        __half const* trailSrc = static_cast<__half const*>(mProjectedBuffer.rawPointer())
            + (assistantSeg.startPos + kAssistantTrailingOffset) * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(trailingTextHidden.rawPointer(), trailSrc,
            trailingCount * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    LOG_INFO("buildTalkerPrefillFromSegments: outSeqLen=%ld (user=%ld + assistant=%d), trailing=%d", outSeqLen,
        userTotalLen, kAssistantRestructuredLen, trailingCount);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//        Incremental Trailing Hidden Helpers (for streaming)
// ═══════════════════════════════════════════════════════════════════════════

void Qwen3OmniTTSRuntime::appendTrailingToken(int32_t tokenId, rt::Tensor const& thinkerEmbedTable,
    rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;

    // Upload token ID (reuse pre-allocated GPU buffer)
    CUDA_CHECK(
        cudaMemcpyAsync(mStreamingTokenId.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // embed_tokens(tokenId) → mStreamingTokenEmbed [1, thinkerH]
    // embeddingLookup expects [1, 1, H] output; mStreamingTokenEmbed is [1, H] — same memory, just reshape for kernel
    rt::Tensor embedView(mStreamingTokenEmbed.rawPointer(), rt::Coords{1, 1, mTalkerConfig.thinkerHiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::embeddingLookup(mStreamingTokenId, thinkerEmbedTable, std::nullopt, embedView, stream);

    // text_projection: mStreamingTokenEmbed [1, thinkerH] → mStreamingProjOut [1, talkerH]
    kernel::invokeTalkerMLP(mStreamingTokenEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mStreamingProjOut, mStreamingMlpWork, stream);

    // Write to trailingTextHidden[trailingIdx]
    __half* dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mStreamingProjOut.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

void Qwen3OmniTTSRuntime::finalizeTrailing(rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    __half* dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mTtsEosEmbed.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

// ═══════════════════════════════════════════════════════════════════════════
//        Thinker-Talker Streaming Pipeline (single CUDA stream)
// ═══════════════════════════════════════════════════════════════════════════

bool Qwen3OmniTTSRuntime::handleStreamingGeneration(LLMInferenceSpecDecodeRuntime& thinkerRuntime,
    LLMGenerationRequest& thinkerRequest, LLMGenerationResponse& thinkerResponse,
    ThinkerTalkerStreamingConfig const& streamingConfig, OmniGenerationRequest const& omniBaseRequest,
    TalkerGenerationResponse& talkerResponse, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TTSRuntime::handleStreamingGeneration", nvtx_colors::PURPLE);

    LOG_INFO(
        "Starting Thinker-Talker streaming pipeline (prefillThreshold=%d)", streamingConfig.talkerPrefillThreshold);

    talkerResponse.batchRvqCodes.clear();
    talkerResponse.numFramesPerSample.clear();
    talkerResponse.success = false;

    float const talkerTemperature = (omniBaseRequest.talkerTemperature > 0) ? omniBaseRequest.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (omniBaseRequest.talkerTopK > 0) ? omniBaseRequest.talkerTopK : 50;
    float const talkerTopP = (omniBaseRequest.talkerTopP > 0) ? omniBaseRequest.talkerTopP : 1.0f;
    float const repetitionPenalty = omniBaseRequest.repetitionPenalty;

    SamplingParams talkerSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t numSeenTokens = 0;
    std::unordered_set<int32_t> seenTokenSet;

    struct StreamingState
    {
        std::vector<int32_t> assistantTokens;
        bool thinkerFinished{false};
        bool talkerPrefillDone{false};
        bool talkerError{false};
        int32_t talkerFrames{0};
        int32_t codecToken{-1};
        int32_t trailingIdx{0};
        int32_t lastChunkEnd{0};
        std::vector<std::vector<int32_t>> rvqCodes;
        std::vector<int32_t> inputIds;
    };
    StreamingState state;
    state.rvqCodes.reserve(omniBaseRequest.maxAudioLength);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
    int32_t const maxTrailingLen = static_cast<int32_t>(trailingStride);

    // Streaming uses batch slot 0 of the trailing buffer
    size_t const slot0Bytes = trailingStride * hiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(mStreamingTrailingHidden.rawPointer(), 0, slot0Bytes, stream));

    rt::Tensor const& thinkerEmbedTable = thinkerRuntime.getEmbeddingTable();

    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (omniBaseRequest.speakerId >= 0)
    {
        speakerId = omniBaseRequest.speakerId;
    }
    else if (!omniBaseRequest.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(omniBaseRequest.speakerName);
    }

    int32_t const prefillThreshold = streamingConfig.talkerPrefillThreshold;
    int32_t const maxAudioLength = omniBaseRequest.maxAudioLength;

    // Reset reuse lengths to batch=1 for streaming (per-batch independent Talker)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    reuseData[0] = 0;

    auto makeTrailingView = [&]() -> rt::Tensor {
        return rt::Tensor(mStreamingTrailingHidden.rawPointer(),
            rt::Coords{static_cast<int64_t>(state.trailingIdx), hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF);
    };

    // ===== Per-token callback for the Thinker decode loop =====
    // SAFETY: This lambda captures stack-local state by reference. It is ONLY safe because
    // thinkerRuntime.handleRequest() invokes the callback synchronously on the same thread
    // (inside cudaStreamSynchronize in the decode loop). Never call this callback asynchronously.
    auto userCallback = thinkerRequest.onTokenGenerated;

    thinkerRequest.onTokenGenerated = [&, userCallback](TokenCallbackInfo const& info) {
        if (userCallback.has_value())
        {
            userCallback.value()(info);
        }
        if (info.batchIdx != 0 || state.talkerError)
        {
            return;
        }

        state.assistantTokens.push_back(info.tokenId);
        state.thinkerFinished = info.isFinished;

        int32_t const numAssistantTokens = static_cast<int32_t>(state.assistantTokens.size());

        if (!state.talkerPrefillDone && numAssistantTokens >= prefillThreshold)
        {
            LOG_INFO("Thinker produced %d assistant tokens, triggering Talker prefill", numAssistantTokens);

            // Reset KV caches
            auto& talkerCacheManager = mTalkerLLMRunner->getCacheManager();
            auto& cpCacheManager = mCodePredictorRunner->getCacheManager();
            talkerCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
            cpCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
            {
                auto& talkerKVManager = talkerCacheManager.getKVCacheManager();
                for (int32_t i = 0; i < talkerKVManager.numLayers(); ++i)
                {
                    rt::Tensor& layerKV = talkerKVManager.getCombinedKVCache(i);
                    CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
                }
                auto& cpKVManager = cpCacheManager.getKVCacheManager();
                for (int32_t i = 0; i < cpKVManager.numLayers(); ++i)
                {
                    rt::Tensor& layerKV = cpKVManager.getCombinedKVCache(i);
                    CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
                }
            }

            // Build combined token IDs — fetch Thinker input IDs from the runtime portal.
            auto const& thinkerInputs = thinkerRuntime.getBaseModelInputTokenIds();
            auto& textTokenIds = state.inputIds;
            if (textTokenIds.empty() && !thinkerInputs.empty())
            {
                textTokenIds = thinkerInputs[0];
            }
            textTokenIds.insert(textTokenIds.end(), state.assistantTokens.begin(), state.assistantTokens.end());

            // Use buildTalkerPrefillFromSegments for segment parsing, MLP projection, and prefill assembly
            auto prefillEmbedPtr = thinkerRuntime.getBaseModelHiddenStates(0);
            auto prefillHiddenPtr = thinkerRuntime.getBaseModelHiddenStates(thinkerRequest.acceptHiddenLayer);
            int32_t const prefillLen = thinkerRuntime.getBaseModelPrefillLength();

            int64_t outSeqLen = 0;
            if (!buildTalkerPrefillFromSegments(textTokenIds, prefillEmbedPtr, prefillHiddenPtr, prefillLen,
                    thinkerEmbedTable, speakerId, mStreamingTrailingHidden, state.trailingIdx, outSeqLen, stream))
            {
                state.talkerError = true;
                return;
            }

            // Talker prefill
            check::check(mTalkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
            check::check(mTalkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");

            {
                TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
                if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
                {
                    LOG_ERROR("Talker prefill failed during streaming pipeline");
                    state.talkerError = true;
                    return;
                }
            }

            kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
                mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
            trt_edgellm::topKtopPSamplingFromLogits(
                mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
                sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            state.codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
            if (seenTokenSet.insert(state.codecToken).second)
            {
                CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
                    mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
                ++numSeenTokens;
            }
            state.talkerPrefillDone = true;
            if (getProfilingEnabled())
            {
                CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
            }
            LOG_INFO("Talker prefill done, first codec token: %d", state.codecToken);

            // Generate frame 0 immediately after prefill
            if (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                rt::Tensor trailingView = makeTrailingView();
                rt::Tensor const* trailingPtr = (state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;
            }
        }
        else if (state.talkerPrefillDone && numAssistantTokens > prefillThreshold)
        {
            // Incremental: append new trailing token + run Talker decode step
            if (state.trailingIdx >= maxTrailingLen - 1)
            {
                LOG_WARNING("Trailing buffer full (%d/%d), skipping token append", state.trailingIdx, maxTrailingLen);
            }
            else
            {
                appendTrailingToken(
                    info.tokenId, thinkerEmbedTable, mStreamingTrailingHidden, state.trailingIdx, stream);
                state.trailingIdx++;
            }

            if (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                rt::Tensor trailingView = makeTrailingView();
                rt::Tensor const* trailingPtr = (state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;

                if (streamingConfig.codecChunkFrames > 0 && streamingConfig.onAudioChunkReady
                    && (state.talkerFrames - state.lastChunkEnd) >= streamingConfig.codecChunkFrames)
                {
                    std::vector<std::vector<int32_t>> chunk(
                        state.rvqCodes.begin() + state.lastChunkEnd, state.rvqCodes.begin() + state.talkerFrames);
                    streamingConfig.onAudioChunkReady(chunk, 0, false);
                    state.lastChunkEnd = state.talkerFrames;
                }
            }
        }
    };

    // ===== Run Thinker with the callback installed =====
    auto hiddenLayers = getThinkerHiddenLayerIndices();
    thinkerRequest.generateAudio = true;
    thinkerRequest.acceptHiddenLayer = hiddenLayers[1];
    bool thinkerSuccess = thinkerRuntime.handleRequest(thinkerRequest, thinkerResponse, stream, true);

    if (!thinkerSuccess)
    {
        LOG_ERROR("Thinker handleRequest failed in streaming pipeline");
        return false;
    }

    // ===== After Thinker finishes: finalize trailing and flush remaining Talker frames =====
    if (state.talkerPrefillDone && !state.talkerError)
    {
        if (state.trailingIdx < maxTrailingLen)
        {
            finalizeTrailing(mStreamingTrailingHidden, state.trailingIdx, stream);
            state.trailingIdx++;
        }
        else
        {
            LOG_WARNING("Trailing buffer full, cannot append tts_eos");
        }

        LOG_INFO("Thinker done. Flushing remaining Talker frames (current: %d, codec=%d, trailingIdx=%d)",
            state.talkerFrames, state.codecToken, state.trailingIdx);

        rt::Tensor flushTrailingView = makeTrailingView();
        rt::Tensor const* flushTrailingPtr = (state.trailingIdx > 0) ? &flushTrailingView : nullptr;

        int32_t const chunkSize = streamingConfig.codecChunkFrames;

        while (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
        {
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        flushTrailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty,
                        state.rvqCodes, stream))
                {
                    break;
                }
            }
            state.talkerFrames++;

            if (chunkSize > 0 && streamingConfig.onAudioChunkReady
                && (state.talkerFrames - state.lastChunkEnd) >= chunkSize)
            {
                std::vector<std::vector<int32_t>> chunk(
                    state.rvqCodes.begin() + state.lastChunkEnd, state.rvqCodes.begin() + state.talkerFrames);
                streamingConfig.onAudioChunkReady(chunk, 0, false);
                state.lastChunkEnd = state.talkerFrames;
            }
        }

        if (streamingConfig.onAudioChunkReady && state.lastChunkEnd < state.talkerFrames)
        {
            std::vector<std::vector<int32_t>> remainder(
                state.rvqCodes.begin() + state.lastChunkEnd, state.rvqCodes.begin() + state.talkerFrames);
            streamingConfig.onAudioChunkReady(remainder, 0, true);
        }
    }
    else
    {
        LOG_WARNING("Thinker finished but Talker prefill was never triggered (only %zu assistant tokens)",
            state.assistantTokens.size());
    }

    bool const hitEos = (state.codecToken == codecEosId);
    LOG_INFO("Streaming pipeline: %d audio frames (exit: %s, codec=%d)", state.talkerFrames,
        hitEos ? "EOS" : "maxAudioLength", state.codecToken);

    talkerResponse.batchRvqCodes.push_back(std::move(state.rvqCodes));
    talkerResponse.numFramesPerSample.push_back(state.talkerFrames);
    talkerResponse.success = state.talkerPrefillDone && !state.talkerError;

    mMultimodalMetrics.recordRun(0, 0, 1, state.talkerFrames);

    if (getProfilingEnabled())
    {
        int32_t const codesPerFrame = mTalkerConfig.numCodeGroups;
        auto talkerPrefillData = gTimer.getTimingData(metrics::StageNames::kTALKER_PREFILL);
        float prefillMs = talkerPrefillData ? talkerPrefillData->getTotalGpuTimeMs() : 0.0f;

        mOmniTalkerMetrics.recordRun(state.talkerFrames, state.talkerFrames * codesPerFrame, prefillMs,
            state.talkerPrefillDone ? static_cast<int32_t>(state.assistantTokens.size() + 30) : 0,
            hitEos ? "eos" : "max_length", true);

        auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;
        float audioDurationS = static_cast<float>(state.talkerFrames * 1920) / 24000.0f;
        mOmniLatencyMetrics.audioDurationSeconds = audioDurationS;
        mOmniLatencyMetrics.audioSamples = static_cast<int64_t>(state.talkerFrames) * 1920;
        mOmniLatencyMetrics.sampleRate = 24000;
        mOmniLatencyMetrics.realTimeFactor = (talkerGenMs > 0.0f) ? (audioDurationS / (talkerGenMs / 1000.0f)) : 0.0f;
    }

    return true;
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
