/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

// MOSS codec streaming state (Chunk 4). Ping-pong device buffers for
// transformer_offset_N (×4) + per-attention-layer (×12) caches. Schema is
// discovered at runtime from codec_browser_onnx_meta.json.
struct MossCodecState
{
    struct AttnCache
    {
        int32_t* offsetIn{nullptr};
        int32_t* offsetOut{nullptr};
        float* keysIn{nullptr};
        float* keysOut{nullptr};
        float* valuesIn{nullptr};
        float* valuesOut{nullptr};
        int32_t* positionsIn{nullptr};
        int32_t* positionsOut{nullptr};
        size_t cacheElements{0};
        size_t positionsElements{0};
    };

    std::array<int32_t*, 4> transformerOffsetIn{};
    std::array<int32_t*, 4> transformerOffsetOut{};
    std::array<AttnCache, 12> attn{};

    int32_t* audioCodesDevice{nullptr};       // int32[1, maxFrames, num_quantizers]
    int32_t* audioCodeLengthsDevice{nullptr}; // int32[1]
    float* audioOutDevice{nullptr};           // float[1, channels, maxFrames * downsample_rate]
    int32_t* audioOutLengthsDevice{nullptr};  // int32[1]

    int32_t maxFrames{8};
};

// MOSS-TTS-Nano per-slot mutable state.
// Layout for global KV: [num_layers=12, 2 (K=0,V=1), maxSeqLen, numHeads=12, headDim=64] FP16
// See docs/specs/moss-tts-nano-paged-kv-cpp.md and memory moss-tts-nano-port-recon.
struct MossTtsNanoSlot
{
    int32_t slotId{-1};
    bool inUse{false};

    void* globalKvDevice{nullptr};
    int32_t pastLen{0};            // engine-reported present_key_0 dim1; drives past_key shape + buffer copy size
    int32_t cumulativePastLen{0};  // monotonic counter; drives past_valid_lengths attention mask (matches Python past_valid_length)

    void* localKvDevice{nullptr};
    int32_t localPastLen{0};

    void* presentScratchK{nullptr};
    void* presentScratchV{nullptr};

    nvinfer1::IExecutionContext* prefillCtx{nullptr};
    nvinfer1::IExecutionContext* decodeCtx{nullptr};
    nvinfer1::IExecutionContext* localDecoderCtx{nullptr};
    nvinfer1::IExecutionContext* localCachedStepCtx{nullptr};
    nvinfer1::IExecutionContext* localFixedSampledFrameCtx{nullptr};

    cudaStream_t stream{nullptr};
    int32_t* pastValidLengthsDevice{nullptr};

    void* inputIdsDevice{nullptr};
    void* attentionMaskDevice{nullptr};
    void* globalHiddenDevice{nullptr};
    void* prevHiddenDevice{nullptr};

    // Local LLM (Chunk 3) — MOSS uses local_fixed_sampled_frame as a one-shot
    // sampler that internally runs the 16-step RVQ loop. See ort_cpu_runtime.py
    // for the official call pattern (sample_mode=FIXED branch).
    void* repetitionSeenMaskDevice{nullptr};   // int32[n_vq, audio_codebook_size]
    void* assistantRandomUDevice{nullptr};     // float[1]
    void* audioRandomUDevice{nullptr};         // float[n_vq]
    void* shouldContinueDevice{nullptr};       // int32[1, 1]
    void* frameTokenIdsDevice{nullptr};        // int32[1, n_vq]

    // Chunk 4: codec stateful streaming context + state (per slot).
    nvinfer1::IExecutionContext* codecDecodeCtx{nullptr};
    std::unique_ptr<MossCodecState> codec;

    // Per-slot deterministic RNG (N=2 concurrency parity gate). Was a
    // ``static thread_local std::mt19937 rng{42}`` inside sampleFrame, which
    // worked at N=1 but became non-deterministic at N=2: the second-served
    // worker thread observed an already-advanced RNG state because both
    // threads share neither the same RNG nor the same starting point
    // determinstically. Per-slot RNG, re-seeded on acquirePoolSlot, makes
    // each individual request byte-identical to its single-client baseline.
    std::mt19937 rng{42};
};

class MossTtsNanoRuntime
{
public:
    class RequestGuard
    {
    public:
        RequestGuard() = default;
        RequestGuard(MossTtsNanoRuntime* runtime, int32_t slotId);
        RequestGuard(RequestGuard const&) = delete;
        RequestGuard& operator=(RequestGuard const&) = delete;
        RequestGuard(RequestGuard&& other) noexcept;
        RequestGuard& operator=(RequestGuard&& other) noexcept;
        ~RequestGuard();

        MossTtsNanoSlot& slot() const;
        int32_t slotId() const { return mSlotId; }
        explicit operator bool() const { return mRuntime != nullptr && mSlotId >= 0; }

    private:
        void reset();
        MossTtsNanoRuntime* mRuntime{nullptr};
        int32_t mSlotId{-1};
    };

    MossTtsNanoRuntime(std::string const& engineDir, int32_t maxSlots, int32_t maxSeqLen);
    ~MossTtsNanoRuntime();

    MossTtsNanoRuntime(MossTtsNanoRuntime const&) = delete;
    MossTtsNanoRuntime& operator=(MossTtsNanoRuntime const&) = delete;

    int32_t acquirePoolSlot();
    void releasePoolSlot(int32_t slotId);
    RequestGuard beginRequest();
    void endRequest();

    MossTtsNanoSlot& slot(int32_t slotId);
    MossTtsNanoSlot const& slot(int32_t slotId) const;

    // Global LLM inference (Chunk 2)
    void* prefill(MossTtsNanoSlot& slot, std::vector<int32_t> const& inputIds,
        std::vector<int32_t> const& attentionMask);
    void* decodeStep(MossTtsNanoSlot& slot, std::vector<int32_t> const& inputIds, void const* prevHidden);

    // Local LLM (Chunk 3): runs local_fixed_sampled_frame.plan once per audio frame.
    // globalHidden: device ptr produced by prefill()/decodeStep()
    // frameTokenIdsHost: output buffer of size n_vq (caller-allocated)
    // returns: shouldContinue flag (true = keep generating, false = EOS)
    bool sampleFrame(MossTtsNanoSlot& slot, void const* globalHidden, std::vector<int32_t>& frameTokenIdsHost);

    // Codec (Chunk 4): reset stateful streaming + decode 1..maxFrames worth of RVQ frame rows.
    // pcmOut receives 48kHz stereo interleaved FP32 samples appended to whatever was already there.
    void resetCodec(MossTtsNanoSlot& slot);
    size_t decodeFrames(MossTtsNanoSlot& slot,
        std::vector<std::vector<int32_t>> const& frameRows, std::vector<float>& pcmOut);

    // Chunk 5: end-to-end orchestrator. Drives prefill → loop {decodeStep,
    // sampleFrame, decodeFrames} until shouldContinue=false or maxNewFrames hit.
    // pcmOut accumulates 48kHz stereo interleaved FP32 samples.
    struct GenerateConfig
    {
        int32_t maxNewFrames{1000};
        int32_t codecChunkFrames{4};
        std::function<void()> onFirstChunk{};
    };

    void generate(MossTtsNanoSlot& slot, std::vector<int32_t> const& inputIds,
        std::vector<int32_t> const& attentionMask, GenerateConfig const& cfg, std::vector<float>& pcmOut);

    int32_t numVq() const { return mNumVq; }
    int32_t codecSampleRate() const { return mCodecSampleRate; }
    int32_t codecChannels() const { return mCodecChannels; }
    int32_t codecDownsampleRate() const { return mCodecDownsampleRate; }
    int32_t audioCodebookSize() const { return mAudioCodebookSize; }

private:
    struct TrtDeleter
    {
        template <typename T>
        void operator()(T* ptr) const { delete ptr; }
    };

    using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;
    using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;

    static constexpr int32_t kGlobalLayers = 12;
    static constexpr int32_t kNumHeads = 12;
    static constexpr int32_t kHeadDim = 64;
    static constexpr int32_t kDefaultHiddenSize = kNumHeads * kHeadDim;

    void loadEngines(std::string const& engineDir);
    void readMetadata(std::string const& engineDir);
    void allocateSlot(MossTtsNanoSlot& slot);
    void destroySlot(MossTtsNanoSlot& slot) noexcept;
    nvinfer1::IExecutionContext* createContext(nvinfer1::ICudaEngine& engine, char const* label);

    void* layerKvPtr(MossTtsNanoSlot const& slot, int32_t layer, int32_t kvIdx) const;
    void* presentKPtr(MossTtsNanoSlot const& slot, int32_t layer, int32_t seqLen) const;
    void* presentVPtr(MossTtsNanoSlot const& slot, int32_t layer, int32_t seqLen) const;
    bool hasTensor(nvinfer1::ICudaEngine const& engine, char const* name) const;
    std::string findOutputTensor(nvinfer1::ICudaEngine const& engine) const;
    int32_t inferHiddenSize(nvinfer1::ICudaEngine const& engine, std::string const& outputName) const;
    size_t tensorElementSize(nvinfer1::DataType dtype) const;
    void setGlobalBindings(nvinfer1::IExecutionContext& ctx, nvinfer1::ICudaEngine const& engine,
        MossTtsNanoSlot& slot, int32_t seqLen, int32_t pastLen, bool isPrefill, void const* prevHidden);
    void setLocalFixedSampledFrameBindings(nvinfer1::IExecutionContext& ctx,
        nvinfer1::ICudaEngine const& engine, MossTtsNanoSlot& slot, void const* globalHidden);
    void loadCodecEngine(std::string const& engineDir);
    void allocateCodecState(MossTtsNanoSlot& slot);
    void destroyCodecState(MossTtsNanoSlot& slot) noexcept;
    void setCodecBindings(MossTtsNanoSlot& slot, int32_t frameCount);
    void checkCuda(cudaError_t status, char const* what) const;
    void checkTrt(bool status, std::string const& what) const;

    RuntimePtr mRuntime;
    EnginePtr mPrefillEngine;
    EnginePtr mDecodeEngine;
    EnginePtr mLocalDecoderEngine;
    EnginePtr mLocalCachedStepEngine;
    EnginePtr mLocalFixedSampledFrameEngine;
    EnginePtr mCodecDecodeStepEngine;

    int32_t mMaxSlots{0};
    int32_t mMaxSeqLen{0};
    int32_t mLocalLayers{0};
    int32_t mNumVq{16};                  // RVQ codebook count; read from meta when available.
    int32_t mAudioCodebookSize{2048};    // Per-codebook vocabulary; read from meta when available.
    int32_t mHiddenSize{kDefaultHiddenSize};
    nvinfer1::DataType mGlobalHiddenDtype{nvinfer1::DataType::kHALF};
    std::string mGlobalHiddenOutputName;

    int32_t mCodecSampleRate{48000};
    int32_t mCodecChannels{2};
    int32_t mCodecDownsampleRate{3840};
    int32_t mCodecNumQuantizers{16};
    std::array<size_t, 12> mCodecCacheElements{};
    std::array<size_t, 12> mCodecPositionsElements{};

    size_t mGlobalLayerKvBytes{0};
    size_t mGlobalKvBytes{0};
    size_t mLocalLayerKvBytes{0};
    size_t mLocalKvBytes{0};
    size_t mMaxPresentLayerBytes{0};
    size_t mMaxHiddenBytes{0};
    // KV element size queried from decode engine past_key_0 dtype at load time.
    // v16 rebuild emits KV as FP32 (sizeof=4); older FP16 engines need sizeof=2.
    // Was hardcoded sizeof(half)=2; buffer half-sized → KV corruption.
    size_t mKvElementSize{2};

    std::mutex mPoolMutex;
    std::condition_variable mPoolCv;
    std::vector<std::unique_ptr<MossTtsNanoSlot>> mSlots;
    std::vector<int32_t> mFreeSlots;

    std::mutex mActiveMutex;
    std::unordered_map<std::thread::id, int32_t> mActiveSlots;
};

} // namespace rt
} // namespace trt_edgellm
