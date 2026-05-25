/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mossTtsNanoRuntime.h"

#include "common/logger.h"
#include "kernels/kvCacheUtilKernels/mossLinearKvKernels.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

namespace
{
using Json = nlohmann::json;

std::vector<char> readBinary(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open TensorRT engine: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

template <typename T>
void cudaFreeIf(T*& ptr) noexcept
{
    if (ptr != nullptr) { cudaFree(ptr); ptr = nullptr; }
}

// ---- Diagnostic helpers (Phase B diag binary only) ------------------------
// Activated by setting env MOSS_DIAG=1. RNG override via MOSS_RNG_SEQ_FILE.

bool diagEnabled()
{
    static bool const enabled = []() {
        char const* v = std::getenv("MOSS_DIAG");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::vector<float>& diagRngBuf()
{
    static std::vector<float> buf = []() {
        std::vector<float> b;
        char const* path = std::getenv("MOSS_RNG_SEQ_FILE");
        if (path == nullptr || path[0] == '\0') return b;
        std::ifstream f(path, std::ios::binary);
        if (!f) return b;
        f.seekg(0, std::ios::end);
        size_t bytes = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        b.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(b.data()), bytes);
        std::fprintf(stderr, "[moss_diag] loaded RNG seq from %s: %zu floats\n", path, b.size());
        return b;
    }();
    return buf;
}

size_t& diagRngCursor()
{
    static thread_local size_t c = 0;
    return c;
}

void dumpBufToFile(void const* devicePtr, size_t bytes, char const* path, cudaStream_t stream)
{
    std::vector<char> host(bytes);
    cudaMemcpyAsync(host.data(), devicePtr, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::ofstream f(path, std::ios::binary);
    f.write(host.data(), static_cast<std::streamsize>(bytes));
}

void dumpHalfFirst8(void const* devicePtr, char const* tag, cudaStream_t stream)
{
    std::vector<__half> tmp(8);
    cudaMemcpyAsync(tmp.data(), devicePtr, 8 * sizeof(__half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::fprintf(stderr, "[moss_diag] %s first8:", tag);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %.6f", __half2float(tmp[i]));
    std::fprintf(stderr, "\n");
}

void dumpFloatFirst8(void const* devicePtr, char const* tag, cudaStream_t stream)
{
    std::vector<float> tmp(8);
    cudaMemcpyAsync(tmp.data(), devicePtr, 8 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::fprintf(stderr, "[moss_diag] %s first8:", tag);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %.6f", tmp[i]);
    std::fprintf(stderr, "\n");
}

void dumpInt32First8(void const* devicePtr, char const* tag, cudaStream_t stream)
{
    std::vector<int32_t> tmp(8);
    cudaMemcpyAsync(tmp.data(), devicePtr, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::fprintf(stderr, "[moss_diag] %s first8:", tag);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %d", tmp[i]);
    std::fprintf(stderr, "\n");
}
} // namespace

// ---- RequestGuard ---------------------------------------------------------

MossTtsNanoRuntime::RequestGuard::RequestGuard(MossTtsNanoRuntime* runtime, int32_t slotId)
    : mRuntime(runtime), mSlotId(slotId) {}

MossTtsNanoRuntime::RequestGuard::RequestGuard(RequestGuard&& other) noexcept
    : mRuntime(other.mRuntime), mSlotId(other.mSlotId)
{
    other.mRuntime = nullptr;
    other.mSlotId = -1;
}

MossTtsNanoRuntime::RequestGuard& MossTtsNanoRuntime::RequestGuard::operator=(RequestGuard&& other) noexcept
{
    if (this != &other)
    {
        reset();
        mRuntime = other.mRuntime;
        mSlotId = other.mSlotId;
        other.mRuntime = nullptr;
        other.mSlotId = -1;
    }
    return *this;
}

MossTtsNanoRuntime::RequestGuard::~RequestGuard() { reset(); }

void MossTtsNanoRuntime::RequestGuard::reset()
{
    if (mRuntime != nullptr)
    {
        mRuntime->endRequest();
        mRuntime = nullptr;
        mSlotId = -1;
    }
}

MossTtsNanoSlot& MossTtsNanoRuntime::RequestGuard::slot() const
{
    if (mRuntime == nullptr || mSlotId < 0)
        throw std::runtime_error("MossTtsNanoRuntime::RequestGuard has no active slot");
    return mRuntime->slot(mSlotId);
}

// ---- Construction --------------------------------------------------------

MossTtsNanoRuntime::MossTtsNanoRuntime(std::string const& engineDir, int32_t maxSlots, int32_t maxSeqLen)
    : mMaxSlots(maxSlots), mMaxSeqLen(maxSeqLen)
{
    if (mMaxSlots <= 0) throw std::runtime_error("MossTtsNanoRuntime maxSlots must be positive");
    if (mMaxSeqLen <= 0) throw std::runtime_error("MossTtsNanoRuntime maxSeqLen must be positive");

    readMetadata(engineDir);
    loadEngines(engineDir);
    loadCodecEngine(engineDir);

    mGlobalHiddenOutputName = findOutputTensor(*mPrefillEngine);
    mHiddenSize = inferHiddenSize(*mPrefillEngine, mGlobalHiddenOutputName);
    mGlobalHiddenDtype = mPrefillEngine->getTensorDataType(mGlobalHiddenOutputName.c_str());

    mGlobalLayerKvBytes = static_cast<size_t>(2) * mMaxSeqLen * kNumHeads * kHeadDim * mKvElementSize;
    mGlobalKvBytes = static_cast<size_t>(kGlobalLayers) * mGlobalLayerKvBytes;
    mLocalLayerKvBytes = static_cast<size_t>(2) * mMaxSeqLen * kNumHeads * kHeadDim * mKvElementSize;
    mLocalKvBytes = static_cast<size_t>(std::max(0, mLocalLayers)) * mLocalLayerKvBytes;
    mMaxPresentLayerBytes = static_cast<size_t>(mMaxSeqLen) * kNumHeads * kHeadDim * mKvElementSize;
    mMaxHiddenBytes = static_cast<size_t>(mMaxSeqLen) * mHiddenSize * tensorElementSize(mGlobalHiddenDtype);

    mSlots.reserve(static_cast<size_t>(mMaxSlots));
    mFreeSlots.reserve(static_cast<size_t>(mMaxSlots));
    for (int32_t i = 0; i < mMaxSlots; ++i)
    {
        auto s = std::make_unique<MossTtsNanoSlot>();
        s->slotId = i;
        allocateSlot(*s);
        allocateCodecState(*s);
        mFreeSlots.push_back(i);
        mSlots.push_back(std::move(s));
    }
}

MossTtsNanoRuntime::~MossTtsNanoRuntime()
{
    for (auto& s : mSlots)
        if (s) destroySlot(*s);
}

void MossTtsNanoRuntime::readMetadata(std::string const& engineDir)
{
    std::filesystem::path const metaPath = std::filesystem::path(engineDir) / "tts_browser_onnx_meta.json";
    if (!std::filesystem::exists(metaPath)) { mLocalLayers = 0; return; }
    std::ifstream file(metaPath);
    if (!file) throw std::runtime_error("Failed to open MOSS metadata: " + metaPath.string());
    Json meta = Json::parse(file);
    // MOSS exporter (export_moss_tts_browser_onnx.py line ~1340) writes model
    // dimensions under "model_config" — n_vq, audio_codebook_sizes (list of 16),
    // audio_pad_token_id, row_width, local_layers, etc.
    if (meta.contains("model_config"))
    {
        auto const& mc = meta["model_config"];
        if (mc.contains("local_layers")) mLocalLayers = mc["local_layers"].get<int32_t>();
        if (mc.contains("n_vq")) mNumVq = mc["n_vq"].get<int32_t>();
        if (mc.contains("audio_codebook_sizes"))
        {
            auto const& sizes = mc["audio_codebook_sizes"];
            if (sizes.is_array() && !sizes.empty())
                mAudioCodebookSize = sizes[0].get<int32_t>();
        }
    }
    else
    {
        mLocalLayers = meta.value("local_layers", 0);
    }
}

void MossTtsNanoRuntime::loadEngines(std::string const& engineDir)
{
    std::filesystem::path const dir(engineDir);
    mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime) throw std::runtime_error("Failed to create TensorRT runtime for MOSS-TTS-Nano");

    auto load = [&](char const* fileName) -> EnginePtr
    {
        std::filesystem::path const path = dir / fileName;
        std::vector<char> blob = readBinary(path);
        EnginePtr engine(mRuntime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine) throw std::runtime_error(std::string("Failed to deserialize MOSS engine: ") + path.string());
        return engine;
    };

    mPrefillEngine = load("moss_tts_prefill.plan");
    mDecodeEngine = load("moss_tts_decode_step.plan");
    mLocalDecoderEngine = load("moss_tts_local_decoder.plan");
    mLocalCachedStepEngine = load("moss_tts_local_cached_step.plan");
    mLocalFixedSampledFrameEngine = load("moss_tts_local_fixed_sampled_frame.plan");

    // Prefill engine consumes no past KV (turn 0 has nothing to attend to);
    // it only produces present_* outputs. Decode engine consumes both.
    if (!hasTensor(*mPrefillEngine, "present_key_0"))
        throw std::runtime_error("MOSS prefill engine must expose present_key_0");
    if (!hasTensor(*mDecodeEngine, "past_key_0") || !hasTensor(*mDecodeEngine, "present_key_0"))
        throw std::runtime_error("MOSS decode engine must expose past_key_0 and present_key_0");

    // Query KV element size from decode engine. v16 rebuild emits FP32 KV;
    // earlier rebuilds emitted FP16. Hardcoding sizeof(half) breaks FP32 path
    // (buffer half-sized → KV corruption across layers).
    nvinfer1::DataType const kvDtype = mDecodeEngine->getTensorDataType("past_key_0");
    mKvElementSize = tensorElementSize(kvDtype);
    std::fprintf(stderr,
        "[moss] KV element dtype=%d size=%zu bytes (FP32=%d FP16=%d)\n",
        static_cast<int>(kvDtype), mKvElementSize,
        static_cast<int>(nvinfer1::DataType::kFLOAT),
        static_cast<int>(nvinfer1::DataType::kHALF));
}

void MossTtsNanoRuntime::allocateSlot(MossTtsNanoSlot& slot)
{
    checkCuda(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking), "create MOSS slot stream");
    checkCuda(cudaMalloc(&slot.globalKvDevice, mGlobalKvBytes), "allocate MOSS global KV");
    checkCuda(cudaMemsetAsync(slot.globalKvDevice, 0, mGlobalKvBytes, slot.stream), "clear MOSS global KV");
    if (mLocalKvBytes > 0)
    {
        checkCuda(cudaMalloc(&slot.localKvDevice, mLocalKvBytes), "allocate MOSS local KV");
        checkCuda(cudaMemsetAsync(slot.localKvDevice, 0, mLocalKvBytes, slot.stream), "clear MOSS local KV");
    }
    checkCuda(cudaMalloc(&slot.presentScratchK,
                 static_cast<size_t>(kGlobalLayers) * mMaxPresentLayerBytes),
        "allocate MOSS present K scratch");
    checkCuda(cudaMalloc(&slot.presentScratchV,
                 static_cast<size_t>(kGlobalLayers) * mMaxPresentLayerBytes),
        "allocate MOSS present V scratch");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&slot.pastValidLengthsDevice), sizeof(int32_t)),
        "allocate MOSS past_valid_lengths");
    // input_ids carries row_width = n_vq + 1 = 17 int32 columns per sequence token.
    constexpr int32_t kRowWidth = 17;
    checkCuda(cudaMalloc(&slot.inputIdsDevice,
                  static_cast<size_t>(mMaxSeqLen) * static_cast<size_t>(kRowWidth) * sizeof(int32_t)),
        "allocate MOSS input_ids");
    checkCuda(cudaMalloc(&slot.attentionMaskDevice, static_cast<size_t>(mMaxSeqLen) * sizeof(int32_t)),
        "allocate MOSS attention_mask");
    checkCuda(cudaMalloc(&slot.globalHiddenDevice, mMaxHiddenBytes), "allocate MOSS global hidden");
    checkCuda(cudaMalloc(&slot.prevHiddenDevice,
                 static_cast<size_t>(mHiddenSize) * tensorElementSize(mGlobalHiddenDtype)),
        "allocate MOSS prev hidden");

    // Local LLM (Chunk 3) staging.
    size_t const repMaskBytes
        = static_cast<size_t>(mNumVq) * static_cast<size_t>(mAudioCodebookSize) * sizeof(int32_t);
    checkCuda(cudaMalloc(&slot.repetitionSeenMaskDevice, repMaskBytes), "allocate MOSS repetition_seen_mask");
    checkCuda(cudaMemsetAsync(slot.repetitionSeenMaskDevice, 0, repMaskBytes, slot.stream),
        "clear MOSS repetition_seen_mask");
    checkCuda(cudaMalloc(&slot.assistantRandomUDevice, sizeof(float)), "allocate MOSS assistant_random_u");
    checkCuda(cudaMalloc(&slot.audioRandomUDevice, static_cast<size_t>(mNumVq) * sizeof(float)),
        "allocate MOSS audio_random_u");
    checkCuda(cudaMalloc(&slot.shouldContinueDevice, sizeof(int32_t)), "allocate MOSS should_continue");
    checkCuda(cudaMalloc(&slot.frameTokenIdsDevice, static_cast<size_t>(mNumVq) * sizeof(int32_t)),
        "allocate MOSS frame_token_ids");

    slot.prefillCtx = createContext(*mPrefillEngine, "prefill");
    slot.decodeCtx = createContext(*mDecodeEngine, "decode_step");
    slot.localDecoderCtx = createContext(*mLocalDecoderEngine, "local_decoder");
    slot.localCachedStepCtx = createContext(*mLocalCachedStepEngine, "local_cached_step");
    slot.localFixedSampledFrameCtx = createContext(*mLocalFixedSampledFrameEngine, "local_fixed_sampled_frame");

    checkCuda(cudaStreamSynchronize(slot.stream), "synchronize MOSS slot allocation");
}

void MossTtsNanoRuntime::destroySlot(MossTtsNanoSlot& slot) noexcept
{
    destroyCodecState(slot);
    delete slot.prefillCtx;                slot.prefillCtx = nullptr;
    delete slot.decodeCtx;                 slot.decodeCtx = nullptr;
    delete slot.localDecoderCtx;           slot.localDecoderCtx = nullptr;
    delete slot.localCachedStepCtx;        slot.localCachedStepCtx = nullptr;
    delete slot.localFixedSampledFrameCtx; slot.localFixedSampledFrameCtx = nullptr;

    cudaFreeIf(slot.globalKvDevice);
    cudaFreeIf(slot.localKvDevice);
    cudaFreeIf(slot.presentScratchK);
    cudaFreeIf(slot.presentScratchV);
    cudaFreeIf(slot.pastValidLengthsDevice);
    cudaFreeIf(slot.inputIdsDevice);
    cudaFreeIf(slot.attentionMaskDevice);
    cudaFreeIf(slot.globalHiddenDevice);
    cudaFreeIf(slot.prevHiddenDevice);
    cudaFreeIf(slot.repetitionSeenMaskDevice);
    cudaFreeIf(slot.assistantRandomUDevice);
    cudaFreeIf(slot.audioRandomUDevice);
    cudaFreeIf(slot.shouldContinueDevice);
    cudaFreeIf(slot.frameTokenIdsDevice);
    if (slot.stream != nullptr) { cudaStreamDestroy(slot.stream); slot.stream = nullptr; }
}

nvinfer1::IExecutionContext* MossTtsNanoRuntime::createContext(nvinfer1::ICudaEngine& engine, char const* label)
{
    nvinfer1::IExecutionContext* ctx = engine.createExecutionContext();
    if (ctx == nullptr)
        throw std::runtime_error(std::string("Failed to create MOSS-TTS-Nano ") + label + " execution context");
    return ctx;
}

// ---- Pool management -----------------------------------------------------

int32_t MossTtsNanoRuntime::acquirePoolSlot()
{
    std::unique_lock<std::mutex> lock(mPoolMutex);
    mPoolCv.wait(lock, [this] { return !mFreeSlots.empty(); });
    int32_t const slotId = mFreeSlots.back();
    mFreeSlots.pop_back();
    MossTtsNanoSlot& s = *mSlots.at(static_cast<size_t>(slotId));
    s.inUse = true; s.pastLen = 0; s.cumulativePastLen = 0; s.localPastLen = 0;
    // Reset per-slot RNG so every request starts from the same deterministic
    // PRNG state regardless of which slot it lands on or how many prior
    // requests the slot handled — N=2 parity hard gate.
    s.rng.seed(42);
    checkCuda(cudaMemsetAsync(s.globalKvDevice, 0, mGlobalKvBytes, s.stream), "clear MOSS global KV on acquire");
    if (s.localKvDevice != nullptr && mLocalKvBytes > 0)
        checkCuda(cudaMemsetAsync(s.localKvDevice, 0, mLocalKvBytes, s.stream), "clear MOSS local KV on acquire");
    size_t const repMaskBytes
        = static_cast<size_t>(mNumVq) * static_cast<size_t>(mAudioCodebookSize) * sizeof(int32_t);
    checkCuda(cudaMemsetAsync(s.repetitionSeenMaskDevice, 0, repMaskBytes, s.stream),
        "clear MOSS repetition_seen_mask on acquire");
    return slotId;
}

void MossTtsNanoRuntime::releasePoolSlot(int32_t slotId)
{
    if (slotId < 0 || slotId >= static_cast<int32_t>(mSlots.size()))
        throw std::runtime_error("MossTtsNanoRuntime::releasePoolSlot invalid slotId");
    {
        std::lock_guard<std::mutex> lock(mPoolMutex);
        auto& s = *mSlots.at(static_cast<size_t>(slotId));
        s.inUse = false; s.pastLen = 0; s.cumulativePastLen = 0; s.localPastLen = 0;
        mFreeSlots.push_back(slotId);
    }
    mPoolCv.notify_one();
}

MossTtsNanoRuntime::RequestGuard MossTtsNanoRuntime::beginRequest()
{
    int32_t const slotId = acquirePoolSlot();
    {
        std::lock_guard<std::mutex> lock(mActiveMutex);
        mActiveSlots[std::this_thread::get_id()] = slotId;
    }
    return RequestGuard(this, slotId);
}

void MossTtsNanoRuntime::endRequest()
{
    int32_t slotId = -1;
    {
        std::lock_guard<std::mutex> lock(mActiveMutex);
        auto it = mActiveSlots.find(std::this_thread::get_id());
        if (it == mActiveSlots.end()) return;
        slotId = it->second;
        mActiveSlots.erase(it);
    }
    releasePoolSlot(slotId);
}

MossTtsNanoSlot& MossTtsNanoRuntime::slot(int32_t slotId)
{
    if (slotId < 0 || slotId >= static_cast<int32_t>(mSlots.size()))
        throw std::runtime_error("MossTtsNanoRuntime::slot invalid slotId");
    return *mSlots.at(static_cast<size_t>(slotId));
}

MossTtsNanoSlot const& MossTtsNanoRuntime::slot(int32_t slotId) const
{
    if (slotId < 0 || slotId >= static_cast<int32_t>(mSlots.size()))
        throw std::runtime_error("MossTtsNanoRuntime::slot invalid slotId");
    return *mSlots.at(static_cast<size_t>(slotId));
}

// ---- prefill / decodeStep -------------------------------------------------

void* MossTtsNanoRuntime::prefill(
    MossTtsNanoSlot& slot, std::vector<int32_t> const& inputIds, std::vector<int32_t> const& attentionMask)
{
    constexpr int32_t kRowWidth = 17;
    if (inputIds.empty() || inputIds.size() % static_cast<size_t>(kRowWidth) != 0)
        throw std::runtime_error("MOSS prefill inputIds size must be multiple of row_width=17");
    int32_t const seqLen = static_cast<int32_t>(inputIds.size() / static_cast<size_t>(kRowWidth));
    if (seqLen <= 0 || seqLen > mMaxSeqLen)
        throw std::runtime_error("MOSS prefill seqLen out of range");
    if (attentionMask.size() != static_cast<size_t>(seqLen))
        throw std::runtime_error("MOSS prefill attentionMask size must equal seqLen");
    if (slot.pastLen != 0)
        throw std::runtime_error("MOSS prefill requires an empty global KV cache");

    checkCuda(cudaMemcpyAsync(slot.inputIdsDevice, inputIds.data(),
                  inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS prefill input_ids");
    checkCuda(cudaMemcpyAsync(slot.attentionMaskDevice, attentionMask.data(),
                  attentionMask.size() * sizeof(int32_t), cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS prefill attention_mask");

    int32_t const pastLen = 0;
    int32_t const pastValidLength = 0;
    checkCuda(cudaMemcpyAsync(slot.pastValidLengthsDevice, &pastValidLength, sizeof(int32_t),
                  cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS prefill past_valid_lengths");

    setGlobalBindings(*slot.prefillCtx, *mPrefillEngine, slot, seqLen, pastLen, /*isPrefill=*/true, nullptr);
    if (!slot.prefillCtx->enqueueV3(slot.stream))
        throw std::runtime_error("MOSS prefill enqueueV3 failed");

    // Read engine's actual present_key_0 dim 1 at runtime — MOSS engine produces
    // a non-trivial seq length that doesn't match either pure delta or pure
    // full-sequence convention. Trust the engine, mirror Python past_by_name=present.
    nvinfer1::Dims const presentShape = slot.prefillCtx->getTensorShape("present_key_0");
    int32_t const presentSeqLen = (presentShape.nbDims > 1) ? static_cast<int32_t>(presentShape.d[1]) : seqLen;

    size_t const kvLayerBytes
        = static_cast<size_t>(presentSeqLen) * kNumHeads * kHeadDim * mKvElementSize;
    for (int32_t layer = 0; layer < kGlobalLayers; ++layer)
    {
        checkCuda(cudaMemcpyAsync(layerKvPtr(slot, layer, /*kvIdx=*/0),
                      presentKPtr(slot, layer, presentSeqLen), kvLayerBytes,
                      cudaMemcpyDeviceToDevice, slot.stream),
            "MOSS prefill copy global K from present");
        checkCuda(cudaMemcpyAsync(layerKvPtr(slot, layer, /*kvIdx=*/1),
                      presentVPtr(slot, layer, presentSeqLen), kvLayerBytes,
                      cudaMemcpyDeviceToDevice, slot.stream),
            "MOSS prefill copy global V from present");
    }
    slot.pastLen = presentSeqLen;       // buffer fill = engine-reported
    slot.cumulativePastLen = seqLen;    // matches Python past_valid_length after prefill
    // Return pointer to the LAST token's hidden state, not the buffer base.
    // Prefill engine outputs full sequence [1, S, H]; downstream local_fixed_sampled_frame
    // expects [1, H]. Matches Python ort_cpu_runtime _extract_last_hidden(global_hidden).
    size_t const lastTokenOffsetBytes
        = static_cast<size_t>(seqLen - 1) * static_cast<size_t>(mHiddenSize)
            * tensorElementSize(mGlobalHiddenDtype);
    return static_cast<unsigned char*>(slot.globalHiddenDevice) + lastTokenOffsetBytes;
}

void* MossTtsNanoRuntime::decodeStep(
    MossTtsNanoSlot& slot, std::vector<int32_t> const& inputIds, void const* prevHidden)
{
    static thread_local int diagDecodeFrame = 0;
    constexpr int32_t kRowWidth = 17;
    if (inputIds.size() != static_cast<size_t>(kRowWidth))
        throw std::runtime_error("MOSS decodeStep requires exactly row_width=17 int32s for one token");
    if (slot.pastLen <= 0 || slot.pastLen >= mMaxSeqLen)
        throw std::runtime_error("MOSS decodeStep pastLen out of range");

    if (diagEnabled() && diagDecodeFrame <= 1)
    {
        std::fprintf(stderr, "[moss_diag] decode#%d ENTER pastLen=%d cumPastLen=%d\n",
            diagDecodeFrame, slot.pastLen, slot.cumulativePastLen);
        std::fprintf(stderr, "[moss_diag] decode#%d input_ids:", diagDecodeFrame);
        for (auto v : inputIds) std::fprintf(stderr, " %d", v);
        std::fprintf(stderr, "\n");
        // past_key_0 first 8 (half) — slot.globalKvDevice base (layer 0 K)
        dumpHalfFirst8(layerKvPtr(slot, 0, 0), "decode_in past_key_0", slot.stream);
        // prev_hidden first 8 floats
        if (prevHidden != nullptr)
        {
            if (mGlobalHiddenDtype == nvinfer1::DataType::kHALF)
                dumpHalfFirst8(prevHidden, "decode_in prev_hidden(half)", slot.stream);
            else
                dumpFloatFirst8(prevHidden, "decode_in prev_hidden(fp32)", slot.stream);
        }
        // dump past_key_0 layer-0 K full buffer (max_seq_len * heads * dim * kv_elem)
        size_t const kvLayerKBytes = static_cast<size_t>(mMaxSeqLen) * kNumHeads * kHeadDim * mKvElementSize;
        char path[256];
        std::snprintf(path, sizeof(path), "/tmp/cpp_f%d_past_key_0_full.bin", diagDecodeFrame);
        dumpBufToFile(layerKvPtr(slot, 0, 0), kvLayerKBytes, path, slot.stream);
    }

    int32_t const seqLen = 1;
    // attention_mask length = cumulative + 1 (matches setGlobalBindings attentionLen calc).
    int32_t const attnLen = slot.cumulativePastLen + 1;
    std::vector<int32_t> attentionMask(static_cast<size_t>(attnLen), 1);

    checkCuda(cudaMemcpyAsync(slot.inputIdsDevice, inputIds.data(),
                  inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS decode input_ids");
    checkCuda(cudaMemcpyAsync(slot.attentionMaskDevice, attentionMask.data(),
                  attentionMask.size() * sizeof(int32_t), cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS decode attention_mask");

    // v7 (ground truth from Python trace):
    //   past_valid_lengths = cumulative cursor (monotonic +1 per step, init = prefill_seq_len)
    //   past_key shape = engine-reported (remaining capacity, decreases each step)
    //   sum(past_valid_lengths + past_key.dim1) = constant capacity
    //   Python does NOT feed attention_mask in decode (we still bind but treat as no-op)
    int32_t const pastLen = slot.pastLen;          // engine-reported, shrinking — used for past_key shape
    int32_t const pastValidLength = slot.cumulativePastLen;  // monotonic cursor
    checkCuda(cudaMemcpyAsync(slot.pastValidLengthsDevice, &pastValidLength, sizeof(int32_t),
                  cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS decode past_valid_lengths");

    setGlobalBindings(*slot.decodeCtx, *mDecodeEngine, slot, seqLen, pastLen, /*isPrefill=*/false, prevHidden);
    if (!slot.decodeCtx->enqueueV3(slot.stream))
        throw std::runtime_error("MOSS decodeStep enqueueV3 failed");

    if (diagEnabled() && diagDecodeFrame <= 1)
    {
        cudaStreamSynchronize(slot.stream);
        nvinfer1::Dims const ps = slot.decodeCtx->getTensorShape("present_key_0");
        int32_t const psLen = (ps.nbDims > 1) ? static_cast<int32_t>(ps.d[1]) : 1;
        std::fprintf(stderr, "[moss_diag] decode#%d engine_out present_seq_len=%d\n",
            diagDecodeFrame, psLen);
        dumpHalfFirst8(presentKPtr(slot, 0, psLen), "decode_out present_key_0", slot.stream);
        // global_hidden output first 8
        if (mGlobalHiddenDtype == nvinfer1::DataType::kHALF)
            dumpHalfFirst8(slot.globalHiddenDevice, "decode_out global_hidden(half)", slot.stream);
        else
            dumpFloatFirst8(slot.globalHiddenDevice, "decode_out global_hidden(fp32)", slot.stream);
        // dump full layer-0 K present + global_hidden
        size_t const presentKBytes = static_cast<size_t>(psLen) * kNumHeads * kHeadDim * mKvElementSize;
        char p[256];
        std::snprintf(p, sizeof(p), "/tmp/cpp_f%d_present_key_0.bin", diagDecodeFrame);
        dumpBufToFile(presentKPtr(slot, 0, psLen), presentKBytes, p, slot.stream);
        size_t const hiddenBytes = static_cast<size_t>(mHiddenSize) *
            tensorElementSize(mGlobalHiddenDtype);
        std::snprintf(p, sizeof(p), "/tmp/cpp_f%d_global_hidden.bin", diagDecodeFrame);
        dumpBufToFile(slot.globalHiddenDevice, hiddenBytes, p, slot.stream);
    }
    diagDecodeFrame++;

    // CLEANED BASELINE (matches production binary 7be68fe0): loose check +
    // full cudaMemcpy device->device replace, mirrors prefill path.
    // Decode engine (unpatched ONNX) emits full present_key_0 of shape
    // [1, presentSeqLen, 12, 64]; replace entire KV cache base.
    nvinfer1::Dims const presentShape = slot.decodeCtx->getTensorShape("present_key_0");
    int32_t const presentSeqLen
        = (presentShape.nbDims > 1) ? static_cast<int32_t>(presentShape.d[1]) : 1;
    if (presentSeqLen <= 0 || presentSeqLen > mMaxSeqLen)
        throw std::runtime_error("MOSS decode present_seq_len out of range");

    size_t const decodeLayerBytes
        = static_cast<size_t>(presentSeqLen) * kNumHeads * kHeadDim * mKvElementSize;
    for (int32_t layer = 0; layer < kGlobalLayers; ++layer)
    {
        checkCuda(cudaMemcpyAsync(layerKvPtr(slot, layer, /*kvIdx=*/0),
                      presentKPtr(slot, layer, presentSeqLen), decodeLayerBytes,
                      cudaMemcpyDeviceToDevice, slot.stream),
            "MOSS decode copy global K from present");
        checkCuda(cudaMemcpyAsync(layerKvPtr(slot, layer, /*kvIdx=*/1),
                      presentVPtr(slot, layer, presentSeqLen), decodeLayerBytes,
                      cudaMemcpyDeviceToDevice, slot.stream),
            "MOSS decode copy global V from present");
    }
    slot.pastLen = presentSeqLen;       // engine-reported, matches prefill convention
    slot.cumulativePastLen += 1;        // next call's past_valid_lengths
    return slot.globalHiddenDevice;
}

void MossTtsNanoRuntime::setGlobalBindings(nvinfer1::IExecutionContext& ctx, nvinfer1::ICudaEngine const& engine,
    MossTtsNanoSlot& slot, int32_t seqLen, int32_t pastLen, bool isPrefill, void const* prevHidden)
{
    constexpr int32_t kRowWidth = 17; // MOSS row_width = n_vq + 1 = 17
    // Python infer feeds attention_mask ONLY for prefill, not decode (validated via Python ORT trace).
    // For decode, past_valid_lengths drives the attention mask semantics. Skip binding it.
    int32_t const attentionLen = isPrefill ? seqLen : (slot.cumulativePastLen + 1);

    if (hasTensor(engine, "input_ids"))
    {
        // input_ids is 3D: [batch=1, seq, row_width=17]
        checkTrt(ctx.setInputShape("input_ids", nvinfer1::Dims3{1, seqLen, kRowWidth}),
            "set input_ids shape");
        checkTrt(ctx.setTensorAddress("input_ids", slot.inputIdsDevice), "set input_ids address");
    }
    if (hasTensor(engine, "attention_mask"))
    {
        // For decode, use cumulative+1 (full attention over all valid past + current).
        // attentionLen was set above based on isPrefill / cumulative.
        checkTrt(ctx.setInputShape("attention_mask", nvinfer1::Dims2{1, attentionLen}), "set attention_mask shape");
        checkTrt(ctx.setTensorAddress("attention_mask", slot.attentionMaskDevice), "set attention_mask address");
    }
    if (hasTensor(engine, "past_valid_lengths"))
    {
        checkTrt(ctx.setInputShape("past_valid_lengths", nvinfer1::Dims{1, {1}}), "set past_valid_lengths shape");
        checkTrt(ctx.setTensorAddress("past_valid_lengths", slot.pastValidLengthsDevice),
            "set past_valid_lengths address");
    }
    if (!isPrefill && prevHidden != nullptr && hasTensor(engine, "prev_hidden"))
    {
        checkTrt(ctx.setInputShape("prev_hidden", nvinfer1::Dims3{1, 1, mHiddenSize}), "set prev_hidden shape");
        checkTrt(ctx.setTensorAddress("prev_hidden", const_cast<void*>(prevHidden)), "set prev_hidden address");
    }

    nvinfer1::Dims4 const pastShape{1, pastLen, kNumHeads, kHeadDim};
    for (int32_t layer = 0; layer < kGlobalLayers; ++layer)
    {
        std::string const pk = "past_key_" + std::to_string(layer);
        std::string const pv = "past_value_" + std::to_string(layer);
        std::string const nk = "present_key_" + std::to_string(layer);
        std::string const nv = "present_value_" + std::to_string(layer);

        // Prefill engine has no past_* inputs; decode engine has both.
        if (hasTensor(engine, pk.c_str()))
        {
            checkTrt(ctx.setInputShape(pk.c_str(), pastShape), "set " + pk + " shape");
            checkTrt(ctx.setInputShape(pv.c_str(), pastShape), "set " + pv + " shape");
            checkTrt(ctx.setTensorAddress(pk.c_str(), layerKvPtr(slot, layer, 0)), "set " + pk + " address");
            checkTrt(ctx.setTensorAddress(pv.c_str(), layerKvPtr(slot, layer, 1)), "set " + pv + " address");
        }
        checkTrt(ctx.setTensorAddress(nk.c_str(), presentKPtr(slot, layer, seqLen)), "set " + nk + " address");
        checkTrt(ctx.setTensorAddress(nv.c_str(), presentVPtr(slot, layer, seqLen)), "set " + nv + " address");
    }

    checkTrt(ctx.setTensorAddress(mGlobalHiddenOutputName.c_str(), slot.globalHiddenDevice),
        "set global hidden output address");
}

void* MossTtsNanoRuntime::layerKvPtr(MossTtsNanoSlot const& slot, int32_t layer, int32_t kvIdx) const
{
    auto* base = static_cast<unsigned char*>(slot.globalKvDevice);
    return base + static_cast<size_t>(layer) * mGlobalLayerKvBytes
        + static_cast<size_t>(kvIdx) * mMaxSeqLen * kNumHeads * kHeadDim * mKvElementSize;
}

void* MossTtsNanoRuntime::presentKPtr(MossTtsNanoSlot const& slot, int32_t layer, int32_t /*seqLen*/) const
{
    auto* base = static_cast<unsigned char*>(slot.presentScratchK);
    return base + static_cast<size_t>(layer) * mMaxPresentLayerBytes;
}

void* MossTtsNanoRuntime::presentVPtr(MossTtsNanoSlot const& slot, int32_t layer, int32_t /*seqLen*/) const
{
    auto* base = static_cast<unsigned char*>(slot.presentScratchV);
    return base + static_cast<size_t>(layer) * mMaxPresentLayerBytes;
}

bool MossTtsNanoRuntime::hasTensor(nvinfer1::ICudaEngine const& engine, char const* name) const
{
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
        if (std::string(engine.getIOTensorName(i)) == name) return true;
    return false;
}

std::string MossTtsNanoRuntime::findOutputTensor(nvinfer1::ICudaEngine const& engine) const
{
    for (char const* candidate : {"global_hidden", "last_hidden", "hidden_states", "hidden", "output_hidden_states"})
        if (hasTensor(engine, candidate)) return candidate;

    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
    {
        char const* name = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            std::string const s(name);
            if (s.rfind("present_key_", 0) != 0 && s.rfind("present_value_", 0) != 0) return s;
        }
    }
    throw std::runtime_error("Unable to find MOSS global hidden output tensor");
}

int32_t MossTtsNanoRuntime::inferHiddenSize(nvinfer1::ICudaEngine const& engine, std::string const& outputName) const
{
    nvinfer1::Dims const dims = engine.getTensorShape(outputName.c_str());
    if (dims.nbDims > 0 && dims.d[dims.nbDims - 1] > 0) return dims.d[dims.nbDims - 1];
    return kDefaultHiddenSize;
}

size_t MossTtsNanoRuntime::tensorElementSize(nvinfer1::DataType dtype) const
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kBF16:  return 2;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kFP8:   return 1;
    default: throw std::runtime_error("Unsupported MOSS TensorRT dtype");
    }
}

void MossTtsNanoRuntime::checkCuda(cudaError_t status, char const* what) const
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA failure in ") + what + ": " + cudaGetErrorString(status));
}

void MossTtsNanoRuntime::checkTrt(bool status, std::string const& what) const
{
    if (!status) throw std::runtime_error("TensorRT failure in MOSS runtime: " + what);
}

// ---- Local LLM (Chunk 3) --------------------------------------------------
// local_fixed_sampled_frame is a one-shot sampler that internally runs the
// 16-step RVQ loop and returns frame_token_ids[n_vq] + should_continue.
// See vendor/moss-tts-nano/onnx_tts_runtime.py run_local_fixed_sampled_frame
// and ort_cpu_runtime.py SAMPLE_MODE_FIXED branch.

void MossTtsNanoRuntime::setLocalFixedSampledFrameBindings(nvinfer1::IExecutionContext& ctx,
    nvinfer1::ICudaEngine const& engine, MossTtsNanoSlot& slot, void const* globalHidden)
{
    checkTrt(ctx.setInputShape("global_hidden", nvinfer1::Dims2{1, mHiddenSize}),
        "set local_fixed_sampled_frame global_hidden shape");
    checkTrt(ctx.setTensorAddress("global_hidden", const_cast<void*>(globalHidden)),
        "set local_fixed_sampled_frame global_hidden address");

    if (hasTensor(engine, "repetition_seen_mask"))
    {
        checkTrt(ctx.setInputShape("repetition_seen_mask",
                     nvinfer1::Dims3{1, mNumVq, mAudioCodebookSize}),
            "set repetition_seen_mask shape");
        checkTrt(ctx.setTensorAddress("repetition_seen_mask", slot.repetitionSeenMaskDevice),
            "set repetition_seen_mask address");
    }
    if (hasTensor(engine, "assistant_random_u"))
    {
        checkTrt(ctx.setInputShape("assistant_random_u", nvinfer1::Dims{1, {1}}),
            "set assistant_random_u shape");
        checkTrt(ctx.setTensorAddress("assistant_random_u", slot.assistantRandomUDevice),
            "set assistant_random_u address");
    }
    if (hasTensor(engine, "audio_random_u"))
    {
        checkTrt(ctx.setInputShape("audio_random_u", nvinfer1::Dims2{1, mNumVq}),
            "set audio_random_u shape");
        checkTrt(ctx.setTensorAddress("audio_random_u", slot.audioRandomUDevice),
            "set audio_random_u address");
    }
    checkTrt(ctx.setTensorAddress("should_continue", slot.shouldContinueDevice),
        "set should_continue address");
    checkTrt(ctx.setTensorAddress("frame_token_ids", slot.frameTokenIdsDevice),
        "set frame_token_ids address");
}

bool MossTtsNanoRuntime::sampleFrame(
    MossTtsNanoSlot& slot, void const* globalHidden, std::vector<int32_t>& frameTokenIdsHost)
{
    if (globalHidden == nullptr) throw std::runtime_error("MOSS sampleFrame globalHidden must not be null");
    if (slot.localFixedSampledFrameCtx == nullptr)
        throw std::runtime_error("MOSS sampleFrame requires loaded local_fixed_sampled_frame engine");

    frameTokenIdsHost.assign(static_cast<size_t>(mNumVq), 0);
    static thread_local int diagSampleFrame = 0;

    // Sample fresh random U values for this frame.
    // If MOSS_RNG_SEQ_FILE is set, consume floats from file (used to match ORT/PCG64).
    // Else fall back to the slot's mt19937 (re-seeded to 42 on acquirePoolSlot;
    // see header comment on MossTtsNanoSlot::rng for N=2 parity rationale).
    auto& rng = slot.rng;
    static thread_local std::uniform_real_distribution<float> u01(0.0f, 0.99999994f);
    float assistantU;
    std::vector<float> audioU(static_cast<size_t>(mNumVq));
    {
        auto& buf = diagRngBuf();
        auto& cur = diagRngCursor();
        size_t const need = 1 + static_cast<size_t>(mNumVq);
        if (!buf.empty() && cur + need <= buf.size())
        {
            assistantU = buf[cur++];
            for (auto& v : audioU) v = buf[cur++];
            if (diagEnabled() && diagSampleFrame <= 1)
                std::fprintf(stderr, "[moss_diag] sample#%d RNG from file cursor=%zu\n",
                    diagSampleFrame, cur);
        }
        else
        {
            assistantU = u01(rng);
            for (auto& v : audioU) v = u01(rng);
        }
    }

    if (diagEnabled() && diagSampleFrame <= 1)
    {
        std::fprintf(stderr, "[moss_diag] sample#%d assistantU=%.9f audioU:",
            diagSampleFrame, assistantU);
        for (auto v : audioU) std::fprintf(stderr, " %.9f", v);
        std::fprintf(stderr, "\n");
        // dump global_hidden input
        if (mGlobalHiddenDtype == nvinfer1::DataType::kHALF)
            dumpHalfFirst8(globalHidden, "sample_in global_hidden(half)", slot.stream);
        else
            dumpFloatFirst8(globalHidden, "sample_in global_hidden(fp32)", slot.stream);
        // dump repetition_seen_mask first 8 + full buf
        dumpInt32First8(slot.repetitionSeenMaskDevice, "sample_in rep_mask", slot.stream);
        size_t const repBytes = static_cast<size_t>(mNumVq) * mAudioCodebookSize * sizeof(int32_t);
        char p[256];
        std::snprintf(p, sizeof(p), "/tmp/cpp_f%d_rep_mask.bin", diagSampleFrame);
        dumpBufToFile(slot.repetitionSeenMaskDevice, repBytes, p, slot.stream);
    }

    checkCuda(cudaMemcpyAsync(slot.assistantRandomUDevice, &assistantU, sizeof(float),
                  cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS assistant_random_u");
    checkCuda(cudaMemcpyAsync(slot.audioRandomUDevice, audioU.data(),
                  audioU.size() * sizeof(float), cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS audio_random_u");

    setLocalFixedSampledFrameBindings(
        *slot.localFixedSampledFrameCtx, *mLocalFixedSampledFrameEngine, slot, globalHidden);
    if (!slot.localFixedSampledFrameCtx->enqueueV3(slot.stream))
        throw std::runtime_error("MOSS local_fixed_sampled_frame enqueueV3 failed");

    int32_t shouldContinue = 0;
    checkCuda(cudaMemcpyAsync(&shouldContinue, slot.shouldContinueDevice, sizeof(int32_t),
                  cudaMemcpyDeviceToHost, slot.stream),
        "copy MOSS should_continue back");
    {
        static thread_local int frameNum = 0;
        std::fprintf(stderr, "[moss_probe] sampleFrame#%d pastLen=%d cumulativePastLen=%d shouldContinue=%d\n",
            frameNum++, slot.pastLen, slot.cumulativePastLen, shouldContinue);
    }
    checkCuda(cudaMemcpyAsync(frameTokenIdsHost.data(), slot.frameTokenIdsDevice,
                  static_cast<size_t>(mNumVq) * sizeof(int32_t), cudaMemcpyDeviceToHost, slot.stream),
        "copy MOSS frame_token_ids back");
    checkCuda(cudaStreamSynchronize(slot.stream), "synchronize MOSS sampleFrame");

    if (diagEnabled() && diagSampleFrame <= 1)
    {
        std::fprintf(stderr, "[moss_diag] sample#%d frame_tokens:", diagSampleFrame);
        for (auto v : frameTokenIdsHost) std::fprintf(stderr, " %d", v);
        std::fprintf(stderr, " shouldContinue=%d\n", shouldContinue);
    }
    diagSampleFrame++;

    // Update repetition_seen_mask for next frame: mark each sampled token as seen.
    if (shouldContinue != 0)
    {
        int32_t const one = 1;
        for (int32_t ch = 0; ch < mNumVq; ++ch)
        {
            int32_t const tok = frameTokenIdsHost[static_cast<size_t>(ch)];
            if (tok < 0 || tok >= mAudioCodebookSize) continue;
            size_t const offset = (static_cast<size_t>(ch) * mAudioCodebookSize + tok) * sizeof(int32_t);
            checkCuda(cudaMemcpyAsync(
                          static_cast<unsigned char*>(slot.repetitionSeenMaskDevice) + offset,
                          &one, sizeof(int32_t), cudaMemcpyHostToDevice, slot.stream),
                "update MOSS repetition_seen_mask");
        }
    }
    return shouldContinue != 0;
}

// ---- Codec stateful streaming (Chunk 4) -----------------------------------

void MossTtsNanoRuntime::loadCodecEngine(std::string const& engineDir)
{
    std::filesystem::path const dir(engineDir);
    std::filesystem::path const metaPath = dir / "codec_browser_onnx_meta.json";
    if (!std::filesystem::exists(metaPath))
        throw std::runtime_error("Missing MOSS codec metadata: " + metaPath.string());

    std::ifstream file(metaPath);
    if (!file) throw std::runtime_error("Failed to open MOSS codec metadata: " + metaPath.string());

    Json meta = Json::parse(file);
    auto const& codec = meta.at("codec_config");
    mCodecSampleRate = codec.at("sample_rate").get<int32_t>();
    mCodecChannels = codec.at("channels").get<int32_t>();
    mCodecDownsampleRate = codec.at("downsample_rate").get<int32_t>();
    mCodecNumQuantizers = codec.at("num_quantizers").get<int32_t>();

    if (mCodecChannels != 2)
        throw std::runtime_error("MOSS codec wrapper currently expects stereo codec output");
    if (mCodecNumQuantizers != mNumVq)
        throw std::runtime_error("MOSS codec num_quantizers does not match TTS n_vq");

    auto const& streaming = meta.at("streaming_decode");
    auto const& transformerOffsets = streaming.at("transformer_offsets");
    auto const& attentionCaches = streaming.at("attention_caches");
    if (transformerOffsets.size() != 4)
        throw std::runtime_error("MOSS codec metadata must contain 4 transformer offsets");
    if (attentionCaches.size() != 12)
        throw std::runtime_error("MOSS codec metadata must contain 12 attention caches");

    for (auto const& cache : attentionCaches)
    {
        int32_t const index = cache.at("index").get<int32_t>();
        if (index < 0 || index >= 12)
            throw std::runtime_error("MOSS codec attention cache index out of range");

        auto const cacheShape = cache.at("cache_shape").get<std::vector<int64_t>>();
        auto const positionsShape = cache.at("positions_shape").get<std::vector<int64_t>>();
        if (cacheShape.size() != 4 || cacheShape[0] != 1 || cacheShape[1] != 4 || cacheShape[3] != 64)
            throw std::runtime_error(
                "Unexpected MOSS codec attention cache shape at index " + std::to_string(index));
        if (positionsShape.size() != 2 || positionsShape[0] != 1 || positionsShape[1] != cacheShape[2])
            throw std::runtime_error("Unexpected MOSS codec positions shape at index " + std::to_string(index));

        size_t cacheElements = 1;
        for (int64_t dim : cacheShape) cacheElements *= static_cast<size_t>(dim);
        size_t positionsElements = 1;
        for (int64_t dim : positionsShape) positionsElements *= static_cast<size_t>(dim);

        mCodecCacheElements[static_cast<size_t>(index)] = cacheElements;
        mCodecPositionsElements[static_cast<size_t>(index)] = positionsElements;
    }

    std::vector<char> blob = readBinary(dir / "codec_decode_step.plan");
    mCodecDecodeStepEngine.reset(mRuntime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!mCodecDecodeStepEngine)
        throw std::runtime_error("Failed to deserialize MOSS codec decode_step engine");

    for (char const* name : {"audio_codes", "audio_code_lengths", "audio", "audio_lengths"})
    {
        if (!hasTensor(*mCodecDecodeStepEngine, name))
            throw std::runtime_error(std::string("MOSS codec decode_step engine missing tensor: ") + name);
    }
}

void MossTtsNanoRuntime::allocateCodecState(MossTtsNanoSlot& slot)
{
    if (!mCodecDecodeStepEngine)
        throw std::runtime_error("MOSS codec engine must be loaded before allocating codec state");

    slot.codec = std::make_unique<MossCodecState>();
    MossCodecState& state = *slot.codec;

    for (int32_t i = 0; i < 4; ++i)
    {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&state.transformerOffsetIn[i]), sizeof(int32_t)),
            "allocate MOSS codec transformer offset in");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&state.transformerOffsetOut[i]), sizeof(int32_t)),
            "allocate MOSS codec transformer offset out");
    }

    for (int32_t i = 0; i < 12; ++i)
    {
        auto& cache = state.attn[i];
        cache.cacheElements = mCodecCacheElements[i];
        cache.positionsElements = mCodecPositionsElements[i];

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.offsetIn), sizeof(int32_t)),
            "allocate MOSS codec attn offset in");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.offsetOut), sizeof(int32_t)),
            "allocate MOSS codec attn offset out");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.keysIn), cache.cacheElements * sizeof(float)),
            "allocate MOSS codec attn keys in");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.keysOut), cache.cacheElements * sizeof(float)),
            "allocate MOSS codec attn keys out");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.valuesIn), cache.cacheElements * sizeof(float)),
            "allocate MOSS codec attn values in");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.valuesOut), cache.cacheElements * sizeof(float)),
            "allocate MOSS codec attn values out");
        checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&cache.positionsIn), cache.positionsElements * sizeof(int32_t)),
            "allocate MOSS codec attn positions in");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&cache.positionsOut),
                      cache.positionsElements * sizeof(int32_t)),
            "allocate MOSS codec attn positions out");
    }

    size_t const maxCodeElements
        = static_cast<size_t>(state.maxFrames) * static_cast<size_t>(mCodecNumQuantizers);
    size_t const maxAudioElements = static_cast<size_t>(mCodecChannels)
        * static_cast<size_t>(state.maxFrames) * static_cast<size_t>(mCodecDownsampleRate);

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&state.audioCodesDevice), maxCodeElements * sizeof(int32_t)),
        "allocate MOSS codec audio_codes");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&state.audioCodeLengthsDevice), sizeof(int32_t)),
        "allocate MOSS codec audio_code_lengths");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&state.audioOutDevice), maxAudioElements * sizeof(float)),
        "allocate MOSS codec audio output");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&state.audioOutLengthsDevice), sizeof(int32_t)),
        "allocate MOSS codec audio_lengths");

    slot.codecDecodeCtx = createContext(*mCodecDecodeStepEngine, "codec_decode_step");
    resetCodec(slot);
}

void MossTtsNanoRuntime::destroyCodecState(MossTtsNanoSlot& slot) noexcept
{
    delete slot.codecDecodeCtx;
    slot.codecDecodeCtx = nullptr;

    if (!slot.codec) return;
    auto& state = *slot.codec;

    for (auto*& ptr : state.transformerOffsetIn) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }
    for (auto*& ptr : state.transformerOffsetOut) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }

    for (auto& cache : state.attn)
    {
        if (cache.offsetIn) { cudaFree(cache.offsetIn); cache.offsetIn = nullptr; }
        if (cache.offsetOut) { cudaFree(cache.offsetOut); cache.offsetOut = nullptr; }
        if (cache.keysIn) { cudaFree(cache.keysIn); cache.keysIn = nullptr; }
        if (cache.keysOut) { cudaFree(cache.keysOut); cache.keysOut = nullptr; }
        if (cache.valuesIn) { cudaFree(cache.valuesIn); cache.valuesIn = nullptr; }
        if (cache.valuesOut) { cudaFree(cache.valuesOut); cache.valuesOut = nullptr; }
        if (cache.positionsIn) { cudaFree(cache.positionsIn); cache.positionsIn = nullptr; }
        if (cache.positionsOut) { cudaFree(cache.positionsOut); cache.positionsOut = nullptr; }
        cache.cacheElements = 0;
        cache.positionsElements = 0;
    }

    if (state.audioCodesDevice) { cudaFree(state.audioCodesDevice); state.audioCodesDevice = nullptr; }
    if (state.audioCodeLengthsDevice) { cudaFree(state.audioCodeLengthsDevice); state.audioCodeLengthsDevice = nullptr; }
    if (state.audioOutDevice) { cudaFree(state.audioOutDevice); state.audioOutDevice = nullptr; }
    if (state.audioOutLengthsDevice) { cudaFree(state.audioOutLengthsDevice); state.audioOutLengthsDevice = nullptr; }

    slot.codec.reset();
}

void MossTtsNanoRuntime::resetCodec(MossTtsNanoSlot& slot)
{
    if (!slot.codec) throw std::runtime_error("MOSS codec state is not allocated");
    auto& state = *slot.codec;

    for (int32_t i = 0; i < 4; ++i)
    {
        checkCuda(cudaMemsetAsync(state.transformerOffsetIn[i], 0, sizeof(int32_t), slot.stream),
            "reset MOSS codec transformer offset in");
        checkCuda(cudaMemsetAsync(state.transformerOffsetOut[i], 0, sizeof(int32_t), slot.stream),
            "reset MOSS codec transformer offset out");
    }

    for (auto& cache : state.attn)
    {
        checkCuda(cudaMemsetAsync(cache.offsetIn, 0, sizeof(int32_t), slot.stream),
            "reset MOSS codec attn offset in");
        checkCuda(cudaMemsetAsync(cache.offsetOut, 0, sizeof(int32_t), slot.stream),
            "reset MOSS codec attn offset out");
        checkCuda(cudaMemsetAsync(cache.keysIn, 0, cache.cacheElements * sizeof(float), slot.stream),
            "reset MOSS codec attn keys in");
        checkCuda(cudaMemsetAsync(cache.keysOut, 0, cache.cacheElements * sizeof(float), slot.stream),
            "reset MOSS codec attn keys out");
        checkCuda(cudaMemsetAsync(cache.valuesIn, 0, cache.cacheElements * sizeof(float), slot.stream),
            "reset MOSS codec attn values in");
        checkCuda(cudaMemsetAsync(cache.valuesOut, 0, cache.cacheElements * sizeof(float), slot.stream),
            "reset MOSS codec attn values out");
        // Positions init to -1 (0xff bytes → 0xffffffff = -1 in two's complement int32).
        checkCuda(cudaMemsetAsync(cache.positionsIn, 0xff, cache.positionsElements * sizeof(int32_t), slot.stream),
            "reset MOSS codec attn positions in");
        checkCuda(cudaMemsetAsync(cache.positionsOut, 0xff, cache.positionsElements * sizeof(int32_t), slot.stream),
            "reset MOSS codec attn positions out");
    }

    checkCuda(cudaStreamSynchronize(slot.stream), "synchronize MOSS codec reset");
}

void MossTtsNanoRuntime::setCodecBindings(MossTtsNanoSlot& slot, int32_t frameCount)
{
    if (!slot.codec || slot.codecDecodeCtx == nullptr)
        throw std::runtime_error("MOSS codec state is not allocated");
    if (frameCount <= 0 || frameCount > slot.codec->maxFrames)
        throw std::runtime_error("MOSS codec frameCount out of range");

    auto& state = *slot.codec;
    auto& ctx = *slot.codecDecodeCtx;

    checkTrt(ctx.setInputShape("audio_codes", nvinfer1::Dims3{1, frameCount, mCodecNumQuantizers}),
        "set codec audio_codes shape");
    checkTrt(ctx.setInputShape("audio_code_lengths", nvinfer1::Dims{1, {1}}),
        "set codec audio_code_lengths shape");
    checkTrt(ctx.setTensorAddress("audio_codes", state.audioCodesDevice), "set codec audio_codes address");
    checkTrt(ctx.setTensorAddress("audio_code_lengths", state.audioCodeLengthsDevice),
        "set codec audio_code_lengths address");
    checkTrt(ctx.setTensorAddress("audio", state.audioOutDevice), "set codec audio address");
    checkTrt(ctx.setTensorAddress("audio_lengths", state.audioOutLengthsDevice), "set codec audio_lengths address");

    for (int32_t i = 0; i < 4; ++i)
    {
        std::string const input = "transformer_offset_" + std::to_string(i);
        std::string const output = "transformer_offset_out_" + std::to_string(i);
        checkTrt(ctx.setInputShape(input.c_str(), nvinfer1::Dims{1, {1}}), "set " + input + " shape");
        checkTrt(ctx.setTensorAddress(input.c_str(), state.transformerOffsetIn[i]), "set " + input + " address");
        checkTrt(ctx.setTensorAddress(output.c_str(), state.transformerOffsetOut[i]), "set " + output + " address");
    }

    for (int32_t i = 0; i < 12; ++i)
    {
        auto& cache = state.attn[i];
        int32_t const context = static_cast<int32_t>(cache.positionsElements);

        std::string const offsetIn = "attn_offset_" + std::to_string(i);
        std::string const offsetOut = "attn_offset_out_" + std::to_string(i);
        std::string const keysIn = "attn_cached_keys_" + std::to_string(i);
        std::string const keysOut = "attn_cached_keys_out_" + std::to_string(i);
        std::string const valuesIn = "attn_cached_values_" + std::to_string(i);
        std::string const valuesOut = "attn_cached_values_out_" + std::to_string(i);
        std::string const positionsIn = "attn_cached_positions_" + std::to_string(i);
        std::string const positionsOut = "attn_cached_positions_out_" + std::to_string(i);

        checkTrt(ctx.setInputShape(offsetIn.c_str(), nvinfer1::Dims{1, {1}}), "set " + offsetIn + " shape");
        checkTrt(ctx.setInputShape(keysIn.c_str(), nvinfer1::Dims4{1, 4, context, 64}), "set " + keysIn + " shape");
        checkTrt(ctx.setInputShape(valuesIn.c_str(), nvinfer1::Dims4{1, 4, context, 64}), "set " + valuesIn + " shape");
        checkTrt(ctx.setInputShape(positionsIn.c_str(), nvinfer1::Dims2{1, context}), "set " + positionsIn + " shape");

        checkTrt(ctx.setTensorAddress(offsetIn.c_str(), cache.offsetIn), "set " + offsetIn + " address");
        checkTrt(ctx.setTensorAddress(keysIn.c_str(), cache.keysIn), "set " + keysIn + " address");
        checkTrt(ctx.setTensorAddress(valuesIn.c_str(), cache.valuesIn), "set " + valuesIn + " address");
        checkTrt(ctx.setTensorAddress(positionsIn.c_str(), cache.positionsIn), "set " + positionsIn + " address");

        checkTrt(ctx.setTensorAddress(offsetOut.c_str(), cache.offsetOut), "set " + offsetOut + " address");
        checkTrt(ctx.setTensorAddress(keysOut.c_str(), cache.keysOut), "set " + keysOut + " address");
        checkTrt(ctx.setTensorAddress(valuesOut.c_str(), cache.valuesOut), "set " + valuesOut + " address");
        checkTrt(ctx.setTensorAddress(positionsOut.c_str(), cache.positionsOut), "set " + positionsOut + " address");
    }
}

size_t MossTtsNanoRuntime::decodeFrames(MossTtsNanoSlot& slot,
    std::vector<std::vector<int32_t>> const& frameRows, std::vector<float>& pcmOut)
{
    if (!slot.codec) throw std::runtime_error("MOSS codec state is not allocated");
    auto& state = *slot.codec;

    int32_t const frameCount = static_cast<int32_t>(frameRows.size());
    if (frameCount <= 0) return 0;
    if (frameCount > state.maxFrames)
        throw std::runtime_error("MOSS codec decodeFrames exceeds maxFrames");

    std::vector<int32_t> flat(static_cast<size_t>(frameCount) * static_cast<size_t>(mCodecNumQuantizers));
    for (int32_t frame = 0; frame < frameCount; ++frame)
    {
        auto const& row = frameRows[static_cast<size_t>(frame)];
        if (row.size() != static_cast<size_t>(mCodecNumQuantizers))
            throw std::runtime_error("MOSS codec frame row must contain num_quantizers codes");
        std::copy(row.begin(), row.end(),
            flat.begin() + static_cast<ptrdiff_t>(frame) * mCodecNumQuantizers);
    }

    checkCuda(cudaMemcpyAsync(state.audioCodesDevice, flat.data(),
                  flat.size() * sizeof(int32_t), cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS codec audio_codes");
    checkCuda(cudaMemcpyAsync(state.audioCodeLengthsDevice, &frameCount, sizeof(int32_t),
                  cudaMemcpyHostToDevice, slot.stream),
        "copy MOSS codec audio_code_lengths");

    setCodecBindings(slot, frameCount);
    if (!slot.codecDecodeCtx->enqueueV3(slot.stream))
        throw std::runtime_error("MOSS codec decode_step enqueueV3 failed");

    for (int32_t i = 0; i < 4; ++i)
        std::swap(state.transformerOffsetIn[i], state.transformerOffsetOut[i]);
    for (auto& cache : state.attn)
    {
        std::swap(cache.offsetIn, cache.offsetOut);
        std::swap(cache.keysIn, cache.keysOut);
        std::swap(cache.valuesIn, cache.valuesOut);
        std::swap(cache.positionsIn, cache.positionsOut);
    }

    int32_t audioLength = 0;
    checkCuda(cudaMemcpyAsync(&audioLength, state.audioOutLengthsDevice, sizeof(int32_t),
                  cudaMemcpyDeviceToHost, slot.stream),
        "copy MOSS codec audio_lengths back");
    checkCuda(cudaStreamSynchronize(slot.stream), "synchronize MOSS codec audio_lengths");

    if (audioLength < 0 || audioLength > frameCount * mCodecDownsampleRate)
        throw std::runtime_error("MOSS codec audio_lengths out of range");

    std::vector<float> planar(
        static_cast<size_t>(mCodecChannels) * static_cast<size_t>(audioLength));
    checkCuda(cudaMemcpyAsync(planar.data(), state.audioOutDevice,
                  planar.size() * sizeof(float), cudaMemcpyDeviceToHost, slot.stream),
        "copy MOSS codec audio back");
    checkCuda(cudaStreamSynchronize(slot.stream), "synchronize MOSS codec audio");

    size_t const oldSize = pcmOut.size();
    pcmOut.resize(oldSize + static_cast<size_t>(audioLength) * static_cast<size_t>(mCodecChannels));
    float const* left = planar.data();
    float const* right = planar.data() + static_cast<size_t>(audioLength);
    for (int32_t i = 0; i < audioLength; ++i)
    {
        size_t const out = oldSize + static_cast<size_t>(i) * 2;
        pcmOut[out] = left[i];
        pcmOut[out + 1] = right[i];
    }
    return static_cast<size_t>(audioLength) * static_cast<size_t>(mCodecChannels);
}

// ---- Chunk 5: generate() orchestrator -------------------------------------

void MossTtsNanoRuntime::generate(MossTtsNanoSlot& slot, std::vector<int32_t> const& inputIds,
    std::vector<int32_t> const& attentionMask, GenerateConfig const& cfg, std::vector<float>& pcmOut)
{
    constexpr int32_t kRowWidth = 17;
    constexpr int32_t kAudioPadTokenId = 0; // TODO: read from meta tts_config.audio_pad_token_id

    resetCodec(slot);

    void* globalHidden = prefill(slot, inputIds, attentionMask);
    std::vector<std::vector<int32_t>> frameBuffer;
    int32_t const maxFrames = cfg.maxNewFrames > 0 ? cfg.maxNewFrames : 0;
    int32_t const chunkFrames = cfg.codecChunkFrames > 0 ? cfg.codecChunkFrames : 1;
    frameBuffer.reserve(static_cast<size_t>(chunkFrames));

    bool firstChunkEmitted = false;
    auto flushFrames = [&]() {
        if (frameBuffer.empty()) return;
        decodeFrames(slot, frameBuffer, pcmOut);
        frameBuffer.clear();
        if (!firstChunkEmitted)
        {
            firstChunkEmitted = true;
            if (cfg.onFirstChunk) cfg.onFirstChunk();
        }
    };

    for (int32_t frameIdx = 0; frameIdx < maxFrames; ++frameIdx)
    {
        std::vector<int32_t> frameTokens(static_cast<size_t>(mNumVq), 0);
        bool const shouldContinue = sampleFrame(slot, globalHidden, frameTokens);
        if (!shouldContinue) break;

        frameBuffer.push_back(std::move(frameTokens));

        bool const reachedChunk = static_cast<int32_t>(frameBuffer.size()) >= chunkFrames;
        bool const reachedLastFrame = (frameIdx + 1) >= maxFrames;
        if (reachedChunk || reachedLastFrame) flushFrames();
        if (reachedLastFrame) break;

        // TODO: feed actual last-frame audio tokens; MVP uses audio_pad_token.
        std::vector<int32_t> nextInputIds(static_cast<size_t>(kRowWidth), kAudioPadTokenId);
        globalHidden = decodeStep(slot, nextInputIds, globalHidden);
    }

    flushFrames();
}

} // namespace rt
} // namespace trt_edgellm
