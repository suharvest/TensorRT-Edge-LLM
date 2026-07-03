/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * qwen3_tts_streaming_worker (v0.9.0 re-port, M1)
 * -----------------------------------------------
 * Standalone TTS streaming worker for Qwen3-TTS.
 *
 * - Reads JSON request lines on stdin.
 * - For each request, invokes the runtime's handleAudioGeneration() with the
 *   v0.9.0 NATIVE streaming API (TalkerGenerationRequest::streamingChunkFrames
 *   + onChunkReady(codes, isFinal)) and vocodes each RVQ chunk through
 *   Code2Wav on the worker side, emitting base64 PCM_S16LE chunk events on
 *   stdout.
 *
 * Protocol (UNCHANGED from the v0.8.0 worker — this is the voxedge consumer
 * contract): ready / chunk (audio_b64, is_final) / done / error events with
 * request_id+id, pool saturation error with status 4429.
 *
 * M1 deltas vs the v0.8.0 worker (see docs spec edgellm-v090-tts-re-port):
 *   * Native streaming: the runtime emits uniform streamingChunkFrames =
 *     `first_chunk_frames` (the TTFA-critical knob). Differentiated chunk
 *     sizes are recovered WORKER-SIDE (M3): after the first chunk, runtime
 *     chunks are aggregated up to `chunk_frames` frames before each Code2Wav
 *     call — no runtime chunk-schedule surgery (unlike the v0.8.0 port).
 *   * Cancel (M3): {"type":"cancel","id":X} trips the per-id atomic flag; the
 *     runtime polls TalkerGenerationRequest::shouldCancel per decoded frame,
 *     the stream ends with is_final=true, and the worker emits `cancelled`
 *     (same events/fields as the v0.8.0 worker: cancel_ack{tripped} →
 *     final chunk → cancelled).
 *   * Slots (M3): slot 0 deserializes the engines; slots 1..N-1 share slot 0's
 *     read-only ICudaEngines via the shared-engine ctors (Talker,
 *     CodePredictor, Code2Wav), paying only per-slot context/KV/workspace
 *     memory. EDGE_LLM_TTS_SHARED_ENGINE=0 forces independent per-slot
 *     deserialization (memory A/B baseline).
 *   * `language` (CustomVoice 9-row language conditioning) and
 *     `speaker_embedding_b64` (external voice-clone embedding, base64
 *     LE float32) are passed through to the runtime (M2 patches). Omitting
 *     them keeps the upstream 8-row named-speaker behavior.
 *
 * Concurrency model (unchanged): N worker threads, each bound to ONE slot
 * (runtime + stateless Code2Wav + CUDA stream). A runtime instance is never
 * invoked from more than one thread. The stdin reader routes generation
 * requests to a free slot via SlotPool's CAS acquire; saturation returns the
 * 4429-style structured error.
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include "slotPool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

struct Args
{
    std::string talkerEngineDir;
    std::string codePredictorEngineDir;
    std::string code2wavEngineDir;
    std::string tokenizerDir;
    int32_t maxSlots{1}; //!< TTS slot-pool size. Default 1 == single-instance behavior (M1 validated config).
    bool debug{false};
};

enum OptionId : int
{
    HELP = 1000,
    TALKER_ENGINE_DIR,
    CODE_PREDICTOR_ENGINE_DIR,
    CODE2WAV_ENGINE_DIR,
    TOKENIZER_DIR,
    MAX_SLOTS,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --talkerEngineDir=<path> --code2wavEngineDir=<path>"
              << " [--codePredictorEngineDir=<path>] [--tokenizerDir=<path>] [--max_slots=<N>] [--debug]\n\n"
              << "Standalone Qwen3-TTS streaming worker (v0.9.0 native streaming API).\n"
              << "Reads JSON request lines from stdin and emits JSON events to stdout.\n"
              << "--max_slots=<N> sets the TTS slot-pool size (default 1). Each slot is a\n"
              << "  full TTS runtime + Code2Wav + CUDA stream + worker thread.\n"
              << "Request schema:\n"
              << "  {\"id\":\"...\",\"text\":\"...\",\"speaker\":\"Vivian\",\n"
              << "   \"stream\":true,\"first_chunk_frames\":8,\"chunk_frames\":10,\n"
              << "   \"chunk_format\":\"pcm_s16le\",\"chunk_transport\":\"base64\"}\n"
              << "Cancel (trips the in-flight request with that id; stream ends with is_final=true):\n"
              << "  {\"type\":\"cancel\",\"id\":\"...\"}\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {
        {"help", no_argument, 0, HELP},
        {"talkerEngineDir", required_argument, 0, TALKER_ENGINE_DIR},
        {"codePredictorEngineDir", required_argument, 0, CODE_PREDICTOR_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, CODE2WAV_ENGINE_DIR},
        {"tokenizerDir", required_argument, 0, TOKENIZER_DIR},
        {"max_slots", required_argument, 0, MAX_SLOTS},
        {"debug", no_argument, 0, DEBUG},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case CODE_PREDICTOR_ENGINE_DIR: args.codePredictorEngineDir = optarg; break;
        case CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case TOKENIZER_DIR: args.tokenizerDir = optarg; break;
        case MAX_SLOTS:
        {
            int const v = std::atoi(optarg);
            args.maxSlots = (v >= 1) ? v : 1; // clamp to >=1; 1 == single-instance behavior
            break;
        }
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }

    if (args.codePredictorEngineDir.empty() && !args.talkerEngineDir.empty())
    {
        args.codePredictorEngineDir
            = (std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor").string();
    }
    if (args.tokenizerDir.empty() && !args.talkerEngineDir.empty())
    {
        args.tokenizerDir = args.talkerEngineDir;
    }
    // Slot-pool size env fallback (matches the OVS EDGE_LLM_TTS_MAX_CONCURRENT
    // convention). CLI --max_slots takes precedence when given (!= default 1).
    if (args.maxSlots == 1)
    {
        if (char const* p = std::getenv("EDGE_LLM_TTS_MAX_CONCURRENT"))
        {
            int const v = std::atoi(p);
            if (v >= 1) args.maxSlots = v;
        }
    }
    return !args.talkerEngineDir.empty() && !args.codePredictorEngineDir.empty() && !args.code2wavEngineDir.empty();
}

// ===== base64 + PCM helpers =====

std::string base64Encode(uint8_t const* data, size_t len)
{
    static constexpr char kTable[]
        = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3)
    {
        uint32_t const b0 = data[i];
        uint32_t const b1 = (i + 1 < len) ? data[i + 1] : 0;
        uint32_t const b2 = (i + 2 < len) ? data[i + 2] : 0;
        uint32_t const triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? kTable[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? kTable[triple & 0x3F] : '=');
    }
    return out;
}

// Decode base64 -> raw bytes (used for the precomputed speaker embedding).
std::vector<uint8_t> base64Decode(std::string const& in)
{
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1; // '=' padding or whitespace
    };
    std::vector<uint8_t> out;
    out.reserve((in.size() / 4) * 3);
    int buf = 0, bits = 0;
    for (char const c : in)
    {
        int const d = val(c);
        if (d < 0) continue;
        buf = (buf << 6) | d;
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

// Decode a base64 little-endian float32 array -> float vector (speaker embedding).
std::vector<float> base64ToFloatVec(std::string const& b64)
{
    std::vector<uint8_t> const bytes = base64Decode(b64);
    size_t const n = bytes.size() / sizeof(float);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        union
        {
            uint32_t u;
            float f;
        } conv;
        conv.u = static_cast<uint32_t>(bytes[i * 4]) | (static_cast<uint32_t>(bytes[i * 4 + 1]) << 8)
            | (static_cast<uint32_t>(bytes[i * 4 + 2]) << 16) | (static_cast<uint32_t>(bytes[i * 4 + 3]) << 24);
        out[i] = conv.f;
    }
    return out;
}

std::vector<int16_t> floatToPcm16(std::vector<float> const& samples)
{
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        float const clipped = std::clamp(samples[i], -1.0f, 1.0f);
        pcm[i] = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
    }
    return pcm;
}

//! v0.9.0: Code2Wav returns the waveform in the ENGINE's output dtype (FP16 on
//! the Jetson engines, FP32 possible on others) — see Code2WavRunner
//! mWaveformDtype. Convert per-dtype; do NOT assume float like the v0.8.0
//! worker did.
std::vector<float> audioToFloatSamples(rt::audioUtils::AudioData const& audio)
{
    if (!audio.waveform || audio.waveform->isEmpty())
    {
        return {};
    }
    int64_t const samples = audio.waveform->getShape()[1];
    std::vector<float> out(static_cast<size_t>(samples));
    if (audio.waveform->getDataType() == nvinfer1::DataType::kHALF)
    {
        __half const* data = static_cast<__half const*>(audio.waveform->rawPointer());
        for (int64_t i = 0; i < samples; ++i)
        {
            out[static_cast<size_t>(i)] = __half2float(data[i]);
        }
    }
    else
    {
        float const* data = static_cast<float const*>(audio.waveform->rawPointer());
        std::copy(data, data + samples, out.begin());
    }
    return out;
}

std::vector<std::vector<int32_t>> transposeFrames(std::vector<std::vector<int32_t>> const& frames)
{
    if (frames.empty())
    {
        return {};
    }
    size_t const numFrames = frames.size();
    size_t const numLayers = frames[0].size();
    std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
    for (size_t f = 0; f < numFrames; ++f)
    {
        for (size_t l = 0; l < numLayers; ++l)
        {
            transposed[l][f] = frames[f][l];
        }
    }
    return transposed;
}

// ===== stdout serialization =====
//
// Every worker thread emits events through emitEvent(); coutMutex serializes
// the line writes so concurrent slots never interleave bytes on stdout.

std::mutex coutMutex;

void emitEvent(Json payload)
{
    std::string const line = payload.dump();
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << line << std::endl;
}

// ===== cancel map (per-id → slot's cancel flag) =====
//
// A cancel line {"type":"cancel","id":X} trips the atomic<bool> belonging to
// the in-flight request with that id. The flag lives on the slot (see TtsSlot)
// and the request's shouldCancel lambda polls it from the runtime decode loop.
// cancelMapMu guards the map; the worker thread registers its flag when it
// starts a request and unregisters at request end. (Verbatim re-port of the
// v0.8.0 worker cancel protocol on top of the M3 runtime shouldCancel hook.)

std::mutex cancelMapMu;
std::unordered_map<std::string, std::atomic<bool>*> cancelMap;

void registerCancel(std::string const& id, std::atomic<bool>* flag)
{
    std::lock_guard<std::mutex> lk(cancelMapMu);
    cancelMap[id] = flag;
}

void unregisterCancel(std::string const& id)
{
    std::lock_guard<std::mutex> lk(cancelMapMu);
    cancelMap.erase(id);
}

bool tripCancel(std::string const& id)
{
    std::lock_guard<std::mutex> lk(cancelMapMu);
    auto it = cancelMap.find(id);
    if (it == cancelMap.end())
    {
        return false;
    }
    it->second->store(true, std::memory_order_release);
    return true;
}

// ===== request builder =====

Qwen3OmniTTSRuntime::TalkerGenerationRequest buildRequest(Json const& item)
{
    Qwen3OmniTTSRuntime::TalkerGenerationRequest req;
    req.maxAudioLength = item.value("max_audio_length", 4096);
    req.talkerTemperature = item.value("talker_temperature", 0.9f);
    req.talkerTopK = item.value("talker_top_k", 50);
    req.talkerTopP = item.value("talker_top_p", 1.0f);
    req.repetitionPenalty = item.value("repetition_penalty", 1.05f);
    req.speakerName = item.value("speaker", "");
    req.speakerId = item.value("speaker_id", -1);

    // CustomVoice language conditioning: pass per-request language through to the runtime, which
    // maps it to a codec language id and selects the 9-row prefix. Empty/omitted => langId=-1 =>
    // the upstream 8-row path, unchanged.
    req.language = item.value("language", "");
    // Optional precomputed external speaker embedding (base64 LE float32 array). When present, the
    // runtime uses it as the speaker-row conditioning vector instead of a named-speaker token.
    {
        std::string const spkB64 = item.value("speaker_embedding_b64", "");
        if (!spkB64.empty())
        {
            req.speakerEmbedding = base64ToFloatVec(spkB64);
        }
    }

    Message msg;
    // Qwen3-TTS talker prefill expects the assistant role prefix
    // ([<|im_start|>, assistant(77091), \n]) at input_ids[:3], not the user
    // prefix (token 872). The text to synthesize is the assistant's content.
    // v0.9.0 keeps `messages` as the only text entry point (applyChatTemplate
    // must stay true: the runtime unconditionally strips the 3-token prefix and
    // 5-token suffix added by the template).
    msg.role = "assistant";
    Message::MessageContent content;
    content.type = "text";
    content.content = item.value("text", "");
    msg.contents.push_back(std::move(content));
    req.messages.push_back(std::move(msg));
    return req;
}

// ===========================================================================
// TTS slot-pool.
//
// Each TtsSlot is a self-contained, single-threaded TTS lane:
//   * runtime    — independent Qwen3OmniTTSRuntime. Slot 0 deserializes the
//                  Talker + CodePredictor engines; slots 1..N-1 share those
//                  ICudaEngines (shared-engine ctor). Each slot has its own
//                  IExecutionContexts, KV/cache state, workspace tensors.
//   * code2wav   — independent (STATELESS) Code2WavRunner per slot; slots
//                  1..N-1 share slot 0's Code2Wav engine weights, contexts
//                  and buffers stay per-slot.
//   * stream     — per-slot CUDA stream so independent slots can overlap.
//   * worker     — the ONE OS thread that drives this slot's runtime.
//   * queue/cv   — single-element handoff from the stdin reader to the worker.
// ===========================================================================

struct WorkItem
{
    Json request; //!< Parsed generation request JSON.
};

struct TtsSlot
{
    int32_t slotId{-1};
    std::atomic<bool> inUse{false};    //!< True from enqueue until request completion.
    std::atomic<bool> shutdown{false}; //!< Set at teardown so the worker thread exits.
    std::atomic<bool> cancelled{false}; //!< Per-slot cancel flag, registered in cancelMap while a request runs.

    std::unique_ptr<Qwen3OmniTTSRuntime> runtime;
    std::unique_ptr<Code2WavRunner> code2wav; //!< Per-slot stateless vocoder.
    cudaStream_t stream{nullptr};

    // Single-slot handoff queue (reader thread -> this slot's worker thread).
    std::mutex queueMu;
    std::condition_variable queueCv;
    std::deque<WorkItem> queue;

    std::thread worker;

    TtsSlot() = default;
    // Non-copyable / non-movable: holds std::atomic + CUDA stream + thread.
    TtsSlot(TtsSlot const&) = delete;
    TtsSlot& operator=(TtsSlot const&) = delete;
};

int32_t gMaxSlots{1};

namespace rt_slotpool = tensorrt_edge_llm::runtime;
std::unique_ptr<rt_slotpool::SlotPool<TtsSlot>> gPool;

void bindSession(std::string const& id, int32_t slotId)
{
    gPool->bind(id, slotId);
}

void unbindSession(std::string const& id)
{
    gPool->unbind(id);
}

// ===== sample rate (read once at init; identical across slots) =====
int32_t gSampleRate{24000};

// ---------------------------------------------------------------------------
// processRequest — run ONE generation request on a given slot.
//
// Streaming path: v0.9.0 native TalkerGenerationRequest::streamingChunkFrames
// + onChunkReady(codes, isFinal). The worker vocodes each chunk through the
// slot's Code2Wav and emits the protocol `chunk` event. The optional async
// vocode thread (EDGE_LLM_TTS_ASYNC_VOCODE=1) overlaps Code2Wav with the next
// talker decode chunk, exactly as in the v0.8.0 worker.
// ---------------------------------------------------------------------------
void processRequest(TtsSlot& slot, Json const& item, bool useStateful, bool useAsyncVocode)
{
    Qwen3OmniTTSRuntime& ttsRuntime = *slot.runtime;
    Code2WavRunner* code2wavRunner = slot.code2wav.get();
    cudaStream_t const stream = slot.stream;

    std::string const requestId = item.value("id", "");
    // Reset and register this slot's cancel flag for the in-flight request.
    slot.cancelled.store(false, std::memory_order_release);
    std::atomic<bool>& cancelled = slot.cancelled;
    if (!requestId.empty())
    {
        registerCancel(requestId, &cancelled);
    }

    int32_t chunkIndex = 0;
    auto const requestStart = std::chrono::steady_clock::now();

    struct VocodeJob
    {
        bool poison{false};
        bool isFinal{false};
        std::vector<std::vector<int32_t>> frames;
        int32_t chunkIndex{0};
        std::chrono::steady_clock::time_point enqueuedAt;
    };
    std::mutex vocodeMu;
    std::condition_variable vocodeCv;
    std::deque<VocodeJob> vocodeQ;
    std::exception_ptr vocodeError{nullptr};
    std::thread vocodeThread;
    cudaStream_t vocodeStream{};
    bool vocodeStreamCreated{false};

    try
    {
        auto request = buildRequest(item);
        // Cooperative cancel: the runtime decode loop polls this flag once per
        // frame; when tripped the stream ends with isFinal=true and
        // handleAudioGeneration returns normally.
        request.shouldCancel = [&cancelled]() { return cancelled.load(std::memory_order_acquire); };
        // Differentiated chunk sizes (M3, worker-side): the runtime's native
        // streamingChunkFrames is set to first_chunk_frames (small — the
        // TTFA-critical knob), and the worker AGGREGATES subsequent runtime
        // chunks up to chunk_frames before vocoding (bigger Code2Wav calls =
        // better throughput). The runtime chunk schedule itself is untouched.
        // chunk_frames <= first_chunk_frames disables aggregation (uniform
        // chunking, M1 behavior).
        int32_t const firstChunkFrames = std::max(1, item.value("first_chunk_frames", 8));
        int32_t const subsequentChunkFrames = std::max(0, item.value("chunk_frames", 10));
        bool const streaming = item.value("stream", true);
        std::string const chunkFormat = item.value("chunk_format", "pcm_s16le");
        std::string const chunkTransport = item.value("chunk_transport", "base64");

        // Vocode lambda — uses THIS slot's code2wav (stateless generateWaveform).
        auto runVocode = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes, cudaStream_t s,
                             std::vector<float>& samplesOut) -> bool {
            rt::audioUtils::AudioData audioOutput;
            auto const transposed = transposeFrames(chunkRvqCodes);
            if (!code2wavRunner->generateWaveform(transposed, audioOutput, s))
            {
                return false;
            }
            samplesOut = audioToFloatSamples(audioOutput);
            return true;
        };

        auto emitChunk = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal, int32_t idx,
                             std::chrono::steady_clock::time_point startTs, cudaStream_t vocStream) {
            int32_t const frames = static_cast<int32_t>(chunkRvqCodes.size());
            auto const c2wStart = std::chrono::steady_clock::now();
            std::vector<float> samples;
            if (frames > 0)
            {
                if (!runVocode(chunkRvqCodes, vocStream, samples))
                {
                    emitEvent(Json{{"event", "error"}, {"ok", false},
                        {"request_id", requestId}, {"id", requestId},
                        {"error", "Code2Wav generateWaveform failed"}});
                    return;
                }
            }
            auto const c2wEnd = std::chrono::steady_clock::now();
            int64_t const c2wMs
                = std::chrono::duration_cast<std::chrono::milliseconds>(c2wEnd - c2wStart).count();

            std::vector<int16_t> const pcm = floatToPcm16(samples);
            std::string const b64 = base64Encode(
                reinterpret_cast<uint8_t const*>(pcm.data()), pcm.size() * sizeof(int16_t));

            int64_t const elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                c2wEnd - requestStart).count();
            int64_t const queueMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                c2wStart - startTs).count();

            Json evt = {
                {"event", "chunk"},
                {"ok", true},
                {"request_id", requestId},
                {"id", requestId},
                {"chunk_index", idx},
                {"chunk_format", chunkFormat},
                {"chunk_transport", chunkTransport},
                {"frames", frames},
                {"samples", static_cast<int64_t>(pcm.size())},
                {"sample_rate", gSampleRate},
                {"is_final", isFinal},
                {"code2wav_ms", c2wMs},
                {"queue_ms", queueMs},
                {"elapsed_ms", elapsedMs},
                {"audio_b64", b64},
            };
            emitEvent(std::move(evt));
        };

        if (streaming && useAsyncVocode)
        {
            CUDA_CHECK(cudaStreamCreate(&vocodeStream));
            vocodeStreamCreated = true;
            vocodeThread = std::thread([&] {
                while (true)
                {
                    VocodeJob job;
                    {
                        std::unique_lock<std::mutex> lk(vocodeMu);
                        vocodeCv.wait(lk, [&] { return !vocodeQ.empty(); });
                        job = std::move(vocodeQ.front());
                        vocodeQ.pop_front();
                    }
                    if (job.poison)
                    {
                        break;
                    }
                    try
                    {
                        emitChunk(job.frames, job.isFinal, job.chunkIndex, job.enqueuedAt, vocodeStream);
                    }
                    catch (...)
                    {
                        vocodeError = std::current_exception();
                    }
                }
            });
        }

        // Deliver one protocol chunk (vocode + emit): async path enqueues to the
        // overlap vocode thread, sync path vocodes inline on the slot stream.
        std::function<void(std::vector<std::vector<int32_t>> const&, bool)> deliverChunk;
        // Worker-side aggregation state for differentiated chunk sizes. Touched
        // only from the runtime generation thread (onChunkReady caller), so no
        // locking is needed.
        std::vector<std::vector<int32_t>> aggBuffer;
        bool firstChunkDelivered = false;

        if (streaming)
        {
            // v0.9.0 native streaming API. The runtime invokes onChunkReady from
            // the talker generation loop every streamingChunkFrames frames, and
            // exactly once with isFinal=true at end-of-stream (possibly with an
            // empty codes vector) — the worker relays that as the protocol's
            // is_final chunk event.
            request.streamingChunkFrames = firstChunkFrames;
            if (useAsyncVocode)
            {
                deliverChunk = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal) {
                    VocodeJob job;
                    job.poison = false;
                    job.isFinal = isFinal;
                    job.frames = chunkRvqCodes;
                    job.chunkIndex = chunkIndex++;
                    job.enqueuedAt = std::chrono::steady_clock::now();
                    {
                        std::lock_guard<std::mutex> lk(vocodeMu);
                        vocodeQ.push_back(std::move(job));
                    }
                    vocodeCv.notify_one();
                };
            }
            else
            {
                deliverChunk = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal) {
                    emitChunk(chunkRvqCodes, isFinal, chunkIndex++, std::chrono::steady_clock::now(), stream);
                };
            }

            bool const aggregate = subsequentChunkFrames > firstChunkFrames;
            if (!aggregate)
            {
                // Uniform chunking (M1 behavior): relay every runtime chunk 1:1.
                request.onChunkReady = deliverChunk;
            }
            else
            {
                // First runtime chunk goes out immediately (TTFA); subsequent
                // runtime chunks accumulate until subsequentChunkFrames frames,
                // then vocode as ONE bigger chunk. isFinal flushes the remainder
                // (possibly empty) as the protocol's is_final event.
                request.onChunkReady = [&, subsequentChunkFrames](
                                           std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal) {
                    if (!firstChunkDelivered)
                    {
                        firstChunkDelivered = true;
                        deliverChunk(chunkRvqCodes, isFinal);
                        return;
                    }
                    aggBuffer.insert(aggBuffer.end(), chunkRvqCodes.begin(), chunkRvqCodes.end());
                    if (isFinal)
                    {
                        deliverChunk(aggBuffer, true);
                        aggBuffer.clear();
                        return;
                    }
                    if (static_cast<int32_t>(aggBuffer.size()) >= subsequentChunkFrames)
                    {
                        deliverChunk(aggBuffer, false);
                        aggBuffer.clear();
                    }
                };
            }
        }

        Qwen3OmniTTSRuntime::TalkerGenerationResponse response;
        bool const ok = ttsRuntime.handleAudioGeneration(
            std::vector<Qwen3OmniTTSRuntime::TalkerGenerationRequest>{request}, response, stream);

        // Drain async vocode thread before emitting done/error.
        if (vocodeThread.joinable())
        {
            {
                std::lock_guard<std::mutex> lk(vocodeMu);
                VocodeJob poison;
                poison.poison = true;
                vocodeQ.push_back(std::move(poison));
            }
            vocodeCv.notify_one();
            vocodeThread.join();
        }
        if (vocodeStreamCreated)
        {
            cudaStreamDestroy(vocodeStream);
            vocodeStreamCreated = false;
        }
        if (vocodeError)
        {
            std::rethrow_exception(vocodeError);
        }

        if (cancelled.load(std::memory_order_acquire))
        {
            emitEvent(Json{{"event", "cancelled"}, {"ok", true},
                {"request_id", requestId}, {"id", requestId}, {"reason", "cancelled"}});
        }
        else if (!ok)
        {
            emitEvent(Json{{"event", "error"}, {"ok", false},
                {"request_id", requestId}, {"id", requestId}, {"error", "handleAudioGeneration failed"}});
        }
        else
        {
            int64_t const totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - requestStart)
                                        .count();
            int32_t const totalFrames
                = response.numFramesPerSample.empty() ? 0 : response.numFramesPerSample[0];
            emitEvent(Json{{"event", "done"}, {"ok", true},
                {"request_id", requestId}, {"id", requestId},
                {"chunks_emitted", chunkIndex},
                {"total_frames", totalFrames},
                {"sample_rate", gSampleRate},
                {"elapsed_ms", totalMs}});
        }
    }
    catch (std::exception const& e)
    {
        if (vocodeThread.joinable())
        {
            {
                std::lock_guard<std::mutex> lk(vocodeMu);
                VocodeJob poison;
                poison.poison = true;
                vocodeQ.push_back(std::move(poison));
            }
            vocodeCv.notify_one();
            vocodeThread.join();
        }
        if (vocodeStreamCreated)
        {
            cudaStreamDestroy(vocodeStream);
            vocodeStreamCreated = false;
        }
        emitEvent(Json{{"event", "error"}, {"ok", false},
            {"request_id", requestId}, {"id", requestId}, {"error", e.what()}});
    }

    if (!requestId.empty())
    {
        unregisterCancel(requestId);
        unbindSession(requestId);
    }
    // Release the slot for reuse. Order matters: drop the id→slot mapping FIRST
    // (above, via unbindSession→pool->unbind), THEN clear inUse (pool->release),
    // so a freed slot is never reachable by a stale routing entry pointing at a
    // now-idle slot.
    gPool->release(slot.slotId);
}

//! Worker thread body: blocks on the slot's queue, runs each request to
//! completion (serially within the slot — one request at a time per slot).
void slotWorkerLoop(TtsSlot* slot, bool useStateful, bool useAsyncVocode)
{
    while (true)
    {
        WorkItem item;
        {
            std::unique_lock<std::mutex> lk(slot->queueMu);
            slot->queueCv.wait(lk, [&] { return !slot->queue.empty() || slot->shutdown.load(); });
            if (slot->shutdown.load() && slot->queue.empty())
            {
                return;
            }
            item = std::move(slot->queue.front());
            slot->queue.pop_front();
        }
        processRequest(*slot, item.request, useStateful, useAsyncVocode);
    }
}

//! Build the N-slot pool. Slot 0 deserializes the Talker + CodePredictor +
//! Code2Wav engines via the path ctors; slots 1..N-1 SHARE those read-only
//! ICudaEngines through the shared-engine ctors (D2), paying only per-slot
//! context/KV/workspace memory. Set EDGE_LLM_TTS_SHARED_ENGINE=0 to force
//! independent per-slot deserialization (memory A/B baseline). Each slot gets
//! its own Code2WavRunner (stateless), CUDA stream, and worker thread.
bool initSlotPool(Args const& args, bool useStateful, bool useAsyncVocode, int64_t& initMsOut)
{
    int32_t const n = std::max(1, gMaxSlots);
    gPool = std::make_unique<rt_slotpool::SlotPool<TtsSlot>>(n);
    auto& slots = gPool->slots();

    bool const enableCudaGraph = std::getenv("EDGE_LLM_TTS_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_TTS_CUDA_GRAPH")) != "0";
    bool const useSharedEngine = std::getenv("EDGE_LLM_TTS_SHARED_ENGINE") == nullptr
        || std::string(std::getenv("EDGE_LLM_TTS_SHARED_ENGINE")) != "0";

    auto const initStart = std::chrono::steady_clock::now();
    try
    {
        for (int32_t i = 0; i < n; ++i)
        {
            auto slot = std::make_unique<TtsSlot>();
            slot->slotId = i;
            slot->inUse.store(false);
            CUDA_CHECK(cudaStreamCreate(&slot->stream));

            if (i > 0 && useSharedEngine)
            {
                // Shared-engine ctor: reuse slot 0's deserialized (read-only) engines.
                // Per-slot execution contexts / KV caches / workspace stay private.
                Qwen3OmniTTSRuntime& owner = *slots[0]->runtime;
                slot->runtime = std::make_unique<Qwen3OmniTTSRuntime>(owner.getTalkerEngine(),
                    owner.getCodePredictorEngine(), args.talkerEngineDir, args.codePredictorEngineDir,
                    args.tokenizerDir, slot->stream);
                slot->code2wav = std::make_unique<Code2WavRunner>(
                    slots[0]->code2wav->getEnginePtr(), args.code2wavEngineDir, slot->stream);
            }
            else
            {
                slot->runtime = std::make_unique<Qwen3OmniTTSRuntime>(
                    args.talkerEngineDir, args.codePredictorEngineDir, args.tokenizerDir, slot->stream);

                // Per-slot STATELESS Code2Wav. Each owns its own engine+context+buffers,
                // so concurrent slots never contend. useStateful only affects `ready` meta.
                slot->code2wav = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, slot->stream);
            }

            if (enableCudaGraph && !slot->runtime->captureDecodingCUDAGraph(slot->stream))
            {
                std::cerr << "warning: failed to capture talker decoding CUDA graph for slot " << i << std::endl;
            }
            if (i == 0)
            {
                gSampleRate = slot->code2wav->getConfig().sampleRate;
            }
            slots.push_back(std::move(slot));
        }
    }
    catch (std::exception const& e)
    {
        emitEvent(Json{{"event", "error"}, {"ok", false}, {"error", std::string("init failed: ") + e.what()}});
        return false;
    }

    initMsOut = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - initStart).count();

    // Spawn one worker thread per slot AFTER all slots are constructed (so a
    // worker never observes a half-built pool).
    for (auto& slotPtr : gPool->slots())
    {
        slotPtr->worker = std::thread(slotWorkerLoop, slotPtr.get(), useStateful, useAsyncVocode);
    }
    return true;
}

//! Stop + join every worker thread, destroy CUDA streams, free runtimes.
void destroySlotPool()
{
    if (!gPool)
    {
        return;
    }
    for (auto& slotPtr : gPool->slots())
    {
        slotPtr->shutdown.store(true);
        slotPtr->queueCv.notify_all();
    }
    for (auto& slotPtr : gPool->slots())
    {
        if (slotPtr->worker.joinable())
        {
            slotPtr->worker.join();
        }
    }
    // Free slots in REVERSE order: slots 1..N-1 borrow slot 0's engines (shared-engine ctor),
    // so every borrower's contexts must be destroyed before the owning slot 0 releases the engines.
    auto& allSlots = gPool->slots();
    for (auto it = allSlots.rbegin(); it != allSlots.rend(); ++it)
    {
        auto& slotPtr = *it;
        slotPtr->code2wav.reset();
        slotPtr->runtime.reset();
        if (slotPtr->stream != nullptr)
        {
            cudaStreamDestroy(slotPtr->stream);
            slotPtr->stream = nullptr;
        }
    }
    gPool->clear();
    gPool.reset();
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    gLogger.setLevel(
        args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);

    auto pluginHandles = loadEdgellmPluginLib();

    auto envIsOne = [](char const* name) {
        auto* v = std::getenv(name);
        return v != nullptr && std::string(v) == "1";
    };
    bool const useStateful = envIsOne("EDGE_LLM_TTS_STATEFUL_CODE2WAV");
    bool const useAsyncVocode = envIsOne("EDGE_LLM_TTS_ASYNC_VOCODE");

    gMaxSlots = args.maxSlots;
    int64_t initMs = 0;
    if (!initSlotPool(args, useStateful, useAsyncVocode, initMs))
    {
        return EXIT_FAILURE;
    }

    emitEvent(Json{{"event", "ready"}, {"request_id", "__worker__"}, {"id", "__worker__"},
        {"init_ms", initMs}, {"max_slots", gMaxSlots},
        {"stateful_code2wav", useStateful}, {"async_vocode", useAsyncVocode}});

    // Single stdin reader thread: parse each JSON line and route it. Generation
    // requests are dispatched to a free slot's worker thread (non-blocking).
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;

        Json item;
        try
        {
            item = Json::parse(line);
        }
        catch (std::exception const& e)
        {
            emitEvent(Json{{"event", "error"}, {"ok", false}, {"error", std::string("invalid JSON: ") + e.what()}});
            continue;
        }

        // Cancel message: {"type":"cancel","id":"..."} — trips the per-id atomic
        // flag registered by the in-flight request; the runtime decode loop polls
        // it per frame, the stream ends with is_final=true, and the worker emits a
        // `cancelled` event. tripped=false when no in-flight request has that id.
        if (item.is_object() && item.value("type", "") == "cancel")
        {
            std::string const cid = item.value("id", "");
            bool const tripped = tripCancel(cid);
            emitEvent(Json{{"event", "cancel_ack"}, {"request_id", cid}, {"id", cid}, {"tripped", tripped}});
            continue;
        }

        // Generation request: claim a free slot and hand it to that slot's worker.
        std::string const requestId = item.value("id", "");
        int32_t const slotId = gPool->acquireFree();
        if (slotId < 0)
        {
            // Pool saturated: every slot has an in-flight request. Surface a
            // 4429-style structured error so OVS / the session-limiter backs off.
            Json ev = {{"event", "error"}, {"ok", false}, {"error", "pool_saturated"},
                {"status", 4429}, {"max_slots", gMaxSlots}};
            if (!requestId.empty())
            {
                ev["request_id"] = requestId;
                ev["id"] = requestId;
            }
            emitEvent(std::move(ev));
            continue;
        }

        if (!requestId.empty())
        {
            bindSession(requestId, slotId);
        }
        TtsSlot& slot = *gPool->get(slotId);
        {
            std::lock_guard<std::mutex> lk(slot.queueMu);
            slot.queue.push_back(WorkItem{std::move(item)});
        }
        slot.queueCv.notify_one();
    }

    // EOF on stdin: drain + join all worker threads, free GPU resources.
    destroySlotPool();
    return EXIT_SUCCESS;
}
