/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * qwen3_tts_streaming_worker
 * --------------------------
 * Standalone TTS streaming worker for Qwen3-TTS.
 *
 * - Reads JSON request lines on stdin.
 * - For each request, invokes the runtime's streaming-aware
 *   handleAudioGeneration() with a per-chunk callback that vocodes the
 *   accumulated RVQ codes through Code2Wav and emits a base64 PCM_S16LE chunk
 *   event on stdout.
 * - Supports per-id mid-stream cancel via {"type":"cancel","id":"..."} lines.
 *
 * Concurrency (TTS slot-pool, D2-2)
 * ---------------------------------
 * Unlike the ASR worker (single stdin poll thread that interleaves short
 * chunk-level work), TTS requests are LONG-RUNNING (seconds each). Serializing
 * them on one thread would force the second concurrent request to wait for the
 * first to finish. So this worker runs N WORKER THREADS, each bound to ONE slot:
 *
 *   * TtsSlot   = { Qwen3OmniTTSRuntime runtime, Code2WavRunner code2wav,
 *                   cudaStream_t stream, std::thread worker, atomic<bool> inUse }.
 *   * Slot 0 deserializes the Talker + CodePredictor engines once via the
 *     path-based constructor; slots 1..N-1 SHARE those ICudaEngines via the
 *     D2-1 shared-engine constructor (weight memory paid once; each slot keeps
 *     its own IExecutionContexts, KV/cache state, workspace tensors, CUDA
 *     stream). Each slot ALSO owns its own (stateless) Code2WavRunner so the
 *     vocoder contexts never contend — this deliberately avoids the
 *     StatefulCode2WavRunner concurrent-cudaMemset hazard at N>1.
 *   * The stdin reader thread (main) parses each JSON line and routes it: a
 *     generation request is enqueued onto the next free slot's queue (id→slot
 *     binding); a cancel line trips the per-id cancel flag.
 *   * A runtime instance is NEVER invoked from more than one thread: each slot
 *     has exactly one worker thread, and a slot is bound to at most one
 *     in-flight request at a time.
 *
 * Backward compatibility: --max_slots=1 yields a single slot + single worker
 * thread == the original single-in-flight behavior. The OVS protocol
 * (ready/chunk/done/cancelled/error events, request_id+id) is unchanged.
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "multimodal/statefulCode2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include "runtime/slotPool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    int32_t maxSlots{1}; //!< TTS slot-pool size (D2 concurrency). Default 1 == original single-instance behavior.
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
              << "Standalone Qwen3-TTS streaming worker.\n"
              << "Reads JSON request lines from stdin and emits JSON events to stdout.\n"
              << "--max_slots=<N> sets the TTS slot-pool size (default 1; D2 concurrency). Each slot is a\n"
              << "  full TTS runtime + Code2Wav + CUDA stream + worker thread, sharing the Talker/CodePredictor\n"
              << "  engine weights with slot 0.\n"
              << "Request schema:\n"
              << "  {\"id\":\"...\",\"text\":\"...\",\"speaker\":\"Vivian\",\n"
              << "   \"stream\":true,\"chunk_frames\":13,\n"
              << "   \"chunk_format\":\"pcm_s16le\",\"chunk_transport\":\"base64\"}\n"
              << "Cancel:\n"
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
    // D2 slot-pool size env fallback (matches the OVS EDGE_LLM_TTS_MAX_CONCURRENT
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

std::vector<float> audioToFloatSamples(rt::audioUtils::AudioData const& audio)
{
    if (!audio.waveform || audio.waveform->isEmpty())
    {
        return {};
    }
    int64_t const samples = audio.waveform->getShape()[1];
    float const* data = static_cast<float const*>(audio.waveform->rawPointer());
    return std::vector<float>(data, data + samples);
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
// and the request's shouldCancel lambda polls it. cancelMapMu guards the map;
// the worker thread registers its flag when it starts a request and
// unregisters at request end.

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
    req.codecEosLogitOffset = item.value("codec_eos_logit_offset", 0.0f);
    req.predictorTemperature = item.value("predictor_temperature", 0.0f);
    req.predictorTopK = item.value("predictor_top_k", 0);
    req.predictorTopP = item.value("predictor_top_p", 0.0f);
    req.language = item.value("language", "");
    req.speakerName = item.value("speaker", "");
    req.speakerId = item.value("speaker_id", -1);

    Message msg;
    // Qwen3-TTS talker prefill expects the assistant role prefix
    // ([<|im_start|>, assistant(77091), \n]) at input_ids[:3], not the user
    // prefix (token 872). The text to synthesize is the assistant's content.
    msg.role = "assistant";
    Message::MessageContent content;
    content.type = "text";
    content.content = item.value("text", "");
    msg.contents.push_back(std::move(content));
    req.messages.push_back(std::move(msg));
    return req;
}

// ===========================================================================
// TTS slot-pool (D2-2).
//
// Each TtsSlot is a self-contained, single-threaded TTS lane:
//   * runtime    — independent Qwen3OmniTTSRuntime. Slot 0 deserializes the
//                  Talker + CodePredictor engines; slots 1..N-1 share those
//                  ICudaEngines (D2-1 shared-engine ctor). Each slot has its own
//                  IExecutionContexts, KV/cache state, workspace tensors.
//   * code2wav   — independent (STATELESS) Code2WavRunner. Each slot owns its
//                  own vocoder engine+context+buffers, so concurrent slots
//                  never contend on Code2Wav state. We deliberately do NOT use
//                  StatefulCode2WavRunner here: its per-request reset() does a
//                  cudaMemset that is unsafe under concurrent slots (known N>1
//                  hazard). Stateless generateWaveform() is pure given codes.
//   * stream     — per-slot CUDA stream so independent slots can overlap on GPU.
//   * worker     — the ONE OS thread that drives this slot's runtime. A runtime
//                  instance is therefore never touched by >1 thread.
//   * cancel     — per-slot atomic flag, registered in cancelMap under the
//                  in-flight request id while a request runs.
//   * queue/cv   — single-element handoff from the stdin reader to the worker.
//                  inUse marks the slot busy from enqueue until the worker
//                  finishes the request (so the reader won't double-assign).
// ===========================================================================

struct WorkItem
{
    Json request; //!< Parsed generation request JSON.
};

struct TtsSlot
{
    int32_t slotId{-1};
    std::atomic<bool> inUse{false};   //!< True from enqueue until request completion.
    std::atomic<bool> shutdown{false};//!< Set at teardown so the worker thread exits.

    std::unique_ptr<Qwen3OmniTTSRuntime> runtime;
    std::unique_ptr<Code2WavRunner> code2wav; //!< Per-slot stateless vocoder.
    cudaStream_t stream{nullptr};

    // Single-slot handoff queue (reader thread -> this slot's worker thread).
    std::mutex queueMu;
    std::condition_variable queueCv;
    std::deque<WorkItem> queue;

    // Per-slot cancel flag for the request currently running on this slot.
    std::atomic<bool> cancelled{false};

    std::thread worker;

    TtsSlot() = default;
    // Non-copyable / non-movable: holds std::atomic + CUDA stream + thread.
    TtsSlot(TtsSlot const&) = delete;
    TtsSlot& operator=(TtsSlot const&) = delete;
};

int32_t gMaxSlots{1};

// D-2.5: slot vector ownership, capacity, id→slot routing map (+its mutex),
// CAS acquire, bind/unbind/lookup, and saturation are now provided by the
// generic SlotPool<TtsSlot> template. Slot construction (runtime/Code2Wav/
// stream/worker-thread), cancel map, queue, and JSON emit stay in this worker.
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

[[maybe_unused]] int32_t lookupSlot(std::string const& id)
{
    return gPool->lookup(id);
}

// ===== sample rate (read once at init; identical across slots) =====
int32_t gSampleRate{24000};

// ---------------------------------------------------------------------------
// processRequest — run ONE generation request on a given slot.
//
// This is exactly the original main()-loop per-request body, factored out and
// parameterized by slot so each worker thread runs it on its own runtime +
// code2wav + stream + cancel flag. The async-vocode env path is preserved.
// ---------------------------------------------------------------------------
void processRequest(TtsSlot& slot, Json const& item, bool useStateful, bool useAsyncVocode,
    StatefulCode2WavRunner* /*statefulShared (unused; per-slot path below)*/)
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
        int32_t const firstChunkFrames = std::max(1, item.value("first_chunk_frames", 8));
        int32_t const subsequentChunkFrames = std::max(0, item.value("chunk_frames", 10));
        bool const streaming = item.value("stream", true);
        std::string const chunkFormat = item.value("chunk_format", "pcm_s16le");
        std::string const chunkTransport = item.value("chunk_transport", "base64");

        // Vocode lambda — uses THIS slot's code2wav. useStateful is not used at
        // N>=1 in the slot-pool path (per-slot stateless vocoder avoids the
        // StatefulCode2WavRunner concurrent-cudaMemset hazard). The flag is kept
        // for the emitted `ready` metadata only.
        auto runVocode = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes,
                              bool /*isFinal*/, cudaStream_t s, std::vector<float>& samplesOut) -> bool {
            rt::audioUtils::AudioData audioOutput;
            auto const transposed = transposeFrames(chunkRvqCodes);
            if (!code2wavRunner->generateWaveform(transposed, audioOutput, s))
            {
                return false;
            }
            samplesOut = audioToFloatSamples(audioOutput);
            return true;
        };

        auto emitChunk = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal,
                              int32_t idx, std::chrono::steady_clock::time_point startTs,
                              cudaStream_t vocStream) {
            int32_t const frames = static_cast<int32_t>(chunkRvqCodes.size());
            auto const c2wStart = std::chrono::steady_clock::now();
            std::vector<float> samples;
            if (frames > 0)
            {
                if (!runVocode(chunkRvqCodes, isFinal, vocStream, samples))
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

        if (streaming)
        {
            request.codecChunkFrames = firstChunkFrames;
            request.subsequentChunkFrames = subsequentChunkFrames;
            request.shouldCancel = [&cancelled]() { return cancelled.load(std::memory_order_acquire); };
            if (useAsyncVocode)
            {
                request.onAudioChunkReady = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes,
                                                 int32_t /*batchIdx*/, bool isFinal) {
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
                request.onAudioChunkReady = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes,
                                                 int32_t /*batchIdx*/, bool isFinal) {
                    emitChunk(chunkRvqCodes, isFinal, chunkIndex++,
                        std::chrono::steady_clock::now(), stream);
                };
            }
        }

        Qwen3OmniTTSRuntime::TalkerGenerationResponse response;
        bool const ok = ttsRuntime.handleAudioGeneration(request, response, stream);

        // Drain async vocode thread before emitting done/cancelled/error.
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
    // Release the slot for reuse. Order matters (RISK POINT 1): drop the id→slot
    // mapping FIRST (above, via unbindSession→pool->unbind), THEN clear inUse
    // (pool->release), so a freed slot is never reachable by a stale routing
    // entry pointing at a now-idle slot. release() clears inUse ONLY — the
    // worker keeps its own cancel/queue teardown above.
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
        processRequest(*slot, item.request, useStateful, useAsyncVocode, nullptr);
    }
}

//! Build the N-slot pool sharing the Talker + CodePredictor engines. Slot 0
//! deserializes via the path-based ctor; slots 1..N-1 share its engines via the
//! D2-1 shared-engine ctor. Each slot gets its own Code2WavRunner + CUDA stream
//! + worker thread. Returns false on init failure.
bool initSlotPool(Args const& args, bool useStateful, bool useAsyncVocode, int64_t& initMsOut)
{
    int32_t const n = std::max(1, gMaxSlots);
    gPool = std::make_unique<rt_slotpool::SlotPool<TtsSlot>>(n);
    auto& slots = gPool->slots();

    bool const enableCudaGraph = std::getenv("EDGE_LLM_TTS_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_TTS_CUDA_GRAPH")) != "0";

    auto const initStart = std::chrono::steady_clock::now();
    try
    {
        for (int32_t i = 0; i < n; ++i)
        {
            auto slot = std::make_unique<TtsSlot>();
            slot->slotId = i;
            slot->inUse.store(false);
            CUDA_CHECK(cudaStreamCreate(&slot->stream));

            if (i == 0)
            {
                // Slot 0: path-based ctor deserializes the Talker + CodePredictor
                // engines once and loads weights / tokenizer / embeddings.
                slot->runtime = std::make_unique<Qwen3OmniTTSRuntime>(
                    args.talkerEngineDir, args.codePredictorEngineDir, args.tokenizerDir, slot->stream);
            }
            else
            {
                // Slots 1..N-1: SHARE slot 0's deserialized Talker + CodePredictor
                // ICudaEngines (D2-1). Each still allocates its own execution
                // contexts, KV/cache state, workspace tensors, tokenizer, embeds.
                std::shared_ptr<nvinfer1::ICudaEngine> sharedTalker = slots[0]->runtime->getTalkerEngine();
                std::shared_ptr<nvinfer1::ICudaEngine> sharedCp = slots[0]->runtime->getCodePredictorEngine();
                slot->runtime = std::make_unique<Qwen3OmniTTSRuntime>(
                    sharedTalker, sharedCp, args.talkerEngineDir, args.codePredictorEngineDir,
                    args.tokenizerDir, slot->stream);
            }

            // Per-slot STATELESS Code2Wav. Each owns its own engine+context+buffers,
            // so concurrent slots never contend (avoids the StatefulCode2WavRunner
            // concurrent-cudaMemset hazard). useStateful only affects `ready` meta.
            slot->code2wav = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, slot->stream);

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
    // worker never observes a half-built pool). (void)useStateful: the slot-pool
    // path always uses per-slot stateless Code2Wav; the flag is kept for `ready`.
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
    // Signal shutdown + wake every worker thread, then join. The slot vector is
    // only mutated at init/teardown (no concurrent acquire here), so iterating
    // gPool->slots() directly is safe.
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
    // Worker keeps ownership of runtime/stream teardown (OUT of the template).
    for (auto& slotPtr : gPool->slots())
    {
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
    // requests are dispatched to a free slot's worker thread (non-blocking — the
    // reader keeps reading so cancel lines for in-flight requests are honored
    // promptly). Cancel lines trip the per-id flag synchronously.
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

        // Cancel message: {"type":"cancel","id":"..."}
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
