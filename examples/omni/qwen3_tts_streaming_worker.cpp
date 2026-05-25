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
 * This worker is intentionally simple (single in-flight request, single
 * Code2Wav runner) — it exists to validate the runtime streaming hook plumbing
 * end-to-end on Orin NX hardware. Concurrency / SlotPool plumbing is handled
 * by the heavier qwen3_tts_worker target.
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "multimodal/statefulCode2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"

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
    bool debug{false};
};

enum OptionId : int
{
    HELP = 1000,
    TALKER_ENGINE_DIR,
    CODE_PREDICTOR_ENGINE_DIR,
    CODE2WAV_ENGINE_DIR,
    TOKENIZER_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --talkerEngineDir=<path> --code2wavEngineDir=<path>"
              << " [--codePredictorEngineDir=<path>] [--tokenizerDir=<path>] [--debug]\n\n"
              << "Standalone Qwen3-TTS streaming worker.\n"
              << "Reads JSON request lines from stdin and emits JSON events to stdout.\n"
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

std::mutex coutMutex;

void emitEvent(Json payload)
{
    std::string const line = payload.dump();
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << line << std::endl;
}

// ===== cancel map =====

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
    msg.role = "user";
    Message::MessageContent content;
    content.type = "text";
    content.content = item.value("text", "");
    msg.contents.push_back(std::move(content));
    req.messages.push_back(std::move(msg));
    return req;
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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto envIsOne = [](char const* name) {
        auto* v = std::getenv(name);
        return v != nullptr && std::string(v) == "1";
    };
    bool const useStateful = envIsOne("EDGE_LLM_TTS_STATEFUL_CODE2WAV");
    bool const useAsyncVocode = envIsOne("EDGE_LLM_TTS_ASYNC_VOCODE");
    std::string statefulEngineDir = args.code2wavEngineDir;
    if (auto* env = std::getenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR"))
    {
        statefulEngineDir = env;
    }

    std::unique_ptr<Qwen3OmniTTSRuntime> ttsRuntime;
    std::unique_ptr<Code2WavRunner> code2wavRunner;
    std::unique_ptr<StatefulCode2WavRunner> statefulCode2wavRunner;

    auto const initStart = std::chrono::steady_clock::now();
    try
    {
        Qwen3OmniTTSRuntime::RuntimeOptions runtimeOptions;
        ttsRuntime = std::make_unique<Qwen3OmniTTSRuntime>(
            args.talkerEngineDir, args.codePredictorEngineDir, args.tokenizerDir, stream, runtimeOptions);
        if (useStateful)
        {
            statefulCode2wavRunner = std::make_unique<StatefulCode2WavRunner>(statefulEngineDir, stream);
        }
        else
        {
            code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
        }
        if (std::getenv("EDGE_LLM_TTS_CUDA_GRAPH") == nullptr
            || std::string(std::getenv("EDGE_LLM_TTS_CUDA_GRAPH")) != "0")
        {
            if (!ttsRuntime->captureDecodingCUDAGraph(stream))
            {
                std::cerr << "warning: failed to capture talker decoding CUDA graph" << std::endl;
            }
        }
    }
    catch (std::exception const& e)
    {
        emitEvent(Json{{"event", "error"}, {"ok", false}, {"error", std::string("init failed: ") + e.what()}});
        return EXIT_FAILURE;
    }

    auto const initEnd = std::chrono::steady_clock::now();
    int64_t const initMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(initEnd - initStart).count();
    emitEvent(Json{{"event", "ready"}, {"request_id", "__worker__"}, {"id", "__worker__"},
        {"init_ms", initMs}, {"stateful_code2wav", useStateful}, {"async_vocode", useAsyncVocode}});

    int32_t const sampleRate
        = useStateful ? statefulCode2wavRunner->getConfig().sampleRate : code2wavRunner->getConfig().sampleRate;

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

        std::string const requestId = item.value("id", "");
        std::atomic<bool> cancelled{false};
        if (!requestId.empty())
        {
            registerCancel(requestId, &cancelled);
        }

        // Streaming bookkeeping (per-request, thread-local since we run one request at a time).
        int32_t chunkIndex = 0;
        auto const requestStart = std::chrono::steady_clock::now();

        // Async vocode worker plumbing (per-request, only constructed when enabled).
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
            // first_chunk_frames drives the runtime's codecChunkFrames so the
            // first emitted RVQ chunk arrives ASAP for low TTFA. chunk_frames
            // is accepted for forward-compat with the adaptive-growth design
            // (see codex spec §A); current implementation = "method ii" where
            // codecChunkFrames is fixed at first_chunk_frames.
            int32_t const firstChunkFrames = std::max(1, item.value("first_chunk_frames",
                                                                    item.value("chunk_frames", 8)));
            bool const streaming = item.value("stream", true);
            std::string const chunkFormat = item.value("chunk_format", "pcm_s16le");
            std::string const chunkTransport = item.value("chunk_transport", "base64");

            // Reset stateful Code2Wav state at request start so each request is independent.
            if (useStateful && statefulCode2wavRunner)
            {
                statefulCode2wavRunner->reset(stream);
            }

            // Vocode lambda — runs on whichever stream is active (sync = main stream,
            // async = dedicated vocodeStream).
            auto runVocode = [&](std::vector<std::vector<int32_t>> const& chunkRvqCodes,
                                  bool isFinal, cudaStream_t s, std::vector<float>& samplesOut) -> bool {
                rt::audioUtils::AudioData audioOutput;
                auto const transposed = transposeFrames(chunkRvqCodes);
                bool ok = false;
                if (useStateful)
                {
                    ok = statefulCode2wavRunner->generateChunk(transposed, isFinal, audioOutput, s);
                }
                else
                {
                    ok = code2wavRunner->generateWaveform(transposed, audioOutput, s);
                }
                if (!ok)
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
                    {"sample_rate", sampleRate},
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
                            // drain remaining jobs but keep loop alive until poison.
                        }
                    }
                });
            }

            if (streaming)
            {
                request.codecChunkFrames = firstChunkFrames;
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
            bool const ok = ttsRuntime->handleAudioGeneration(request, response, stream);

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
                    {"sample_rate", sampleRate},
                    {"elapsed_ms", totalMs}});
            }
        }
        catch (std::exception const& e)
        {
            // Best-effort drain of async vocode thread on exception so we don't
            // leak a detached worker into the next request.
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
        }
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
