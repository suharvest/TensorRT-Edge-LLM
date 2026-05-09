/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audioWriter.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{
struct Args
{
    std::string talkerEngineDir;
    std::string qwen3TtsTalkerBackend{"auto"};
    std::string qwen3TtsTalkerEngine;
    std::string codePredictorEngineDir;
    std::string codePredictorBackend{"auto"};
    std::string qwen3TtsTextProjection{"auto"};
    bool qwen3TtsPromptKvCache{false};
    std::string code2wavEngineDir;
    std::string tokenizerDir;
    bool debug{false};
};

enum OptionId : int
{
    HELP = 1000,
    TALKER_ENGINE_DIR,
    QWEN3_TTS_TALKER_BACKEND,
    QWEN3_TTS_TALKER_ENGINE,
    CODE_PREDICTOR_ENGINE_DIR,
    CODE_PREDICTOR_BACKEND,
    QWEN3_TTS_TEXT_PROJECTION,
    QWEN3_TTS_PROMPT_KV_CACHE,
    CODE2WAV_ENGINE_DIR,
    TOKENIZER_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --talkerEngineDir=<path> --code2wavEngineDir=<path>"
              << " [--qwen3TtsTalkerBackend=<auto|qwen3_tts_explicit_kv|generic>]"
              << " [--qwen3TtsTalkerEngine=<path>]"
              << " [--codePredictorEngineDir=<path>] [--codePredictorBackend=<auto|qwen3_tts_native|generic>]"
              << " [--qwen3TtsTextProjection=<auto|host_fp32|device>]"
              << " [--qwen3TtsPromptKvCache=<0|1>]"
              << " [--tokenizerDir=<path>] [--debug]\n\n"
              << "Reads JSON lines from stdin and writes JSON events to stdout.\n"
              << "Streaming requests set stream=true and stream_only=true.\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"talkerEngineDir", required_argument, 0, TALKER_ENGINE_DIR},
        {"qwen3TtsTalkerBackend", required_argument, 0, QWEN3_TTS_TALKER_BACKEND},
        {"qwen3TtsTalkerEngine", required_argument, 0, QWEN3_TTS_TALKER_ENGINE},
        {"codePredictorEngineDir", required_argument, 0, CODE_PREDICTOR_ENGINE_DIR},
        {"codePredictorBackend", required_argument, 0, CODE_PREDICTOR_BACKEND},
        {"qwen3TtsTextProjection", required_argument, 0, QWEN3_TTS_TEXT_PROJECTION},
        {"qwen3TtsPromptKvCache", required_argument, 0, QWEN3_TTS_PROMPT_KV_CACHE},
        {"code2wavEngineDir", required_argument, 0, CODE2WAV_ENGINE_DIR},
        {"tokenizerDir", required_argument, 0, TOKENIZER_DIR}, {"debug", no_argument, 0, DEBUG}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case QWEN3_TTS_TALKER_BACKEND: args.qwen3TtsTalkerBackend = optarg; break;
        case QWEN3_TTS_TALKER_ENGINE: args.qwen3TtsTalkerEngine = optarg; break;
        case CODE_PREDICTOR_ENGINE_DIR: args.codePredictorEngineDir = optarg; break;
        case CODE_PREDICTOR_BACKEND: args.codePredictorBackend = optarg; break;
        case QWEN3_TTS_TEXT_PROJECTION: args.qwen3TtsTextProjection = optarg; break;
        case QWEN3_TTS_PROMPT_KV_CACHE:
            args.qwen3TtsPromptKvCache = std::string(optarg) == "1" || std::string(optarg) == "true"
                || std::string(optarg) == "yes" || std::string(optarg) == "on";
            break;
        case CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case TOKENIZER_DIR: args.tokenizerDir = optarg; break;
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }

    if (args.codePredictorEngineDir.empty() && !args.talkerEngineDir.empty())
    {
        args.codePredictorEngineDir = (std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor").string();
    }
    return !args.talkerEngineDir.empty() && !args.codePredictorEngineDir.empty() && !args.code2wavEngineDir.empty();
}

bool parseCodePredictorBackend(
    std::string const& value, Qwen3OmniTTSRuntime::CodePredictorBackend& backend)
{
    if (value == "auto")
    {
        backend = Qwen3OmniTTSRuntime::CodePredictorBackend::kAuto;
        return true;
    }
    if (value == "qwen3_tts_native" || value == "native")
    {
        backend = Qwen3OmniTTSRuntime::CodePredictorBackend::kQwen3TTSNative;
        return true;
    }
    if (value == "generic" || value == "generic_llm_runner")
    {
        backend = Qwen3OmniTTSRuntime::CodePredictorBackend::kGeneric;
        return true;
    }
    return false;
}

bool parseTalkerBackend(std::string const& value, Qwen3OmniTTSRuntime::TalkerBackend& backend)
{
    if (value == "auto")
    {
        backend = Qwen3OmniTTSRuntime::TalkerBackend::kAuto;
        return true;
    }
    if (value == "qwen3_tts_explicit_kv" || value == "explicit_kv" || value == "direct")
    {
        backend = Qwen3OmniTTSRuntime::TalkerBackend::kQwen3TTSExplicitKV;
        return true;
    }
    if (value == "generic" || value == "generic_llm_runner" || value == "official")
    {
        backend = Qwen3OmniTTSRuntime::TalkerBackend::kGeneric;
        return true;
    }
    return false;
}

bool parseTextProjectionMode(std::string const& value, Qwen3OmniTTSRuntime::TextProjectionMode& mode)
{
    if (value == "auto")
    {
        mode = Qwen3OmniTTSRuntime::TextProjectionMode::kAuto;
        return true;
    }
    if (value == "host_fp32")
    {
        mode = Qwen3OmniTTSRuntime::TextProjectionMode::kHostFP32;
        return true;
    }
    if (value == "device")
    {
        mode = Qwen3OmniTTSRuntime::TextProjectionMode::kDevice;
        return true;
    }
    return false;
}

long readProcValueKb(char const* path, char const* wantedKey)
{
    std::ifstream file(path);
    std::string key;
    long value = -1;
    std::string unit;
    while (file >> key >> value >> unit)
    {
        if (key == wantedKey)
        {
            return value;
        }
    }
    return -1;
}

long kbToMb(long kb)
{
    return kb < 0 ? -1 : kb / 1024;
}

void logMemTag(char const* tag)
{
    long const memAvailableKb = readProcValueKb("/proc/meminfo", "MemAvailable:");
    long const memFreeKb = readProcValueKb("/proc/meminfo", "MemFree:");
    long const swapFreeKb = readProcValueKb("/proc/meminfo", "SwapFree:");
    long const rssKb = readProcValueKb("/proc/self/status", "VmRSS:");
    std::cerr << "[JV_MEM] tag=" << tag << " mem_available_mb=" << kbToMb(memAvailableKb)
              << " mem_free_mb=" << kbToMb(memFreeKb) << " swap_free_mb=" << kbToMb(swapFreeKb)
              << " rss_mb=" << kbToMb(rssKb) << std::endl;
}

bool envIsOne(char const* name)
{
    char const* value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

int32_t envIntOr(char const* name, int32_t fallback)
{
    char const* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }
    try
    {
        return std::stoi(value);
    }
    catch (...)
    {
        LOG_WARNING("Ignoring invalid integer value for %s=%s", name, value);
        return fallback;
    }
}

std::vector<std::vector<int32_t>> transposeFrameWindow(
    std::vector<std::vector<int32_t>> const& frames, size_t begin, size_t end)
{
    if (begin >= end || end > frames.size())
    {
        return {};
    }
    size_t const numFrames = end - begin;
    size_t const numLayers = frames[begin].size();
    std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
    for (size_t f = 0; f < numFrames; ++f)
    {
        for (size_t l = 0; l < numLayers; ++l)
        {
            transposed[l][f] = frames[begin + f][l];
        }
    }
    return transposed;
}

std::vector<std::vector<int32_t>> transposeCodes(std::vector<std::vector<int32_t>> const& frames)
{
    return transposeFrameWindow(frames, 0, frames.size());
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

std::vector<int16_t> floatSamplesToPcm16(std::vector<float> const& samples)
{
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        float const clipped = std::clamp(samples[i], -1.0f, 1.0f);
        pcm[i] = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
    }
    return pcm;
}

std::string base64Encode(uint8_t const* data, size_t len)
{
    static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

bool savePcm16(std::string const& filepath, std::vector<int16_t> const& samples)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
    {
        return false;
    }
    file.write(reinterpret_cast<char const*>(samples.data()),
        static_cast<std::streamsize>(samples.size() * sizeof(int16_t)));
    return static_cast<bool>(file);
}

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

    Message msg;
    msg.role = "user";
    Message::MessageContent content;
    content.type = "text";
    content.content = item.value("text", "");
    msg.contents.push_back(std::move(content));
    req.messages.push_back(std::move(msg));
    return req;
}

std::vector<float> synthesizeWindow(
    Code2WavRunner& code2wavRunner, std::vector<std::vector<int32_t>> const& windowCodes, int32_t skipContextFrames,
    cudaStream_t stream)
{
    rt::audioUtils::AudioData audioOutput;
    if (!code2wavRunner.generateWaveform(windowCodes, audioOutput, stream))
    {
        throw std::runtime_error("Code2Wav chunk generation failed");
    }
    auto samples = audioToFloatSamples(audioOutput);
    int64_t const skipSamples = static_cast<int64_t>(std::max(0, skipContextFrames)) * code2wavRunner.getConfig().upsampleRate;
    if (skipSamples <= 0)
    {
        return samples;
    }
    if (skipSamples >= static_cast<int64_t>(samples.size()))
    {
        return {};
    }
    return std::vector<float>(samples.begin() + skipSamples, samples.end());
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

    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    logMemTag("worker_entry_before_plugin");
    auto pluginHandles = loadEdgellmPluginLib();
    logMemTag("worker_after_plugin");

    cudaStream_t stream;
    logMemTag("worker_before_cuda_stream");
    CUDA_CHECK(cudaStreamCreate(&stream));
    logMemTag("worker_after_cuda_stream");

    std::unique_ptr<Qwen3OmniTTSRuntime> ttsRuntime;
    std::unique_ptr<Code2WavRunner> code2wavRunner;
    cudaStream_t asyncCode2WavStream{};
    std::unique_ptr<Code2WavRunner> asyncCode2wavRunner;
    bool const lazyCode2Wav = envIsOne("EDGE_LLM_TTS_LAZY_CODE2WAV");
    int32_t const code2WavContextFrameCap = envIntOr("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", -1);
    auto getAsyncCode2WavRunner = [&]() -> Code2WavRunner& {
        if (!asyncCode2wavRunner)
        {
            logMemTag("worker_before_async_code2wav_stream");
            CUDA_CHECK(cudaStreamCreate(&asyncCode2WavStream));
            logMemTag("worker_after_async_code2wav_stream");
            logMemTag("worker_before_async_code2wav");
            asyncCode2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, asyncCode2WavStream);
            logMemTag("worker_after_async_code2wav");
        }
        return *asyncCode2wavRunner;
    };
    auto const initStart = std::chrono::steady_clock::now();
    try
    {
        Qwen3OmniTTSRuntime::RuntimeOptions runtimeOptions;
        if (!parseTalkerBackend(args.qwen3TtsTalkerBackend, runtimeOptions.talkerBackend))
        {
            throw std::runtime_error("Invalid --qwen3TtsTalkerBackend: " + args.qwen3TtsTalkerBackend);
        }
        runtimeOptions.qwen3TtsTalkerEnginePath = args.qwen3TtsTalkerEngine;
        if (!parseCodePredictorBackend(args.codePredictorBackend, runtimeOptions.codePredictorBackend))
        {
            throw std::runtime_error("Invalid --codePredictorBackend: " + args.codePredictorBackend);
        }
        if (!parseTextProjectionMode(args.qwen3TtsTextProjection, runtimeOptions.textProjectionMode))
        {
            throw std::runtime_error("Invalid --qwen3TtsTextProjection: " + args.qwen3TtsTextProjection);
        }
        runtimeOptions.qwen3TtsPromptKvCache = args.qwen3TtsPromptKvCache;
        logMemTag("worker_before_tts_runtime");
        ttsRuntime = std::make_unique<Qwen3OmniTTSRuntime>(
            args.talkerEngineDir, args.codePredictorEngineDir, args.tokenizerDir, stream, runtimeOptions);
        logMemTag("worker_after_tts_runtime");
        if (lazyCode2Wav)
        {
            logMemTag("worker_skip_code2wav_lazy");
        }
        else
        {
            logMemTag("worker_before_code2wav");
            code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
            logMemTag("worker_after_code2wav");
        }
        if (std::getenv("EDGE_LLM_TTS_CUDA_GRAPH") == nullptr
            || std::string(std::getenv("EDGE_LLM_TTS_CUDA_GRAPH")) != "0")
        {
            logMemTag("worker_before_cuda_graph");
            if (!ttsRuntime->captureDecodingCUDAGraph(stream))
            {
                LOG_WARNING("CUDA graph capture failed for TTS worker, proceeding without.");
            }
            logMemTag("worker_after_cuda_graph");
        }
        else
        {
            logMemTag("worker_skip_cuda_graph");
        }
    }
    catch (std::exception const& e)
    {
        logMemTag("worker_init_error");
        std::cerr << e.what() << std::endl;
        CUDA_CHECK(cudaStreamDestroy(stream));
        if (asyncCode2WavStream)
        {
            CUDA_CHECK(cudaStreamDestroy(asyncCode2WavStream));
        }
        return EXIT_FAILURE;
    }

    double const initMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    logMemTag("worker_before_ready");
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}}.dump() << std::endl;
    logMemTag("worker_after_ready");

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }

        Json response;
        auto const requestStart = std::chrono::steady_clock::now();
        try
        {
            Json item = Json::parse(line);
            std::string const id = item.value("id", "");
            bool const streamOutput = item.value("stream", false);
            bool const streamOnly = item.value("stream_only", false);
            bool const asyncCode2Wav = item.value("async_code2wav", false);
            int32_t const firstChunkFrames = std::max(1, item.value("first_chunk_frames", 25));
            int32_t const chunkFrames = std::max(1, item.value("chunk_frames", 25));
            bool const adaptiveChunks = item.value("adaptive_chunks", false);
            int32_t const maxChunkFrames = std::max(chunkFrames, item.value("max_chunk_frames", chunkFrames));
            int32_t const chunkGrowthFrames = std::max(0, item.value("chunk_growth_frames", 0));
            std::string const chunkFormat = item.value("chunk_format", "pcm_s16le");
            std::string const chunkTransport = item.value("chunk_transport", "base64");
            std::string outputFile = item.value("output_file", "/tmp/qwen3_tts_worker_" + id + ".wav");
            if (streamOnly && !streamOutput)
            {
                throw std::runtime_error("stream_only requires stream=true");
            }
            if (chunkFormat != "pcm_s16le")
            {
                throw std::runtime_error("Only pcm_s16le streaming chunks are supported");
            }
            if (chunkTransport != "base64" && chunkTransport != "file")
            {
                throw std::runtime_error("Unsupported chunk_transport: " + chunkTransport);
            }

            auto request = buildRequest(item);
            Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResponse;
            std::vector<std::vector<int32_t>> streamedFrames;
            int32_t lastEmittedFrames = 0;
            int32_t nextChunkAt = firstChunkFrames;
            int32_t currentChunkFrames = chunkFrames;
            int32_t chunkIndex = 0;
            int64_t streamedSamples = 0;
            double streamedCode2WavMs = 0.0;
            int64_t code2wavInputFrames = 0;
            int64_t code2wavContextFrames = 0;
            std::chrono::steady_clock::time_point firstChunkAt{};

            auto scheduleNextChunk = [&]() {
                nextChunkAt = lastEmittedFrames + currentChunkFrames;
                if (adaptiveChunks && chunkGrowthFrames > 0)
                {
                    currentChunkFrames = std::min(maxChunkFrames, currentChunkFrames + chunkGrowthFrames);
                }
            };

            auto writeChunk = [&](int32_t outputChunkIndex, bool isFinal, int32_t totalFrames,
                                  std::vector<int16_t> const& pcm, double code2wavMs,
                                  std::chrono::steady_clock::time_point chunkEnd, int32_t sampleRate) {
                Json chunk = Json{{"id", id},
                    {"event", "chunk"},
                    {"ok", true},
                    {"chunk_index", outputChunkIndex},
                    {"chunk_format", chunkFormat},
                    {"chunk_transport", chunkTransport},
                    {"frames", totalFrames},
                    {"samples", pcm.size()},
                    {"sample_rate", sampleRate},
                    {"is_final", isFinal},
                    {"code2wav_ms", code2wavMs},
                    {"elapsed_ms", std::chrono::duration<double, std::milli>(chunkEnd - requestStart).count()}};
                if (chunkTransport == "base64")
                {
                    chunk["audio_b64"] = base64Encode(
                        reinterpret_cast<uint8_t const*>(pcm.data()), pcm.size() * sizeof(int16_t));
                }
                else
                {
                    std::filesystem::path chunkPath(outputFile);
                    chunkPath.replace_filename(
                        chunkPath.stem().string() + ".chunk" + std::to_string(outputChunkIndex) + ".pcm");
                    if (!savePcm16(chunkPath.string(), pcm))
                    {
                        throw std::runtime_error("Failed to save chunk PCM: " + chunkPath.string());
                    }
                    chunk["chunk_file"] = chunkPath.string();
                }
                std::cout << chunk.dump() << std::endl;
            };

            auto emitChunk = [&](bool isFinal) {
                int32_t const totalFrames = static_cast<int32_t>(streamedFrames.size());
                if (totalFrames <= lastEmittedFrames)
                {
                    return;
                }

                if (!code2wavRunner)
                {
                    logMemTag("worker_before_lazy_code2wav");
                    code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
                    logMemTag("worker_after_lazy_code2wav");
                }
                int32_t const naturalLeftContext = code2wavRunner->getConfig().leftContextSize;
                int32_t const leftContext = code2WavContextFrameCap >= 0
                    ? std::min(naturalLeftContext, code2WavContextFrameCap)
                    : naturalLeftContext;
                int32_t const windowStart = std::max(0, lastEmittedFrames - leftContext);
                int32_t const skipContextFrames = lastEmittedFrames - windowStart;
                auto const windowCodes = transposeFrameWindow(
                    streamedFrames, static_cast<size_t>(windowStart), static_cast<size_t>(totalFrames));

                auto const chunkStart = std::chrono::steady_clock::now();
                logMemTag(isFinal ? "worker_before_code2wav_final_chunk" : "worker_before_code2wav_chunk");
                auto samples = synthesizeWindow(*code2wavRunner, windowCodes, skipContextFrames, stream);
                auto const chunkEnd = std::chrono::steady_clock::now();
                logMemTag(isFinal ? "worker_after_code2wav_final_chunk" : "worker_after_code2wav_chunk");
                if (samples.empty())
                {
                    lastEmittedFrames = totalFrames;
                    scheduleNextChunk();
                    return;
                }
                if (chunkIndex == 0)
                {
                    firstChunkAt = chunkEnd;
                }

                auto pcm = floatSamplesToPcm16(samples);
                streamedSamples += static_cast<int64_t>(pcm.size());
                double const code2wavMs = std::chrono::duration<double, std::milli>(chunkEnd - chunkStart).count();
                streamedCode2WavMs += code2wavMs;
                code2wavInputFrames += static_cast<int64_t>(windowCodes.empty() ? 0 : windowCodes[0].size());
                code2wavContextFrames += static_cast<int64_t>(std::max(0, skipContextFrames));

                writeChunk(chunkIndex, isFinal, totalFrames, pcm, code2wavMs, chunkEnd,
                    code2wavRunner->getConfig().sampleRate);

                lastEmittedFrames = totalFrames;
                scheduleNextChunk();
                ++chunkIndex;
            };

            auto const genStart = std::chrono::steady_clock::now();
            bool ok = false;
            std::chrono::steady_clock::time_point genEnd{};
            if (streamOutput && asyncCode2Wav)
            {
                Code2WavRunner& asyncRunner = getAsyncCode2WavRunner();
                std::mutex streamMutex;
                std::condition_variable streamCv;
                bool generationDone = false;
                std::exception_ptr asyncError;

                std::thread code2wavThread([&]() {
                    try
                    {
                        while (true)
                        {
                            int32_t totalFrames = 0;
                            int32_t emitUntil = 0;
                            int32_t outputChunkIndex = 0;
                            int32_t skipContextFrames = 0;
                            bool isFinal = false;
                            std::vector<std::vector<int32_t>> windowCodes;

                            {
                                std::unique_lock<std::mutex> lock(streamMutex);
                                streamCv.wait(lock, [&]() {
                                    return generationDone
                                        || static_cast<int32_t>(streamedFrames.size()) >= nextChunkAt || asyncError;
                                });
                                if (asyncError)
                                {
                                    return;
                                }
                                totalFrames = static_cast<int32_t>(streamedFrames.size());
                                if (totalFrames <= lastEmittedFrames)
                                {
                                    if (generationDone)
                                    {
                                        return;
                                    }
                                    continue;
                                }
                                if (totalFrames >= nextChunkAt)
                                {
                                    emitUntil = nextChunkAt;
                                }
                                else if (generationDone)
                                {
                                    emitUntil = totalFrames;
                                }
                                else
                                {
                                    continue;
                                }

                                int32_t const naturalLeftContext = asyncRunner.getConfig().leftContextSize;
                                int32_t const leftContext = code2WavContextFrameCap >= 0
                                    ? std::min(naturalLeftContext, code2WavContextFrameCap)
                                    : naturalLeftContext;
                                int32_t const windowStart = std::max(0, lastEmittedFrames - leftContext);
                                skipContextFrames = lastEmittedFrames - windowStart;
                                outputChunkIndex = chunkIndex;
                                isFinal = generationDone && emitUntil == totalFrames;
                                windowCodes = transposeFrameWindow(
                                    streamedFrames, static_cast<size_t>(windowStart), static_cast<size_t>(emitUntil));
                            }

                            auto const chunkStart = std::chrono::steady_clock::now();
                            auto samples
                                = synthesizeWindow(asyncRunner, windowCodes, skipContextFrames, asyncCode2WavStream);
                            auto const chunkEnd = std::chrono::steady_clock::now();
                            double const code2wavMs
                                = std::chrono::duration<double, std::milli>(chunkEnd - chunkStart).count();
                            auto pcm = floatSamplesToPcm16(samples);

                            {
                                std::lock_guard<std::mutex> lock(streamMutex);
                                if (samples.empty())
                                {
                                    lastEmittedFrames = emitUntil;
                                    scheduleNextChunk();
                                    continue;
                                }
                                if (chunkIndex == 0)
                                {
                                    firstChunkAt = chunkEnd;
                                }
                                streamedSamples += static_cast<int64_t>(pcm.size());
                                streamedCode2WavMs += code2wavMs;
                                code2wavInputFrames
                                    += static_cast<int64_t>(windowCodes.empty() ? 0 : windowCodes[0].size());
                                code2wavContextFrames += static_cast<int64_t>(std::max(0, skipContextFrames));
                                lastEmittedFrames = emitUntil;
                                scheduleNextChunk();
                                ++chunkIndex;
                            }
                            writeChunk(outputChunkIndex, isFinal, emitUntil, pcm, code2wavMs, chunkEnd,
                                asyncRunner.getConfig().sampleRate);
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(streamMutex);
                        asyncError = std::current_exception();
                    }
                });

                auto asyncFrameCallback = [&](std::vector<int32_t> const& frameCodes, int32_t) {
                    {
                        std::lock_guard<std::mutex> lock(streamMutex);
                        streamedFrames.push_back(frameCodes);
                    }
                    streamCv.notify_one();
                };

                ok = ttsRuntime->handleAudioGeneration(request, talkerResponse, stream, asyncFrameCallback);
                genEnd = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lock(streamMutex);
                    generationDone = true;
                }
                streamCv.notify_one();
                code2wavThread.join();
                if (asyncError)
                {
                    std::rethrow_exception(asyncError);
                }
            }
            else
            {
                auto frameCallback = [&](std::vector<int32_t> const& frameCodes, int32_t totalFrames) {
                    streamedFrames.push_back(frameCodes);
                    if (streamOutput && totalFrames >= nextChunkAt)
                    {
                        emitChunk(false);
                    }
                };
                ok = streamOutput ? ttsRuntime->handleAudioGeneration(request, talkerResponse, stream, frameCallback)
                                  : ttsRuntime->handleAudioGeneration(request, talkerResponse, stream);
                genEnd = std::chrono::steady_clock::now();
                if (streamOutput)
                {
                    emitChunk(true);
                }
            }
            if (!ok || talkerResponse.rvqCodes.empty())
            {
                throw std::runtime_error("TTS generation failed");
            }

            if (streamOnly)
            {
                auto const doneAt = std::chrono::steady_clock::now();
                int32_t const sampleRate = code2wavRunner->getConfig().sampleRate;
                double const audioSeconds = static_cast<double>(streamedSamples) / sampleRate;
                double const totalMs = std::chrono::duration<double, std::milli>(doneAt - requestStart).count();
                response = Json{{"id", id},
                    {"event", "done"},
                    {"ok", true},
                    {"stream_only", true},
                    {"frames", talkerResponse.numFrames},
                    {"samples", streamedSamples},
                    {"sample_rate", sampleRate},
                    {"audio_s", audioSeconds},
                    {"async_code2wav", asyncCode2Wav},
                    {"adaptive_chunks", adaptiveChunks},
                    {"chunk_frames", chunkFrames},
                    {"chunk_growth_frames", chunkGrowthFrames},
                    {"max_chunk_frames", maxChunkFrames},
                    {"chunk_count", chunkIndex},
                    {"code2wav_input_frames", code2wavInputFrames},
                    {"code2wav_context_frames", code2wavContextFrames},
                    {"code2wav_context_ratio",
                        code2wavInputFrames > 0 ? static_cast<double>(code2wavContextFrames) / code2wavInputFrames
                                                : 0.0},
                    {"generation_ms", std::chrono::duration<double, std::milli>(genEnd - genStart).count()},
                    {"code2wav_ms", streamedCode2WavMs},
                    {"first_chunk_ms",
                        (firstChunkAt.time_since_epoch().count() != 0)
                            ? std::chrono::duration<double, std::milli>(firstChunkAt - requestStart).count()
                            : 0.0},
                    {"total_ms", totalMs},
                    {"rtf", audioSeconds > 0.0 ? totalMs / 1000.0 / audioSeconds : 0.0}};
                std::cout << response.dump() << std::endl;
                continue;
            }

            rt::audioUtils::AudioData audioOutput;
            auto const wavStart = std::chrono::steady_clock::now();
            if (!code2wavRunner)
            {
                logMemTag("worker_before_lazy_code2wav");
                code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
                logMemTag("worker_after_lazy_code2wav");
            }
            logMemTag("worker_before_code2wav_full");
            if (!code2wavRunner->generateWaveform(transposeCodes(talkerResponse.rvqCodes), audioOutput, stream))
            {
                throw std::runtime_error("Code2Wav failed");
            }
            auto const wavEnd = std::chrono::steady_clock::now();
            logMemTag("worker_after_code2wav_full");
            if (!saveAudioToWav(outputFile, audioOutput))
            {
                throw std::runtime_error("Failed to save WAV: " + outputFile);
            }
            int64_t const samples
                = (audioOutput.waveform && !audioOutput.waveform->isEmpty()) ? audioOutput.waveform->getShape()[1] : 0;
            double const audioSeconds = static_cast<double>(samples) / audioOutput.sampleRate;
            double const totalMs = std::chrono::duration<double, std::milli>(wavEnd - requestStart).count();
            response = Json{{"id", id},
                {"event", "done"},
                {"ok", true},
                {"output_file", outputFile},
                {"frames", talkerResponse.numFrames},
                {"samples", samples},
                {"sample_rate", audioOutput.sampleRate},
                {"audio_s", audioSeconds},
                {"generation_ms", std::chrono::duration<double, std::milli>(genEnd - genStart).count()},
                {"code2wav_ms", std::chrono::duration<double, std::milli>(wavEnd - wavStart).count()},
                {"total_ms", totalMs},
                {"rtf", audioSeconds > 0.0 ? totalMs / 1000.0 / audioSeconds : 0.0}};
        }
        catch (std::exception const& e)
        {
            response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
        }
        std::cout << response.dump() << std::endl;
    }

    asyncCode2wavRunner.reset();
    if (asyncCode2WavStream)
    {
        CUDA_CHECK(cudaStreamDestroy(asyncCode2WavStream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
