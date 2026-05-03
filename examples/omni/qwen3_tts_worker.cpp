/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audioWriter.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{
struct Args
{
    std::string talkerEngineDir;
    std::string code2wavEngineDir;
    std::string tokenizerDir;
    bool debug{false};
};

enum OptionId : int
{
    HELP = 1000,
    TALKER_ENGINE_DIR,
    CODE2WAV_ENGINE_DIR,
    TOKENIZER_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --talkerEngineDir=<path> --code2wavEngineDir=<path>"
              << " [--tokenizerDir=<path>] [--debug]\n\n"
              << "Reads JSON lines from stdin:\n"
              << "  {\"id\":\"1\",\"text\":\"你好。\",\"output_file\":\"/tmp/out.wav\",\"max_audio_length\":80}\n"
              << "Writes JSON lines to stdout.\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"talkerEngineDir", required_argument, 0, TALKER_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, CODE2WAV_ENGINE_DIR},
        {"tokenizerDir", required_argument, 0, TOKENIZER_DIR}, {"debug", no_argument, 0, DEBUG}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case TOKENIZER_DIR: args.tokenizerDir = optarg; break;
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }

    return !args.talkerEngineDir.empty() && !args.code2wavEngineDir.empty();
}

std::vector<std::vector<int32_t>> transposeCodes(std::vector<std::vector<int32_t>> const& frames)
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

Qwen3OmniTTSRuntime::TalkerGenerationRequest buildRequest(Json const& item)
{
    Qwen3OmniTTSRuntime::TalkerGenerationRequest req;
    req.talkerTemperature = item.value("talker_temperature", 0.9f);
    req.talkerTopK = item.value("talker_top_k", 50);
    req.talkerTopP = item.value("talker_top_p", 1.0f);
    req.repetitionPenalty = item.value("repetition_penalty", 1.05f);
    req.maxAudioLength = item.value("max_audio_length", 4096);
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
    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::filesystem::path const codePredictorDir
        = std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor";

    auto const initStart = std::chrono::steady_clock::now();
    auto ttsRuntime = std::make_unique<Qwen3OmniTTSRuntime>(
        args.talkerEngineDir, codePredictorDir.string(), args.tokenizerDir, stream);
    bool const lazyCode2Wav = std::getenv("EDGE_LLM_TTS_LAZY_CODE2WAV") != nullptr
        && std::string(std::getenv("EDGE_LLM_TTS_LAZY_CODE2WAV")) == "1";
    std::unique_ptr<Code2WavRunner> code2wavRunner;
    if (!lazyCode2Wav)
    {
        code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
    }
    bool const enableGraph = std::getenv("EDGE_LLM_TTS_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_TTS_CUDA_GRAPH")) != "0";
    if (enableGraph && !ttsRuntime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for TTS worker, proceeding without.");
    }
    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}}.dump() << std::endl;

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
            std::string outputFile = item.value("output_file", "");
            if (outputFile.empty())
            {
                outputFile = "/tmp/qwen3_tts_worker_" + id + ".wav";
            }

            auto request = buildRequest(item);
            Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResponse;
            auto const genStart = std::chrono::steady_clock::now();
            bool ok = ttsRuntime->handleAudioGeneration(request, talkerResponse, stream);
            auto const genEnd = std::chrono::steady_clock::now();
            if (!ok || talkerResponse.rvqCodes.empty())
            {
                throw std::runtime_error("TTS generation failed");
            }

            rt::audioUtils::AudioData audioOutput;
            auto transposed = transposeCodes(talkerResponse.rvqCodes);
            auto const wavStart = std::chrono::steady_clock::now();
            if (!code2wavRunner)
            {
                code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
            }
            if (!code2wavRunner->generateWaveform(transposed, audioOutput, stream))
            {
                throw std::runtime_error("Code2Wav failed");
            }
            auto const wavEnd = std::chrono::steady_clock::now();
            if (lazyCode2Wav)
            {
                code2wavRunner.reset();
            }
            if (!saveAudioToWav(outputFile, audioOutput))
            {
                throw std::runtime_error("Failed to save WAV: " + outputFile);
            }

            int64_t samples = (audioOutput.waveform && !audioOutput.waveform->isEmpty())
                ? audioOutput.waveform->getShape()[1]
                : 0;
            double const audioSeconds = static_cast<double>(samples) / audioOutput.sampleRate;
            double const totalMs
                = std::chrono::duration<double, std::milli>(wavEnd - requestStart).count();
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
