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

#include "audioWriter.h"
#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "multimodal/code2WavRunner.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

struct ParsedInput
{
    // One entry per request; each entry is the list of messages for that request.
    std::vector<std::vector<Message>> requests;
    // Per-request speaker/language metadata. Falls back to top-level defaults.
    std::vector<std::string> requestSpeakers;
    std::vector<std::string> requestLanguages;
    bool applyChatTemplate{true};
    bool addGenerationPrompt{true};
    bool enableThinking{false};
    float talkerTemperature{0.9f};
    int32_t talkerTopK{40};
    float talkerTopP{0.8f};
    float repetitionPenalty{1.05f};
    float codecEosLogitOffset{0.0f};
    float predictorTemperature{0.0f};
    int32_t predictorTopK{0};
    float predictorTopP{0.0f};
    std::string speakerName{""};
    std::string language{""};
    int32_t maxAudioLength{4096};
};

ParsedInput parseInputFile(std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1)
{
    ParsedInput result;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    check::check(inputFileStream.is_open(), "Failed to open input file: " + inputFilePath.string());
    try
    {
        inputData = Json::parse(inputFileStream);
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            format::fmtstr("Failed to parse input file %s: %s", inputFilePath.string().c_str(), e.what()));
    }

    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size: %d", batchSize));
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("batch_size %d exceeds limit %d", batchSize, limits::security::kReasonableMaxBatchSize));

    result.applyChatTemplate = inputData.value("apply_chat_template", true);
    result.addGenerationPrompt = inputData.value("add_generation_prompt", true);
    result.enableThinking = inputData.value("enable_thinking", false);
    result.talkerTemperature = inputData.value("talker_temperature", 0.9f);
    result.talkerTopK = inputData.value("talker_top_k", 50);
    result.talkerTopP = inputData.value("talker_top_p", 1.0f);
    result.repetitionPenalty = inputData.value("repetition_penalty", 1.05f);
    result.codecEosLogitOffset = inputData.value("codec_eos_logit_offset", 0.0f);
    result.predictorTemperature = inputData.value("predictor_temperature", 0.0f);
    result.predictorTopK = inputData.value("predictor_top_k", 0);
    result.predictorTopP = inputData.value("predictor_top_p", 0.0f);
    result.speakerName = inputData.value("speaker", "");
    result.language = inputData.value("language", "");
    result.maxAudioLength = inputData.value("max_audio_length", 4096);

    check::check(
        inputData.contains("requests") && inputData["requests"].is_array(), "'requests' array not found in input file");

    auto const& requestsArray = inputData["requests"];
    size_t const numRequests = requestsArray.size();

    for (size_t i = 0; i < numRequests; ++i)
    {
        auto const& requestItem = requestsArray[i];
        check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
            "Each request must contain a 'messages' array");

        std::string requestSpeaker = requestItem.value("speaker", result.speakerName);
        std::string requestLanguage = requestItem.value("language", result.language);

        auto const& messagesArray = requestItem["messages"];
        check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
            format::fmtstr("Too many messages in request %zu", i));

        std::vector<Message> messages;
        for (auto const& messageJson : messagesArray)
        {
            check::check(messageJson.contains("role") && messageJson.contains("content"),
                "Each message must have 'role' and 'content' fields");

            Message msg;
            msg.role = messageJson["role"].get<std::string>();

            auto const& contentJson = messageJson["content"];
            Message::MessageContent mc;
            mc.type = "text";
            if (contentJson.is_string())
            {
                mc.content = contentJson.get<std::string>();
            }
            else if (contentJson.is_array())
            {
                for (auto const& item : contentJson)
                {
                    check::check(item.contains("type") && item["type"] == "text", "Only 'text' content is supported");
                    mc.content += item["text"].get<std::string>();
                }
            }
            else
            {
                throw std::runtime_error("Message content must be a string or array");
            }
            check::check(mc.content.size() <= limits::security::kMaxMessageContentSizeBytes,
                format::fmtstr("Message content too large: %zu bytes", mc.content.size()));

            msg.contents.push_back(std::move(mc));
            messages.push_back(std::move(msg));
        }
        result.requests.push_back(std::move(messages));
        result.requestSpeakers.push_back(std::move(requestSpeaker));
        result.requestLanguages.push_back(std::move(requestLanguage));
    }

    return result;
}

enum Qwen3TTSOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    TALKER_ENGINE_DIR = 903,
    CODE_PREDICTOR_ENGINE_DIR = 914,
    CODE2WAV_ENGINE_DIR = 904,
    OUTPUT_FILE = 905,
    OUTPUT_AUDIO_DIR = 906,
    DEBUG = 907,
    DUMP_PROFILE = 908,
    PROFILE_OUTPUT_FILE = 909,
    DUMP_OUTPUT = 911,
    BATCH_SIZE = 912,
    TOKENIZER_DIR = 915,
    CODE_PREDICTOR_BACKEND = 916,
    QWEN3_TTS_TEXT_PROJECTION = 917
};

struct Qwen3TTSInferenceArgs
{
    bool help{false};
    std::string talkerEngineDir{""};
    std::string codePredictorEngineDir{""};
    std::string codePredictorBackend{"auto"};
    std::string qwen3TtsTextProjection{"auto"};
    std::string code2wavEngineDir{""};
    std::string tokenizerDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string outputAudioDir{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    bool dumpOutput{false};
    int32_t batchSize{-1};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [OPTIONS]\n\n"
              << "Main Options:\n"
              << "  --help                       Display this help message\n"
              << "  --inputFile=<path>           Path to input JSON file (text messages only)\n"
              << "  --talkerEngineDir=<path>     Path to Talker engine directory\n"
              << "  --codePredictorEngineDir=<path> Path to CodePredictor engine directory\n"
              << "                               Defaults to --talkerEngineDir/../code_predictor\n"
              << "  --codePredictorBackend=<auto|qwen3_tts_native|generic>\n"
              << "                               CodePredictor runtime backend (default: auto)\n"
              << "  --qwen3TtsTextProjection=<auto|host_fp32|device>\n"
              << "                               Qwen3-TTS text projection precision path (default: auto)\n"
              << "  --code2wavEngineDir=<path>   Path to Code2Wav engine directory\n"
              << "  --tokenizerDir=<path>        Path to tokenizer directory\n"
              << "                               Defaults to --talkerEngineDir/../\n"
              << "  --outputFile=<path>          Path to output JSON file\n"
              << "  --outputAudioDir=<path>      Directory to save generated audio (.wav) files\n\n"
              << "Performance Options:\n"
              << "  --batchSize=<number>         Override batch size from input file\n\n"
              << "Debug Options:\n"
              << "  --debug                      Enable verbose logging\n"
              << "  --dumpOutput                 Print inference output to console\n"
              << "  --dumpProfile                Print performance summary to console\n"
              << "  --profileOutputFile=<path>   Path to profile JSON output\n"
              << std::endl;
}

bool parseArgs(Qwen3TTSInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, Qwen3TTSOptionId::HELP},
        {"inputFile", required_argument, 0, Qwen3TTSOptionId::INPUT_FILE},
        {"talkerEngineDir", required_argument, 0, Qwen3TTSOptionId::TALKER_ENGINE_DIR},
        {"codePredictorEngineDir", required_argument, 0, Qwen3TTSOptionId::CODE_PREDICTOR_ENGINE_DIR},
        {"codePredictorBackend", required_argument, 0, Qwen3TTSOptionId::CODE_PREDICTOR_BACKEND},
        {"qwen3TtsTextProjection", required_argument, 0, Qwen3TTSOptionId::QWEN3_TTS_TEXT_PROJECTION},
        {"code2wavEngineDir", required_argument, 0, Qwen3TTSOptionId::CODE2WAV_ENGINE_DIR},
        {"tokenizerDir", required_argument, 0, Qwen3TTSOptionId::TOKENIZER_DIR},
        {"outputFile", required_argument, 0, Qwen3TTSOptionId::OUTPUT_FILE},
        {"outputAudioDir", required_argument, 0, Qwen3TTSOptionId::OUTPUT_AUDIO_DIR},
        {"debug", no_argument, 0, Qwen3TTSOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, Qwen3TTSOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, Qwen3TTSOptionId::PROFILE_OUTPUT_FILE},
        {"dumpOutput", no_argument, 0, Qwen3TTSOptionId::DUMP_OUTPUT},
        {"batchSize", required_argument, 0, Qwen3TTSOptionId::BATCH_SIZE}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case Qwen3TTSOptionId::HELP: args.help = true; return true;
        case Qwen3TTSOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case Qwen3TTSOptionId::TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case Qwen3TTSOptionId::CODE_PREDICTOR_ENGINE_DIR: args.codePredictorEngineDir = optarg; break;
        case Qwen3TTSOptionId::CODE_PREDICTOR_BACKEND: args.codePredictorBackend = optarg; break;
        case Qwen3TTSOptionId::QWEN3_TTS_TEXT_PROJECTION: args.qwen3TtsTextProjection = optarg; break;
        case Qwen3TTSOptionId::CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case Qwen3TTSOptionId::TOKENIZER_DIR: args.tokenizerDir = optarg; break;
        case Qwen3TTSOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case Qwen3TTSOptionId::OUTPUT_AUDIO_DIR: args.outputAudioDir = optarg; break;
        case Qwen3TTSOptionId::DEBUG: args.debug = true; break;
        case Qwen3TTSOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case Qwen3TTSOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case Qwen3TTSOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case Qwen3TTSOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("batchSize must be positive, got: %s", optarg);
                    return false;
                }
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        default: LOG_ERROR("Unknown option: %c", opt); return false;
        }
    }

    if (!args.help)
    {
        if (args.inputFile.empty())
        {
            LOG_ERROR("--inputFile is required");
            return false;
        }
        if (args.talkerEngineDir.empty())
        {
            LOG_ERROR("--talkerEngineDir is required");
            return false;
        }
    }

    return true;
}

bool parseCodePredictorBackend(
    std::string const& value, rt::Qwen3OmniTTSRuntime::CodePredictorBackend& backend)
{
    if (value == "auto")
    {
        backend = rt::Qwen3OmniTTSRuntime::CodePredictorBackend::kAuto;
        return true;
    }
    if (value == "qwen3_tts_native" || value == "native")
    {
        backend = rt::Qwen3OmniTTSRuntime::CodePredictorBackend::kQwen3TTSNative;
        return true;
    }
    if (value == "generic" || value == "generic_llm_runner")
    {
        backend = rt::Qwen3OmniTTSRuntime::CodePredictorBackend::kGeneric;
        return true;
    }
    return false;
}

bool parseTextProjectionMode(std::string const& value, rt::Qwen3OmniTTSRuntime::TextProjectionMode& mode)
{
    if (value == "auto")
    {
        mode = rt::Qwen3OmniTTSRuntime::TextProjectionMode::kAuto;
        return true;
    }
    if (value == "host_fp32")
    {
        mode = rt::Qwen3OmniTTSRuntime::TextProjectionMode::kHostFP32;
        return true;
    }
    if (value == "device")
    {
        mode = rt::Qwen3OmniTTSRuntime::TextProjectionMode::kDevice;
        return true;
    }
    return false;
}

int main(int argc, char** argv)
{
    Qwen3TTSInferenceArgs args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kINFO);
    LOG_INFO("=== Qwen3-TTS Inference ===");

    auto pluginHandles = loadEdgellmPluginLib();

    LOG_INFO("Talker Engine:  %s", args.talkerEngineDir.c_str());
    if (!args.code2wavEngineDir.empty())
    {
        LOG_INFO("Code2Wav Engine: %s", args.code2wavEngineDir.c_str());
    }
    LOG_INFO("Input File:     %s", args.inputFile.c_str());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize TTS Runtime (loads tokenizer, engines, and weights internally)
    std::unique_ptr<rt::Qwen3OmniTTSRuntime> ttsRuntime;
    try
    {
        std::filesystem::path const codePredictorDir = args.codePredictorEngineDir.empty()
            ? std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor"
            : std::filesystem::path(args.codePredictorEngineDir);
        rt::Qwen3OmniTTSRuntime::RuntimeOptions runtimeOptions;
        if (!parseCodePredictorBackend(args.codePredictorBackend, runtimeOptions.codePredictorBackend))
        {
            throw std::runtime_error("Invalid --codePredictorBackend: " + args.codePredictorBackend);
        }
        if (!parseTextProjectionMode(args.qwen3TtsTextProjection, runtimeOptions.textProjectionMode))
        {
            throw std::runtime_error("Invalid --qwen3TtsTextProjection: " + args.qwen3TtsTextProjection);
        }
        ttsRuntime = std::make_unique<rt::Qwen3OmniTTSRuntime>(
            args.talkerEngineDir, codePredictorDir.string(), args.tokenizerDir, stream, runtimeOptions);
        LOG_INFO("TTS runtime initialized");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize TTS Runtime: %s", e.what());
        return EXIT_FAILURE;
    }

    // Initialize Code2Wav Runner
    std::unique_ptr<Code2WavRunner> code2wavRunner;
    std::filesystem::path const code2wavDir = args.code2wavEngineDir.empty()
        ? std::filesystem::path(args.talkerEngineDir).parent_path() / "code2wav"
        : std::filesystem::path(args.code2wavEngineDir);

    if (std::filesystem::exists(code2wavDir))
    {
        LOG_INFO("Initializing Code2Wav Runner from %s...", code2wavDir.string().c_str());
        try
        {
            code2wavRunner = std::make_unique<Code2WavRunner>(code2wavDir.string(), stream);
            LOG_INFO("Code2Wav Runner initialized");
        }
        catch (std::exception const& e)
        {
            LOG_WARNING("Failed to initialize Code2Wav: %s. Will output RVQ codes only.", e.what());
        }
    }
    else
    {
        LOG_INFO("Code2Wav engine not found at %s. Will output RVQ codes only.", code2wavDir.string().c_str());
    }

    if (!args.outputAudioDir.empty())
    {
        std::filesystem::create_directories(args.outputAudioDir);
    }

    if (!ttsRuntime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for TTS decoding, proceeding without.");
    }

    bool const profilerEnabled = args.dumpProfile || !args.profileOutputFile.empty();
    MemoryMonitor memoryMonitor;
    if (profilerEnabled)
    {
        setProfilingEnabled(true);
        memoryMonitor.start();
    }

    LOG_INFO("Parsing input file...");
    auto input = parseInputFile(args.inputFile, args.batchSize);

    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    size_t failedCount = 0;

    LOG_INFO("Processing %zu request(s)...", input.requests.size());
    for (size_t requestIdx = 0; requestIdx < input.requests.size(); ++requestIdx)
    {
        rt::Qwen3OmniTTSRuntime::TalkerGenerationRequest talkerReq;
        talkerReq.talkerTemperature = input.talkerTemperature;
        talkerReq.talkerTopK = input.talkerTopK;
        talkerReq.talkerTopP = input.talkerTopP;
        talkerReq.repetitionPenalty = input.repetitionPenalty;
        talkerReq.codecEosLogitOffset = input.codecEosLogitOffset;
        talkerReq.predictorTemperature = input.predictorTemperature;
        talkerReq.predictorTopK = input.predictorTopK;
        talkerReq.predictorTopP = input.predictorTopP;
        talkerReq.applyChatTemplate = input.applyChatTemplate;
        talkerReq.addGenerationPrompt = input.addGenerationPrompt;
        talkerReq.enableThinking = input.enableThinking;
        talkerReq.speakerName = input.requestSpeakers[requestIdx];
        talkerReq.language = input.requestLanguages[requestIdx];
        talkerReq.maxAudioLength = input.maxAudioLength;
        talkerReq.messages = input.requests[requestIdx];

        rt::Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResp;
        bool const requestStatus = ttsRuntime->handleAudioGeneration(talkerReq, talkerResp, stream);

        if (!requestStatus)
        {
            LOG_WARNING("TTS generation failed for request %zu", requestIdx);
            hasFailedRequest = true;
            failedCount++;
        }

        // Run Code2Wav
        rt::audioUtils::AudioData audioOutput;
        bool hasAudio = false;
        if (requestStatus && code2wavRunner && !talkerResp.rvqCodes.empty())
        {
            // Transpose [frames][layers] → [layers][frames]
            size_t const numFrames = talkerResp.rvqCodes.size();
            size_t const numLayers = talkerResp.rvqCodes[0].size();
            std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
            for (size_t f = 0; f < numFrames; ++f)
            {
                for (size_t l = 0; l < numLayers; ++l)
                {
                    transposed[l][f] = talkerResp.rvqCodes[f][l];
                }
            }

            if (code2wavRunner->generateWaveform(transposed, audioOutput, stream))
            {
                hasAudio = true;
                if (!args.outputAudioDir.empty())
                {
                    std::string filename = format::fmtstr("audio_req%zu.wav", requestIdx);
                    std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                    if (!saveAudioToWav(audioPath.string(), audioOutput))
                    {
                        LOG_WARNING("Failed to save audio: %s", audioPath.string().c_str());
                    }
                }
            }
            else
            {
                LOG_WARNING("Code2Wav failed for request %zu", requestIdx);
            }
        }

        if (args.dumpOutput && requestStatus && hasAudio)
        {
            int64_t samples
                = (!audioOutput.waveform || audioOutput.waveform->isEmpty()) ? 0 : audioOutput.waveform->getShape()[1];
            LOG_INFO("[%zu] Audio: %ld samples (%.2fs)", requestIdx, samples,
                static_cast<float>(samples) / audioOutput.sampleRate);
        }

        // Build JSON output
        {
            nlohmann::json responseJson;
            responseJson["request_idx"] = requestIdx;
            responseJson["output_text"] = requestStatus ? "" : "FAILED";

            nlohmann::json messagesJson = nlohmann::json::array();
            for (auto const& msg : input.requests[requestIdx])
            {
                nlohmann::json m;
                m["role"] = msg.role;
                m["content"] = msg.contents.empty() ? "" : msg.contents[0].content;
                messagesJson.push_back(std::move(m));
            }
            responseJson["messages"] = std::move(messagesJson);

            if (requestStatus && hasAudio && !args.outputAudioDir.empty())
            {
                std::string filename = format::fmtstr("audio_req%zu.wav", requestIdx);
                std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                int64_t samples = (audioOutput.waveform && !audioOutput.waveform->isEmpty())
                    ? audioOutput.waveform->getShape()[1]
                    : 0;
                responseJson["audio_file"] = audioPath.string();
                responseJson["audio_samples"] = samples;
                responseJson["audio_sample_rate"] = audioOutput.sampleRate;
                responseJson["audio_duration_ms"] = static_cast<int64_t>(1000.0 * samples / audioOutput.sampleRate);
            }

            if (requestStatus && !talkerResp.rvqCodes.empty() && !args.outputAudioDir.empty())
            {
                auto const& frames = talkerResp.rvqCodes;
                int64_t const numFrames = static_cast<int64_t>(frames.size());
                int64_t const numCodes = frames.empty() ? 0 : static_cast<int64_t>(frames[0].size());
                // Flatten [numFrames][numCodes] into a contiguous buffer for the Tensor wrapper
                std::vector<int32_t> flat;
                flat.reserve(numFrames * numCodes);
                for (auto const& frame : frames)
                {
                    flat.insert(flat.end(), frame.begin(), frame.end());
                }
                std::vector<rt::Tensor> tensors;
                tensors.emplace_back(flat.data(), rt::Coords{numFrames, numCodes}, rt::DeviceType::kCPU,
                    nvinfer1::DataType::kINT32, "rvq_codes");
                std::string filename = format::fmtstr("rvq_req%zu.safetensors", requestIdx);
                std::filesystem::path stPath = std::filesystem::path(args.outputAudioDir) / filename;
                safetensors::saveSafetensors(stPath, tensors, stream);
                responseJson["rvq_file"] = stPath.string();
            }

            outputData["responses"].push_back(responseJson);
        }
    }

    LOG_INFO("Done: %zu/%zu requests succeeded", input.requests.size() - failedCount, input.requests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("%zu request(s) failed", failedCount);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    if (args.dumpProfile)
    {
        std::ostringstream ss;
        ss << "\n=== Performance Summary ===\n";
        outputTalkerProfile(ss, ttsRuntime->getMetrics());
        outputMemoryProfile(ss, memoryMonitor);
        ss << "===========================\n";
        LOG_INFO("%s", ss.str().c_str());
    }

    if (!args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;
            addJsonTalkerSummary(profileJson, ttsRuntime->getMetrics());
            addJsonTimingStages(profileJson);
            addJsonMemorySummary(profileJson, memoryMonitor);

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2);
                LOG_INFO("Profile saved to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile: %s", e.what());
        }
    }

    if (!args.outputFile.empty())
    {
        try
        {
            std::ofstream out(args.outputFile);
            if (out.is_open())
            {
                out << outputData.dump(2);
                LOG_INFO("Output saved to: %s", args.outputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write output file: %s", e.what());
        }
    }

    cudaStreamDestroy(stream);
    LOG_INFO("=== Done ===");
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
