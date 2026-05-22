/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audioWriter.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "multimodal/statefulCode2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <getopt.h>
#include <iostream>
#include <list>
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

std::vector<uint8_t> base64Decode(std::string const& input)
{
    std::array<int8_t, 256> table{};
    table.fill(-1);
    for (int i = 0; i < 26; ++i)
    {
        table[static_cast<uint8_t>('A' + i)] = i;
        table[static_cast<uint8_t>('a' + i)] = i + 26;
    }
    for (int i = 0; i < 10; ++i)
    {
        table[static_cast<uint8_t>('0' + i)] = i + 52;
    }
    table[static_cast<uint8_t>('+')] = 62;
    table[static_cast<uint8_t>('/')] = 63;

    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    int val = 0;
    int bits = -8;
    for (unsigned char c : input)
    {
        if (c == '=')
        {
            break;
        }
        int8_t decoded = table[c];
        if (decoded < 0)
        {
            continue;
        }
        val = (val << 6) + decoded;
        bits += 6;
        if (bits >= 0)
        {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

std::vector<float> float32VectorFromBytes(std::vector<uint8_t> const& bytes)
{
    if (bytes.size() % sizeof(float) != 0)
    {
        throw std::runtime_error("speaker_embedding_b64 size is not a float32 vector");
    }
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
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
    req.speakerId = item.value("speaker_id", -1);
    if (item.contains("speaker_embedding_b64") && item["speaker_embedding_b64"].is_string())
    {
        req.speakerEmbedding = float32VectorFromBytes(base64Decode(item["speaker_embedding_b64"].get<std::string>()));
    }
    if (!req.speakerEmbedding.empty() && (req.speakerId >= 0 || !req.speakerName.empty()))
    {
        throw std::runtime_error("speaker_embedding_b64 cannot be combined with speaker or speaker_id");
    }

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

std::vector<float> synthesizeStatefulChunk(
    StatefulCode2WavRunner& code2wavRunner, std::vector<std::vector<int32_t>> const& chunkCodes, bool isFinal,
    cudaStream_t stream)
{
    rt::audioUtils::AudioData audioOutput;
    if (!code2wavRunner.generateChunk(chunkCodes, isFinal, audioOutput, stream))
    {
        throw std::runtime_error("Stateful Code2Wav chunk generation failed");
    }
    return audioToFloatSamples(audioOutput);
}

// Serializes stdout writes so concurrent emitters (Phase 3b-B-2 worker
// dispatch threads) cannot interleave JSON lines. Even at N=1 this is harmless
// and centralizes the per-event envelope so Phase 1 protocol additions
// ("request_id") stay consistent across every emit site.
std::mutex coutMutex;

// [Cancel protocol step 1] Process-wide map of in-flight request_id ->
// per-request atomic cancel flag. The flag's storage lives on the
// worker thread's stack (inside handleRequest); the map holds a raw
// pointer to it, valid only while the worker thread is active.
// Lifetime guaranteed by RAII inside handleRequest (CancelMapEntry
// registers in the lambda body and erases on scope exit, including
// the exception path).
//
// Reads (worker thread checking its own flag) use acquire ordering;
// writes (main/dispatcher thread setting another thread's flag on
// "cancel" message) use release ordering. The map mutex serializes
// insert / erase / lookup; once a worker thread has acquired the raw
// atomic* via the map under the lock, subsequent flag stores/loads
// go straight through the atomic without re-taking the map mutex.
std::mutex cancelMapMu;
std::unordered_map<std::string, std::atomic<bool>*> cancelMap;

// [Phase B C5b — empirical probe] Worker-level mutex around the
// entire ttsRuntime->handleAudioGeneration() call. Codex audit
// (docs/specs/tts-n2-shared-tensor-audit.md §1) lists ~14 shared
// mutable Qwen3OmniTTSRuntime members that race at N=2 (mTalkerLogits,
// mTalkerHiddenStatesBuffer, mCodecHiddensBuffer-already-fixed,
// mCodePredictor*, etc.). The N=2 probe with only C5 (Code2Wav
// mutex) still crashed inside StatefulCode2WavRunner::reset, which
// is a downstream symptom — runtime scratch races corrupt the
// codes BEFORE Code2Wav runs, then Code2Wav's allocator returns
// already-poisoned memory. Until C2/C3/C4 lands the proper per-slot
// solution, serialize the whole runtime path at the worker.
std::mutex runtimeMutex;

// [Phase B C5] Code2Wav serialization — empirically required at
// N=2 even with per-slot Code2Wav runners (Phase 3b-B-4 part-2).
// The crash signature was:
//   CUDA runtime error in cudaMemsetAsync(state.read.rawPointer(),
//   ...) an illegal memory access was encountered
//   CUDA runtime error in cudaMemcpyAsync(mInputCodesDevice...) ditto
// observed in StatefulCode2WavRunner::reset / generateChunk when two
// requests reach the worker concurrently. Per-slot runners theory
// said this should be safe; reality says otherwise. Until per-instance
// state buffer ownership is fully fixed (see
// docs/specs/tts-n2-phase-b-patches.md §5), serialize at the worker
// level around all Code2Wav GPU ops. Note that Code2Wav runs AFTER
// all token generation completes, so slow-client TTFA is driven by
// the first audio chunk emitted before this contention point —
// throughput cost is tolerable.
std::mutex code2WavMutex;

// Phase 3b-B-4 part-2: Code2Wav runners are now per-slot (mirroring the
// engine SlotPool capacity). Each in-flight request acquires a Code2Wav
// slot index from Code2WavSlotPool below and uses the matching per-slot
// runner; runners no longer share a singleton or a mutex. With
// OVS_TTS_WORKER_CONCURRENCY=N we instantiate N (Stateful)Code2WavRunner
// objects, each with its own CUDA stream so kernels actually overlap on
// the GPU instead of serializing on a shared stream. The old
// the prior process-wide Code2Wav mutex (Phase 3b-B-2) has been removed — concurrency is bounded
// by the slot pool itself and each slot's runner is touched by exactly
// one worker thread at a time.
//
// Counted index pool: blocks acquire() until a free slot exists. Capacity
// matches the dispatcher's readConcurrencyEnv() so there's always a slot
// for any thread the dispatcher hands a request to (no extra waiting).
class Code2WavSlotPool
{
public:
    explicit Code2WavSlotPool(size_t capacity) : mCapacity(capacity)
    {
        mFree.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i)
        {
            mFree.push_back(static_cast<int>(capacity - 1 - i)); // pop from back -> 0,1,2...
        }
    }

    size_t capacity() const
    {
        return mCapacity;
    }

    int acquire()
    {
        std::unique_lock<std::mutex> lk(mMu);
        mCv.wait(lk, [&]() { return !mFree.empty(); });
        int idx = mFree.back();
        mFree.pop_back();
        return idx;
    }

    void release(int idx)
    {
        {
            std::lock_guard<std::mutex> lk(mMu);
            mFree.push_back(idx);
        }
        mCv.notify_one();
    }

private:
    size_t mCapacity;
    std::vector<int> mFree;
    std::mutex mMu;
    std::condition_variable mCv;
};

// RAII handle so an exception inside handleRequest can't leak a slot.
class Code2WavSlotGuard
{
public:
    Code2WavSlotGuard(Code2WavSlotPool& pool, int idx) : mPool(pool), mIdx(idx) {}
    ~Code2WavSlotGuard()
    {
        if (mIdx >= 0)
        {
            mPool.release(mIdx);
        }
    }
    Code2WavSlotGuard(Code2WavSlotGuard const&) = delete;
    Code2WavSlotGuard& operator=(Code2WavSlotGuard const&) = delete;
    int index() const
    {
        return mIdx;
    }

private:
    Code2WavSlotPool& mPool;
    int mIdx;
};

// Phase 3b-B-2: worker-side concurrency env reader. Mirrors the engine-side
// readTtsWorkerConcurrencyEnv() in cpp/runtime/qwen3OmniTTSRuntime.cpp (Phase
// 3b-B-1). Both read the SAME env (OVS_TTS_WORKER_CONCURRENCY) so the worker
// dispatcher and the engine SlotPools stay in sync. Duplicated (not shared via
// header) to avoid an API change for one small helper; Phase 3b-B-3 may
// consolidate.
inline size_t readConcurrencyEnv()
{
    size_t capacity = 1;
    char const* env = std::getenv("OVS_TTS_WORKER_CONCURRENCY");
    if (env != nullptr && *env != '\0')
    {
        try
        {
            int const parsed = std::stoi(env);
            if (parsed < 1)
            {
                LOG_WARNING("[Worker] OVS_TTS_WORKER_CONCURRENCY=%d below min=1; clamping to 1", parsed);
                capacity = 1;
            }
            else if (parsed > 8)
            {
                LOG_WARNING("[Worker] OVS_TTS_WORKER_CONCURRENCY=%d above max=8; clamping to 8", parsed);
                capacity = 8;
            }
            else
            {
                capacity = static_cast<size_t>(parsed);
            }
        }
        catch (std::exception const& e)
        {
            LOG_WARNING("[Worker] OVS_TTS_WORKER_CONCURRENCY=\"%s\" not parseable (%s); defaulting to 1",
                env, e.what());
            capacity = 1;
        }
    }
    LOG_INFO("[Worker] dispatcher concurrency=%zu (OVS_TTS_WORKER_CONCURRENCY)", capacity);
    return capacity;
}

// Phase 1 protocol helper (see docs/specs/tts-worker-concurrency.md §4.3).
// Stamps every stdout event with both "request_id" and "id" fields holding
// the same value, then writes one JSON line under coutMutex. The dual field
// is intentional: "id" stays for back-compat with older Python clients that
// may rely on it; "request_id" is the new demux key for the future scheduler.
// For events emitted outside a request scope (currently only the "ready"
// event), pass requestId="__worker__".
void emitEvent(std::string const& requestId, std::string const& kind, Json payload)
{
    payload["event"] = kind;
    payload["request_id"] = requestId;
    // Preserve "id" as an alias. If the caller already set it (chunk/done
    // build their JSON with "id" inline), keep that value; otherwise mirror
    // request_id so every line carries both fields.
    if (!payload.contains("id"))
    {
        payload["id"] = requestId;
    }
    std::string const line = payload.dump();
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << line << std::endl;
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
    // Phase 3b-B-4 part-2: per-slot Code2Wav runners. Vectors are sized to
    // the dispatcher concurrency below; index 0 is constructed eagerly during
    // init (preserving single-client behaviour and the original lazy-init
    // log tags), and slots [1..N-1] are constructed lazily on first use by a
    // higher slot index. Each per-slot runner owns its own CUDA stream so
    // GPU kernels actually overlap instead of serializing on the shared
    // main `stream` (which is reserved for the talker + code-predictor).
    std::vector<std::unique_ptr<Code2WavRunner>> code2wavRunners;
    std::vector<std::unique_ptr<StatefulCode2WavRunner>> statefulCode2wavRunners;
    std::vector<std::unique_ptr<Code2WavRunner>> asyncCode2wavRunners;
    std::vector<cudaStream_t> code2wavStreams;       // streams for sync runners (slot 0 reuses main `stream`)
    std::vector<cudaStream_t> asyncCode2wavStreams;  // dedicated streams for the async branch
    // Per-slot lazy-init guards (each slot constructed at most once). At
    // capacity 1 these contend with nothing; at N>1 they only contend during
    // a single one-shot init per slot, never on hot paths.
    std::vector<std::mutex> code2wavSlotInitMu; // sized later
    std::vector<std::mutex> statefulCode2wavSlotInitMu;
    std::vector<std::mutex> asyncCode2wavSlotInitMu;
    bool const lazyCode2Wav = envIsOne("EDGE_LLM_TTS_LAZY_CODE2WAV");
    bool const statefulCode2Wav = envIsOne("EDGE_LLM_TTS_STATEFUL_CODE2WAV");
    std::string const statefulCode2WavEngineDir = std::getenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR") != nullptr
        ? std::string(std::getenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR"))
        : args.code2wavEngineDir;
    int32_t const code2WavContextFrameCap = envIntOr("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", -1);
    // Helper: lazy-init the async Code2Wav runner + its CUDA stream for slot
    // `slot`. Each slot has its own mutex so concurrent first-uses of
    // distinct slots don't serialize. Once initialized, the slot's runner is
    // reused on every subsequent acquisition of that slot index.
    auto getAsyncCode2WavRunner = [&](int slot) -> Code2WavRunner& {
        std::lock_guard<std::mutex> lk(asyncCode2wavSlotInitMu[slot]);
        if (!asyncCode2wavRunners[slot])
        {
            logMemTag("worker_before_async_code2wav_stream");
            CUDA_CHECK(cudaStreamCreate(&asyncCode2wavStreams[slot]));
            logMemTag("worker_after_async_code2wav_stream");
            logMemTag("worker_before_async_code2wav");
            asyncCode2wavRunners[slot]
                = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, asyncCode2wavStreams[slot]);
            logMemTag("worker_after_async_code2wav");
        }
        return *asyncCode2wavRunners[slot];
    };
    // Phase 3b-B-4 part-2: read worker concurrency early so we can size the
    // per-slot Code2Wav vectors before the init try{} runs. The same env
    // (OVS_TTS_WORKER_CONCURRENCY) drives both this and the engine SlotPools
    // so dispatcher capacity == per-engine slot count by construction.
    size_t const concurrency = readConcurrencyEnv();
    LOG_INFO("[Worker] startup capacity sync: dispatcher=%zu, talker/code-predictor SlotPool capacity reads "
             "the same OVS_TTS_WORKER_CONCURRENCY env (see engine ctor logs above)",
        concurrency);
    code2wavRunners.resize(concurrency);
    statefulCode2wavRunners.resize(concurrency);
    asyncCode2wavRunners.resize(concurrency);
    code2wavStreams.assign(concurrency, cudaStream_t{});
    asyncCode2wavStreams.assign(concurrency, cudaStream_t{});
    std::vector<std::mutex> tmpA(concurrency);
    std::vector<std::mutex> tmpB(concurrency);
    std::vector<std::mutex> tmpC(concurrency);
    code2wavSlotInitMu.swap(tmpA);
    statefulCode2wavSlotInitMu.swap(tmpB);
    asyncCode2wavSlotInitMu.swap(tmpC);
    Code2WavSlotPool code2wavSlotPool(concurrency);
    // Slot 0 reuses the main `stream` to preserve byte-equivalent N=1 init
    // ordering (same CUDA stream as Phase 3b-B-2). Slots >=1 get their own
    // dedicated stream so kernels overlap on the GPU rather than serializing.
    code2wavStreams[0] = stream;

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
        // Phase 3b-B-4 part-2: eager-init slot 0 only (matches the prior
        // single-runner init). Slots >=1 init lazily on first acquisition by
        // a concurrent request — keeps cold-start VRAM unchanged at N=1 and
        // only pays the cost when concurrency actually picks up.
        if (statefulCode2Wav)
        {
            logMemTag("worker_before_stateful_code2wav");
            statefulCode2wavRunners[0] = std::make_unique<StatefulCode2WavRunner>(statefulCode2WavEngineDir, stream);
            logMemTag("worker_after_stateful_code2wav");
        }
        else if (lazyCode2Wav)
        {
            logMemTag("worker_skip_code2wav_lazy");
        }
        else
        {
            logMemTag("worker_before_code2wav");
            code2wavRunners[0] = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
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
        for (auto& s : asyncCode2wavStreams)
        {
            if (s)
            {
                CUDA_CHECK(cudaStreamDestroy(s));
            }
        }
        for (size_t i = 1; i < code2wavStreams.size(); ++i)
        {
            if (code2wavStreams[i])
            {
                CUDA_CHECK(cudaStreamDestroy(code2wavStreams[i]));
            }
        }
        return EXIT_FAILURE;
    }

    double const initMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    logMemTag("worker_before_ready");
    emitEvent("__worker__", "ready", Json{{"init_ms", initMs}});
    logMemTag("worker_after_ready");

    // Phase 3b-B-4 part-2: worker-level request dispatcher. Each request is
    // handed to a worker std::thread bounded by `concurrency` (read above).
    // Code2Wav is now per-slot (Code2WavSlotPool + per-slot runners), so the
    // talker + code-predictor + Code2Wav stages all overlap across requests.
    std::list<std::thread> workers;
    std::mutex workersMu;
    std::condition_variable workersCv;
    size_t inFlight = 0;

    // Per-request lambda. Captures every long-lived local from main() by
    // reference (they all outlive every worker thread since we join below
    // before returning). The 'line' is captured by value so the reader can
    // overwrite it for the next request immediately.
    auto handleRequest = [&](std::string const& reqLine) {
        Json response;
        auto const requestStart = std::chrono::steady_clock::now();
        // Lifted above the try{} so the catch block can stamp the failing
        // request's id onto the error event via emitEvent(). If JSON parse
        // fails before we extract "id", fall back to "__worker__".
        std::string id = "__worker__";
        std::string const& line = reqLine;
        // [Cancel protocol step 1] Per-request cancel flag. Its address is
        // published into cancelMap once we know the real request id (after
        // JSON parse below). The dispatcher's main loop looks the address
        // up by id when a {"type":"cancel"} message arrives, then stores
        // true with release ordering. The chunk emit loop loads with
        // acquire ordering between chunks; on observed-true it stops
        // emitting further chunks and emits a "cancelled" terminal event
        // in lieu of "done". Worst-case latency = one chunk boundary
        // (~30-100 ms on Orin NX) — we never abort an in-flight CUDA
        // kernel mid-enqueue.
        std::atomic<bool> cancelled{false};
        // RAII guard for cancelMap registration. Constructed AFTER the id
        // is known (see below); destructor unconditionally erases the
        // entry on every exit path including thrown exceptions. If the
        // dispatcher never registered us (e.g. JSON parse threw before
        // registration), mRegistered stays false and dtor is a no-op.
        struct CancelMapEntry
        {
            std::string id;
            bool registered{false};
            ~CancelMapEntry()
            {
                if (!registered)
                {
                    return;
                }
                std::lock_guard<std::mutex> lk(cancelMapMu);
                cancelMap.erase(id);
            }
        } cancelEntry;
        // Phase 3b-B-4 part-2: acquire a Code2Wav slot for the lifetime of
        // the request. The pool is sized to match dispatcher concurrency, so
        // at steady state acquire() never blocks; it only orders init when
        // two requests race for the same fresh slot. The RAII guard releases
        // the slot when the request handler returns OR throws.
        int const c2wSlot = code2wavSlotPool.acquire();
        Code2WavSlotGuard c2wGuard(code2wavSlotPool, c2wSlot);
        // Lazy-init this slot's CUDA stream if it's a non-zero slot. Slot 0
        // reuses the main `stream` (set above) for byte-equivalence at N=1.
        if (c2wSlot != 0 && code2wavStreams[c2wSlot] == cudaStream_t{})
        {
            std::lock_guard<std::mutex> lk(code2wavSlotInitMu[c2wSlot]);
            if (code2wavStreams[c2wSlot] == cudaStream_t{})
            {
                cudaStream_t s{};
                CUDA_CHECK(cudaStreamCreate(&s));
                code2wavStreams[c2wSlot] = s;
            }
        }
        cudaStream_t const c2wStream = code2wavStreams[c2wSlot];
        try
        {
            Json item = Json::parse(line);
            id = item.value("id", "");
            if (id.empty())
            {
                id = "__worker__";
            }
            // [Cancel protocol step 1] Publish our cancel flag now that
            // we know the real request id. Done BEFORE entering the
            // chunk emit loop so a cancel arriving milliseconds after
            // the request line can still race in and be observed at
            // the first chunk boundary. cancelEntry's dtor erases the
            // entry on EVERY exit path (normal, throw, or cancel).
            {
                std::lock_guard<std::mutex> lk(cancelMapMu);
                cancelMap[id] = &cancelled;
            }
            cancelEntry.id = id;
            cancelEntry.registered = true;
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
            if (statefulCode2Wav && asyncCode2Wav)
            {
                throw std::runtime_error("Stateful Code2Wav does not support async_code2wav yet");
            }
            if (statefulCode2Wav && !streamOutput)
            {
                throw std::runtime_error("Stateful Code2Wav currently requires stream=true");
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
                emitEvent(id, "chunk", std::move(chunk));
            };

            auto emitChunk = [&](bool isFinal) {
                // [Cancel protocol step 1] Cooperative cancel checkpoint.
                // Placed at the START of every chunk emit, BEFORE any
                // synthesizeStatefulChunk / synthesizeWindow call, so the
                // previous chunk's CUDA work completes naturally but no
                // new vocoder kernel is enqueued after cancel is observed.
                // Acquire ordering pairs with the dispatcher's release
                // store. On observed-true we skip emit; the outer
                // generation loop will continue but its frameCallback
                // invocations become no-ops here.
                if (cancelled.load(std::memory_order_acquire))
                {
                    return;
                }
                int32_t const totalFrames = static_cast<int32_t>(streamedFrames.size());
                if (totalFrames <= lastEmittedFrames)
                {
                    return;
                }

                if (statefulCode2Wav)
                {
                    // Phase 3b-B-4 part-2: per-slot stateful runner. Each
                    // slot is touched by exactly one in-flight request at a
                    // time (Code2WavSlotPool acquires above), so no mutex is
                    // needed around the runner itself. Slot-init mutex only
                    // serializes the first-ever construction of THIS slot.
                    if (!statefulCode2wavRunners[c2wSlot])
                    {
                        std::lock_guard<std::mutex> lk(statefulCode2wavSlotInitMu[c2wSlot]);
                        if (!statefulCode2wavRunners[c2wSlot])
                        {
                            logMemTag("worker_before_stateful_code2wav");
                            // [Phase B C5b probe] runtime mutex (upstream) prevents
                            // the upstream corruption that was crashing Code2Wav.
                            // Per-slot runners should now be safe on their own —
                            // no C5 lock needed here.
                            statefulCode2wavRunners[c2wSlot]
                                = std::make_unique<StatefulCode2WavRunner>(statefulCode2WavEngineDir, c2wStream);
                            logMemTag("worker_after_stateful_code2wav");
                        }
                    }
                    auto& runner = *statefulCode2wavRunners[c2wSlot];
                    auto const chunkCodes = transposeFrameWindow(
                        streamedFrames, static_cast<size_t>(lastEmittedFrames), static_cast<size_t>(totalFrames));
                    auto const chunkStart = std::chrono::steady_clock::now();
                    logMemTag(isFinal ? "worker_before_stateful_code2wav_final_chunk"
                                      : "worker_before_stateful_code2wav_chunk");
                    // [Phase B C5b probe] no C5 lock — upstream runtime mutex
                    // prevents corruption; per-slot runners are safe.
                    auto samples = synthesizeStatefulChunk(runner, chunkCodes, isFinal, c2wStream);
                    auto const chunkEnd = std::chrono::steady_clock::now();
                    logMemTag(isFinal ? "worker_after_stateful_code2wav_final_chunk"
                                      : "worker_after_stateful_code2wav_chunk");
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
                    code2wavInputFrames += static_cast<int64_t>(chunkCodes.empty() ? 0 : chunkCodes[0].size());

                    writeChunk(chunkIndex, isFinal, totalFrames, pcm, code2wavMs, chunkEnd,
                        runner.getConfig().sampleRate);

                    lastEmittedFrames = totalFrames;
                    scheduleNextChunk();
                    ++chunkIndex;
                    return;
                }

                // Phase 3b-B-4 part-2: per-slot sync Code2Wav runner.
                if (!code2wavRunners[c2wSlot])
                {
                    std::lock_guard<std::mutex> lk(code2wavSlotInitMu[c2wSlot]);
                    if (!code2wavRunners[c2wSlot])
                    {
                        logMemTag("worker_before_lazy_code2wav");
                        code2wavRunners[c2wSlot]
                            = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, c2wStream);
                        logMemTag("worker_after_lazy_code2wav");
                    }
                }
                auto& runner = *code2wavRunners[c2wSlot];
                int32_t const naturalLeftContext = runner.getConfig().leftContextSize;
                int32_t const leftContext = code2WavContextFrameCap >= 0
                    ? std::min(naturalLeftContext, code2WavContextFrameCap)
                    : naturalLeftContext;
                int32_t const windowStart = std::max(0, lastEmittedFrames - leftContext);
                int32_t const skipContextFrames = lastEmittedFrames - windowStart;
                auto const windowCodes = transposeFrameWindow(
                    streamedFrames, static_cast<size_t>(windowStart), static_cast<size_t>(totalFrames));

                auto const chunkStart = std::chrono::steady_clock::now();
                logMemTag(isFinal ? "worker_before_code2wav_final_chunk" : "worker_before_code2wav_chunk");
                auto samples = synthesizeWindow(runner, windowCodes, skipContextFrames, c2wStream);
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

                writeChunk(chunkIndex, isFinal, totalFrames, pcm, code2wavMs, chunkEnd, runner.getConfig().sampleRate);

                lastEmittedFrames = totalFrames;
                scheduleNextChunk();
                ++chunkIndex;
            };

            auto const genStart = std::chrono::steady_clock::now();
            bool ok = false;
            std::chrono::steady_clock::time_point genEnd{};
            if (statefulCode2Wav && statefulCode2wavRunners[c2wSlot])
            {
                // [Phase B C5b probe] no C5 lock — upstream runtime mutex
                // prevents corruption; per-slot reset is safe.
                statefulCode2wavRunners[c2wSlot]->reset(c2wStream);
            }
            if (streamOutput && asyncCode2Wav)
            {
                Code2WavRunner& asyncRunner = getAsyncCode2WavRunner(c2wSlot);
                cudaStream_t const asyncSlotStream = asyncCode2wavStreams[c2wSlot];
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

                            // [Cancel protocol step 1] Cooperative cancel
                            // checkpoint for the async Code2Wav path —
                            // same semantics as emitChunk above. Checked
                            // AFTER the streamCv wait and BEFORE the next
                            // synthesizeWindow enqueue so the previous
                            // chunk's kernels finish naturally.
                            if (cancelled.load(std::memory_order_acquire))
                            {
                                return;
                            }
                            auto const chunkStart = std::chrono::steady_clock::now();
                            // Phase 3b-B-4 part-2: per-slot async runner +
                            // its dedicated CUDA stream; no cross-request
                            // sharing, so no mutex.
                            std::vector<float> samples
                                = synthesizeWindow(asyncRunner, windowCodes, skipContextFrames, asyncSlotStream);
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

                {
                    std::lock_guard<std::mutex> runtimeLock(runtimeMutex);
                    ok = ttsRuntime->handleAudioGeneration(request, talkerResponse, stream, asyncFrameCallback);
                }
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
                {
                    std::lock_guard<std::mutex> runtimeLock(runtimeMutex);
                    ok = streamOutput ? ttsRuntime->handleAudioGeneration(request, talkerResponse, stream, frameCallback)
                                      : ttsRuntime->handleAudioGeneration(request, talkerResponse, stream);
                }
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
                // Phase 3b-B-4 part-2: read from THIS request's slot runner
                // (always populated by emitChunk above when streamOnly).
                int32_t const sampleRate = statefulCode2Wav
                    ? statefulCode2wavRunners[c2wSlot]->getConfig().sampleRate
                    : (asyncCode2Wav ? asyncCode2wavRunners[c2wSlot]->getConfig().sampleRate
                                     : code2wavRunners[c2wSlot]->getConfig().sampleRate);
                double const audioSeconds = static_cast<double>(streamedSamples) / sampleRate;
                double const totalMs = std::chrono::duration<double, std::milli>(doneAt - requestStart).count();
                response = Json{{"id", id},
                    {"ok", true},
                    {"stream_only", true},
                    {"frames", talkerResponse.numFrames},
                    {"samples", streamedSamples},
                    {"sample_rate", sampleRate},
                    {"audio_s", audioSeconds},
                    {"async_code2wav", asyncCode2Wav},
                    {"stateful_code2wav", statefulCode2Wav},
                    {"adaptive_chunks", adaptiveChunks},
                    {"chunk_frames", chunkFrames},
                    {"chunk_growth_frames", chunkGrowthFrames},
                    {"max_chunk_frames", maxChunkFrames},
                    {"chunk_count", chunkIndex},
                    {"audio_complete", true},
                    {"final_chunk_index", chunkIndex > 0 ? chunkIndex - 1 : -1},
                    {"last_chunk_was_final", chunkIndex > 0 && lastEmittedFrames == talkerResponse.numFrames},
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
                // [Cancel protocol step 1] If cancellation was observed
                // at any point during streaming, emit a terminal
                // "cancelled" event in lieu of "done". ok:true per spec
                // §4.1 — cancel is a normal control-flow event, not an
                // error; Python's _WorkerIO.request() must treat it as
                // terminal-non-error.
                if (cancelled.load(std::memory_order_acquire))
                {
                    emitEvent(id, "cancelled",
                        Json{{"ok", true}, {"reason", "client_disconnect"}});
                    return;
                }
                emitEvent(id, "done", std::move(response));
                return;
            }

            rt::audioUtils::AudioData audioOutput;
            auto const wavStart = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point wavEnd;
            {
                // Phase 3b-B-4 part-2: per-slot full-waveform Code2Wav path.
                if (!code2wavRunners[c2wSlot])
                {
                    std::lock_guard<std::mutex> lk(code2wavSlotInitMu[c2wSlot]);
                    if (!code2wavRunners[c2wSlot])
                    {
                        logMemTag("worker_before_lazy_code2wav");
                        code2wavRunners[c2wSlot]
                            = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, c2wStream);
                        logMemTag("worker_after_lazy_code2wav");
                    }
                }
                logMemTag("worker_before_code2wav_full");
                if (!code2wavRunners[c2wSlot]->generateWaveform(
                        transposeCodes(talkerResponse.rvqCodes), audioOutput, c2wStream))
                {
                    throw std::runtime_error("Code2Wav failed");
                }
                wavEnd = std::chrono::steady_clock::now();
                logMemTag("worker_after_code2wav_full");
            }
            if (!saveAudioToWav(outputFile, audioOutput))
            {
                throw std::runtime_error("Failed to save WAV: " + outputFile);
            }
            int64_t const samples
                = (audioOutput.waveform && !audioOutput.waveform->isEmpty()) ? audioOutput.waveform->getShape()[1] : 0;
            double const audioSeconds = static_cast<double>(samples) / audioOutput.sampleRate;
            double const totalMs = std::chrono::duration<double, std::milli>(wavEnd - requestStart).count();
            response = Json{{"id", id},
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
            response = Json{{"ok", false}, {"error", e.what()}};
            emitEvent(id, "error", std::move(response));
            return;
        }
        // [Cancel protocol step 1] Same terminal-event swap as the
        // streamOnly path above: if a cancel was observed during the
        // full-waveform generation, emit "cancelled" in lieu of "done".
        // Note: the full-waveform code path has no chunk-boundary
        // check today (Code2Wav runs as a single call), so cancel can
        // only take effect at the very end of generation. The chunk
        // emit loop is the main observation point; this branch covers
        // the (rare) race where cancel arrives after the last chunk
        // but before the terminal emit.
        if (cancelled.load(std::memory_order_acquire))
        {
            emitEvent(id, "cancelled",
                Json{{"ok", true}, {"reason", "client_disconnect"}});
            return;
        }
        emitEvent(id, "done", std::move(response));
    };

    // Reader loop: pull a line, wait for a free slot, spawn a worker thread.
    // At concurrency=1 this matches the pre-refactor behaviour modulo the
    // extra thread::create+join overhead per request (a few hundred us,
    // amortized over multi-second TTS requests). At N>1 talker / code-pred
    // AND Code2Wav stages all overlap (Phase 3b-B-4 part-2 per-slot Code2Wav
    // runners + per-slot CUDA streams).
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }
        // [Cancel protocol step 1] Parse the request type BEFORE waiting
        // for capacity. At concurrency=1, the single in-flight request
        // owns the only slot; if we waited for capacity before parsing,
        // a cancel for that very request would block on workersCv until
        // the request it's trying to cancel finishes — deadlock.
        // Cancel messages bypass the capacity gate entirely: look up
        // the request_id, set its atomic, continue. No worker thread
        // is spawned for a cancel.
        //
        // Malformed JSON falls through to the legacy path (worker
        // thread + handleRequest's try/catch surfaces the parse error
        // as an "error" event with id="__worker__"). This preserves
        // pre-cancel behaviour for protocol-violating clients.
        try
        {
            Json const peek = Json::parse(line);
            std::string const type = peek.value("type", "");
            if (type == "cancel")
            {
                std::string const cancelId = peek.value("id", "");
                if (!cancelId.empty())
                {
                    std::atomic<bool>* flag = nullptr;
                    {
                        std::lock_guard<std::mutex> lk(cancelMapMu);
                        auto it = cancelMap.find(cancelId);
                        if (it != cancelMap.end())
                        {
                            flag = it->second;
                        }
                    }
                    if (flag != nullptr)
                    {
                        flag->store(true, std::memory_order_release);
                    }
                    // Unknown id (request already finished, or never
                    // existed): silently drop. The spec §4.3 documents
                    // this as a no-op so late-arriving cancels after
                    // natural completion don't break anything.
                }
                continue;
            }
        }
        catch (std::exception const&)
        {
            // Fall through to the worker-thread path; handleRequest
            // will re-parse and emit a structured error event.
        }
        {
            std::unique_lock<std::mutex> lk(workersMu);
            workersCv.wait(lk, [&]() { return inFlight < concurrency; });
            ++inFlight;
        }
        std::string lineCopy = line;
        std::lock_guard<std::mutex> add(workersMu);
        workers.emplace_back([lineCopy, &handleRequest, &inFlight, &workersCv, &workersMu]() {
            try
            {
                handleRequest(lineCopy);
            }
            catch (std::exception const& e)
            {
                // handleRequest catches its own per-request exceptions and
                // emits "error" events; this outer catch is a safety net
                // for anything that escapes (allocation failures, etc.)
                // so the dispatcher slot is always released.
                emitEvent("__worker__", "error", Json{{"ok", false}, {"error", e.what()}});
            }
            {
                std::lock_guard<std::mutex> lk(workersMu);
                if (inFlight > 0)
                {
                    --inFlight;
                }
            }
            workersCv.notify_one();
        });
    }

    // Shutdown: stdin closed -> drain remaining workers. Joining the whole
    // list also pays back the std::thread objects we never reaped during
    // steady-state (kept simple; bounded slot count caps memory anyway).
    for (auto& w : workers)
    {
        if (w.joinable())
        {
            w.join();
        }
    }

    // Phase 3b-B-4 part-2: tear down per-slot runners and their dedicated
    // CUDA streams. Runners reset() first (so device buffers free before
    // their stream is destroyed); then streams destroyed (skip slot 0 — it
    // reused the main `stream`, freed below).
    for (auto& r : asyncCode2wavRunners)
    {
        r.reset();
    }
    for (auto& r : code2wavRunners)
    {
        r.reset();
    }
    for (auto& r : statefulCode2wavRunners)
    {
        r.reset();
    }
    for (auto& s : asyncCode2wavStreams)
    {
        if (s)
        {
            CUDA_CHECK(cudaStreamDestroy(s));
        }
    }
    for (size_t i = 1; i < code2wavStreams.size(); ++i)
    {
        if (code2wavStreams[i])
        {
            CUDA_CHECK(cudaStreamDestroy(code2wavStreams[i]));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
