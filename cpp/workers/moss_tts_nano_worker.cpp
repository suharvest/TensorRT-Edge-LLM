/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MOSS-TTS-Nano stdin/stdout JSON-line worker.
 *
 * Protocol mirrors qwen3_tts_worker (see scripts/run_qwen3_tts_worker_smoke.py).
 *
 * Prompt structure & special tokens sourced from:
 *   - browser_poc_manifest.json (prompt_templates, tts_config, builtin_voices)
 *   - tts_browser_onnx_meta.json (model_config.* token IDs)
 *   - codec_browser_onnx_meta.json (codec sample rate / num_quantizers / encode IO)
 *
 * Runtime: MossTtsNanoRuntime (cpp/runtime/mossTtsNanoRuntime.h).
 */

#include "runtime/mossTtsNanoRuntime.h"

#include <algorithm>
#include <array>
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
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Json = nlohmann::json;
using trt_edgellm::rt::MossTtsNanoRuntime;

namespace
{

// ─── CLI args ────────────────────────────────────────────────────────────────

struct Args
{
    std::string engineDir;        // contains *.plan + tts/codec meta + browser_poc_manifest.json
    std::string tokenizerModel;   // SentencePiece tokenizer.model path
    std::string codecOnnxDir;     // dir containing moss_audio_tokenizer_encode.onnx (for voice clone)
    int32_t maxSlots{1};
    int32_t maxSeqLen{1024};
};

// ─── TTS config (merged from browser_poc_manifest.json + tts_browser_onnx_meta.json) ────

struct TtsConfig
{
    int32_t nVq{16};
    int32_t rowWidth{17};
    int32_t audioPadTokenId{1024};
    int32_t padTokenId{3};
    int32_t imStartTokenId{4};
    int32_t imEndTokenId{5};
    int32_t audioStartTokenId{6};
    int32_t audioEndTokenId{7};
    int32_t audioUserSlotTokenId{8};
    int32_t audioAssistantSlotTokenId{9};
    std::vector<int32_t> userPromptPrefixTokenIds;
    std::vector<int32_t> userPromptAfterReferenceTokenIds;
    std::vector<int32_t> assistantPromptPrefixTokenIds;
    // Default voice conditioning (audio codes from builtin_voices[0].prompt_audio_codes).
    // Python infer_onnx uses these when no ref_audio is provided. Without them, the LM
    // has no voice condition and produces garbage.
    std::vector<std::vector<int32_t>> defaultVoiceAudioCodes;
};

// ─── Codec encode runner (ORT-CPU; voice clone path) ─────────────────────────

struct CodecEncodeRunner
{
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "moss_codec_encode"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    std::string inputName0{"waveform"};
    std::string inputName1{"input_lengths"};
    std::string outputName0{"audio_codes"};
    std::string outputName1{"audio_code_lengths"};
    int32_t sampleRate{48000};
    int32_t channels{2};
    int32_t numQuantizers{16};

    explicit CodecEncodeRunner(std::string const& codecOnnxDir)
    {
        if (codecOnnxDir.empty()) return;
        namespace fs = std::filesystem;
        fs::path const dir(codecOnnxDir);
        fs::path const metaPath = dir / "codec_browser_onnx_meta.json";
        if (!fs::exists(metaPath)) return;

        std::ifstream metaFile(metaPath);
        Json meta = Json::parse(metaFile);
        std::string onnxRelPath = meta.at("files").at("encode").get<std::string>();
        std::string onnxPath = (dir / onnxRelPath).string();

        sampleRate = meta.at("codec_config").at("sample_rate").get<int32_t>();
        channels = meta.at("codec_config").at("channels").get<int32_t>();
        numQuantizers = meta.at("codec_config").at("num_quantizers").get<int32_t>();

        if (meta.contains("onnx"))
        {
            auto const& on = meta["onnx"];
            if (on.contains("encode_input_names"))
            {
                auto const& a = on["encode_input_names"];
                if (a.size() >= 1) inputName0 = a[0].get<std::string>();
                if (a.size() >= 2) inputName1 = a[1].get<std::string>();
            }
            if (on.contains("encode_output_names"))
            {
                auto const& a = on["encode_output_names"];
                if (a.size() >= 1) outputName0 = a[0].get<std::string>();
                if (a.size() >= 2) outputName1 = a[1].get<std::string>();
            }
        }

        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        session = std::make_unique<Ort::Session>(env, onnxPath.c_str(), sessionOptions);
    }

    bool available() const { return static_cast<bool>(session); }
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

int64_t elapsedMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

// N=2 concurrency: a single stdout is shared across multiple worker threads.
// Without serialization, JSON lines from different requests can interleave
// at byte boundaries (event A's "{...}\n" tail mixing with event B's head),
// breaking the Python-side reader that does line-buffered json.loads().
// emit() therefore holds gStdoutMutex around the full dump + newline + flush.
std::mutex gStdoutMutex;

void emit(Json payload)
{
    std::lock_guard<std::mutex> lock(gStdoutMutex);
    std::cout << payload.dump() << '\n' << std::flush;
}

std::string base64Encode(uint8_t const* data, size_t len)
{
    static constexpr char kTable[]
        = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3)
    {
        uint32_t b0 = data[i];
        uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
        uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;
        uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? kTable[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? kTable[triple & 0x3F] : '=');
    }
    return out;
}

std::vector<uint8_t> base64Decode(std::string const& input)
{
    std::array<int8_t, 256> table{};
    table.fill(-1);
    for (int i = 0; i < 26; ++i)
    {
        table[static_cast<uint8_t>('A' + i)] = static_cast<int8_t>(i);
        table[static_cast<uint8_t>('a' + i)] = static_cast<int8_t>(i + 26);
    }
    for (int i = 0; i < 10; ++i)
        table[static_cast<uint8_t>('0' + i)] = static_cast<int8_t>(i + 52);
    table[static_cast<uint8_t>('+')] = 62;
    table[static_cast<uint8_t>('/')] = 63;

    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : input)
    {
        if (c == '=') break;
        int8_t d = table[c];
        if (d < 0) continue;
        val = (val << 6) + d;
        bits += 6;
        if (bits >= 0)
        {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

std::vector<int16_t> floatToPcm16(std::vector<float> const& samples, size_t begin)
{
    std::vector<int16_t> pcm;
    if (begin >= samples.size()) return pcm;
    pcm.reserve(samples.size() - begin);
    for (size_t i = begin; i < samples.size(); ++i)
    {
        float c = std::clamp(samples[i], -1.0f, 1.0f);
        pcm.push_back(static_cast<int16_t>(std::lrint(c * 32767.0f)));
    }
    return pcm;
}

std::vector<int32_t> jsonIntVector(Json const& arr)
{
    std::vector<int32_t> v;
    v.reserve(arr.size());
    for (auto const& el : arr) v.push_back(el.get<int32_t>());
    return v;
}

// ─── Config loading ──────────────────────────────────────────────────────────

TtsConfig loadTtsConfig(std::string const& engineDir)
{
    namespace fs = std::filesystem;
    TtsConfig cfg;
    fs::path const dir(engineDir);

    // 1) Load tts_browser_onnx_meta.json for model_config.* (token IDs)
    fs::path const ttsMetaPath = dir / "tts_browser_onnx_meta.json";
    if (fs::exists(ttsMetaPath))
    {
        std::ifstream f(ttsMetaPath);
        Json meta = Json::parse(f);
        if (meta.contains("model_config"))
        {
            auto const& mc = meta["model_config"];
            cfg.nVq = mc.value("n_vq", cfg.nVq);
            cfg.rowWidth = mc.value("row_width", cfg.rowWidth);
            cfg.audioPadTokenId = mc.value("audio_pad_token_id", cfg.audioPadTokenId);
            cfg.padTokenId = mc.value("pad_token_id", cfg.padTokenId);
            cfg.imStartTokenId = mc.value("im_start_token_id", cfg.imStartTokenId);
            cfg.imEndTokenId = mc.value("im_end_token_id", cfg.imEndTokenId);
            cfg.audioStartTokenId = mc.value("audio_start_token_id", cfg.audioStartTokenId);
            cfg.audioEndTokenId = mc.value("audio_end_token_id", cfg.audioEndTokenId);
            cfg.audioUserSlotTokenId = mc.value("audio_user_slot_token_id", cfg.audioUserSlotTokenId);
            cfg.audioAssistantSlotTokenId
                = mc.value("audio_assistant_slot_token_id", cfg.audioAssistantSlotTokenId);
        }
    }
    if (cfg.nVq + 1 != cfg.rowWidth)
        throw std::runtime_error("Unsupported MOSS row_width; expected n_vq+1=17");

    // 2) Load browser_poc_manifest.json for prompt_templates.*
    fs::path const manifestPath = dir / "browser_poc_manifest.json";
    if (fs::exists(manifestPath))
    {
        std::ifstream f(manifestPath);
        Json manifest = Json::parse(f);
        if (manifest.contains("prompt_templates"))
        {
            auto const& t = manifest["prompt_templates"];
            if (t.contains("user_prompt_prefix_token_ids"))
                cfg.userPromptPrefixTokenIds = jsonIntVector(t["user_prompt_prefix_token_ids"]);
            if (t.contains("user_prompt_after_reference_token_ids"))
                cfg.userPromptAfterReferenceTokenIds
                    = jsonIntVector(t["user_prompt_after_reference_token_ids"]);
            if (t.contains("assistant_prompt_prefix_token_ids"))
                cfg.assistantPromptPrefixTokenIds
                    = jsonIntVector(t["assistant_prompt_prefix_token_ids"]);
        }
    }
    if (cfg.userPromptPrefixTokenIds.empty() || cfg.assistantPromptPrefixTokenIds.empty())
    {
        std::cerr << "[moss_worker] WARNING: prompt_templates missing in browser_poc_manifest.json"
                  << " — generation will likely be incorrect.\n";
    }

    // Load builtin_voices[0].prompt_audio_codes for default voice conditioning.
    // Python infer_onnx defaults to this; without it, LM has no voice condition and produces garbage.
    if (fs::exists(manifestPath))
    {
        std::ifstream f(manifestPath);
        Json manifest = Json::parse(f);
        if (manifest.contains("builtin_voices") && manifest["builtin_voices"].is_array()
            && !manifest["builtin_voices"].empty())
        {
            // Pick voice with most prompt_audio_codes rows (longer = more LM conditioning).
            size_t bestIdx = 0;
            size_t bestRows = 0;
            for (size_t i = 0; i < manifest["builtin_voices"].size(); ++i)
            {
                auto const& v = manifest["builtin_voices"][i];
                if (v.contains("prompt_audio_codes") && v["prompt_audio_codes"].is_array())
                {
                    size_t const rows = v["prompt_audio_codes"].size();
                    if (rows > bestRows) { bestRows = rows; bestIdx = i; }
                }
            }
            auto const& voice = manifest["builtin_voices"][bestIdx];
            if (voice.contains("prompt_audio_codes") && voice["prompt_audio_codes"].is_array())
            {
                for (auto const& row : voice["prompt_audio_codes"])
                {
                    cfg.defaultVoiceAudioCodes.push_back(jsonIntVector(row));
                }
                std::cerr << "[moss_worker] loaded default voice '"
                          << voice.value("voice", std::string("?"))
                          << "' with " << cfg.defaultVoiceAudioCodes.size()
                          << " audio code rows\n";
            }
        }
    }

    return cfg;
}

// ─── Prompt row builder ──────────────────────────────────────────────────────

using Row = std::vector<int32_t>;

Row makeTextRow(int32_t tokenId, TtsConfig const& cfg)
{
    Row row(static_cast<size_t>(cfg.rowWidth), cfg.audioPadTokenId);
    row[0] = tokenId;
    return row;
}

Row makeAudioRow(std::vector<int32_t> const& codes, int32_t slotToken, TtsConfig const& cfg)
{
    Row row(static_cast<size_t>(cfg.rowWidth), cfg.audioPadTokenId);
    row[0] = slotToken;
    for (size_t i = 0; i < std::min(codes.size(), static_cast<size_t>(cfg.nVq)); ++i)
        row[i + 1] = codes[i];
    return row;
}

std::vector<int32_t> flattenRows(std::vector<Row> const& rows, int32_t rowWidth)
{
    std::vector<int32_t> flat;
    flat.reserve(rows.size() * static_cast<size_t>(rowWidth));
    for (auto const& row : rows)
    {
        if (static_cast<int32_t>(row.size()) != rowWidth)
            throw std::runtime_error("MOSS row width mismatch");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

void appendTextRows(std::vector<Row>& rows, std::vector<int32_t> const& ids, TtsConfig const& cfg)
{
    for (int32_t id : ids) rows.push_back(makeTextRow(id, cfg));
}

// Build input_ids matching ort_cpu_runtime.py:494-510 prompt order.
std::vector<int32_t> buildInputIds(std::vector<int32_t> const& textTokenIds,
    std::vector<std::vector<int32_t>> const& refAudioCodes, TtsConfig const& cfg)
{
    std::vector<Row> rows;
    appendTextRows(rows, cfg.userPromptPrefixTokenIds, cfg);
    rows.push_back(makeTextRow(cfg.audioStartTokenId, cfg));
    for (auto const& codeRow : refAudioCodes)
        rows.push_back(makeAudioRow(codeRow, cfg.audioUserSlotTokenId, cfg));
    rows.push_back(makeTextRow(cfg.audioEndTokenId, cfg));
    appendTextRows(rows, cfg.userPromptAfterReferenceTokenIds, cfg);
    appendTextRows(rows, textTokenIds, cfg);
    appendTextRows(rows, cfg.assistantPromptPrefixTokenIds, cfg);
    rows.push_back(makeTextRow(cfg.audioStartTokenId, cfg));
    return flattenRows(rows, cfg.rowWidth);
}

// Per-decode-step row: assistant audio output frame fed back as next input.
// TODO [VERIFY]: confirm against ort_cpu_runtime.py decode loop. Slot token in col 0 may
// alternate between audio_assistant_slot and a different ID for end-of-turn rows.
Row makeNextAudioDecodeRow(std::vector<int32_t> const& frameTokens, TtsConfig const& cfg)
{
    Row row(static_cast<size_t>(cfg.rowWidth), cfg.audioPadTokenId);
    row[0] = cfg.audioAssistantSlotTokenId;
    for (size_t i = 0; i < std::min(frameTokens.size(), static_cast<size_t>(cfg.nVq)); ++i)
        row[i + 1] = frameTokens[i];
    return row;
}

// ─── Codec encode (voice clone) ──────────────────────────────────────────────

std::vector<float> pcm16ToChannelFirst(std::vector<uint8_t> const& bytes, int32_t channels)
{
    if (bytes.size() % sizeof(int16_t) != 0)
        throw std::runtime_error("ref_audio_b64: byte count not even");
    size_t totalSamples = bytes.size() / sizeof(int16_t);
    if (totalSamples % static_cast<size_t>(channels) != 0)
        throw std::runtime_error("ref_audio_b64: sample count not divisible by channels");
    size_t frames = totalSamples / static_cast<size_t>(channels);
    std::vector<float> out(totalSamples);
    for (size_t f = 0; f < frames; ++f)
    {
        for (int32_t ch = 0; ch < channels; ++ch)
        {
            int16_t v;
            std::memcpy(&v,
                bytes.data() + (f * static_cast<size_t>(channels) + ch) * sizeof(int16_t),
                sizeof(int16_t));
            out[static_cast<size_t>(ch) * frames + f] = static_cast<float>(v) / 32768.0f;
        }
    }
    return out;
}

std::vector<std::vector<int32_t>> encodeReferenceAudio(
    CodecEncodeRunner& codec, std::string const& refAudioB64, int32_t refSampleRate)
{
    if (!codec.available() || refAudioB64.empty()) return {};
    if (refSampleRate != codec.sampleRate)
        throw std::runtime_error("ref_audio_sample_rate mismatch: expected "
            + std::to_string(codec.sampleRate) + " got " + std::to_string(refSampleRate));

    auto bytes = base64Decode(refAudioB64);
    auto waveform = pcm16ToChannelFirst(bytes, codec.channels);
    int64_t frames = static_cast<int64_t>(waveform.size() / static_cast<size_t>(codec.channels));

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> waveformShape{1, codec.channels, frames};
    std::array<int64_t, 1> lengthShape{1};
    int32_t inputLength = static_cast<int32_t>(frames);

    char const* inNames[2] = {codec.inputName0.c_str(), codec.inputName1.c_str()};
    char const* outNames[2] = {codec.outputName0.c_str(), codec.outputName1.c_str()};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memInfo, waveform.data(), waveform.size(), waveformShape.data(), waveformShape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(
        memInfo, &inputLength, 1, lengthShape.data(), lengthShape.size()));

    auto outputs = codec.session->Run(
        Ort::RunOptions{nullptr}, inNames, inputs.data(), 2, outNames, 2);

    int32_t* codes = outputs[0].GetTensorMutableData<int32_t>();
    int32_t* lengths = outputs[1].GetTensorMutableData<int32_t>();
    int32_t codeLength = lengths[0];

    // TODO [VERIFY]: codec encode output layout [1, frames, n_vq] vs [1, n_vq, frames].
    // Currently assumed row-major [B, T, Q]. If garbled clone audio, transpose.
    std::vector<std::vector<int32_t>> rows;
    rows.reserve(static_cast<size_t>(codeLength));
    for (int32_t f = 0; f < codeLength; ++f)
    {
        std::vector<int32_t> row(static_cast<size_t>(codec.numQuantizers));
        for (int32_t q = 0; q < codec.numQuantizers; ++q)
            row[static_cast<size_t>(q)] = codes[
                static_cast<size_t>(f) * static_cast<size_t>(codec.numQuantizers)
                + static_cast<size_t>(q)];
        rows.push_back(std::move(row));
    }
    return rows;
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

std::vector<int32_t> tokenize(sentencepiece::SentencePieceProcessor& sp, std::string const& text)
{
    std::vector<int> ids;
    auto const st = sp.Encode(text, &ids);
    if (!st.ok()) throw std::runtime_error("SentencePiece encode failed: " + st.ToString());
    return std::vector<int32_t>(ids.begin(), ids.end());
}

// ─── CLI parsing ─────────────────────────────────────────────────────────────

void printUsage(char const* prog)
{
    std::cerr << "Usage: " << prog
              << " --engine-dir=PATH --tokenizer-model=PATH"
                 " [--codec-onnx-dir=PATH] [--max-slots=N] [--max-seq-len=N]\n";
}

bool parseArgs(int argc, char** argv, Args& args)
{
    enum OptionId { HELP = 1000, ENGINE_DIR, TOKENIZER_MODEL, CODEC_ONNX_DIR, MAX_SLOTS, MAX_SEQ_LEN };
    static option opts[] = {
        {"help", no_argument, nullptr, HELP},
        {"engine-dir", required_argument, nullptr, ENGINE_DIR},
        {"tokenizer-model", required_argument, nullptr, TOKENIZER_MODEL},
        {"codec-onnx-dir", required_argument, nullptr, CODEC_ONNX_DIR},
        {"max-slots", required_argument, nullptr, MAX_SLOTS},
        {"max-seq-len", required_argument, nullptr, MAX_SEQ_LEN},
        {nullptr, 0, nullptr, 0},
    };
    while (true)
    {
        int o = getopt_long(argc, argv, "", opts, nullptr);
        if (o == -1) break;
        switch (o)
        {
        case HELP: printUsage(argv[0]); std::exit(0);
        case ENGINE_DIR: args.engineDir = optarg; break;
        case TOKENIZER_MODEL: args.tokenizerModel = optarg; break;
        case CODEC_ONNX_DIR: args.codecOnnxDir = optarg; break;
        case MAX_SLOTS: args.maxSlots = std::max(1, std::atoi(optarg)); break;
        case MAX_SEQ_LEN: args.maxSeqLen = std::max(1, std::atoi(optarg)); break;
        default: return false;
        }
    }
    return !args.engineDir.empty() && !args.tokenizerModel.empty();
}

// ─── Request handler ─────────────────────────────────────────────────────────

void handleRequest(Json const& item, MossTtsNanoRuntime& runtime,
    sentencepiece::SentencePieceProcessor& sp, std::mutex& spMutex,
    CodecEncodeRunner& codec, std::mutex& codecMutex, TtsConfig const& cfg)
{
    // Prefer "request_id" (spec §1 — TTS worker concurrency framework), fall
    // back to the legacy "id" field that older Python callers still use.
    // Backward compat: emit() events carry BOTH "request_id" and "id" so
    // either reader format works.
    std::string id = item.value("request_id", std::string{});
    if (id.empty())
        id = item.value("id", std::string("__request__"));
    auto const requestStart = std::chrono::steady_clock::now();

    if (!item.value("stream", true) || !item.value("stream_only", true))
        throw std::runtime_error("Worker only supports stream=true, stream_only=true");
    if (item.value("chunk_transport", std::string("base64")) != "base64")
        throw std::runtime_error("Worker only supports chunk_transport=base64");
    if (item.value("chunk_format", std::string("pcm_s16le")) != "pcm_s16le")
        throw std::runtime_error("Worker only supports chunk_format=pcm_s16le");

    std::string const text = item.value("text", std::string(""));
    if (text.empty()) throw std::runtime_error("text must not be empty");

    emit(Json{{"event", "ready"}, {"id", id}, {"request_id", id}, {"ok", true},
        {"sample_rate", runtime.codecSampleRate()}, {"channels", runtime.codecChannels()}});

    std::vector<std::vector<int32_t>> refAudioCodes;
    if (item.contains("ref_audio_b64") && item["ref_audio_b64"].is_string()
        && !item["ref_audio_b64"].get<std::string>().empty())
    {
        try {
            // codec is shared (single ORT session); serialize encode across worker threads.
            std::lock_guard<std::mutex> lock(codecMutex);
            refAudioCodes = encodeReferenceAudio(codec, item["ref_audio_b64"].get<std::string>(),
                item.value("ref_audio_sample_rate", codec.sampleRate));
            std::fprintf(stderr, "[moss_worker] encoded ref audio → %zu code rows (codec.available=%d)\n",
                refAudioCodes.size(), int(codec.available()));
        } catch (std::exception const& e) {
            std::fprintf(stderr, "[moss_worker] ref audio encode failed: %s\n", e.what());
            refAudioCodes.clear();
        }
    }
    if (refAudioCodes.empty())
    {
        // Either no ref_audio_b64 in request, OR encode returned empty (codec unavailable / failed).
        // Fall back to default voice conditioning — match Python infer_onnx behavior.
        refAudioCodes = cfg.defaultVoiceAudioCodes;
        std::fprintf(stderr, "[moss_worker] using default voice → %zu code rows\n", refAudioCodes.size());
    }

    std::vector<int32_t> textIds;
    {
        // SentencePiece is shared; Encode is not documented as thread-safe — lock.
        std::lock_guard<std::mutex> lock(spMutex);
        textIds = tokenize(sp, text);
    }
    std::vector<int32_t> inputIds = buildInputIds(textIds, refAudioCodes, cfg);
    std::vector<int32_t> attentionMask(inputIds.size() / static_cast<size_t>(cfg.rowWidth), 1);

    auto guard = runtime.beginRequest();
    auto& slot = guard.slot();
    runtime.resetCodec(slot);

    // Codec engine was built with --maxShapes=audio_codes:1x8x16; decodeFrames
    // throws if a single batch exceeds 8 frames. Clamp chunk budgets to that ceiling.
    // To raise: rebuild codec engine with bigger maxShapes and lift
    // MossCodecState::maxFrames in mossTtsNanoRuntime.h.
    constexpr int32_t kCodecMaxFramesPerBatch = 8;
    int32_t const firstChunkBudget = std::clamp(
        item.value("first_chunk_frames", 4), 1, kCodecMaxFramesPerBatch);
    int32_t const chunkBudget = std::clamp(
        item.value("chunk_frames", 8), 1, kCodecMaxFramesPerBatch);

    // decode_step engine was built with --maxShapes=past_key_*:1x256x12x64; setInputShape
    // fails the moment pastLen would exceed 256. Decode budget = 256 - prefillSeqLen.
    // To raise: rebuild prefill/decode engines with bigger past_key max profile.
    constexpr int32_t kKvProfileMaxPast = 512;
    int32_t const prefillSeqLen = static_cast<int32_t>(attentionMask.size());
    int32_t const decodeBudget = std::max(0, kKvProfileMaxPast - prefillSeqLen);
    int32_t const maxNewFrames = std::min(
        std::max(1, item.value("max_new_frames", 1000)), decodeBudget);

    void* globalHidden = runtime.prefill(slot, inputIds, attentionMask);

    std::vector<std::vector<int32_t>> frameBuffer;
    frameBuffer.reserve(static_cast<size_t>(chunkBudget));
    std::vector<float> pcmOut;
    size_t emittedSamples = 0;
    int32_t frameIdx = 0;
    int32_t chunkIdx = 0;
    int64_t ttfaMs = -1;

    auto flushBuffer = [&](bool force) {
        int32_t budget = (chunkIdx == 0) ? firstChunkBudget : chunkBudget;
        if (!force && static_cast<int32_t>(frameBuffer.size()) < budget) return;
        if (frameBuffer.empty()) return;

        runtime.decodeFrames(slot, frameBuffer, pcmOut);
        frameBuffer.clear();

        auto now = std::chrono::steady_clock::now();
        if (ttfaMs < 0) ttfaMs = elapsedMs(requestStart, now);

        std::vector<int16_t> pcm = floatToPcm16(pcmOut, emittedSamples);
        emittedSamples = pcmOut.size();

        emit(Json{{"event", "chunk"}, {"id", id}, {"request_id", id}, {"ok", true},
            {"audio_b64",
                base64Encode(reinterpret_cast<uint8_t const*>(pcm.data()), pcm.size() * sizeof(int16_t))},
            {"frame_index", chunkIdx}, {"samples", static_cast<int32_t>(pcm.size())}});
        ++chunkIdx;
    };

    for (frameIdx = 0; frameIdx < maxNewFrames; ++frameIdx)
    {
        std::vector<int32_t> frameTokens;
        bool const cont = runtime.sampleFrame(slot, globalHidden, frameTokens);
        if (!cont) break;

        frameBuffer.push_back(frameTokens);
        flushBuffer(false);

        if (frameIdx + 1 >= maxNewFrames) break;

        Row nextRow = makeNextAudioDecodeRow(frameTokens, cfg);
        globalHidden = runtime.decodeStep(slot, nextRow, globalHidden);
    }

    flushBuffer(true);

    auto doneAt = std::chrono::steady_clock::now();
    emit(Json{{"event", "done"}, {"id", id}, {"request_id", id}, {"ok", true},
        {"total_samples", static_cast<int32_t>(pcmOut.size())}, {"ttfa_ms", ttfaMs},
        {"wall_ms", elapsedMs(requestStart, doneAt)}});
}

// ─── Request dispatch (N=2 concurrency) ───────────────────────────────────────
//
// Why a worker-thread pool instead of round-robin per request?
//
// Runtime is already designed for parallel callers — slot pool +
// std::this_thread::get_id()→slotId map (see mossTtsNanoRuntime.h
// mActiveSlots) means each worker thread acquires its own slot and runs
// prefill/decode/codec concurrently on independent CUDA streams + TRT
// contexts. The main thread only reads stdin and dispatches.

struct PendingRequest
{
    Json item;
    std::string id;
};

struct WorkerPool
{
    std::deque<PendingRequest> queue;
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    void push(PendingRequest&& req)
    {
        {
            std::lock_guard<std::mutex> lock(mu);
            queue.push_back(std::move(req));
        }
        cv.notify_one();
    }

    void shutdown()
    {
        stop.store(true);
        cv.notify_all();
        for (auto& t : threads)
            if (t.joinable()) t.join();
        threads.clear();
    }
};

void workerThreadLoop(WorkerPool& pool, MossTtsNanoRuntime& runtime,
    sentencepiece::SentencePieceProcessor& sp, std::mutex& spMutex,
    CodecEncodeRunner& codec, std::mutex& codecMutex, TtsConfig const& cfg)
{
    while (true)
    {
        PendingRequest req;
        {
            std::unique_lock<std::mutex> lock(pool.mu);
            pool.cv.wait(lock, [&] { return pool.stop.load() || !pool.queue.empty(); });
            if (pool.stop.load() && pool.queue.empty()) return;
            req = std::move(pool.queue.front());
            pool.queue.pop_front();
        }
        try
        {
            handleRequest(req.item, runtime, sp, spMutex, codec, codecMutex, cfg);
        }
        catch (std::exception const& e)
        {
            emit(Json{{"event", "error"}, {"id", req.id}, {"request_id", req.id},
                {"ok", false}, {"error", e.what()}});
        }
        catch (...)
        {
            emit(Json{{"event", "error"}, {"id", req.id}, {"request_id", req.id},
                {"ok", false}, {"error", "unknown exception"}});
        }
    }
}

} // anonymous namespace

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    try
    {
        Args args;
        if (!parseArgs(argc, argv, args))
        {
            printUsage(argv[0]);
            emit(Json{{"event", "error"}, {"id", "__worker__"}, {"ok", false},
                {"error", "invalid arguments"}});
            return 1;
        }

        TtsConfig cfg = loadTtsConfig(args.engineDir);

        sentencepiece::SentencePieceProcessor sp;
        auto const spSt = sp.Load(args.tokenizerModel);
        if (!spSt.ok())
            throw std::runtime_error("Failed to load tokenizer: " + spSt.ToString());

        CodecEncodeRunner codec(args.codecOnnxDir);
        MossTtsNanoRuntime runtime(args.engineDir, args.maxSlots, args.maxSeqLen);

        emit(Json{{"event", "worker_ready"}, {"ok", true},
            {"sample_rate", runtime.codecSampleRate()}, {"channels", runtime.codecChannels()},
            {"engine_dir", args.engineDir},
            {"voice_clone_enabled", codec.available()},
            {"prompt_template_loaded", !cfg.userPromptPrefixTokenIds.empty()},
            {"max_slots", args.maxSlots}});

        // N=2 concurrency: spawn maxSlots worker threads sharing the runtime's
        // slot pool. Main thread reads stdin and pushes requests into the queue.
        // At maxSlots=1 the pool degenerates to single-thread serial (same
        // behavior as the pre-N=2 baseline, byte-equivalent output expected).
        std::mutex spMutex;
        std::mutex codecMutex;
        WorkerPool pool;
        for (int32_t i = 0; i < args.maxSlots; ++i)
        {
            pool.threads.emplace_back(workerThreadLoop, std::ref(pool), std::ref(runtime),
                std::ref(sp), std::ref(spMutex), std::ref(codec), std::ref(codecMutex), std::cref(cfg));
        }

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty()) continue;
            std::string id = "__worker__";
            try
            {
                Json item = Json::parse(line);
                // Prefer request_id, fall back to id for protocol back-compat.
                std::string rid = item.value("request_id", std::string{});
                if (rid.empty()) rid = item.value("id", id);
                id = rid;
                pool.push(PendingRequest{std::move(item), id});
            }
            catch (std::exception const& e)
            {
                emit(Json{{"event", "error"}, {"id", id}, {"request_id", id},
                    {"ok", false}, {"error", e.what()}});
            }
            catch (...)
            {
                emit(Json{{"event", "error"}, {"id", id}, {"request_id", id},
                    {"ok", false}, {"error", "unknown exception"}});
            }
        }
        // stdin EOF — drain queue and join worker threads.
        pool.shutdown();
        return 0;
    }
    catch (std::exception const& e)
    {
        emit(Json{{"event", "error"}, {"id", "__worker__"}, {"ok", false}, {"error", e.what()}});
        return 1;
    }
    catch (...)
    {
        emit(Json{{"event", "error"}, {"id", "__worker__"}, {"ok", false},
            {"error", "unknown exception"}});
        return 1;
    }
}
