// Streaming-ASR Milestone 3.6 empirical LCS test.
//
// Why this test exists
// --------------------
// M3.5 proved the audio encoder is per-block-causal in CNN, but the transformer
// attention inside the audio encoder spans up to 8 mel-blocks (the
// `n_window_infer=800` block-diagonal mask). Splitting audio mid-attention-window
// at the per-chunk encodeMelChunk boundary will produce slightly different audio
// embeddings near boundary tokens. The 0.106 max-abs diff measured in M3.5's
// initial naive-split path quantifies the embedding drift, but does NOT answer
// the load-bearing question: does the drift change the decoded TEXT?
//
// If the LCS-similarity between Path A (single chunk over full mel) and
// Path B (K aligned chunks fed via appendPrefillEmbeds) is >= 0.95 for our
// production chunk sizes, the streaming worker can ship as-is — no overlap,
// no attention-cache. If LCS is decisively below threshold, we must build an
// overlap-and-discard or a cross-chunk attention cache.
//
// Test layout
// -----------
// All paths share the same plumbing: beginAsrSession → for each mel chunk
// (encodeMelChunk + appendPrefillEmbeds) → decodeAfterChunkedPrefillForTesting
// → endAsrSession. This eliminates noise from handleRequest-vs-chunked path
// divergence (M1 already proved single-chunk appendPrefillEmbeds matches
// one-shot prefill bit-exact on logits).
//
// Path A (reference) = chunk_blocks=8 → single chunk (because total ≤ 8 blocks).
// Path B candidates  = chunk_blocks ∈ {1, 2, 4}.
//
// We compare decoded greedy token sequences. LCS-similarity =
// LCS_length / max(len_A, len_B).

#include "common/tensor.h"
#include "multimodal/audioRunner.h"
#include "runtime/audioUtils.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "tokenizer/tokenizer.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        auto _e = (x);                                                                                                 \
        if (_e != cudaSuccess)                                                                                         \
        {                                                                                                              \
            fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));                        \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

// Paths — engines + pre-chunked mel sidecars produced by /tmp/make_chunks.py.
constexpr char const* kEngineDir = "/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_thinker_full_fp8embed";
constexpr char const* kMultimodalEngineDir
    = "/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_audio_encoder";

// Pre-chunked mel sidecar layout: /tmp/m36_mel/cb<K>/chunk_<i>.safetensors
constexpr char const* kMelRoot = "/tmp/m36_mel";
constexpr char const* kAudioPath = "/home/harvest/qwen3-asr-hlm-validate/mel/zh_audio_input.safetensors";
constexpr int32_t kTotalMelTimesteps = 560;  // zh sample. 560/100 = 5.6 mel-blocks. <=8 so cb=8 is one chunk.
constexpr int32_t kMaxNewTokens = 64;

// Qwen3-Omni audio special-token IDs (mirror audioRunner.h defaults).
constexpr int32_t kAudioBosId = 151669; // <|audio_start|>
constexpr int32_t kAudioEosId = 151670; // <|audio_end|>
constexpr int32_t kAudioPadId = 151676; // <|audio_pad|> (from thinker config audio_token_id)

// Mel preprocessing constants (from audioRunner: 100 mel timesteps → ~13 audio tokens).
// One mel-block = 100 mel timesteps; this is the encoder's chunk granularity.
constexpr int32_t kMelTimestepsPerBlock = 100;

// Build per-chunk mel-timestep partitioning matching make_chunks.py.
std::vector<int32_t> partitionMelTimesteps(int32_t total, int32_t chunkBlocks)
{
    std::vector<int32_t> sizes;
    int32_t step = chunkBlocks * kMelTimestepsPerBlock;
    for (int32_t s = 0; s < total; s += step)
    {
        sizes.push_back(std::min(step, total - s));
    }
    return sizes;
}

// Tokenize the full ASR prompt using the runtime tokenizer, then split into
// (prefix, suffix) at the <|audio_pad|> placeholder, replacing it with the
// audio_start / audio_pad-block / audio_end triple that the production audio
// preprocess emits. Returns {prefixTokens, suffixTokens} — the audio pad-block
// is filled in per-chunk by the caller.
struct PromptSplits
{
    std::vector<int32_t> prefix; // <bos>... text ... <|audio_start|>
    std::vector<int32_t> suffix; // <|audio_end|> ... <|im_end|> ... <|im_start|>assistant
};

PromptSplits buildPromptSplits(tokenizer::Tokenizer& tok)
{
    // Production chat template for ASR (matches qwen3_asr_worker / verify scripts):
    //   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_pad|><|im_end|>\n<|im_start|>assistant\n
    std::string pre = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
    std::string suf = "<|im_end|>\n<|im_start|>assistant\n";
    auto preTok = tok.encode(pre);
    auto sufTok = tok.encode(suf);
    PromptSplits ps;
    ps.prefix = std::move(preTok);
    ps.prefix.push_back(kAudioBosId);
    ps.suffix.assign(1, kAudioEosId);
    for (auto t : sufTok) ps.suffix.push_back(t);
    return ps;
}

// Run one chunked ASR session and return generated tokens.
bool runChunkedSession(rt::LLMInferenceSpecDecodeRuntime& runtime, rt::Qwen3OmniAudioRunner& audioRunner,
    PromptSplits const& prompt, int32_t chunkBlocks, std::vector<int32_t>& outTokens, int32_t& outAudioTokenTotal,
    cudaStream_t stream)
{
    outTokens.clear();
    outAudioTokenTotal = 0;

    // Init session context.
    rt::SpecDecodeInferenceContext ctx;
    ctx.initialize(/*batch*/ 1, /*maxGen*/ kMaxNewTokens, rt::OptionalInputTensor{std::nullopt},
        rt::OptionalInputTensors{}, /*lora*/ "", stream);
    ctx.systemPrompts[0] = "";
    ctx.rawBatchedInputIds.assign(1, {});
    ctx.tokenIds[0].clear();

    if (!runtime.beginAsrSession(ctx, stream))
    {
        fprintf(stderr, "  beginAsrSession FAILED (chunkBlocks=%d)\n", chunkBlocks);
        return false;
    }

    auto melSizes = partitionMelTimesteps(kTotalMelTimesteps, chunkBlocks);
    int32_t const numChunks = static_cast<int32_t>(melSizes.size());
    printf("  chunkBlocks=%d → %d chunks  melSizes=[", chunkBlocks, numChunks);
    for (size_t i = 0; i < melSizes.size(); ++i)
        printf("%s%d", (i ? ", " : ""), melSizes[i]);
    printf("]\n");

    // Per-chunk scratch tensor for audio embeddings (worst case: 1 chunk × full).
    int64_t constexpr kHidden = 2560;
    int64_t constexpr kMaxRows = 256;
    rt::Tensor chunkEmbed({kMaxRows, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "chunkEmbed");

    int32_t audioBase = 0;
    for (int32_t ci = 0; ci < numChunks; ++ci)
    {
        // Build mel chunk path.
        char path[256];
        std::snprintf(path, sizeof(path), "%s/cb%d/chunk_%d.safetensors", kMelRoot, chunkBlocks, ci);

        rt::audioUtils::AudioData ad;
        ad.melSpectrogramPath = path;
        ad.melSpectrogramFormat = "safetensors";

        if (!audioRunner.encodeMelChunk(ad, chunkEmbed, stream))
        {
            fprintf(stderr, "  encodeMelChunk(%s) FAILED\n", path);
            runtime.endAsrSession(ctx, stream);
            return false;
        }
        auto shape = chunkEmbed.getShape();
        int32_t chunkAudioTokens = static_cast<int32_t>(shape[0]);

        // Build tokenSliceDelta for this chunk.
        std::vector<int32_t> tokenSlice;
        if (ci == 0)
        {
            // First chunk: prompt prefix + N audio_pads.
            tokenSlice = prompt.prefix;
        }
        for (int32_t k = 0; k < chunkAudioTokens; ++k) tokenSlice.push_back(kAudioPadId);
        if (ci == numChunks - 1)
        {
            for (auto t : prompt.suffix) tokenSlice.push_back(t);
        }

        if (!runtime.appendPrefillEmbeds(ctx, chunkEmbed, audioBase, tokenSlice, stream))
        {
            fprintf(stderr, "  appendPrefillEmbeds FAILED at chunk %d (status=%d)\n", ci,
                static_cast<int>(runtime.getLastAppendStatus()));
            runtime.endAsrSession(ctx, stream);
            return false;
        }
        audioBase += chunkAudioTokens;
    }
    outAudioTokenTotal = audioBase;

    // Drive decode from the last-prefill logits.
    if (!runtime.decodeAfterChunkedPrefillForTesting(ctx, kMaxNewTokens, outTokens, stream))
    {
        fprintf(stderr, "  decodeAfterChunkedPrefillForTesting FAILED\n");
        runtime.endAsrSession(ctx, stream);
        return false;
    }
    runtime.endAsrSession(ctx, stream);
    return true;
}

// LCS length over int32 sequences (O(n*m) DP).
size_t lcsLen(std::vector<int32_t> const& a, std::vector<int32_t> const& b)
{
    size_t n = a.size(), m = b.size();
    if (n == 0 || m == 0) return 0;
    std::vector<size_t> prev(m + 1, 0), curr(m + 1, 0);
    for (size_t i = 1; i <= n; ++i)
    {
        for (size_t j = 1; j <= m; ++j)
        {
            if (a[i - 1] == b[j - 1])
                curr[j] = prev[j - 1] + 1;
            else
                curr[j] = std::max(prev[j], curr[j - 1]);
        }
        std::swap(prev, curr);
        std::fill(curr.begin(), curr.end(), 0);
    }
    return prev[m];
}

void printTokens(std::vector<int32_t> const& v, int maxItems = 32)
{
    printf("[");
    int n = std::min<int>(static_cast<int>(v.size()), maxItems);
    for (int i = 0; i < n; ++i) printf("%s%d", (i ? "," : ""), v[i]);
    if (static_cast<int>(v.size()) > n) printf(", ...");
    printf("]");
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
{
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    printf("=== M3.6 empirical LCS test ===\n");
    printf("Audio: %s\n", kAudioPath);
    printf("  total mel timesteps: %d  (=%g mel-blocks at 100/block)\n", kTotalMelTimesteps,
        static_cast<double>(kTotalMelTimesteps) / kMelTimestepsPerBlock);

    std::unordered_map<std::string, std::string> loraMap;
    auto runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(kEngineDir, kMultimodalEngineDir, loraMap,
        stream);
    auto* baseAudioRunner = runtime->getAudioRunnerForTesting();
    if (!baseAudioRunner)
    {
        fprintf(stderr, "FATAL: runtime has no audio runner\n");
        return 1;
    }
    auto* audioRunner = dynamic_cast<rt::Qwen3OmniAudioRunner*>(baseAudioRunner);
    if (!audioRunner)
    {
        fprintf(stderr, "FATAL: audio runner is not Qwen3OmniAudioRunner\n");
        return 1;
    }
    auto* tok = runtime->getTokenizerForTesting();
    if (!tok)
    {
        fprintf(stderr, "FATAL: no tokenizer\n");
        return 1;
    }
    auto prompt = buildPromptSplits(*tok);
    printf("  prompt prefix tokens (%zu): ", prompt.prefix.size()); printTokens(prompt.prefix, 16); printf("\n");
    printf("  prompt suffix tokens (%zu): ", prompt.suffix.size()); printTokens(prompt.suffix, 16); printf("\n");

    struct Result
    {
        int chunkBlocks;
        std::vector<int32_t> tokens;
        std::string text;
        int32_t audioTokens{0};
        bool ok{false};
    };
    std::vector<Result> results;

    // Run reference (cb=8 = single chunk) first.
    for (int cb : {8, 1, 2, 4})
    {
        Result r;
        r.chunkBlocks = cb;
        r.ok = runChunkedSession(*runtime, *audioRunner, prompt, cb, r.tokens, r.audioTokens, stream);
        if (r.ok)
        {
            r.text = tok->decode(r.tokens, true);
        }
        printf("\n--- chunk_blocks=%d ---\n", cb);
        printf("  generated tokens (%zu): ", r.tokens.size()); printTokens(r.tokens, 64); printf("\n");
        printf("  text: \"%s\"\n", r.text.c_str());
        printf("  audio tokens total: %d\n", r.audioTokens);
        results.push_back(std::move(r));
    }

    // Compare each to the reference (chunk_blocks=8).
    auto const& ref = results[0];
    if (!ref.ok)
    {
        fprintf(stderr, "\nFATAL: reference (cb=8) FAILED — cannot compute LCS\n");
        return 1;
    }
    printf("\n=== LCS-similarity (vs chunk_blocks=8 reference) ===\n");
    printf("Path A (chunk_blocks=8, num_chunks=1):\n");
    printf("  text: \"%s\"\n", ref.text.c_str());
    printf("  len: %zu tokens\n", ref.tokens.size());
    for (size_t i = 1; i < results.size(); ++i)
    {
        auto const& r = results[i];
        if (!r.ok)
        {
            printf("\nPath B (chunk_blocks=%d): FAILED\n", r.chunkBlocks);
            continue;
        }
        size_t L = lcsLen(ref.tokens, r.tokens);
        double sim = static_cast<double>(L) / std::max<size_t>(ref.tokens.size(), r.tokens.size());
        printf("\nPath B (chunk_blocks=%d, num_chunks=%zu):\n", r.chunkBlocks,
            partitionMelTimesteps(kTotalMelTimesteps, r.chunkBlocks).size());
        printf("  text: \"%s\"\n", r.text.c_str());
        printf("  len: %zu tokens, LCS=%zu, LCS-similarity=%.4f %s\n", r.tokens.size(), L, sim,
            sim >= 0.95 ? "[PASS]" : "[FAIL]");
    }

    printf("\nVERDICT:\n");
    bool anyFail = false;
    bool allBelow09 = true;
    for (size_t i = 1; i < results.size(); ++i)
    {
        auto const& r = results[i];
        if (!r.ok) { anyFail = true; continue; }
        size_t L = lcsLen(ref.tokens, r.tokens);
        double sim = static_cast<double>(L) / std::max<size_t>(ref.tokens.size(), r.tokens.size());
        printf("  - chunk_blocks=%d: LCS-sim=%.4f → %s\n", r.chunkBlocks, sim, sim >= 0.95 ? "PASS" : "FAIL");
        if (sim < 0.95) anyFail = true;
        if (sim >= 0.90) allBelow09 = false;
    }

    printf("\nRECOMMENDATION: ");
    if (!anyFail) printf("ship-as-is\n");
    else if (allBelow09) printf("build-attention-cache\n");
    else printf("add-overlap-and-discard\n");

    cudaStreamDestroy(stream);
    return 0;
}
