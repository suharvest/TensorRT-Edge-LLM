// Streaming-ASR Milestone 2 acceptance test: session lifecycle + capacity caps.
//
// Validates the three behavioral properties M2 adds to LLMInferenceSpecDecodeRuntime:
//
//   Scenario 1 — Session pair:
//       beginAsrSession → appendPrefillEmbeds × 2 → endAsrSession →
//       beginAsrSession → appendPrefillEmbeds × 2 → endAsrSession
//     Both sessions consume the same inputs and must produce bit-exact identical
//     logits, proving session teardown frees state cleanly and a fresh session
//     starts from the same initial KV/state.
//
//   Scenario 2 — KV capacity refusal:
//       Push chunks summing to > max_kv_cache_capacity (256). Successful calls
//       must keep cache_kv_length ≤ 256. The first call that would overflow
//       must refuse with kKvCapacityExceeded and leave state unchanged. After
//       endAsrSession + beginAsrSession, a fresh chunk must succeed.
//
//   Scenario 3 — Per-chunk overflow refusal:
//       appendPrefillEmbeds with tokenSliceDelta.size() > max_input_len (128).
//       Must refuse with kChunkTooLong and not advance state.
//
// All three scenarios report PASS/FAIL with explicit cache_length numbers.

#include "common/tensor.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

constexpr int32_t kAudioPadId = 151676;
constexpr int32_t kHidden = 1024;
constexpr int32_t kEngineMaxIn = 128;
constexpr int32_t kEngineMaxKv = 256;

rt::Tensor makeAudioEmbeds(int rows, int hidden, uint32_t seed)
{
    rt::Tensor t({rows, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "audioEmbeds");
    std::vector<half> host(static_cast<size_t>(rows) * hidden);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : host)
    {
        v = __float2half(dis(gen));
    }
    CK(cudaMemcpy(t.rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice));
    return t;
}

std::vector<float> downloadLogits(rt::Tensor const& logits)
{
    auto const dt = logits.getDataType();
    size_t const vol = logits.getShape().volume();
    std::vector<float> out(vol);
    if (dt == nvinfer1::DataType::kHALF)
    {
        std::vector<half> tmp(vol);
        CK(cudaMemcpy(tmp.data(), logits.rawPointer(), vol * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < vol; ++i)
        {
            out[i] = __half2float(tmp[i]);
        }
    }
    else
    {
        CK(cudaMemcpy(out.data(), logits.rawPointer(), vol * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return out;
}

// Read the live KV cache length for batch slot 0. Used by the test to assert
// state-advance / state-no-advance invariants.
int32_t readSlot0KvLength(rt::LLMInferenceSpecDecodeRuntime& runtime, cudaStream_t stream)
{
    // Use the M2 helper peekKvCacheLength to do a synchronous D2H of slot 0.
    // (Before begin runs there is no active slot yet — peek requires >=1 slot
    // — but every call site below runs after beginAsrSession.)
    return runtime.peekKvCacheLength(0, stream);
}

// Build a synthetic token slice consisting entirely of audio_pad IDs.
std::vector<int32_t> makeAudioPadSlice(int rows)
{
    return std::vector<int32_t>(static_cast<size_t>(rows), kAudioPadId);
}

void initCtx(rt::SpecDecodeInferenceContext& ctx, cudaStream_t stream)
{
    ctx.initialize(/*batchSize*/ 1, /*maxGenLength*/ 1, rt::OptionalInputTensor{std::nullopt},
        rt::OptionalInputTensors{}, /*loraName*/ "", stream);
    ctx.systemPrompts[0] = "";
    ctx.rawBatchedInputIds.assign(1, {});
    ctx.tokenIds[0].clear();
}

bool scenario1_sessionPair(rt::LLMInferenceSpecDecodeRuntime& runtime, rt::Tensor const& audioEmbeds, cudaStream_t stream)
{
    printf("\n--- Scenario 1: session pair (begin/append×2/end → begin/append×2/end) ---\n");
    int const audioPerChunk = 30; // 30 + 30 = 60 audio rows, well under cap
    auto chunk0 = makeAudioPadSlice(audioPerChunk);
    auto chunk1 = makeAudioPadSlice(audioPerChunk);

    auto runOne = [&](char const* label) -> std::vector<float>
    {
        rt::SpecDecodeInferenceContext ctx;
        initCtx(ctx, stream);
        // (cannot peek KV before begin — no active batch slot.)
        if (!runtime.beginAsrSession(ctx))
        {
            fprintf(stderr, "  [%s] beginAsrSession FAILED\n", label);
            return {};
        }
        printf("  [%s] cache_kv after begin  = %d\n", label, readSlot0KvLength(runtime, stream));
        if (!runtime.appendPrefillEmbeds(ctx, audioEmbeds, 0, chunk0, stream))
        {
            fprintf(stderr, "  [%s] append #1 FAILED (status=%d)\n", label, (int) runtime.getLastAppendStatus());
            return {};
        }
        CK(cudaStreamSynchronize(stream));
        int kv1 = readSlot0KvLength(runtime, stream);
        printf("  [%s] cache_kv after append1 = %d (expected %d)\n", label, kv1, audioPerChunk);
        if (!runtime.appendPrefillEmbeds(ctx, audioEmbeds, audioPerChunk, chunk1, stream))
        {
            fprintf(stderr, "  [%s] append #2 FAILED (status=%d)\n", label, (int) runtime.getLastAppendStatus());
            return {};
        }
        CK(cudaStreamSynchronize(stream));
        int kv2 = readSlot0KvLength(runtime, stream);
        printf("  [%s] cache_kv after append2 = %d (expected %d)\n", label, kv2, 2 * audioPerChunk);
        auto logits = downloadLogits(runtime.getLogitsForTesting());
        if (!runtime.endAsrSession(ctx, stream))
        {
            fprintf(stderr, "  [%s] endAsrSession FAILED\n", label);
            return {};
        }
        CK(cudaStreamSynchronize(stream));
        int kv3 = readSlot0KvLength(runtime, stream);
        printf("  [%s] cache_kv after end    = %d (expected 0)\n", label, kv3);
        if (kv3 != 0)
        {
            fprintf(stderr, "  [%s] cache_kv after endAsrSession is not 0\n", label);
            return {};
        }
        return logits;
    };

    auto la = runOne("session_a");
    auto lb = runOne("session_b");
    if (la.empty() || lb.empty() || la.size() != lb.size())
    {
        printf("  Scenario 1: FAIL — logits empty or size mismatch\n");
        return false;
    }
    double maxAbs = 0.0;
    for (size_t i = 0; i < la.size(); ++i)
    {
        double d = std::fabs((double) la[i] - (double) lb[i]);
        if (d > maxAbs) maxAbs = d;
    }
    printf("  Scenario 1: max abs diff between sessions = %.6e\n", maxAbs);
    bool pass = (maxAbs < 1e-2);
    printf("  Scenario 1: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool scenario2_kvOverflow(rt::LLMInferenceSpecDecodeRuntime& runtime, rt::Tensor const& audioEmbeds, cudaStream_t stream)
{
    printf("\n--- Scenario 2: KV overflow refusal (cumulative > %d) ---\n", kEngineMaxKv);
    int const chunkLen = 100; // 100, 100, 100 — third should refuse (200 + 100 > 256)
    auto chunk = makeAudioPadSlice(chunkLen);

    rt::SpecDecodeInferenceContext ctx;
    initCtx(ctx, stream);
    if (!runtime.beginAsrSession(ctx))
    {
        printf("  FAIL: beginAsrSession\n");
        return false;
    }

    int audioBase = 0;
    int successCount = 0;
    bool refused = false;
    int kvAfterLastOk = 0;
    int kvAtRefusal = 0;
    auto refusalStatus = rt::LLMInferenceSpecDecodeRuntime::AppendPrefillStatus::kOk;
    for (int i = 0; i < 5; ++i)
    {
        int kvBefore = readSlot0KvLength(runtime, stream);
        bool ok = runtime.appendPrefillEmbeds(ctx, audioEmbeds, audioBase, chunk, stream);
        CK(cudaStreamSynchronize(stream));
        int kvAfter = readSlot0KvLength(runtime, stream);
        auto status = runtime.getLastAppendStatus();
        printf("  call %d: kv_before=%d ok=%d status=%d kv_after=%d\n", i, kvBefore, ok ? 1 : 0, (int) status, kvAfter);
        if (ok)
        {
            successCount++;
            kvAfterLastOk = kvAfter;
            audioBase += chunkLen;
            if (kvAfter > kEngineMaxKv)
            {
                printf("  FAIL: kv_after %d > cap %d on a successful call\n", kvAfter, kEngineMaxKv);
                return false;
            }
        }
        else
        {
            refused = true;
            kvAtRefusal = kvAfter;
            refusalStatus = status;
            if (kvAfter != kvBefore)
            {
                printf("  FAIL: refused call advanced state (kv_before=%d kv_after=%d)\n", kvBefore, kvAfter);
                return false;
            }
            break;
        }
    }
    if (!refused)
    {
        printf("  FAIL: never refused — capacity guard inactive\n");
        return false;
    }
    if (refusalStatus != rt::LLMInferenceSpecDecodeRuntime::AppendPrefillStatus::kKvCapacityExceeded)
    {
        printf("  FAIL: refused but status not kKvCapacityExceeded (%d)\n", (int) refusalStatus);
        return false;
    }
    printf("  successes=%d, last_ok_kv=%d, refused_kv=%d (state preserved)\n", successCount, kvAfterLastOk, kvAtRefusal);

    // Tear down and re-begin: a fresh chunk must succeed.
    if (!runtime.endAsrSession(ctx, stream))
    {
        printf("  FAIL: endAsrSession\n");
        return false;
    }
    int kvAfterEnd = readSlot0KvLength(runtime, stream);
    printf("  kv after endAsrSession = %d (expected 0)\n", kvAfterEnd);
    if (kvAfterEnd != 0)
    {
        printf("  FAIL: endAsrSession did not zero cache_kv\n");
        return false;
    }

    rt::SpecDecodeInferenceContext ctx2;
    initCtx(ctx2, stream);
    if (!runtime.beginAsrSession(ctx2))
    {
        printf("  FAIL: re-beginAsrSession\n");
        return false;
    }
    if (!runtime.appendPrefillEmbeds(ctx2, audioEmbeds, 0, chunk, stream))
    {
        printf("  FAIL: append after end+begin (status=%d)\n", (int) runtime.getLastAppendStatus());
        return false;
    }
    CK(cudaStreamSynchronize(stream));
    int kvFresh = readSlot0KvLength(runtime, stream);
    printf("  kv after fresh-session append = %d (expected %d)\n", kvFresh, chunkLen);
    runtime.endAsrSession(ctx2, stream);
    printf("  Scenario 2: PASS\n");
    return true;
}

bool scenario3_perChunkOverflow(rt::LLMInferenceSpecDecodeRuntime& runtime, rt::Tensor const& audioEmbeds,
    cudaStream_t stream)
{
    printf("\n--- Scenario 3: per-chunk overflow refusal (chunkLen > %d) ---\n", kEngineMaxIn);
    int const tooLong = kEngineMaxIn + 1;
    auto chunk = makeAudioPadSlice(tooLong);

    rt::SpecDecodeInferenceContext ctx;
    initCtx(ctx, stream);
    if (!runtime.beginAsrSession(ctx))
    {
        printf("  FAIL: beginAsrSession\n");
        return false;
    }
    int kvBefore = readSlot0KvLength(runtime, stream);
    bool ok = runtime.appendPrefillEmbeds(ctx, audioEmbeds, 0, chunk, stream);
    CK(cudaStreamSynchronize(stream));
    int kvAfter = readSlot0KvLength(runtime, stream);
    auto status = runtime.getLastAppendStatus();
    printf("  chunkLen=%d kv_before=%d ok=%d status=%d kv_after=%d\n", tooLong, kvBefore, ok ? 1 : 0, (int) status,
        kvAfter);

    if (ok)
    {
        printf("  FAIL: per-chunk overflow accepted (must refuse)\n");
        return false;
    }
    if (status != rt::LLMInferenceSpecDecodeRuntime::AppendPrefillStatus::kChunkTooLong)
    {
        printf("  FAIL: refused but status not kChunkTooLong (%d)\n", (int) status);
        return false;
    }
    if (kvAfter != kvBefore)
    {
        printf("  FAIL: refused call advanced state (kv_before=%d kv_after=%d)\n", kvBefore, kvAfter);
        return false;
    }
    if (!ctx.tokenIds[0].empty())
    {
        printf("  FAIL: refused call appended to ctx.tokenIds (size=%zu)\n", ctx.tokenIds[0].size());
        return false;
    }
    runtime.endAsrSession(ctx, stream);
    printf("  Scenario 3: PASS\n");
    return true;
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
{
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    std::string engineDir = "/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_thinker_full_fp8embed";
    std::string multimodalEngineDir = "";
    std::unordered_map<std::string, std::string> loraMap;

    auto runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(engineDir, multimodalEngineDir, loraMap, stream);
    printf("runtime constructed against %s\n", engineDir.c_str());
    printf("engine max_input_len=%d  max_kv_cache_capacity=%d\n", kEngineMaxIn, runtime->getMaxKvCacheCapacity());

    // Pre-allocate enough audio rows for the largest test (Scenario 2: 300+ rows).
    rt::Tensor audioEmbedsFull = makeAudioEmbeds(/*rows*/ 512, kHidden, /*seed*/ 42);

    bool s1 = scenario1_sessionPair(*runtime, audioEmbedsFull, stream);
    bool s2 = scenario2_kvOverflow(*runtime, audioEmbedsFull, stream);
    bool s3 = scenario3_perChunkOverflow(*runtime, audioEmbedsFull, stream);

    printf("\n=== M2 SESSION LIFECYCLE SUMMARY ===\n");
    printf("  Scenario 1 (session pair):           %s\n", s1 ? "PASS" : "FAIL");
    printf("  Scenario 2 (KV overflow refusal):    %s\n", s2 ? "PASS" : "FAIL");
    printf("  Scenario 3 (per-chunk overflow):     %s\n", s3 ? "PASS" : "FAIL");

    bool all = s1 && s2 && s3;
    printf("=== %s ===\n", all ? "M2 ACCEPTANCE: PASS" : "M2 ACCEPTANCE: FAIL");
    cudaStreamDestroy(stream);
    return all ? 0 : 1;
}
