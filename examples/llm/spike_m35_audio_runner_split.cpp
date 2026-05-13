// Streaming-ASR Milestone 3.5 acceptance test: audio runner split correctness.
//
// Validates the M3.5 refactor of cpp/multimodal/audioRunner: the new public
// encodeMelChunk entry point must produce bit-exact embeddings when the input
// mel is split along chunk-aligned boundaries, vs. encoding the full mel in
// one shot. This is the load-bearing correctness property that lets the M3
// streaming worker encode mel per chunk without quality regression.
//
// Path A (one-shot)  — encodeMelChunk on the FULL mel (560 frames).
// Path B (chunked)   — encodeMelChunk on 3 splits (200, 200, 160 frames),
//                      concatenate along the audio-token axis.
//
// The split boundaries are multiples of n_window*2 = 100 mel frames, so each
// chunk's internal CNN/attention windows align identically with the one-shot
// path. The encoder block-diagonal attention mask has windowAfterCNN = 104
// after-CNN tokens, well above per-split sizes (~26 / 26 / 19), so attention
// is single-window per split — bit-exact equality with one-shot is expected.
//
// NOTE: Path A uses encodeMelChunk on the full mel rather than the public
// preprocess() entry. The two paths share the same private
// encodeOneAudioImpl body byte-for-byte, so this test exercises the same
// encoder forward used by preprocess(). The one-shot path through
// preprocess() is regression-tested via spike_m1_append_prefill_embeds and
// spike_m2_session_lifecycle which still PASS bit-exact post-M3.5 refactor.

#include "multimodal/audioRunner.h"
#include "runtime/audioUtils.h"
#include "common/tensor.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// Path resolution: encoder dir layout is <multimodalEngineDir>/audio/{config.json,audio_encoder.engine}.
// audioRunner constructor takes the .../audio subdir.
constexpr char const* kAudioEngineDir
    = "/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_audio_encoder/audio";

// Pre-generated mel splits (zh_audio_input mel split along time axis):
//   full:    [1, 128, 560]
//   chunk_0: [1, 128, 200]
//   chunk_1: [1, 128, 200]
//   chunk_2: [1, 128, 160]
constexpr char const* kFullMelPath = "/tmp/m35_mel/full.safetensors";
constexpr char const* kChunkPaths[3]
    = {"/tmp/m35_mel/chunk_0.safetensors", "/tmp/m35_mel/chunk_1.safetensors", "/tmp/m35_mel/chunk_2.safetensors"};

rt::audioUtils::AudioData melAudioData(std::string const& path)
{
    rt::audioUtils::AudioData ad;
    ad.melSpectrogramPath = path;
    ad.melSpectrogramFormat = "safetensors";
    return ad;
}

// Download an [N, audioFeatureDim] FP16 device tensor to a host vector<float>
// flat-major for comparison. `rows` is the leading-dim slice to keep.
std::vector<float> downloadEmbedding(rt::Tensor const& embed, int64_t expectedRows, int64_t expectedCols)
{
    auto const shape = embed.getShape();
    if (shape.getNumDims() != 2 || shape[0] != expectedRows || shape[1] != expectedCols)
    {
        fprintf(stderr, "downloadEmbedding: shape mismatch — got [%ld,%ld] expected [%ld,%ld]\n", shape[0], shape[1],
            expectedRows, expectedCols);
        std::exit(1);
    }
    size_t const vol = static_cast<size_t>(expectedRows) * expectedCols;
    std::vector<__half> tmp(vol);
    CK(cudaMemcpy(tmp.data(), embed.rawPointer(), vol * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> out(vol);
    for (size_t i = 0; i < vol; ++i)
    {
        out[i] = __half2float(tmp[i]);
    }
    return out;
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
{
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    printf("Constructing Qwen3OmniAudioRunner from %s\n", kAudioEngineDir);
    rt::Qwen3OmniAudioRunner runner(kAudioEngineDir, stream);

    // ===== Path A — encode full mel one-shot =====
    printf("\n--- Path A — encodeMelChunk on FULL mel ---\n");
    // Pre-allocate enough capacity (max 6 chunks * 13 tokens = 78 audio tokens — leave headroom).
    int64_t constexpr kHidden = 2560;
    int64_t constexpr kMaxRows = 256;
    rt::Tensor outA({kMaxRows, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "outA");
    if (!runner.encodeMelChunk(melAudioData(kFullMelPath), outA, stream))
    {
        fprintf(stderr, "Path A: encodeMelChunk(full) FAILED\n");
        return 1;
    }
    auto const shapeA = outA.getShape();
    int64_t const rowsA = shapeA[0];
    int64_t const colsA = shapeA[1];
    printf("  audio tokens (A) = %ld  hidden = %ld\n", rowsA, colsA);
    auto hostA = downloadEmbedding(outA, rowsA, colsA);

    // ===== Path B — three chunks =====
    printf("\n--- Path B — encodeMelChunk on 3 splits (200+200+160 mel frames) ---\n");
    rt::Tensor outB_chunk({kMaxRows, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "outB_chunk");

    std::vector<float> hostB;
    int64_t totalRowsB = 0;
    for (int i = 0; i < 3; ++i)
    {
        if (!runner.encodeMelChunk(melAudioData(kChunkPaths[i]), outB_chunk, stream))
        {
            fprintf(stderr, "Path B: encodeMelChunk(chunk %d) FAILED\n", i);
            return 1;
        }
        auto const shape = outB_chunk.getShape();
        int64_t const rows = shape[0];
        int64_t const cols = shape[1];
        if (cols != colsA)
        {
            fprintf(stderr, "Path B chunk %d: hidden dim mismatch %ld != %ld\n", i, cols, colsA);
            return 1;
        }
        auto chunkHost = downloadEmbedding(outB_chunk, rows, cols);
        printf("  chunk %d: audio tokens = %ld\n", i, rows);
        hostB.insert(hostB.end(), chunkHost.begin(), chunkHost.end());
        totalRowsB += rows;
    }
    printf("  total audio tokens (B, sum) = %ld\n", totalRowsB);

    // ===== Compare =====
    printf("\n--- Comparison ---\n");
    bool const tokenCountMatch = (rowsA == totalRowsB);
    printf("  total audio tokens (A)        = %ld\n", rowsA);
    printf("  total audio tokens (B, sum)   = %ld  [match: %s]\n", totalRowsB,
        tokenCountMatch ? "yes" : "no");
    if (!tokenCountMatch)
    {
        printf("\n=== M3.5 ACCEPTANCE: FAIL ===\n");
        printf("  Token count mismatch — split boundaries do not align with one-shot encoder.\n");
        cudaStreamDestroy(stream);
        return 1;
    }

    double maxAbsDiff = 0.0;
    int64_t firstDivergeRow = -1;
    int64_t firstDivergeCol = -1;
    double firstDivergeA = 0.0;
    double firstDivergeB = 0.0;
    double maxRowAbsDiff = 0.0;
    int64_t maxRowAbsDiffRow = -1;
    for (int64_t r = 0; r < rowsA; ++r)
    {
        double rowMax = 0.0;
        for (int64_t c = 0; c < colsA; ++c)
        {
            size_t idx = static_cast<size_t>(r) * colsA + c;
            double a = static_cast<double>(hostA[idx]);
            double b = static_cast<double>(hostB[idx]);
            double d = std::fabs(a - b);
            if (d > rowMax) rowMax = d;
            if (d > maxAbsDiff)
            {
                maxAbsDiff = d;
                if (firstDivergeRow == -1)
                {
                    firstDivergeRow = r;
                    firstDivergeCol = c;
                    firstDivergeA = a;
                    firstDivergeB = b;
                }
            }
        }
        if (rowMax > maxRowAbsDiff)
        {
            maxRowAbsDiff = rowMax;
            maxRowAbsDiffRow = r;
        }
    }

    printf("  max abs diff per row: %.6e (worst row idx = %ld)\n", maxRowAbsDiff, maxRowAbsDiffRow);
    printf("  max abs diff overall: %.6e\n", maxAbsDiff);

    bool const bitExact = (maxAbsDiff == 0.0);
    if (!bitExact)
    {
        printf("\n  First divergence row %ld col %ld: A=%.6f B=%.6f (delta=%.6e)\n", firstDivergeRow, firstDivergeCol,
            firstDivergeA, firstDivergeB, firstDivergeA - firstDivergeB);
        printf("  Sample row %ld values (first 6):\n    A:", firstDivergeRow);
        for (int64_t c = 0; c < std::min<int64_t>(6, colsA); ++c)
        {
            printf(" %.6f", static_cast<double>(hostA[static_cast<size_t>(firstDivergeRow) * colsA + c]));
        }
        printf("\n    B:");
        for (int64_t c = 0; c < std::min<int64_t>(6, colsA); ++c)
        {
            printf(" %.6f", static_cast<double>(hostB[static_cast<size_t>(firstDivergeRow) * colsA + c]));
        }
        printf("\n");
    }

    printf("\nM3.5 acceptance:\n");
    printf("  total audio tokens (A) = %ld\n", rowsA);
    printf("  total audio tokens (B, sum) = %ld  [match: %s]\n", totalRowsB, tokenCountMatch ? "yes" : "no");
    printf("  max abs diff per row: %.6e %s\n", maxRowAbsDiff,
        bitExact ? "(bit-exact)" : "(NOT bit-exact)");
    printf("  %s\n", bitExact ? "PASS" : "FAIL");

    printf("\n=== M3.5 ACCEPTANCE: %s ===\n", bitExact ? "PASS" : "FAIL");

    cudaStreamDestroy(stream);
    return bitExact ? 0 : 1;
}
