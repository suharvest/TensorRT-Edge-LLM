// M1 standalone microbench: prove that the INT4-weight GEMV with a BF16 output
// stays finite for an overflow-prone column, while the existing FP16-output path
// overflows to +inf. Builds without TensorRT / edgellmCore — it includes the kernel
// .cu sources directly and embeds the AWQ pack + scaled-weight + naive references
// (lifted verbatim from unittests/references.cpp / naiveGemmReference.cu).
//
// Build (on a CUDA device, sm_87 for Orin):
//   nvcc -O2 -std=c++17 -arch=sm_87 \
//        -I <repo>/cpp -I <repo>/cpp/kernels/int4GroupwiseGemmKernels \
//        bf16_overflow_microbench.cu -o bf16_microbench
//
// The two kernel .cu files are compiled as part of this TU (single-file build).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "int4WoQGemvCuda.cu"
#include "int4WoqGemmCuda.cu"

#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t e = (x);                                                                                           \
        if (e != cudaSuccess)                                                                                          \
        {                                                                                                              \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                       \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// ---- index helpers (verbatim from references.cpp) ----
static inline size_t idx4(int32_t i, int32_t j, int32_t k, int32_t t, int32_t A, int32_t B, int32_t C, int32_t Tlast)
{
    return ((((size_t) i * B + j) * C + k) * Tlast) + t;
}
static inline size_t idx3(int32_t i, int32_t j, int32_t k, int32_t A, int32_t B, int32_t C)
{
    return ((size_t) i * B + j) * C + k;
}
static inline size_t idx2(int32_t i0, int32_t i1, int32_t dim1)
{
    return static_cast<size_t>(i0) * dim1 + i1;
}

// ---- AWQ pack reference (verbatim from references.cpp) ----
static void awqPackReference(int16_t const* kernel_KxN, int32_t N_in, int32_t K_in, int16_t* out_Ndiv4xK)
{
    int32_t const interleave = 4;
    int32_t const kstride = 64;
    int32_t const N = K_in;
    int32_t const K = N_in;
    std::vector<int16_t> A((size_t) K * N);
    for (int32_t k_in = 0; k_in < K_in; ++k_in)
        for (int32_t n_in = 0; n_in < N_in; ++n_in)
        {
            int16_t v = kernel_KxN[idx2(k_in, n_in, N_in)] + 8;
            A[idx2(n_in, k_in, N)] = v;
        }
    std::vector<int16_t> B((size_t) K * N);
    int32_t const K32 = K / 32;
    for (int32_t n = 0; n < N; ++n)
    {
        size_t const in_row = (size_t) n * K;
        size_t const out_row = (size_t) n * K;
        for (int32_t b = 0; b < K32; ++b)
        {
            size_t const in_blk = in_row + (size_t) b * 32;
            size_t const out_blk = out_row + (size_t) b * 32;
            for (int32_t o = 0; o < 32; ++o)
            {
                int32_t const a = o >> 3;
                int32_t const b4 = (o >> 1) & 0x3;
                int32_t const c = o & 1;
                int32_t const o2 = (((b4 << 2) | a) << 1) | c;
                B[out_blk + o2] = A[in_blk + o];
            }
        }
    }
    static int32_t const REORDER8[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<int16_t> C((size_t) K * N);
    for (int32_t n = 0; n < N; ++n)
        for (int32_t q = 0; q < K32; ++q)
        {
            int32_t base = q * 32;
            for (int32_t g = 0; g < 4; ++g)
            {
                int32_t gbase = base + g * 8;
                for (int32_t j = 0; j < 8; ++j)
                {
                    int32_t in_idx = gbase + j;
                    int32_t out_idx = gbase + REORDER8[j];
                    C[idx2(n, out_idx, K)] = B[idx2(n, in_idx, K)];
                }
            }
        }
    int32_t const Ng = K / 4;
    int32_t const Kg = N / 64;
    std::vector<int16_t> D(static_cast<size_t>(Ng) * Kg * interleave * kstride);
    for (int32_t ng = 0; ng < Ng; ++ng)
        for (int32_t ks = 0; ks < Kg; ++ks)
            for (int32_t j = 0; j < interleave; ++j)
                for (int32_t i = 0; i < kstride; ++i)
                    D[idx4(ng, ks, j, i, 0, Kg, interleave, kstride)]
                        = C[idx4(ng, j, ks, i, 0, interleave, Kg, kstride)];
    for (int32_t i = 0; i < Ng; ++i)
        for (int32_t j = 0; j < Kg; ++j)
            for (int32_t k = 0; k < kstride; ++k)
            {
                size_t const base4 = idx4(i, j, k, 0, Ng, Kg, kstride, 4);
                uint16_t const a0 = static_cast<uint16_t>(D[base4 + 0]);
                uint16_t const a1 = static_cast<uint16_t>(D[base4 + 1]);
                uint16_t const a2 = static_cast<uint16_t>(D[base4 + 2]);
                uint16_t const a3 = static_cast<uint16_t>(D[base4 + 3]);
                uint16_t const packed = static_cast<uint16_t>(
                    (a0) | (static_cast<uint16_t>(a1) << 4) | (static_cast<uint16_t>(a2) << 8)
                    | (static_cast<uint16_t>(a3) << 12));
                out_Ndiv4xK[idx3(i, j, k, Ng, Kg, kstride)] = static_cast<int16_t>(packed);
            }
}

static void scaledWeightsReference(int16_t const* kernel_KxN, half const* scales_KdivGxN, int32_t K, int32_t N,
    int32_t group_size, std::vector<half>& out_KxN)
{
    out_KxN.resize(static_cast<size_t>(K) * N);
    for (int32_t k = 0; k < K; ++k)
    {
        int32_t srow = k / group_size;
        for (int32_t n = 0; n < N; ++n)
        {
            float w = static_cast<float>(kernel_KxN[idx2(k, n, N)]);
            float sf = __half2float(scales_KdivGxN[idx2(srow, n, N)]);
            out_KxN[idx2(k, n, N)] = __float2half(w * sf);
        }
    }
}

// ---- naive fp32-accumulate GEMM reference, output fp32 (no overflow) ----
__global__ void naive_gemm_f32(half const* A, half const* B, float* C, int M, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
        return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
    C[m * N + n] = acc;
}

int main()
{
    // Decode-path shape (M=1) sized so a column accumulates well past the FP16 max
    // (65504). N must be divisible by 8 (N/4 packed, N_PER_BLOCK=2); K divisible by
    // group_size(128). All weights/inputs positive so the sum is monotone and large.
    int const M = 1, N = 128, K = 1024, G = 128;

    std::mt19937 rng(123);
    std::vector<half> in(M * K);
    std::vector<int16_t> w(K * N); // int4 in [-8,7]; use +7 (max) -> large positive
    std::vector<half> scales((K / G) * N);

    // inputs ~ 4.0, weights = +7 (max int4), scales = 0.9
    // per-column sum ~ K * 4.0 * (7 * 0.9) = 1024 * 4 * 6.3 ~= 2.58e4 ... bump higher:
    // inputs = 6.0, scales = 1.5 -> 1024 * 6 * (7*1.5)=1024*6*10.5 = 6.45e4 ~ near fp16 max
    // push clearly over: inputs = 8.0, scales = 2.0 -> 1024*8*14 = 1.15e5 > 65504 -> fp16 inf
    for (auto& x : in)
        x = __float2half(8.0f);
    for (auto& x : w)
        x = 7;
    for (auto& x : scales)
        x = __float2half(2.0f);

    // pack + scaled (fp16 reference) weights
    std::vector<int16_t> wpacked((N / 4) * K);
    awqPackReference(w.data(), N, K, wpacked.data());
    std::vector<half> wscaled;
    scaledWeightsReference(w.data(), scales.data(), K, N, G, wscaled);

    // device buffers
    half *dIn, *dScale, *dWref, *dOutFp16;
    __nv_bfloat16* dOutBf16;
    int8_t* dW;
    float* dRefF32;
    CK(cudaMalloc(&dIn, M * K * sizeof(half)));
    CK(cudaMalloc(&dW, (size_t) K * N / 2 * sizeof(int8_t)));
    CK(cudaMalloc(&dScale, (K / G) * N * sizeof(half)));
    CK(cudaMalloc(&dOutFp16, M * N * sizeof(half)));
    CK(cudaMalloc(&dOutBf16, M * N * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dWref, (size_t) K * N * sizeof(half)));
    CK(cudaMalloc(&dRefF32, M * N * sizeof(float)));

    CK(cudaMemcpy(dIn, in.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, reinterpret_cast<int8_t const*>(wpacked.data()), (size_t) K * N / 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dScale, scales.data(), (K / G) * N * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWref, wscaled.data(), (size_t) K * N * sizeof(half), cudaMemcpyHostToDevice));

    // fp32 reference (ground truth, no overflow)
    {
        dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
        naive_gemm_f32<<<grd, blk>>>(dIn, dWref, dRefF32, M, N, K);
        CK(cudaDeviceSynchronize());
    }

    // fp16-output kernel (existing path)
    trt_edgellm::kernel::gemv_forward_cuda_new(dIn, dW, dScale, dOutFp16, M, N, K, G, 0);
    CK(cudaDeviceSynchronize());

    // bf16-output kernel (new mixed-precision path)
    trt_edgellm::kernel::gemv_forward_cuda_new_bf16(dIn, dW, dScale, dOutBf16, M, N, K, G, 0);
    CK(cudaDeviceSynchronize());

    std::vector<half> outFp16(M * N);
    std::vector<__nv_bfloat16> outBf16(M * N);
    std::vector<float> ref(M * N);
    CK(cudaMemcpy(outFp16.data(), dOutFp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outBf16.data(), dOutBf16, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), dRefF32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    int fp16_inf = 0, bf16_inf = 0, bf16_finite = 0;
    float worst_rel = 0.f;
    for (int i = 0; i < M * N; ++i)
    {
        float f16 = __half2float(outFp16[i]);
        float b16 = __bfloat162float(outBf16[i]);
        float r = ref[i];
        if (!std::isfinite(f16))
            fp16_inf++;
        if (!std::isfinite(b16))
            bf16_inf++;
        else
            bf16_finite++;
        if (std::isfinite(b16) && std::fabs(r) > 1.f)
            worst_rel = std::max(worst_rel, std::fabs(b16 - r) / std::fabs(r));
    }

    std::printf("=== M1 BF16 GEMV overflow microbench ===\n");
    std::printf("shape M=%d N=%d K=%d group=%d\n", M, N, K, G);
    std::printf("reference (fp32, ch0)  = %g\n", ref[0]);
    std::printf("fp16 output (ch0)      = %g  [isfinite=%d]\n", __half2float(outFp16[0]),
        (int) std::isfinite(__half2float(outFp16[0])));
    std::printf("bf16 output (ch0)      = %g  [isfinite=%d]\n", __bfloat162float(outBf16[0]),
        (int) std::isfinite(__bfloat162float(outBf16[0])));
    std::printf("fp16 non-finite count  = %d / %d   <-- existing path OVERFLOWS\n", fp16_inf, M * N);
    std::printf("bf16 non-finite count  = %d / %d   <-- new path stays FINITE\n", bf16_inf, M * N);
    std::printf("bf16 worst rel-err vs fp32 ref = %g (expected ~bf16 mantissa ~0.01)\n", worst_rel);

    bool gemvPass = (fp16_inf > 0) && (bf16_inf == 0) && (worst_rel < 0.05f);
    std::printf("RESULT(GEMV): %s\n\n", gemvPass ? "PASS" : "FAIL");

    // ---- GEMM prefill path (M>kGemvMaxM, N % kGemmCtaN == 0 -> tiled GEMM) ----
    int const M2v = 64; // CTA_M tile -> exercises gemm_w4a16_T2(_bf16)
    std::vector<half> in2(M2v * K);
    for (auto& x : in2)
        x = __float2half(8.0f); // same overflow-prone magnitude
    half *dIn2, *dOutFp16_2;
    __nv_bfloat16* dOutBf16_2;
    float* dRefF32_2;
    CK(cudaMalloc(&dIn2, (size_t) M2v * K * sizeof(half)));
    CK(cudaMalloc(&dOutFp16_2, (size_t) M2v * N * sizeof(half)));
    CK(cudaMalloc(&dOutBf16_2, (size_t) M2v * N * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dRefF32_2, (size_t) M2v * N * sizeof(float)));
    CK(cudaMemcpy(dIn2, in2.data(), (size_t) M2v * K * sizeof(half), cudaMemcpyHostToDevice));
    {
        dim3 blk(16, 16), grd((N + 15) / 16, (M2v + 15) / 16);
        naive_gemm_f32<<<grd, blk>>>(dIn2, dWref, dRefF32_2, M2v, N, K);
        CK(cudaDeviceSynchronize());
    }
    trt_edgellm::kernel::gemm_forward_cuda_new(dIn2, dW, dScale, dOutFp16_2, M2v, N, K, G, 0);
    CK(cudaDeviceSynchronize());
    trt_edgellm::kernel::gemm_forward_cuda_new_bf16(dIn2, dW, dScale, dOutBf16_2, M2v, N, K, G, 0);
    CK(cudaDeviceSynchronize());

    std::vector<half> outFp16_2((size_t) M2v * N);
    std::vector<__nv_bfloat16> outBf16_2((size_t) M2v * N);
    std::vector<float> ref2((size_t) M2v * N);
    CK(cudaMemcpy(outFp16_2.data(), dOutFp16_2, (size_t) M2v * N * sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outBf16_2.data(), dOutBf16_2, (size_t) M2v * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref2.data(), dRefF32_2, (size_t) M2v * N * sizeof(float), cudaMemcpyDeviceToHost));

    int g_fp16_inf = 0, g_bf16_inf = 0;
    float g_worst_rel = 0.f;
    for (int i = 0; i < M2v * N; ++i)
    {
        float f16 = __half2float(outFp16_2[i]);
        float b16 = __bfloat162float(outBf16_2[i]);
        float r = ref2[i];
        if (!std::isfinite(f16))
            g_fp16_inf++;
        if (!std::isfinite(b16))
            g_bf16_inf++;
        if (std::isfinite(b16) && std::fabs(r) > 1.f)
            g_worst_rel = std::max(g_worst_rel, std::fabs(b16 - r) / std::fabs(r));
    }
    std::printf("=== M1 BF16 GEMM (prefill) overflow microbench ===\n");
    std::printf("shape M=%d N=%d K=%d group=%d\n", M2v, N, K, G);
    std::printf("reference (fp32, [0])  = %g\n", ref2[0]);
    std::printf("fp16 output ([0])      = %g  [isfinite=%d]\n", __half2float(outFp16_2[0]),
        (int) std::isfinite(__half2float(outFp16_2[0])));
    std::printf("bf16 output ([0])      = %g  [isfinite=%d]\n", __bfloat162float(outBf16_2[0]),
        (int) std::isfinite(__bfloat162float(outBf16_2[0])));
    std::printf("fp16 non-finite count  = %d / %d   <-- existing GEMM OVERFLOWS\n", g_fp16_inf, M2v * N);
    std::printf("bf16 non-finite count  = %d / %d   <-- new GEMM stays FINITE\n", g_bf16_inf, M2v * N);
    std::printf("bf16 worst rel-err vs fp32 ref = %g\n", g_worst_rel);
    bool gemmPass = (g_fp16_inf > 0) && (g_bf16_inf == 0) && (g_worst_rel < 0.05f);
    std::printf("RESULT(GEMM): %s\n", gemmPass ? "PASS" : "FAIL");

    return (gemvPass && gemmPass) ? 0 : 1;
}
