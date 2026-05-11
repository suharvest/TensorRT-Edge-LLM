/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "w8A16Linear.h"

#include <cstdlib>
#include <string>

#include <cuda_runtime.h>
#include <mma.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int kThreadsPerBlock = 256;
constexpr int kKTile = 256;

int getOutputTileOverride() noexcept
{
    char const* value = std::getenv("EDGE_LLM_W8A16_OUTPUT_TILE");
    if (value == nullptr)
    {
        return 32;
    }
    int const parsed = std::atoi(value);
    if (parsed == 32 || parsed == 64 || parsed == 128)
    {
        return parsed;
    }
    return 32;
}

bool envFlagEnabled(char const* name) noexcept
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return false;
    }
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' || value[0] == 't' || value[0] == 'T';
}

__global__ void w8a16_per_output_reference_kernel(
    half const* __restrict__ input, int8_t const* __restrict__ weight, half const* __restrict__ scales,
    half* __restrict__ output, int m, int n, int k)
{
    int const outCol = blockIdx.x;
    int const row = blockIdx.y;
    if (outCol >= n || row >= m)
    {
        return;
    }

    float sum = 0.0F;
    int const inputOffset = row * k;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x)
    {
        float a = __half2float(input[inputOffset + kk]);
        float w = static_cast<float>(weight[kk * n + outCol]);
        sum += a * w;
    }

    __shared__ float shared[kThreadsPerBlock];
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        float scale = __half2float(scales[outCol]);
        output[row * n + outCol] = __float2half_rn(shared[0] * scale);
    }
}

__global__ void w8a16_per_output_output_k_reference_kernel(
    half const* __restrict__ input, int8_t const* __restrict__ weight, half const* __restrict__ scales,
    half* __restrict__ output, int m, int n, int k)
{
    int const outCol = blockIdx.x;
    int const row = blockIdx.y;
    if (outCol >= n || row >= m)
    {
        return;
    }

    float sum = 0.0F;
    int const inputOffset = row * k;
    int const weightOffset = outCol * k;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x)
    {
        float a = __half2float(input[inputOffset + kk]);
        float w = static_cast<float>(weight[weightOffset + kk]);
        sum += a * w;
    }

    __shared__ float shared[kThreadsPerBlock];
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[row * n + outCol] = __float2half_rn(shared[0] * __half2float(scales[outCol]));
    }
}

template <int OutputTile>
__global__ void w8a16_small_m_tiled_kernel(
    half const* __restrict__ input, int8_t const* __restrict__ weight, half const* __restrict__ scales,
    half* __restrict__ output, int m, int n, int k)
{
    constexpr int KThreads = kThreadsPerBlock / OutputTile;
    static_assert(OutputTile == 32 || OutputTile == 64 || OutputTile == 128);
    static_assert(KThreads >= 2);

    int const colBase = blockIdx.x * OutputTile;
    int const row = blockIdx.y;
    int const colLocal = threadIdx.x % OutputTile;
    int const kLane = threadIdx.x / OutputTile;
    int const outCol = colBase + colLocal;
    if (row >= m)
    {
        return;
    }

    __shared__ half inputTile[kKTile];
    __shared__ float partial[OutputTile][KThreads];

    float sum = 0.0F;
    int const inputOffset = row * k;
    for (int kBase = 0; kBase < k; kBase += kKTile)
    {
        int const tileLen = min(kKTile, k - kBase);
        for (int i = threadIdx.x; i < tileLen; i += blockDim.x)
        {
            inputTile[i] = input[inputOffset + kBase + i];
        }
        __syncthreads();

        if (outCol < n)
        {
            for (int kk = kLane; kk < tileLen; kk += KThreads)
            {
                float a = __half2float(inputTile[kk]);
                float w = static_cast<float>(weight[(kBase + kk) * n + outCol]);
                sum += a * w;
            }
        }
        __syncthreads();
    }

    partial[colLocal][kLane] = sum;
    __syncthreads();

    if (kLane == 0 && outCol < n)
    {
        float total = 0.0F;
#pragma unroll
        for (int i = 0; i < KThreads; ++i)
        {
            total += partial[colLocal][i];
        }
        float scale = __half2float(scales[outCol]);
        output[row * n + outCol] = __float2half_rn(total * scale);
    }
}

__global__ void w8a16_m1_output_k_kernel(
    half const* __restrict__ input, int8_t const* __restrict__ weight, half const* __restrict__ scales,
    half* __restrict__ output, int n, int k)
{
    constexpr int WarpsPerBlock = kThreadsPerBlock / 32;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const outCol = blockIdx.x * WarpsPerBlock + warp;
    if (outCol >= n)
    {
        return;
    }

    float sum = 0.0F;
    int const weightOffset = outCol * k;
    for (int kk = lane; kk < k; kk += warpSize)
    {
        float a = __half2float(input[kk]);
        float w = static_cast<float>(weight[weightOffset + kk]);
        sum += a * w;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0)
    {
        output[outCol] = __float2half_rn(sum * __half2float(scales[outCol]));
    }
}

__global__ void w8a16_hmma_m16n16k16_kernel(
    half const* __restrict__ input, int8_t const* __restrict__ weight, half const* __restrict__ scales,
    half* __restrict__ output, int m, int n, int k, int weightLayout)
{
    namespace wmma = nvcuda::wmma;
    constexpr int Tile = 16;

    int const tileN = blockIdx.x * Tile;
    int const tileM = blockIdx.y * Tile;
    int const lane = threadIdx.x;

    __shared__ half aTile[Tile * Tile];
    __shared__ half bTile[Tile * Tile];

    wmma::fragment<wmma::matrix_a, Tile, Tile, Tile, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, Tile, Tile, Tile, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, Tile, Tile, Tile, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0F);

    for (int kBase = 0; kBase < k; kBase += Tile)
    {
        for (int idx = lane; idx < Tile * Tile; idx += 32)
        {
            int const row = idx / Tile;
            int const col = idx % Tile;
            int const globalM = tileM + row;
            int const globalK = kBase + col;
            aTile[idx] = (globalM < m && globalK < k) ? input[globalM * k + globalK] : __float2half(0.0F);
        }

        for (int idx = lane; idx < Tile * Tile; idx += 32)
        {
            int const row = idx / Tile;
            int const col = idx % Tile;
            int const globalK = kBase + row;
            int const globalN = tileN + col;
            float value = 0.0F;
            if (globalK < k && globalN < n)
            {
                int const weightIndex = (weightLayout == 1) ? (globalN * k + globalK) : (globalK * n + globalN);
                value = static_cast<float>(weight[weightIndex]) * __half2float(scales[globalN]);
            }
            bTile[idx] = __float2half_rn(value);
        }
        __syncwarp();

        wmma::load_matrix_sync(aFrag, aTile, Tile);
        wmma::load_matrix_sync(bFrag, bTile, Tile);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        __syncwarp();
    }

    __shared__ float cTile[Tile * Tile];
    wmma::store_matrix_sync(cTile, cFrag, Tile, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < Tile * Tile; idx += 32)
    {
        int const row = idx / Tile;
        int const col = idx % Tile;
        int const globalM = tileM + row;
        int const globalN = tileN + col;
        if (globalM < m && globalN < n)
        {
            output[globalM * n + globalN] = __float2half_rn(cTile[idx]);
        }
    }
}

} // namespace

void w8a16_linear_forward(half const* input, int8_t const* weight, half const* scales, half* output, int m, int n,
    int k, W8A16ScaleMode scaleMode, int groupSize, int weightLayout, cudaStream_t stream) noexcept
{
    if (input == nullptr || weight == nullptr || scales == nullptr || output == nullptr || m <= 0 || n <= 0 || k <= 0)
    {
        return;
    }
    if (scaleMode != W8A16ScaleMode::kPerOutput)
    {
        return;
    }
    if (!(groupSize == 0 || groupSize == 1 || groupSize == k))
    {
        return;
    }

    if (weightLayout == 1 && m == 1)
    {
        dim3 grid((n + 7) / 8);
        w8a16_m1_output_k_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(input, weight, scales, output, n, k);
        return;
    }

    if ((weightLayout == 0 || weightLayout == 1) && m >= 16 && envFlagEnabled("EDGE_LLM_W8A16_HMMA"))
    {
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        w8a16_hmma_m16n16k16_kernel<<<grid, 32, 0, stream>>>(input, weight, scales, output, m, n, k, weightLayout);
        return;
    }

    if (weightLayout == 1)
    {
        dim3 grid(n, m);
        w8a16_per_output_output_k_reference_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
            input, weight, scales, output, m, n, k);
        return;
    }

    if (n >= 32 && k <= 8192)
    {
        int const outputTile = getOutputTileOverride();
        if (outputTile == 128 && n >= 128)
        {
            dim3 grid((n + 127) / 128, m);
            w8a16_small_m_tiled_kernel<128><<<grid, kThreadsPerBlock, 0, stream>>>(
                input, weight, scales, output, m, n, k);
        }
        else if (outputTile == 64 && n >= 64)
        {
            dim3 grid((n + 63) / 64, m);
            w8a16_small_m_tiled_kernel<64><<<grid, kThreadsPerBlock, 0, stream>>>(
                input, weight, scales, output, m, n, k);
        }
        else
        {
            dim3 grid((n + 31) / 32, m);
            w8a16_small_m_tiled_kernel<32><<<grid, kThreadsPerBlock, 0, stream>>>(
                input, weight, scales, output, m, n, k);
        }
        return;
    }

    dim3 grid(n, m);
    w8a16_per_output_reference_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(input, weight, scales, output, m, n, k);
}

} // namespace kernel
} // namespace trt_edgellm
