/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "w8A16Linear.h"

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int kThreadsPerBlock = 256;
constexpr int kOutputTile = 32;
constexpr int kKThreads = kThreadsPerBlock / kOutputTile;

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

__global__ void w8a16_per_output_tiled_kernel(
    half const* __restrict__ input, int8_t const* __restrict__ weight, half const* __restrict__ scales,
    half* __restrict__ output, int m, int n, int k)
{
    int const colBase = blockIdx.x * kOutputTile;
    int const row = blockIdx.y;
    int const colLocal = threadIdx.x % kOutputTile;
    int const kLane = threadIdx.x / kOutputTile;
    int const outCol = colBase + colLocal;
    if (row >= m)
    {
        return;
    }

    float sum = 0.0F;
    int const inputOffset = row * k;
    if (outCol < n)
    {
        for (int kk = kLane; kk < k; kk += kKThreads)
        {
            float a = __half2float(input[inputOffset + kk]);
            float w = static_cast<float>(weight[kk * n + outCol]);
            sum += a * w;
        }
    }

    __shared__ float partial[kOutputTile][kKThreads];
    partial[colLocal][kLane] = sum;
    __syncthreads();

    if (kLane == 0 && outCol < n)
    {
        float total = 0.0F;
#pragma unroll
        for (int i = 0; i < kKThreads; ++i)
        {
            total += partial[colLocal][i];
        }
        float scale = __half2float(scales[outCol]);
        output[row * n + outCol] = __float2half_rn(total * scale);
    }
}

} // namespace

void w8a16_linear_forward(half const* input, int8_t const* weight, half const* scales, half* output, int m, int n,
    int k, W8A16ScaleMode scaleMode, int groupSize, cudaStream_t stream) noexcept
{
    if (input == nullptr || weight == nullptr || scales == nullptr || output == nullptr || m <= 0 || n <= 0 || k <= 0)
    {
        return;
    }
    if (scaleMode != W8A16ScaleMode::kPerOutput)
    {
        return;
    }
    // groupSize is reserved for future group-wise modes. Per-output scaling
    // accepts 0, 1, or K so graph rewriters can choose a neutral value.
    if (!(groupSize == 0 || groupSize == 1 || groupSize == k))
    {
        return;
    }

    if (n >= kOutputTile)
    {
        dim3 grid((n + kOutputTile - 1) / kOutputTile, m);
        w8a16_per_output_tiled_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(input, weight, scales, output, m, n, k);
        return;
    }

    dim3 grid(n, m);
    w8a16_per_output_reference_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(input, weight, scales, output, m, n, k);
}

} // namespace kernel
} // namespace trt_edgellm
