/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernels/kvCacheUtilKernels/mossLinearKvKernels.h"

#include "common/checkMacros.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

__global__ void appendMossLinearKVGenericKernel(half* __restrict__ kvBuffer, half const* __restrict__ delta,
    int32_t seqLen, int32_t pastLen, int32_t maxSeqLen, int32_t numHeads, int32_t headDim, int32_t kvIdx)
{
    int64_t const total = static_cast<int64_t>(seqLen) * numHeads * headDim;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride)
    {
        int32_t const d = static_cast<int32_t>(idx % headDim);
        int64_t const sh = idx / headDim;
        int32_t const h = static_cast<int32_t>(sh % numHeads);
        int32_t const s = static_cast<int32_t>(sh / numHeads);

        int32_t const logicalPos = pastLen + s;
        int64_t const dst = ((static_cast<int64_t>(kvIdx) * maxSeqLen + logicalPos) * numHeads + h) * headDim + d;
        int64_t const src = (static_cast<int64_t>(s) * numHeads + h) * headDim + d;

        kvBuffer[dst] = delta[src];
    }
}

// Fast path for headDim==64: vectorized 16-byte (float4 = 8 halves) loads/stores.
// Grid: seqLen * numHeads CTAs; block: 8 threads, each moves one float4.
__global__ void appendMossLinearKVHeadDim64Kernel(half* __restrict__ kvBuffer, half const* __restrict__ delta,
    int32_t seqLen, int32_t pastLen, int32_t maxSeqLen, int32_t numHeads, int32_t kvIdx)
{
    int32_t const s = blockIdx.x / numHeads;
    int32_t const h = blockIdx.x - s * numHeads;
    int32_t const vecIdx = threadIdx.x;

    if (s >= seqLen || vecIdx >= 8)
    {
        return;
    }

    constexpr int32_t kHeadDim = 64;
    int32_t const logicalPos = pastLen + s;

    int64_t const srcBase = (static_cast<int64_t>(s) * numHeads + h) * kHeadDim;
    int64_t const dstBase = ((static_cast<int64_t>(kvIdx) * maxSeqLen + logicalPos) * numHeads + h) * kHeadDim;

    float4 const* srcVec = reinterpret_cast<float4 const*>(delta + srcBase);
    float4* dstVec = reinterpret_cast<float4*>(kvBuffer + dstBase);

    dstVec[vecIdx] = srcVec[vecIdx];
}

} // namespace

void appendMossLinearKV(half* kvBuffer, half const* delta, int32_t seqLen, int32_t pastLen, int32_t maxSeqLen,
    int32_t numHeads, int32_t headDim, int32_t kvIdx, cudaStream_t stream)
{
    if (kvBuffer == nullptr)
        throw std::invalid_argument("appendMossLinearKV: kvBuffer must not be null.");
    if (delta == nullptr)
        throw std::invalid_argument("appendMossLinearKV: delta must not be null.");
    if (seqLen < 0 || pastLen < 0 || maxSeqLen <= 0 || numHeads <= 0 || headDim <= 0)
        throw std::invalid_argument("appendMossLinearKV: invalid shape argument.");
    if (kvIdx != 0 && kvIdx != 1)
        throw std::invalid_argument("appendMossLinearKV: kvIdx must be 0 for K or 1 for V.");
    if (pastLen + seqLen > maxSeqLen)
        throw std::invalid_argument("appendMossLinearKV: pastLen + seqLen exceeds maxSeqLen.");

    if (seqLen == 0)
        return;

    if (headDim == 64)
    {
        dim3 const grid(static_cast<unsigned int>(seqLen * numHeads));
        dim3 const block(8);
        appendMossLinearKVHeadDim64Kernel<<<grid, block, 0, stream>>>(
            kvBuffer, delta, seqLen, pastLen, maxSeqLen, numHeads, kvIdx);
    }
    else
    {
        int64_t const total = static_cast<int64_t>(seqLen) * numHeads * headDim;
        constexpr int32_t blockSize = 256;
        int32_t const gridSize = static_cast<int32_t>((total + blockSize - 1) / blockSize);
        appendMossLinearKVGenericKernel<<<gridSize, blockSize, 0, stream>>>(
            kvBuffer, delta, seqLen, pastLen, maxSeqLen, numHeads, headDim, kvIdx);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
