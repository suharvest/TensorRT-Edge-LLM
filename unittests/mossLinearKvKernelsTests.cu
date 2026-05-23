/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/checkMacros.h"
#include "kernels/kvCacheUtilKernels/mossLinearKvKernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

namespace
{

int64_t kvOffset(int32_t kvIdx, int32_t pos, int32_t h, int32_t d, int32_t maxSeqLen, int32_t numHeads, int32_t headDim)
{
    return ((static_cast<int64_t>(kvIdx) * maxSeqLen + pos) * numHeads + h) * headDim + d;
}

int64_t deltaOffset(int32_t s, int32_t h, int32_t d, int32_t numHeads, int32_t headDim)
{
    return (static_cast<int64_t>(s) * numHeads + h) * headDim + d;
}

std::vector<half> makeDelta(int32_t seqLen, int32_t numHeads, int32_t headDim)
{
    std::vector<half> delta(static_cast<size_t>(seqLen) * numHeads * headDim);
    for (size_t i = 0; i < delta.size(); ++i)
        delta[i] = __float2half(static_cast<float>((i % 251) + 1) / 16.0f);
    return delta;
}

void runAppendAndVerifyWrittenRange(
    int32_t seqLen, int32_t pastLen, int32_t maxSeqLen, int32_t numHeads, int32_t headDim, int32_t kvIdx)
{
    size_t const kvSize = static_cast<size_t>(2) * maxSeqLen * numHeads * headDim;
    size_t const deltaSize = static_cast<size_t>(seqLen) * numHeads * headDim;

    std::vector<half> deltaHost = makeDelta(seqLen, numHeads, headDim);

    half* kvDevice{nullptr};
    half* deltaDevice{nullptr};

    CUDA_CHECK(cudaMalloc(&kvDevice, kvSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&deltaDevice, deltaSize * sizeof(half)));
    CUDA_CHECK(cudaMemset(kvDevice, 0, kvSize * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(deltaDevice, deltaHost.data(), deltaSize * sizeof(half), cudaMemcpyHostToDevice));

    cudaStream_t stream{nullptr};
    kernel::appendMossLinearKV(kvDevice, deltaDevice, seqLen, pastLen, maxSeqLen, numHeads, headDim, kvIdx, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> kvHost(kvSize);
    CUDA_CHECK(cudaMemcpy(kvHost.data(), kvDevice, kvSize * sizeof(half), cudaMemcpyDeviceToHost));

    for (int32_t s = 0; s < seqLen; ++s)
    {
        int32_t const pos = pastLen + s;
        for (int32_t h = 0; h < numHeads; ++h)
        {
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const dst = kvOffset(kvIdx, pos, h, d, maxSeqLen, numHeads, headDim);
                int64_t const src = deltaOffset(s, h, d, numHeads, headDim);
                ASSERT_EQ(__half2float(kvHost[dst]), __half2float(deltaHost[src]))
                    << "Mismatch at s=" << s << ", h=" << h << ", d=" << d;
            }
        }
    }

    CUDA_CHECK(cudaFree(deltaDevice));
    CUDA_CHECK(cudaFree(kvDevice));
}

} // namespace

TEST(MossLinearKvKernelsTest, DecodeWritesLogicalPastPositionAndLeavesOthersZero)
{
    constexpr int32_t maxSeqLen = 128, numHeads = 12, headDim = 64, pastLen = 10, seqLen = 1, kvIdx = 0;

    size_t const kvSize = static_cast<size_t>(2) * maxSeqLen * numHeads * headDim;
    size_t const deltaSize = static_cast<size_t>(seqLen) * numHeads * headDim;

    std::vector<half> deltaHost = makeDelta(seqLen, numHeads, headDim);

    half* kvDevice{nullptr};
    half* deltaDevice{nullptr};
    CUDA_CHECK(cudaMalloc(&kvDevice, kvSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&deltaDevice, deltaSize * sizeof(half)));
    CUDA_CHECK(cudaMemset(kvDevice, 0, kvSize * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(deltaDevice, deltaHost.data(), deltaSize * sizeof(half), cudaMemcpyHostToDevice));

    cudaStream_t stream{nullptr};
    kernel::appendMossLinearKV(kvDevice, deltaDevice, seqLen, pastLen, maxSeqLen, numHeads, headDim, kvIdx, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> kvHost(kvSize);
    CUDA_CHECK(cudaMemcpy(kvHost.data(), kvDevice, kvSize * sizeof(half), cudaMemcpyDeviceToHost));

    for (int32_t pos = 0; pos < maxSeqLen; ++pos)
    {
        for (int32_t h = 0; h < numHeads; ++h)
        {
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const kIdx = kvOffset(0, pos, h, d, maxSeqLen, numHeads, headDim);
                if (pos == pastLen)
                {
                    int64_t const src = deltaOffset(0, h, d, numHeads, headDim);
                    ASSERT_EQ(__half2float(kvHost[kIdx]), __half2float(deltaHost[src]));
                }
                else
                {
                    ASSERT_EQ(__half2float(kvHost[kIdx]), 0.0f);
                }

                int64_t const vIdx = kvOffset(1, pos, h, d, maxSeqLen, numHeads, headDim);
                ASSERT_EQ(__half2float(kvHost[vIdx]), 0.0f);
            }
        }
    }

    CUDA_CHECK(cudaFree(deltaDevice));
    CUDA_CHECK(cudaFree(kvDevice));
}

TEST(MossLinearKvKernelsTest, PrefillWritesPositionsZeroToSeven)
{
    runAppendAndVerifyWrittenRange(8, 0, 128, 12, 64, 0);
}

TEST(MossLinearKvKernelsTest, WriteToMaxFillsExactlyToCapacity)
{
    runAppendAndVerifyWrittenRange(8, 120, 128, 12, 64, 0);
}

TEST(MossLinearKvKernelsTest, GenericPathHeadDim128)
{
    runAppendAndVerifyWrittenRange(4, 0, 64, 8, 128, 1);
}

TEST(MossLinearKvKernelsTest, RejectsAppendPastMaxSeqLen)
{
    half* kvDevice{nullptr};
    half* deltaDevice{nullptr};
    CUDA_CHECK(cudaMalloc(&kvDevice, static_cast<size_t>(2) * 128 * 12 * 64 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&deltaDevice, static_cast<size_t>(10) * 12 * 64 * sizeof(half)));

    EXPECT_THROW(
        kernel::appendMossLinearKV(kvDevice, deltaDevice, 10, 125, 128, 12, 64, 0, nullptr), std::invalid_argument);

    CUDA_CHECK(cudaFree(deltaDevice));
    CUDA_CHECK(cudaFree(kvDevice));
}
