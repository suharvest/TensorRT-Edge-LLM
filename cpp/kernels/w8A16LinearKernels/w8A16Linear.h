/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace kernel
{

enum class W8A16ScaleMode : int32_t
{
    kPerOutput = 0,
};

/*!
 * @brief Weight-only INT8 dense linear.
 *
 * Computes ``C[M, N] = A[M, K] @ (W[K, N] * scales[N])``.
 * Activations remain FP16; only weights are INT8. The current implementation
 * supports per-output-channel scales and is optimized as a correctness-first
 * small-M path for streaming decode. Faster tiled kernels can use the same ABI.
 */
void w8a16_linear_forward(half const* input, int8_t const* weight, half const* scales, half* output, int m, int n,
    int k, W8A16ScaleMode scaleMode, int groupSize, int weightLayout, cudaStream_t stream) noexcept;

} // namespace kernel
} // namespace trt_edgellm

