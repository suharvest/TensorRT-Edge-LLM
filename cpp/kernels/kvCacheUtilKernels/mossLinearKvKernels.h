/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

// Append a single-layer K or V slice (FP16, dim order [S, H, D]) to the per-slot
// linear KV buffer (FP16, dim order [2, maxSeqLen, H, D]) at logical position pastLen.
// Used by MOSS-TTS-Nano runtime: decode_step / local_cached_step pass S=1 delta;
// prefill passes the full S-token present output.
//
// Layout contract (MOSS):
//   kvBuffer  [2 (K=0,V=1), maxSeqLen, numHeads, headDim] FP16
//   delta     [S, numHeads, headDim]                       FP16
// Caller must ensure pastLen + seqLen <= maxSeqLen and batch size == 1.
void appendMossLinearKV(half* kvBuffer, half const* delta, int32_t seqLen, int32_t pastLen, int32_t maxSeqLen,
    int32_t numHeads, int32_t headDim, int32_t kvIdx, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
