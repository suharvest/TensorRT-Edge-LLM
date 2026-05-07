/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef CUTE_DSL_GDN_ENABLED

#include "kernels/cuteDslRuntimeCompat.h"

#include "cutedsl_all.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/** Device pointers and dimensions; runner fills generated tensor structs from these. */
struct GDNParams
{
    void* q{};
    void* k{};
    void* v{};
    void* a{};
    void* b{};
    void* A_log{};
    void* dt_bias{};
    void* h0_source{};
    void* context_lengths{}; ///< [N] int32 — valid length per batch (decode / sequential prefill)
    void* cu_seqlens{};      ///< [N+1] int32 — prefix-sum of context_lengths (Blackwell prefill)
    void* h0_scratch{};      ///< [N, hv, k, v] f32 — pre-allocated scratch for h0_out (Blackwell prefill);
                             ///<   must be provided by caller (e.g. plugin workspace).
    void* o{};

    int32_t n{};
    int32_t seq_len{};
    int32_t h{};
    int32_t hv{};
    int32_t k_dim{};
    int32_t v_dim{};
    int32_t smVersion{}; // GPU SM version for dispatch (e.g. 87, 110)
};

/** Loads AOT .o, fills tensor structs from GDNParams, calls generated wrapper. */
class CuteDslGDNRunner
{
public:
    CuteDslGDNRunner() = default;
    ~CuteDslGDNRunner() = default;
    CuteDslGDNRunner(CuteDslGDNRunner const&) = delete;
    CuteDslGDNRunner& operator=(CuteDslGDNRunner const&) = delete;

    static bool canImplement(int32_t kDim, int32_t vDim, int32_t smVersion);

    static bool loadKernelModules();
    static void unloadKernelModules();

    /** Run GDN: decode when params.seq_len == 1, else prefill (Blackwell or sequential). */
    int run(GDNParams const& params, cudaStream_t stream);

private:
    int runDecode(GDNParams const& params, cudaStream_t stream);
    int runPrefill(GDNParams const& params, cudaStream_t stream);
    int runPrefillBlackwell(GDNParams const& params, cudaStream_t stream);

    static gdn_decode_Kernel_Module_t sDecodeModule;
    static gdn_prefill_Kernel_Module_t sPrefillModule;
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    static gdn_prefill_blackwell_Kernel_Module_t sBlackwellPrefillModule;
#endif
    static bool sLoaded;
};

} // namespace trt_edgellm

#endif // CUTE_DSL_GDN_ENABLED
