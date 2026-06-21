/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// ── Scope note (Base N>1 serving) ────────────────────────────────────────────
// This header intentionally contains ONLY the static-partition lane allocator
// (SessionLaneManager + LaneOwnerKind) needed by the N>1 ASR voice worker
// (native/edgellm_voice_worker/qwen3_asr_worker.cpp), which runs the vanilla
// rt::LLMInferenceRuntime ONE-SHOT path and uses the lane manager purely for
// lane reservation over the shared HybridCacheManager batch rows.
//
// The full chunked-streaming wrapper (AsrStreamingSessionRuntime: incremental
// appendChunk/beginAsrSession/endAsrSession + per-lane KV reset) is the DEFERRED
// streaming runtime and is NOT part of the Base N>1 serving stack. It must not
// be pulled in here, to avoid dragging the ASR streaming spike (and its
// LLMInferenceRuntime hooks) into the Base build. See engine-overlay
// PATCH-STATE-v080.md.

#include <cstdint>
#include <mutex>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! @brief Which subsystem owns a lane (for the static ASR/TTS partition).
enum class LaneOwnerKind
{
    kNone, //!< Free lane.
    kAsr,  //!< Owned by a streaming-ASR session.
    kTts   //!< Owned by a TTS slot.
};

/*!
 * @brief Static-partition lane allocator over a shared HybridCacheManager's batch rows (spec R1).
 *
 * v0.8.0 has ONE process-lifetime HybridCacheManager per engine (SharedResources::cacheManagers),
 * not per-session. To let concurrent ASR sessions (and later TTS slots) co-reside without a
 * full-batch reset clobbering each other, lanes are statically partitioned:
 *   ASR -> [0, asrMax)   TTS -> [asrMax, asrMax + ttsMax)
 * with asrMax + ttsMax <= maxBatchSize. acquire() hands out a free lane within the owner's
 * partition; release() returns it. The physical laneId IS the HybridCacheManager batch row.
 *
 * Thread-safe (internal mutex); the engine step itself is serialized elsewhere (shared PipelineIO).
 */
class SessionLaneManager
{
public:
    //! @brief One physical lane's bookkeeping.
    struct LaneRecord
    {
        LaneOwnerKind ownerKind{LaneOwnerKind::kNone}; //!< Owning subsystem (or kNone if free).
        int64_t ownerId{-1};                           //!< Caller-supplied owner id (session/slot).
        int32_t slotId{-1};                            //!< Logical slot id within the partition.
        bool inUse{false};                             //!< Whether this lane is currently allocated.
        int32_t kvLength{0};                           //!< Last-known committed KV length (advisory).
    };

    //! @param maxBatchSize Physical lane capacity of the shared cache manager.
    //! @param asrMax       Lanes reserved for ASR ([0, asrMax)).
    //! @param ttsMax       Lanes reserved for TTS ([asrMax, asrMax + ttsMax)).
    SessionLaneManager(int32_t maxBatchSize, int32_t asrMax, int32_t ttsMax);

    //! @brief Reserve a free lane in the owner's partition.
    //! @return Physical laneId (== HybridCacheManager batch row), or -1 if the partition is full.
    int32_t acquire(LaneOwnerKind ownerKind, int64_t ownerId);

    //! @brief Release a previously-acquired lane back to its partition.
    void release(int32_t laneId);

    //! @brief Record the advisory KV length for a lane (introspection only).
    void setKvLength(int32_t laneId, int32_t kvLength);

    //! @brief Snapshot of a lane's record (for tests / introspection).
    LaneRecord recordOf(int32_t laneId) const;

    //! @brief First lane index of the TTS partition (== asrMax).
    int32_t ttsBase() const noexcept
    {
        return mAsrMax;
    }

    //! @brief Physical lane capacity.
    int32_t maxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

private:
    mutable std::mutex mMutex;
    int32_t mMaxBatchSize{0};
    int32_t mAsrMax{0};
    int32_t mTtsMax{0};
    std::vector<LaneRecord> mLanes;
};

} // namespace rt
} // namespace trt_edgellm
