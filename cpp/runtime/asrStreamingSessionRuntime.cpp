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

// Scope: SessionLaneManager only (static-partition lane allocator). See header
// for why the full chunked-streaming wrapper is intentionally excluded from the
// Base N>1 serving stack.

#include "runtime/asrStreamingSessionRuntime.h"
#include "common/logger.h"

#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

// ── SessionLaneManager (spec R1) ─────────────────────────────────────────────

SessionLaneManager::SessionLaneManager(int32_t maxBatchSize, int32_t asrMax, int32_t ttsMax)
    : mMaxBatchSize(maxBatchSize)
    , mAsrMax(asrMax)
    , mTtsMax(ttsMax)
    , mLanes(static_cast<size_t>(maxBatchSize < 0 ? 0 : maxBatchSize))
{
    if (maxBatchSize < 0 || asrMax < 0 || ttsMax < 0)
    {
        throw std::invalid_argument("SessionLaneManager: negative size argument.");
    }
    if (asrMax + ttsMax > maxBatchSize)
    {
        throw std::invalid_argument("SessionLaneManager: asrMax + ttsMax exceeds maxBatchSize.");
    }
}

int32_t SessionLaneManager::acquire(LaneOwnerKind ownerKind, int64_t ownerId)
{
    std::lock_guard<std::mutex> lock(mMutex);
    int32_t begin = 0;
    int32_t end = 0;
    if (ownerKind == LaneOwnerKind::kAsr)
    {
        begin = 0;
        end = mAsrMax;
    }
    else if (ownerKind == LaneOwnerKind::kTts)
    {
        begin = mAsrMax;
        end = mAsrMax + mTtsMax;
    }
    else
    {
        return -1; // kNone is not allocatable.
    }
    int32_t slot = 0;
    for (int32_t lane = begin; lane < end; ++lane)
    {
        if (!mLanes[lane].inUse)
        {
            mLanes[lane].ownerKind = ownerKind;
            mLanes[lane].ownerId = ownerId;
            mLanes[lane].slotId = slot;
            mLanes[lane].inUse = true;
            mLanes[lane].kvLength = 0;
            return lane;
        }
        ++slot;
    }
    return -1; // partition full
}

void SessionLaneManager::release(int32_t laneId)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (laneId < 0 || laneId >= mMaxBatchSize)
    {
        LOG_ERROR("SessionLaneManager::release: laneId=%d out of range [0, %d)", laneId, mMaxBatchSize);
        return;
    }
    mLanes[laneId] = LaneRecord{};
}

void SessionLaneManager::setKvLength(int32_t laneId, int32_t kvLength)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (laneId < 0 || laneId >= mMaxBatchSize)
    {
        return;
    }
    mLanes[laneId].kvLength = kvLength;
}

SessionLaneManager::LaneRecord SessionLaneManager::recordOf(int32_t laneId) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (laneId < 0 || laneId >= mMaxBatchSize)
    {
        return LaneRecord{};
    }
    return mLanes[laneId];
}

} // namespace rt
} // namespace trt_edgellm
