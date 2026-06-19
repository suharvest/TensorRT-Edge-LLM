/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorrt_edge_llm
{
namespace runtime
{

// ===========================================================================
// SlotPool<TSlot> — generic slot-pool orchestration (D-2.5).
//
// Factors out the slot-management plumbing that the qwen3 ASR (single-thread
// poll) and TTS (thread-per-slot) workers duplicated:
//   * ownership of the per-slot object vector (`std::vector<unique_ptr<TSlot>>`)
//   * the configured capacity (max_slots)
//   * id -> slotId routing map (+ its own mutex)
//   * lock-free CAS acquire over the slots' `inUse` flag
//   * idempotent acquire-or-reuse (acquireOrExisting)
//   * bind / unbind / lookup
//   * saturation query
//
// What stays OUT of the template (each worker keeps it):
//   * runtime/CUDA-stream construction + destruction
//   * ASR per-slot AsrSessionState reset (worker resets it BEFORE release/
//     after acquire, mirroring the legacy reset-before-free order)
//   * TTS per-slot queue / worker thread / cancel flag / cancelMap
//   * JSON event emission
//   * routing policy, idle-timeout sweeps, thread model
//
// TSlot contract (compile-time requirements):
//   * `int32_t slotId;`            — stable index, set by the pool's slots()
//   * `std::atomic<bool> inUse;`   — claim flag, CAS'd by acquire/release
//   * default-constructible        — pool reserves but does NOT construct slots
//                                    (worker constructs each TSlot into slots())
//
// THREAD-SAFETY:
//   * acquireFree / acquireOrExisting / release take the pool mutex, so the
//     CAS claim + map bind are atomic w.r.t. each other. TTS drives these from
//     multiple threads (reader + per-slot workers); ASR from one poll thread.
//   * The pool mutex orders ABOVE the id-map mutex (acquireOrExisting takes the
//     pool mutex, then the id-map mutex internally). lookup/bind/unbind take
//     only the id-map mutex, so callers must never hold the pool mutex while
//     calling them (acquireOrExisting is the single helper that needs both and
//     it acquires them in pool->idmap order).
// ===========================================================================
template <typename TSlot>
class SlotPool
{
public:
    explicit SlotPool(int32_t maxSlots = 1)
        : mCapacity(maxSlots >= 1 ? maxSlots : 1)
    {
        mSlots.reserve(static_cast<size_t>(mCapacity));
    }

    SlotPool(SlotPool const&) = delete;
    SlotPool& operator=(SlotPool const&) = delete;

    //! Configured slot count (max_slots).
    int32_t capacity() const
    {
        return mCapacity;
    }

    //! True when every constructed slot has inUse == true.
    bool saturated() const
    {
        std::lock_guard<std::mutex> const guard(mPoolMutex);
        for (auto const& slotPtr : mSlots)
        {
            if (slotPtr && !slotPtr->inUse.load())
            {
                return false;
            }
        }
        return !mSlots.empty();
    }

    //! Direct access to the underlying slot vector. The worker constructs its
    //! TSlots into this vector during init (runtime/stream bring-up) and indexes
    //! it by slotId on the hot path. The pool itself never constructs TSlots.
    std::vector<std::unique_ptr<TSlot>>& slots()
    {
        return mSlots;
    }

    std::vector<std::unique_ptr<TSlot>> const& slots() const
    {
        return mSlots;
    }

    //! Bounds-checked slot accessor. Returns nullptr for out-of-range ids.
    TSlot* get(int32_t slotId)
    {
        if (slotId < 0 || static_cast<size_t>(slotId) >= mSlots.size())
        {
            return nullptr;
        }
        return mSlots[static_cast<size_t>(slotId)].get();
    }

    //! id -> slotId, or -1 when no live mapping exists.
    int32_t lookup(std::string const& id) const
    {
        std::lock_guard<std::mutex> const guard(mMapMutex);
        auto const it = mIdToSlot.find(id);
        return it == mIdToSlot.end() ? -1 : it->second;
    }

    //! Bind id -> slotId. Caller must already own the slot (post-acquire).
    void bind(std::string const& id, int32_t slotId)
    {
        std::lock_guard<std::mutex> const guard(mMapMutex);
        mIdToSlot[id] = slotId;
    }

    //! Drop the id -> slot mapping (end / timeout / error cleanup).
    void unbind(std::string const& id)
    {
        std::lock_guard<std::mutex> const guard(mMapMutex);
        mIdToSlot.erase(id);
    }

    //! Claim a free slot via atomic CAS on each slot's `inUse`. Returns the slot
    //! index or -1 when the pool is saturated. Does NOT bind any id (caller
    //! binds after a successful claim) and does NOT reset per-slot worker state.
    int32_t acquireFree()
    {
        std::lock_guard<std::mutex> const guard(mPoolMutex);
        return acquireFreeLocked();
    }

    //! Idempotent acquire: if @p id already owns a slot, return it (re-use,
    //! NO second claim); otherwise claim a fresh free slot, bind it, and return
    //! it. Returns -1 when saturated. The lookup + claim + bind are performed
    //! atomically under the pool mutex so concurrent begins for the same id
    //! cannot each grab a distinct slot.
    int32_t acquireOrExisting(std::string const& id)
    {
        std::lock_guard<std::mutex> const guard(mPoolMutex);
        {
            std::lock_guard<std::mutex> const mapGuard(mMapMutex);
            auto const it = mIdToSlot.find(id);
            if (it != mIdToSlot.end())
            {
                return it->second; // re-use existing slot for this id
            }
        }
        int32_t const slotId = acquireFreeLocked();
        if (slotId >= 0)
        {
            std::lock_guard<std::mutex> const mapGuard(mMapMutex);
            mIdToSlot[id] = slotId;
        }
        return slotId;
    }

    //! Mark a slot free. Pure slot-management: clears `inUse` ONLY. The worker
    //! must reset/teardown the slot's own per-session state (ASR session reset,
    //! TTS cancel/queue) and drop any id binding BEFORE calling release, exactly
    //! as the legacy reset-before-free / unbind-before-free order required.
    void release(int32_t slotId)
    {
        std::lock_guard<std::mutex> const guard(mPoolMutex);
        if (slotId < 0 || static_cast<size_t>(slotId) >= mSlots.size())
        {
            return;
        }
        mSlots[static_cast<size_t>(slotId)]->inUse.store(false, std::memory_order_release);
    }

    //! Drop all slots + id mappings (teardown). The worker must have already
    //! torn down runtime/stream/thread resources held inside each TSlot.
    void clear()
    {
        std::lock_guard<std::mutex> const guard(mPoolMutex);
        {
            std::lock_guard<std::mutex> const mapGuard(mMapMutex);
            mIdToSlot.clear();
        }
        mSlots.clear();
    }

private:
    //! CAS over inUse, assuming mPoolMutex is held. Returns slotId or -1.
    int32_t acquireFreeLocked()
    {
        for (auto& slotPtr : mSlots)
        {
            bool expected = false;
            if (slotPtr->inUse.compare_exchange_strong(expected, true))
            {
                return slotPtr->slotId;
            }
        }
        return -1;
    }

    int32_t mCapacity{1};
    std::vector<std::unique_ptr<TSlot>> mSlots;
    mutable std::mutex mPoolMutex;                  //!< guards slot vector + CAS claim
    std::unordered_map<std::string, int32_t> mIdToSlot;
    mutable std::mutex mMapMutex;                   //!< guards id -> slot routing map
};

} // namespace runtime
} // namespace tensorrt_edge_llm
