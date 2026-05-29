/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "action/alpamayo1ActionRunner.h"
#include "common/hashUtils.h"
#include "common/tensor.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/eagleDraftEngineRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <cassert>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{

/*! \brief Structure to hold cached system prompt and its KV cache (unified with recurrent state support)
 */
struct SystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    std::vector<rt::Tensor> kvCacheLayers;        //!< Per-layer KV cache tensors for the system prompt
    std::vector<rt::Tensor>
        recurrentStateContents;                //!< Cached recurrent states for hybrid layers (empty if not applicable)
    std::vector<rt::Tensor> convStateContents; //!< Cached conv states for hybrid layers (empty if not applicable)
};

namespace rt
{
/*!
 * @brief Batch result data for a single sequence
 *
 * Encapsulates all data needed to track a batch's execution results,
 * whether it's active or evicted. Groups related fields together for
 * better cache locality and maintainability.
 */
struct BatchResult
{
    std::vector<int32_t> tokenIds;           //!< Generated token IDs
    std::vector<int32_t> rawBatchedInputIds; //!< Original input token IDs
    int32_t generateLength{0};               //!< Number of tokens generated
    int32_t actualIterations{0};             //!< Number of iterations executed
    int32_t effectivePrefillLength{0};       //!< Effective prefill length (excluding reused KVCache length)
};

/*!
 * @brief Execution context for speculative decode runtime
 *
 * Holds execution information and intermediate metadata during inference.
 * Supports multi-batch inference with independent sequence tracking.
 */
struct SpecDecodeInferenceContext
{
    std::vector<std::string> systemPrompts;               //!< System prompts for each sequence in batch
    std::vector<std::vector<int32_t>> rawBatchedInputIds; //!< Original token IDs before preprocessing (includes padding
                                                          //!< and removal of reused system IDs)
    std::vector<std::vector<int32_t>> tokenIds;           //!< Token IDs for each sequence: [batch_size][seq_length]
    std::vector<int32_t> currentGenerateLengths;          //!< Current generation length for each sequence: [batch_size]
    std::vector<int32_t>
        effectivePrefillLengths;        //!< Effective prefill length (excluding reused KVCache length) [batch_size]
    std::vector<int8_t> finishedStates; //!< Finished state for each sequence: [batch_size] (0=not finished, 1=finished)

    // Completed batch results (saved before eviction for final output)
    // Key: original batch index, Value: complete batch result data
    std::unordered_map<int32_t, BatchResult> completedBatches; //!< Results of completed batches (unified storage)
    std::vector<int32_t> batchIndexMapping;                    //!< Maps current batch index to original index
    std::vector<SlotStreamState> slotStreams;                  //!< Per-slot streaming state (parallel to tokenIds).
    rt::OptionalInputTensor visualEmbeddings;                  //!< Optional visual embeddings
    rt::OptionalInputTensor audioEmbeddings;                   //!< Optional audio embeddings
    rt::OptionalInputTensors deepstackFeatures; //!< Deepstack features for Qwen3-VL (raw features before embedding)
    int32_t generationRound;                    //!< Current generation round (shared across all batches)
    int32_t maxGenerateLength;                  //!< Maximum generation length
    int32_t activeBatchSize;                    //!< Current active batch size
    std::string loraWeightsName{""};            //!< LoRA adapter name used by this request
    cudaStream_t stream;                        //!< CUDA stream

    // Sampling parameters (forwarded from request)
    float temperature{1.0f}; //!< Temperature for sampling
    float topP{1.0f};        //!< Top-P (nucleus) sampling parameter
    int64_t topK{0};         //!< Top-K sampling parameter

    // Thinker embedding output (Qwen3-Omni audio generation)
    bool outputThinkerEmbeddings{false}; //!< Whether to capture hidden states for Talker pipeline

    //! Optional per-token callback invoked after each vanilla decode step
    std::optional<TokenCallback> onTokenGenerated;

    /*!
     * @brief Initialize the context with given parameters
     * @param batchSize Active batch size
     * @param maxGenLength Maximum generation length
     * @param visual Optional visual embeddings
     * @param deepstackFeatures Deepstack features for Qwen3-VL (raw features before embedding)
     * @param loraName LoRA weights name used by this request
     * @param cudaStream CUDA stream for operations
     */
    void initialize(int32_t batchSize, int32_t maxGenLength, rt::OptionalInputTensor const& visual,
        rt::OptionalInputTensors const& deepstackFeatures, std::string const& loraName, cudaStream_t cudaStream);
};

/*!
 * @brief Drafting configuration for Eagle speculative decoding
 *
 * Configuration parameters to drive Eagle spec-decoding.
 */
struct EagleDraftingConfig
{
    int32_t draftingTopK;   //!< Tokens to select from one predecessor for next draft tree level
    int32_t draftingStep;   //!< Number of drafting steps with draft model
    int32_t verifyTreeSize; //!< Number of tokens for base model verification
};

/*!
 * @brief Unified LLM inference runtime with optional Eagle speculative decoding
 *
 * Manages inference pipeline for both standard (vanilla) and Eagle speculative decoding modes.
 * When constructed without a drafting config, operates as a pure vanilla decoding runtime
 * (equivalent to the former LLMInferenceRuntime) with zero draft-model memory overhead.
 * Coordinates base model, optional draft model, and multimodal processing (vision + audio).
 */
class LLMInferenceSpecDecodeRuntime
{
public:
    /*!
     * @brief Construct runtime with Eagle speculative decoding
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param draftingConfig Eagle drafting configuration
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, EagleDraftingConfig const& draftingConfig,
        cudaStream_t stream);

    /*!
     * @brief Construct runtime for vanilla-only decoding (no draft model)
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    /*!
     * @brief Construct vanilla-only runtime that SHARES a base ICudaEngine with another runtime (ASR slot-pool, D1).
     *
     * Used to build N decoder/thinker slots that share one deserialized base engine (saving weight memory)
     * while each slot owns an independent IExecutionContext, KV/cache manager state, runtime tensors, and
     * shared-execution-context memory. The supplied @p sharedBaseEngine must come from another live
     * LLMEngineRunner (via getEngine()), whose deserializing owner must outlive this runtime.
     *
     * This path is vanilla-only (no Eagle draft). It still loads its own embedding table, tokenizer, and
     * (if present) multimodal runners from @p engineDir / @p multimodalEngineDir, exactly like the vanilla
     * path-based constructor — only the base LLM engine weights are shared.
     *
     * @param sharedBaseEngine Shared, already-deserialized base TensorRT engine (must be non-null)
     * @param engineDir Directory containing engine files (embedding.safetensors, config.json, etc.)
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceSpecDecodeRuntime(std::shared_ptr<nvinfer1::ICudaEngine> sharedBaseEngine,
        std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    //! @brief Get the shared base TensorRT engine backing this runtime (ASR slot-pool, D1).
    //! @return Shared pointer to the base ICudaEngine, for constructing additional slot runtimes that share it.
    //! @note Valid only while this runtime (and its deserializing owner) is alive.
    std::shared_ptr<nvinfer1::ICudaEngine> getBaseEngine() const noexcept;

    //! @brief Destructor
    ~LLMInferenceSpecDecodeRuntime() noexcept = default;

    //! @brief Capture CUDA graphs for decoding stages to optimize performance.
    //!
    //! When draft model is present, captures graphs for draft proposal, draft accept token,
    //! base verification, and base vanilla decoding. Without draft model, captures only
    //! vanilla decoding graphs.
    //!
    //! @param stream CUDA stream
    //! @return True if all stage captures succeed, false otherwise
    //! @throws std::runtime_error if a tensor reshape operation fails
    //! @note If capture fails for any stage, the inference can proceed without CUDA graph capture,
    //! but at cost of performance degradation.
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*!
     * @brief Handle generation request
     * @param request Generation request with prompts and parameters
     * @param response Output response with generated tokens and text
     * @param stream CUDA stream
     * @return True on success, false on failure
     * @throws std::runtime_error if an LLM or CUDA operation fails
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false);

    /*!
     * @brief Generate and save system prompt KV cache (public API matching standard runtime signature)
     * @param prompt The system prompt to generate the KVCache
     * @param loraWeightsName The name of the LoRA weights
     * @param stream The CUDA stream used for the generation
     * @return True if the KVCache is generated and saved successfully, false otherwise
     * @throws std::runtime_error if a CUDA operation fails
     */
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    /*! \brief Set the random seed used when initializing the action diffusion noise trajectory
     *  \param seed Random seed value; has no effect if no action runner is loaded
     */
    void setActionNoiseSeed(int32_t seed) noexcept;

    /*!
     * @brief One-time per-session setup that must run before any
     *        appendPrefillEmbeds calls. Reuses the OneShot setup path so LoRA
     *        binding, KV-cache reset, system-prompt restore, and reuse-length
     *        bookkeeping happen exactly as they do in handleRequest.
     *
     * After this returns true the caller may invoke appendPrefillEmbeds any
     * number of times (subject to engine max_input_len per chunk and max KV
     * capacity total), then drive decode separately.
     *
     * Caller must populate context.activeBatchSize, context.systemPrompts[i],
     * context.rawBatchedInputIds[i] (initial prompt token IDs — typically the
     * text prefix that precedes any audio), and context.loraWeightsName / stream
     * before calling.
     *
     * @return True on success, false on validation failure.
     */
    /*!
     * @brief Status code reported by the most recent appendPrefillEmbeds call.
     *        M2 introduces the kKvCapacityExceeded / kChunkTooLong refusal
     *        modes so callers can distinguish capacity refusal from engine
     *        prefill failure and emit a structured error event.
     */
    enum class AppendPrefillStatus : int32_t
    {
        kOk = 0,                  //!< Append succeeded.
        kKvCapacityExceeded = 1,  //!< current_kv_length + chunkLen > max_kv_cache_capacity.
        kChunkTooLong = 2,        //!< tokenSliceDelta.size() > engine max_input_len.
        kPreconditionFailed = 3,  //!< Per-chunk validation in setUpForPrefillExecutionForChunk failed.
        kPrefillFailed = 4,       //!< Underlying executePrefillStep returned false.
    };

    //! @brief Status of the most recent appendPrefillEmbeds call (kOk if never called).
    AppendPrefillStatus getLastAppendStatus() const noexcept
    {
        return mLastAppendStatus;
    }

    //! @brief Snapshot of the live KV cache length (batch slot 0) read at entry
    //!        to the most recent appendPrefillEmbeds call. Useful for the
    //!        worker to surface in a structured capacity-error event.
    int32_t getLastObservedKvLength() const noexcept
    {
        return mLastObservedKvLength;
    }

    //! @brief Static read of the engine's max KV cache capacity (256 in the
    //!        shipped ASR thinker engine). Convenience for callers that want
    //!        to advertise the cap in a structured error.
    int32_t getMaxKvCacheCapacity() const noexcept;

    //! @brief Synchronous D2H read of the live KV cache length for batch slot
    //!        @p batchIdx. M2 helper for both the lifecycle acceptance test and
    //!        the M3 worker, which wants to advertise live cache occupancy in
    //!        structured error events.
    int32_t peekKvCacheLength(int32_t batchIdx, cudaStream_t stream);

    //! @brief Begin a streaming-ASR session.
    //!
    //! Wraps the one-shot setup path so LoRA / KV reset / system-prompt
    //! restore / reuse lengths are bound identically to the handleRequest
    //! path. When a `stream` is supplied and an audio runner is loaded, also
    //! initializes the audio runner's MRope cos/sin cache for the worst-case
    //! session length (bounded by max_kv_cache_capacity) so per-chunk
    //! encodeMelChunk calls do not touch MRope state.
    //!
    //! @param context  Inference context to bind.
    //! @param stream   CUDA stream for MRope init. If 0 (default), MRope init
    //!                 is skipped — preserves the M2 spike_m2 API.
    //! @param activeBatchSize  Number of concurrent ASR lanes sharing this
    //!                 single context (single-context batched ASR). Default 1
    //!                 keeps existing single-session callers unchanged; the
    //!                 value is forwarded to the audio runner's MRope session
    //!                 init so the cos/sin cache spans all N lanes.
    bool beginAsrSession(
        SpecDecodeInferenceContext& context, cudaStream_t stream = 0, int32_t activeBatchSize = 1);

    /*!
     * @brief End an in-flight streaming-ASR session and release its state.
     *        Mirror of beginAsrSession.
     *
     * Frees the KV-cache slot bound to this context (via
     * HybridCacheManager::resetForNewSequences with zero reuse lengths),
     * clears the accumulated session token-ID list and per-batch effective
     * prefill lengths so a subsequent beginAsrSession on the same context
     * starts from a clean slate.
     *
     * Safe to call without a paired begin (returns true and is a no-op on
     * unused contexts). Safe to call repeatedly. After endAsrSession returns,
     * the context can be reused via beginAsrSession.
     *
     * @param context  Inference context previously passed to beginAsrSession.
     * @param stream   CUDA stream used to issue the cache-length reset H2D.
     * @return True on success.
     */
    bool endAsrSession(SpecDecodeInferenceContext& context, cudaStream_t stream);

    /*!
     * @brief Append one chunk of prefill embeddings to an in-flight streaming
     *        prefill session. Milestone 1 of the streaming-ASR plan
     *        (design doc §12).
     *
     * Preconditions:
     *  - Caller has already run one OneShot prefill (e.g. via handleRequest /
     *    a session-init path) on context, so LoRA, KV-cache state, system-prompt
     *    restore, and recurrent state init are done.
     *  - The engine's KV cache lengths reflect prior chunks; this function does
     *    NOT reset them. Cache start index is derived inside the engine from
     *    live cache lengths (see design doc §10c-real,
     *    cpp/runtime/llmEngineRunner.cpp:1247-1264).
     *  - Single batch only (activeBatchSize == 1) in M1.
     *  - audioEmbedsDelta layout matches the one-shot path: shape
     *    `[totalAudioTokensSoFar, hiddenSize]` cumulative, OR
     *    `[chunkAudioTokens, hiddenSize]` per-chunk — either is supported via
     *    audioIndexBase. Per-chunk + base=cumulative-so-far is the recommended
     *    call pattern: it avoids the caller having to grow a single tensor.
     *
     * @param context           Inference context that has been used for prior
     *                          chunks. tokenIds[0] is extended in place.
     * @param audioEmbedsDelta  Audio embedding rows visible to the kernel for
     *                          this call. With audioIndexBase==N, the kernel
     *                          reads row indices in [N, N + tokenSliceDelta's
     *                          audio-token count). Must be device FP16.
     * @param audioIndexBase    Number of audio tokens consumed by prior chunks
     *                          in this session (cumulative).
     * @param tokenSliceDelta   New token IDs for this chunk (the audio-pad /
     *                          audio-eos / text suffix slice). Appended to
     *                          context.tokenIds[0].
     * @param stream            CUDA stream (must match context.stream).
     * @return True on success, false on prefill failure.
     */
    bool appendPrefillEmbeds(SpecDecodeInferenceContext& context, Tensor const& audioEmbedsDelta,
        int32_t audioIndexBase, std::vector<int32_t> const& tokenSliceDelta, cudaStream_t stream);

    /*!
     * @brief Per-lane streaming-ASR bookkeeping for the batched append-prefill
     *        path. Modeled after PerBatchTalkerState
     *        (qwen3OmniTTSRuntime.h:398-409): one instance per concurrent ASR
     *        session occupying a batch slot/lane. Owned by the product-level
     *        micro-batch scheduler (qwen3-edgellm-jetson worker), NOT by this
     *        runtime; appendPrefillEmbedsBatched only writes appendStatus /
     *        currentKvLen back into it as a convenience. The remaining fields
     *        (emitted cursors, stream channel) are carried for the scheduler's
     *        per-session decode/emit loop and are not consumed by the prefill
     *        call itself.
     *
     * NOTE: In the single-context + activeBatchSize=N model, every lane shares
     * ONE SpecDecodeInferenceContext (per-lane data lives in that context's
     * [N] structures: tokenIds[b], effectivePrefillLengths[b]). The `laneIndex`
     * field identifies which row b of the shared context this session occupies;
     * `context` points at the (shared) context the lane belongs to.
     */
    struct PerBatchAsrState
    {
        int32_t sessionId{-1};                  //!< Caller-assigned stream/session identifier.
        SpecDecodeInferenceContext* context{};  //!< Shared context this lane belongs to (activeBatchSize=N).
        int32_t laneIndex{0};                   //!< Row b of the shared [N] context structures.
        int32_t audioIndexBase{0};              //!< Cumulative audio tokens consumed by prior chunks.
        int32_t chunkLen{0};                    //!< Token slice length staged for the most recent chunk.
        int32_t currentKvLen{0};                //!< KV length snapshot read at entry of the most recent append.
        bool finished{false};                   //!< True once this session's decode reached EOS / final.
        AppendPrefillStatus appendStatus{AppendPrefillStatus::kOk}; //!< Per-lane status of the last append.
        int32_t emittedTokenCursor{0};          //!< Count of decoded tokens already emitted to the caller.
        int32_t emittedTextCursor{0};           //!< Byte/char offset of partial text already emitted.
        std::shared_ptr<StreamChannel> streamChannel; //!< Per-slot streaming sink (StreamChannel is per-slot).
    };

    /*!
     * @brief Batched variant of appendPrefillEmbeds: append one chunk of
     *        audio-bearing prefill embeddings for N concurrent ASR lanes in a
     *        single executePrefillStep call (Phase D-1a, spec §3).
     *
     * SINGLE-CONTEXT MODEL. One SpecDecodeInferenceContext carries
     * activeBatchSize == N lanes; per-lane data lives in that context's [N]
     * structures (context.tokenIds[b], context.effectivePrefillLengths[b]).
     * This is the SAME model the official batched normal-prefill path uses
     * (runBaseModelPrefill, llmInferenceSpecDecodeRuntime.cpp:1183-1224):
     * a single `context.activeBatchSize` drives [N,...] reshapes of the runtime
     * member tensors, and per-lane lengths/ids are read from the context's [N]
     * vectors. We do NOT take N independent contexts — that would mean N copies
     * of activation state and cannot be executed by a single executePrefillStep.
     *
     * Packing contract (mirrors runBaseModelPrefill:1187-1223):
     *  - N = context.activeBatchSize; maxChunkLen = max over lanes of chunk
     *    length (== max_element(context.effectivePrefillLengths), the same
     *    `inputIdsLength` the normal path computes at :1187-1188).
     *  - Token ids staged as [N, maxChunkLen] in pinned host memory, padded
     *    with 0; padding is excluded from the engine's last-token selection by
     *    per-lane context lengths (:1211-1216).
     *  - Embeddings staged as [N, maxChunkLen, H]; logits bound as [N, vocab].
     *  - contextLengths[b] = chunkLen[b] (the per-chunk effective prefill
     *    length, matching the single-lane contract — engine adds the live KV
     *    start index internally, llmEngineRunner.cpp:1247-1264).
     *  - The N per-lane audioEmbedsDeltas are concatenated row-wise into one
     *    flat audio-embeds buffer because the multimodal kernel indexes a
     *    single audioEmbeds tensor globally (embeddingKernels.cu:414). Lane b's
     *    multimodal indices are biased by (concatRowBase[b] + audioIndexBases[b]).
     *
     * The old single-lane appendPrefillEmbeds is retained as an N=1 wrapper
     * (sets context.activeBatchSize=1, builds 1-element vectors, delegates here).
     *
     * @param context           Single shared context with activeBatchSize == N.
     *                           Lane b's accumulated token list is context.tokenIds[b];
     *                           the same context object must be reused across a
     *                           session's chunks (tokenIds[b] extended in place).
     * @param audioEmbedsDeltas N device-FP16 audio-embedding tensors, one per
     *                          lane, [audioRows_b, H]. Concatenated internally.
     *                          size() must equal context.activeBatchSize.
     * @param audioIndexBases   N cumulative audio-token counts (one per lane).
     * @param tokenSliceDeltas  N token-id slices for this chunk (one per lane).
     *                          Appended to context.tokenIds[b].
     * @param stream            CUDA stream (must match the context's stream).
     * @return True if every lane's append succeeded; false on any per-lane
     *         capacity refusal or engine prefill failure (mLastAppendStatus and
     *         mLastObservedKvLength reflect the FIRST failing lane).
     */
    bool appendPrefillEmbedsBatched(SpecDecodeInferenceContext& context,
        std::vector<Tensor const*> const& audioEmbedsDeltas, std::vector<int32_t> const& audioIndexBases,
        std::vector<std::vector<int32_t>> const& tokenSliceDeltas, cudaStream_t stream);

    //! Get LLM prefill stage metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const noexcept
    {
        return mPrefillMetrics;
    }

    //! Get Eagle generation stage metrics (only meaningful when draft model is present)
    metrics::EagleGenerationMetrics const& getEagleGenerationMetrics() const noexcept
    {
        return mEagleGenerationMetrics;
    }

    //! Get vanilla generation stage metrics (only meaningful when no draft model / vanilla path)
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const noexcept
    {
        return mGenerationMetrics;
    }

    //! Get multimodal metrics (returns empty metrics if no multimodal runner)
    metrics::MultimodalMetrics getMultimodalMetrics() const noexcept
    {
        return mVisionRunner ? mVisionRunner->getMultimodalMetrics()
            : mAudioRunner   ? mAudioRunner->getMultimodalMetrics()
                             : metrics::MultimodalMetrics{};
    }

    //! Get the embedding table (for Talker streaming pipeline)
    rt::Tensor const& getEmbeddingTable() const
    {
        return mEmbedding.table;
    }

    //! @brief Get a base model hidden-states buffer for the requested layer index.
    //!
    //! Buffers are owned by the runtime and reused across requests. Layer 0 corresponds to
    //! the post-multimodal input embeddings (backed up before the decode loop reshapes them);
    //! other layer indices correspond to engine-output hidden states (e.g. acceptHiddenLayer
    //! for the Qwen3-Omni Talker, or future MTP layers).
    //!
    //! Lifetime contract:
    //!   - Buffers are sized to {maxRuntimeBatchSize, maxSupportedInputLength, hiddenSize}.
    //!   - Contents are cleared (overwritten) at the start of each handleRequest() call and
    //!     remain valid until the next handleRequest() begins. The buffer is reshaped to
    //!     {activeBatchSize, prefillLength, hiddenSize} for the most recent request — use
    //!     getBaseModelPrefillLength() to query the valid prefill length.
    //!   - The caller is responsible for consuming the data within that window.
    //!
    //! @param layerIdx Layer index. 0 = input embeddings (post-multimodal); other indices are
    //!                 model-specific (e.g. acceptHiddenLayer for Qwen3-Omni Talker).
    //! @return Pointer to the buffer, or nullptr if no buffer is registered for that layer.
    rt::Tensor const* getBaseModelHiddenStates(int32_t layerIdx) const noexcept
    {
        auto it = mHiddenStatesRegistry.find(layerIdx);
        return it != mHiddenStatesRegistry.end() ? it->second : nullptr;
    }

    //! @brief Number of valid prefill tokens in the hidden-states buffers from the most
    //! recent handleRequest() call. Returns 0 if no hidden-states output was requested.
    int32_t getBaseModelPrefillLength() const noexcept
    {
        return mLastPrefillLength;
    }

    //! @brief Per-batch input token IDs from the most recent handleRequest() call.
    //! Cleared at the start of each handleRequest(); valid until the next one begins.
    std::vector<std::vector<int32_t>> const& getBaseModelInputTokenIds() const noexcept
    {
        return mLastInputTokenIds;
    }

    //! @brief Test-only accessor: read-only view of the most recent prefill
    //! logits binding. Used by the M1 acceptance test to bit-exact compare
    //! single-chunk vs split-chunk paths. Not part of the production API —
    //! production callers receive tokens via handleRequest, not raw logits.
    rt::Tensor const& getLogitsForTesting() const noexcept
    {
        return mLogitsOutput;
    }

    //! @brief Test-only accessor: tokenizer pointer used by the runtime. The
    //! M3.6 empirical-LCS spike needs to encode the request prompt prefix /
    //! suffix outside handleRequest to drive chunked prefill manually.
    //! Production callers should hand text to handleRequest instead.
    tokenizer::Tokenizer* getTokenizerForTesting() const noexcept
    {
        return mTokenizer.get();
    }

    //! @brief Test-only accessor: audio runner pointer used by the runtime.
    //! Returns nullptr if no audio runner is loaded. The M3.6 spike calls
    //! encodeMelChunk through this handle (after dynamic_cast to the concrete
    //! Qwen3OmniAudioRunner) so the runtime's MRope-init issued in
    //! beginAsrSession is the one driving per-chunk encoding.
    //! Production callers should not bypass handleRequest.
    MultimodalRunner* getAudioRunnerForTesting() const noexcept
    {
        return mAudioRunner.get();
    }

    //! @brief Test-only post-prefill decode driver for the M3.6 empirical-LCS
    //! spike. Preconditions:
    //!   - The caller has just finished its last appendPrefillEmbeds call for
    //!     this session. mLogitsOutput holds the logits over the LAST prefill
    //!     token (i.e., the next token to sample).
    //!   - context.activeBatchSize == 1, no draft model, no streaming/cancel
    //!     wiring is attached.
    //!
    //! Behavior: greedy-samples the first generated token from mLogitsOutput,
    //! appends it to context.tokenIds[0], then loops runVanillaDecoding until
    //! EOS or until @p maxNewTokens additional generated tokens have been
    //! produced (inclusive of the first sampled token). Returns the list of
    //! GENERATED token IDs (not the prefill tokens) in @p outGeneratedTokens.
    //!
    //! This duplicates the post-prefill loop of handleRequest in a stripped
    //! form (no spec-decode, no streaming, no eviction). It exists so the
    //! M3.6 spike can compare chunked-prefill text quality without touching
    //! the production handleRequest path.
    bool decodeAfterChunkedPrefillForTesting(SpecDecodeInferenceContext& context, int32_t maxNewTokens,
        std::vector<int32_t>& outGeneratedTokens, cudaStream_t stream);

    //! @brief Check if draft model is loaded and spec-decode is available
    bool hasDraftModel() const noexcept
    {
        return mDraftEngineRunner != nullptr;
    }

private:
    //! @brief Common initialization logic shared between all constructors
    //! @param sharedBaseEngine Optional pre-deserialized base engine (ASR slot-pool, D1). When non-null, the
    //! base LLMEngineRunner is built via the shared-engine overload instead of deserializing a new engine.
    //! Only supported in vanilla mode (draftingConfig must be nullopt when sharedBaseEngine is provided).
    void initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<EagleDraftingConfig> const& draftingConfig, cudaStream_t stream,
        std::shared_ptr<nvinfer1::ICudaEngine> sharedBaseEngine = nullptr);

    rt::Tensor mSharedExecContextMemory{};              //!< Shared device memory for all execution contexts
    int32_t mMaxRuntimeBatchSize{1};                    //!< Maximum runtime batch size
    std::optional<EagleDraftingConfig> mDraftingConfig; //!< Eagle drafting configuration (nullopt = no draft)
    LLMEngineRunnerConfig mBaseEngineConfig;            //!< Base engine configuration
    std::optional<EagleDraftEngineRunnerConfig> mDraftEngineConfig; //!< Draft engine configuration (nullopt = no draft)

    std::unique_ptr<LLMEngineRunner> mBaseEngineRunner;            //!< Base model engine runner
    std::unique_ptr<EagleDraftEngineRunner> mDraftEngineRunner;    //!< Draft model engine runner (nullptr = no draft)
    std::unique_ptr<MultimodalRunner> mVisionRunner{nullptr};      //!< Vision multimodal runner (optional)
    std::unique_ptr<MultimodalRunner> mAudioRunner{nullptr};       //!< Audio multimodal runner (optional)
    std::unique_ptr<Alpamayo1ActionRunner> mActionRunner{nullptr}; //!< Action/diffusion head runner (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;              //!< Tokenizer
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCacheBase; //!< System prompt KVCache for base model
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCacheDraft;         //!< System prompt KVCache for draft model
    std::string mEmptyLoraWeightsName{""}; //!< Empty LoRA weights name for default case

    // Pre-define key runtime GPU tensors and initialize them during construction.
    // [1] I/O Tensors to work with base and eagle draft engine.
    EmbeddingData mEmbedding;                 //!< Embedding table [vocabSize, hiddenSize] and optional FP8 scales
    rt::Tensor mIdsInput;                     //!< Input token IDs (used for embedding lookup)
    rt::Tensor mInputsEmbeds;                 //!< Input embeddings (after embedding lookup)
    std::vector<rt::Tensor> mDeepstackEmbeds; //!< Deepstack embeddings for Qwen3-VL (one per feature)
    rt::Tensor mContextLengthsInput;
    rt::Tensor mLogitsOutput;
    rt::Tensor mDraftTreeSize;
    rt::Tensor mDraftTreeMask;
    rt::Tensor mBaseHiddenStatesOutput;
    // Distinguish draft hidden states input and output since we cannot easily
    // Perform inplace update for hidden states between drafting steps.
    rt::Tensor mDraftHiddenStatesInput;
    rt::Tensor mDraftHiddenStatesOutput;

    // [2] Sampling workspace and output tensors that used across all the sampling operations.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mSamplingIndices;
    rt::Tensor mSamplingScores;
    rt::Tensor mBaseVocabMappingTable; // Vocab mapping table for base model reduced vocab (empty if not used)

    // [3] Data structures used during Draft tree constructions (only allocated when draft model present).
    // Data tables that store the data structure that can completely describe a multi-layer draft tree.
    rt::Tensor mDraftTokenIdsFullTable;
    rt::Tensor mDraftTokenScoreFullTable;
    rt::Tensor mDraftTokenPredecessorFullTable;
    // Store conversion table (offset) to map from draft-model vocab token id to the original token id.
    // base_id = draft_id + mapping_table[draft_id]
    rt::Tensor mDraftVocabMappingTable;

    rt::Tensor mDraftTreeRootTokenId;
    rt::Tensor mDraftTokenIdsTable;
    rt::Tensor mDraftTokenScoresTable;
    rt::Tensor mDraftTokenIntermediateScores;
    rt::Tensor mDraftTokenIntermediateParents;

    // [4] Data structures that used during base model verification (only allocated when draft model present).
    rt::Tensor mAcceptedTokenIds;
    rt::Tensor mAcceptedTokenIndices;
    rt::Tensor mAcceptLength;

    // [5] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;

    // [6] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostPackedTokenIds;      //!< Host pinned memory for packed token IDs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
    rt::Tensor mHostAcceptLengths;       //!< Host pinned memory for accept lengths from verification
    rt::Tensor mHostAcceptedTokenIds;    //!< Host pinned memory for accepted token IDs
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths

    //! @brief Pinned host scratch for D2H copy of the live KV cache lengths
    //!        tensor — populated at the entry of appendPrefillEmbeds so the
    //!        capacity check can run synchronously with respect to in-flight
    //!        prefill commits.
    rt::Tensor mHostKvLengthSnapshot{};

    //! @brief Status of the most recent appendPrefillEmbeds call.
    AppendPrefillStatus mLastAppendStatus{AppendPrefillStatus::kOk};

    //! @brief Snapshot of live KV cache length observed by the most recent
    //!        appendPrefillEmbeds call (batch slot 0). 0 if never called.
    int32_t mLastObservedKvLength{0};

    // [7] Multimodal support tensors for audio/image token indexing
    rt::Tensor mMultimodalIndices; //!< Multimodal indices tensor [batchSize, seqLen] for audio/image embeddings

    // [8] Base model hidden states portal (Qwen3-Omni audio generation, future MTP).
    //     Buffers are pre-allocated to {maxBS, maxISL, H} and reshaped per request.
    //     mHiddenStatesRegistry maps layer index → buffer; populated per handleRequest().
    //     See getBaseModelHiddenStates() for the lifetime contract.
    rt::Tensor mOutputHiddenStates{};  //!< Engine-output hidden states (layer N = acceptHiddenLayer)
    rt::Tensor mPrefillEmbedsBackup{}; //!< Layer-0 input embeddings backup (post-multimodal)
    std::unordered_map<int32_t, rt::Tensor const*> mHiddenStatesRegistry; //!< Per-request layer→buffer map
    int32_t mLastPrefillLength{0};                                        //!< Valid prefill length in buffers
    std::vector<std::vector<int32_t>> mLastInputTokenIds;                 //!< Per-batch input token IDs

    //! @brief Restore recurrent/conv states from a cached system prompt.
    void restoreRecurrentStates(int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream);

    //! @brief Zero all recurrent/conv states for a given batch index.
    void zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    // Key functions to drive the spec-decode runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(SpecDecodeInferenceContext& context);

    //! Validate request shape/runtime compatibility.
    bool validateRequestConfig(LLMGenerationRequest const& request);

    //! Prepare per-request runtime state for models built with multimodal support.
    //! Runs multimodal preprocessing when audio or vision inputs are present.
    //! For text-only requests on MRope-based multimodal models, restores text-only RoPE state
    //! and clears stale multimodal request state.
    bool multiModalRuntimePreprocess(
        LLMGenerationRequest const& request, SpecDecodeInferenceContext& context, cudaStream_t stream);

    // Consume the base model hidden states and input token of the sequence. Produce the draft hidden states and logits
    // for the last token of the sequence.
    //! @throws std::runtime_error if tensor shapes do not match, or a CUDA error occurs
    bool runDraftModelPrefill(SpecDecodeInferenceContext& context);

    // Consume the draft hidden states and logits for the last token of the sequence. Produce a speculative draft tree
    // that described by a sequence of draft tokens and tree mask that describe the tree structure.
    //! @throws std::runtime_error if tensor shapes are invalid, or a CUDA operation fails
    bool constructDraftTree(SpecDecodeInferenceContext& context);

    // Consume the speculative draft tree, produce selected tokens and corresponding hidden states.
    //! @throws std::runtime_error if tensor shapes are invalid, or a CUDA operation fails
    bool runBaseModelVerification(SpecDecodeInferenceContext& context);

    // Consume the selected tokens and base model hidden state, produce the draft hidden states and logits for the last
    // token of the accepted sequence.
    //! @throws std::runtime_error if a CUDA operation fails
    bool runDraftModelAcceptToken(SpecDecodeInferenceContext& context);

    // Consume the token sequence & KVCache to produce the next token directly.
    bool runVanillaDecoding(SpecDecodeInferenceContext& context);

    // Consume system prompt, produce the hash table of system prompt KVCache if kv cache reuse is enabled.
    //! @throws std::runtime_error if a CUDA operation fails
    bool genAndSaveSystemPromptKVCache(SpecDecodeInferenceContext& context, int32_t genAndSaveBatchIdx);

    // Consume batched input ids and the hash table of system prompt KVCache, produce the padded input ids and input
    // lengths. Instantiate the KVCache from the hash table if the system prompt has been cached.
    //
    // OneShot variant: full setup for a fresh request (LoRA switch, reuseKVCacheLengths
    // init, system-prompt cache restore, tokenIds (re)seed, cache-manager reset).
    // Use this on session start / single-request handleRequest path.
    //! @throws std::runtime_error if system prompt is malformed
    bool setUpForPrefillExecutionOneShot(SpecDecodeInferenceContext& context);

    // Per-chunk variant for streaming/chunked prefill (M1, streaming ASR).
    // Only performs the BOTH-classified work: NVTX scope, activeBatchSize fetch,
    // engine-max validation against context.effectivePrefillLengths (per-chunk slice
    // length). All one-shot mutations (LoRA, KV-restore, tokenIds.clear,
    // resetForNewSequences) are skipped. Caller must have already set
    // context.effectivePrefillLengths[i] to this chunk's token-slice length, and must
    // not have reset cache lengths between chunks (engine derives kvcache_start_index
    // from live cache state — see design doc §10c-real).
    //! @throws std::runtime_error if a CUDA error occurs
    bool setUpForPrefillExecutionForChunk(SpecDecodeInferenceContext& context);

    // Batch eviction support
    //! @brief Perform batch eviction
    //! @param context Inference context
    //! @return True on success, false on failure
    //! @throws std::runtime_error if a CUDA error occurs
    bool performBatchEvict(SpecDecodeInferenceContext& context);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::EagleGenerationMetrics mEagleGenerationMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics; //!< Vanilla generation metrics (used when no spec-decode)
};

} // namespace rt
} // namespace trt_edgellm
