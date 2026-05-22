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

#include "common/tensor.h"
#include "profiling/metrics.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <NvInferRuntime.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{

// Forward declaration
struct SamplingParams;

namespace rt
{

// ========== Constants ==========

namespace talker_constants
{
constexpr int32_t kNumRvqLayers = 15;      //!< Number of RVQ codebook layers (fixed architecture)
constexpr int32_t kAssistantPrefixLen = 3; //!< Assistant prefix tokens ([:3])
constexpr int32_t kAssistantTrailingSuffix
    = 5; //!< Trailing tokens to strip from end of sequence ("<|im_end|>\n<|im_start|>assistant\n")
constexpr int32_t kNonStreamingPrefixRows = 8;     //!< Fixed prefix rows in non-streaming prefill (rows 0-7)
constexpr int32_t kCodePredictorPrefillSeqLen = 2; //!< CodePredictor prefill sequence length
constexpr int32_t kCodecEmbeddingCount = 6;        //!< Number of codec embeddings to add
constexpr int32_t kQwen3TTSMinActiveCodePredictorGroups = 12; //!< Quality floor for high-performance CP pruning
constexpr int32_t kQwen3TTSActiveCodePredictorGroups = 15;    //!< Residual groups used by native Qwen3-TTS CP engine
} // namespace talker_constants

/*!
 * @brief Talker runtime for Qwen3-Omni RVQ code generation
 *
 * LLM-based codec encoder that generates RVQ codes from text tokens and hidden states.
 * Manages two LLM engines (Talker + CodePredictor) and MLP projection layers.
 *
 * Pipeline:
 *   1. MLP Projection: thinker embed (layer 0) → talker embeddings via text_projection
 *   2. Talker LLM: generate codec tokens autoregressively
 *   3. CodePredictor: generate 15-layer codebook codes
 *   4. Return RVQ codes (vocoding done separately at example layer)
 *
 * Architecture Philosophy:
 *   - Talker is an LLM decoder, NOT a multimodal input encoder
 *   - Similar to LLMInferenceRuntime, manages multiple LLM engines
 *   - Standalone runtime, not dependent on MultimodalRunner hierarchy
 *   - Code2Wav vocoding is separated for better modularity
 */
class Qwen3OmniTTSRuntime
{
public:
    using FrameCallback = std::function<void(std::vector<int32_t> const& frameCodes, int32_t totalFrames)>;

    enum class TalkerBackend
    {
        kAuto,
        kGeneric,
        kQwen3TTSExplicitKV,
    };

    enum class CodePredictorBackend
    {
        kAuto,
        kGeneric,
        kQwen3TTSNative,
    };

    enum class TextProjectionMode
    {
        kAuto,
        kDevice,
        kHostFP32,
    };

    struct RuntimeOptions
    {
        TalkerBackend talkerBackend{TalkerBackend::kAuto};
        std::string qwen3TtsTalkerEnginePath;
        CodePredictorBackend codePredictorBackend{CodePredictorBackend::kAuto};
        TextProjectionMode textProjectionMode{TextProjectionMode::kAuto};
        bool qwen3TtsPromptKvCache{false};
    };

    /*!
     * @brief Construct and fully initialize the TTS runtime
     * @param talkerEngineDir Directory containing talker engine, MLP weights, embedding table, etc.
     * @param codePredictorEngineDir Directory containing code_predictor engine and codec embeddings
     * @param tokenizerDir Directory containing tokenizer files. If empty, defaults to talkerEngineDir/../
     * @param stream CUDA stream for operations
     * @throws std::runtime_error on any initialization failure
     */
    Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& tokenizerDir, cudaStream_t stream);

    Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& tokenizerDir, cudaStream_t stream, RuntimeOptions const& options);

    //! @brief Destructor
    ~Qwen3OmniTTSRuntime();

    // ========== Core API ==========

    /*!
     * @brief Talker audio generation request structure
     *
     * Contains sampling parameters and input data for audio generation.
     * Sampling parameters are provided per-request (not from config.json).
     */
    struct TalkerGenerationRequest
    {
        int32_t maxAudioLength{4096}; //!< Maximum number of audio codec tokens to generate

        // Talker/CodePredictor sampling parameters (independent from Thinker)
        // 0 = use PyTorch defaults: temperature=0.9, top_k=50, top_p=1.0
        float talkerTemperature{0};     //!< Talker temperature (0 = default 0.9)
        int32_t talkerTopK{0};          //!< Talker top-K (0 = default 50)
        float talkerTopP{0};            //!< Talker top-P (0 = default 1.0)
        float repetitionPenalty{1.05f}; //!< Repetition penalty applied to seen codec tokens (1.0 = disabled)
        float codecEosLogitOffset{0};   //!< Added to codec EOS logit before EOS bias onset
        float predictorTemperature{0};  //!< CodePredictor temperature (0 = talker temperature)
        int32_t predictorTopK{0};        //!< CodePredictor top-K (0 = talker top-K)
        float predictorTopP{0};          //!< CodePredictor top-P (0 = talker top-P)

        // Speaker selection (optional, defaults to config default)
        std::string speakerName{""}; //!< Speaker name (e.g., "f245", "m02") - empty means use default
        std::string language{""};    //!< Language hint (e.g., "chinese", "english")
        int32_t speakerId{-1};       //!< Speaker ID - if >= 0, overrides speakerName
        std::vector<float> speakerEmbedding; //!< Optional raw x-vector embedding [talkerHiddenSize]

        // Input: conversation messages for this request (runtime tokenizes internally)
        std::vector<Message> messages;
        bool applyChatTemplate{true};   //!< Whether to apply chat template formatting
        bool addGenerationPrompt{true}; //!< Whether to add generation prompt at the end
        bool enableThinking{false};     //!< Whether to enable thinking mode
    };

    /*!
     * @brief Talker audio generation response structure
     *
     * Contains generated RVQ codes and metadata.
     */
    struct TalkerGenerationResponse
    {
        // RVQ codes: [numFrames][15 layers]
        std::vector<std::vector<int32_t>> rvqCodes;

        // Metadata
        int32_t numFrames{0}; //!< Number of audio frames generated
        bool success{false};  //!< Whether generation succeeded
    };

    /*!
     * @brief Get required hidden state layer indices from thinker
     * @return Vector containing {0} for layer 0 (embed)
     */
    std::vector<int32_t> getThinkerHiddenLayerIndices() const
    {
        return {0};
    }

    /*!
     * @brief Generate audio with RVQ codes
     *
     * @param request Request containing sampling parameters and input data
     * @param response Response containing generated RVQ codes
     * @param stream CUDA stream for execution
     * @return True if generation succeeded, false otherwise
     */
    bool handleAudioGeneration(TalkerGenerationRequest const& request, TalkerGenerationResponse& response,
        cudaStream_t stream, FrameCallback const& frameCallback = {});

    /*!
     * @brief Get performance metrics for Talker pipeline
     * @return Reference to metrics object
     */
    metrics::MultimodalMetrics const& getMetrics() const
    {
        return mMultimodalMetrics;
    }

    /*!
     * @brief Capture CUDA graphs for decoding steps (same pattern as LLMInferenceRuntime).
     * @param stream CUDA stream for capture
     * @return True if all graphs captured successfully
     */
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*!
     * @brief Get speaker ID by name
     * @param speakerName Speaker name (e.g., "f245", "m02")
     * @return Speaker ID, or default speaker ID if not found
     */
    int32_t getSpeakerIdByName(std::string const& speakerName) const;

    /*!
     * @brief [Phase 2 hook] Create a fresh execution context bound to the loaded Talker engine.
     *
     * Returned context shares engine weights but has independent CUDA state and
     * optimization-profile selection. Phase 3 will pair this with a per-slot CUDA stream
     * and KV cache to enable concurrent N>1 inference. Phase 2 ships the hook only; the
     * default per-instance contexts continue to serve all current call sites.
     *
     * NOTE: Talker uses a single context that switches between prefill (profile 0) and
     * decode (profile 1) at runtime via setOptimizationProfileAsync — a single factory
     * ctx is sufficient as long as Phase 3 also switches profiles per call. This is
     * unlike the CP engine which requires a paired (prefill, decode) ctx set (see
     * @ref createCodePredictorExecutionContextPair).
     *
     * @return Owning pointer to a fresh execution context, or nullptr if the Talker engine
     *         is not loaded (e.g. wrong backend selected).
     */
    std::unique_ptr<nvinfer1::IExecutionContext> createTalkerExecutionContext();

    /*!
     * @brief [Phase 2 must-fix 3] Create a paired (prefill, decode) execution context set
     *        bound to the loaded native Qwen3-TTS CodePredictor engine.
     *
     * CP generate-path uses BOTH a prefill ctx (profile 0, 2-token warmup) and a decode ctx
     * (profile 1, single-token decode for residual groups) per frame. A single factory
     * context cannot service both shapes without per-call profile churn, so we expose the
     * pair directly. Phase 3 will install one such pair per request slot.
     *
     * @return std::pair{prefill_ctx, decode_ctx}. Both nullptr if the native CP engine
     *         is not enabled.
     */
    std::pair<std::unique_ptr<nvinfer1::IExecutionContext>, std::unique_ptr<nvinfer1::IExecutionContext>>
    createCodePredictorExecutionContextPair();

    // ===================================================================
    // [Phase 3a] Per-slot state ownership
    //
    // TalkerSlot / CodePredictorSlot wrap every piece of per-request mutable
    // state previously held on the engine sub-classes (KV double-buffers, IO
    // scratch tensors, sampling RNG, attention mask, prompt KV cache, etc.).
    // The slot owns its own CUDA stream (borrowed reference) and execution
    // context(s). The engine globals stay in place as the default path —
    // every engine method now accepts an optional Slot*; when nullptr it
    // falls back to engine members and behavior is byte-identical to Phase 2
    // (N=1 default path).
    //
    // Phase 3b (out of scope here) will introduce a C++ Scheduler that holds
    // a pool of these slots, one per concurrent request, and dispatches into
    // the engine methods with non-null Slot* arguments.
    //
    // Storage is held in rt::Tensor (UINT8 byte buffer with size baked in)
    // rather than the engine's private nested DeviceBuffer so that the slot
    // types remain fully visible in the header without leaking engine
    // internals.
    // ===================================================================
    struct TalkerSlot
    {
        cudaStream_t stream{nullptr};                       //!< borrowed
        // Contexts are owned by the unique_ptrs below; callers should
        // dereference those (via .get()) rather than caching a raw pointer.
        // Codex round-3 review (Phase 3a) explicitly removed the public raw
        // pointer mirrors to avoid silent dangling when the slot is moved
        // or destroyed.
        std::unique_ptr<nvinfer1::IExecutionContext> prefillCtxOwned;
        std::unique_ptr<nvinfer1::IExecutionContext> decodeCtxOwned;
        // Per-call IO scratch (byte buffers; sizes match the engine's mDevice* members)
        rt::Tensor deviceEmbeds;
        rt::Tensor deviceLogits;
        rt::Tensor deviceHidden;
        rt::Tensor devicePositionIds;
        rt::Tensor deviceAttentionMask;
        // KV double-buffers (2 * numDecoderLayers entries each)
        std::vector<rt::Tensor> kvA;
        std::vector<rt::Tensor> kvB;
        // Prompt-KV cache (optional, lazy-allocated on first hit/store)
        std::vector<rt::Tensor> promptKVs;
        rt::Tensor promptLogits;
        rt::Tensor promptHidden;
        size_t promptKVBytes{0};
        size_t promptHiddenBytes{0};
        bool promptCacheValid{false};
        int32_t promptCacheLen{0};
        uint64_t promptCacheKey{0};
        // Decode-loop state
        int32_t seqLen{0};
        int32_t parity{0};
    };

    struct CodePredictorSlot
    {
        cudaStream_t stream{nullptr};                      //!< borrowed
        // Contexts are owned by the unique_ptrs below; callers should
        // dereference those rather than caching a raw pointer (see
        // TalkerSlot comment for the rationale).
        std::unique_ptr<nvinfer1::IExecutionContext> prefillCtxOwned;
        std::unique_ptr<nvinfer1::IExecutionContext> decodeCtxOwned;
        // Per-call IO scratch
        rt::Tensor deviceEmbeds;
        rt::Tensor deviceLogits;
        rt::Tensor deviceCachePosition;
        rt::Tensor deviceSelectedTokens;
        rt::Tensor deviceGenStep;
        rt::Tensor devicePastLength;
        rt::Tensor deviceSamplingWorkspace; //!< only sized when GPU sampling enabled on parent engine
        // KV double-buffers (2 * numLayers each)
        std::vector<rt::Tensor> kvA;
        std::vector<rt::Tensor> kvB;
        // Host-side sampling scratch (mirrors engine's mSample* members)
        std::vector<float> sampleLogits;
        std::vector<uint16_t> sampleRaw;
        std::vector<std::pair<float, int32_t>> sampleVals;
        std::vector<double> sampleProbs;
        // Full std::mt19937 state (not just a seed). The engine mutates the
        // mt19937 in-place during sampling (cpp ~:1484, ~:1525), so a 64-bit
        // seed is NOT a drop-in replacement — codex round-3 review caught
        // this gap. Initialize via the slot factory using the same seed
        // policy the engine uses today.
        std::mt19937 rng;
        uint64_t gpuSamplingOffset{0};//!< Philox counter for GPU top-k/top-p path
    };

    //! [Phase 3a] Allocate a fully-sized Talker slot bound to the supplied
    //! stream. Stream is borrowed (owner outlives the slot). Returns nullptr
    //! if the explicit-KV Talker engine is not loaded.
    std::unique_ptr<TalkerSlot> createTalkerSlot(cudaStream_t stream);

    //! [Phase 3a] Allocate a fully-sized CodePredictor slot bound to the
    //! supplied stream. Stream is borrowed. Returns nullptr if the native
    //! Qwen3-TTS CP engine is not enabled.
    std::unique_ptr<CodePredictorSlot> createCodePredictorSlot(cudaStream_t stream);

private:
    class Qwen3TTSCodePredictorEngine;
    class Qwen3TTSTalkerEngine;

    // ===================================================================
    // [Phase B C2/C3/C4] Per-request local state for handleAudioGeneration.
    //
    // Bundles every scratch tensor / counter / host buffer that was
    // previously held as a runtime-global member but is only meaningful
    // within a single audio generation request. Moving these to a stack
    // local inside `handleAudioGeneration` removes the cross-request
    // races flagged in docs/specs/tts-n2-shared-tensor-audit.md §1 and
    // permits the worker to drop the global runtime mutex.
    //
    // The corresponding `m*` declarations are intentionally retained
    // below because `captureDecodingCUDAGraph` (init-time path) still
    // refers to them; those legacy paths are gated by
    // `!mQwen3TTSTalkerEngine` / `!mUseQwen3TTSCodePredictorEngine` and
    // are not exercised in the N=2 production configuration.
    // ===================================================================
    struct TalkerLocal
    {
        rt::Tensor thinkerEmbedBuffer;
        rt::Tensor gpuTokenIdsBuffer;
        rt::Tensor mlpWorkspace;
        rt::Tensor projectedBuffer;
        rt::Tensor talkerInputEmbeds;
        rt::Tensor speakerEmbedding;
        rt::Tensor talkerLogits;
        rt::Tensor talkerSelectedIndices;
        rt::Tensor seenCodecTokensBuf;
        rt::Tensor talkerHiddenStatesBuffer;
        rt::Tensor talkerLastHidden;
        rt::Tensor residualEmbedBuffer;
        int32_t trailingTextLen{0};
        std::vector<float> hostProjectedBuffer;
    };

    struct CodePredictorLocal
    {
        rt::Tensor codePredictorPrefillInput;
        rt::Tensor codePredictorCodecIds;
        rt::Tensor codePredictorCodecEmbed;
        rt::Tensor rawCodecEmbed;
        rt::Tensor smallToMtpProjectedHidden;
        // [Phase B C2 fix] These were originally kept runtime-global on the
        // theory that they were legacy-CP only. In fact the native CP frame
        // loop (runCodePredictorGenerationForFrame / executeCodePredictorDecodingStep)
        // writes them on every decode step — so they race at N=2 and corrupt
        // CUDA state, surfacing as cudaMemsetAsync(state.read) errors downstream
        // in Code2Wav.
        rt::Tensor codePredictorLogits;
        std::vector<rt::Tensor> codePredictorLogitsPerHead;
        rt::Tensor codePredictorHiddenStatesBuffer;
        rt::Tensor codePredictorSelectedIndices;
    };

    // ========== Internal Methods ==========

    void initializeTTSEmbeddings(cudaStream_t stream);

    bool executeTalkerPrefillStep(
        rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    bool executeTalkerDecodingStep(
        rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    bool runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
        SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes,
        rt::Tensor& codecHiddensBuffer, TalkerLocal& tlocal, CodePredictorLocal& cplocal, cudaStream_t stream);

    bool computeResidualConnection(std::vector<int32_t> const& codes, rt::Tensor& outputResidual, int32_t frameIdx,
        rt::Tensor const& codecHiddensBuffer, TalkerLocal& tlocal, cudaStream_t stream);
    bool computeResidualConnectionHost(
        std::vector<int32_t> const& codes, rt::Tensor& outputResidual, int32_t frameIdx,
        TalkerLocal& tlocal, cudaStream_t stream);

    bool extractTalkerLastHidden(
        rt::Tensor const& talkerHiddenStates, rt::Tensor& outputLastHidden, cudaStream_t stream);

    // ========== Configuration Structure ==========

    /*!
     * @brief Talker configuration parameters
     */
    struct TalkerConfig
    {
        // Model dimensions (read from config, not hardcoded)
        int32_t thinkerHiddenSize{};       //!< Thinker hidden dimension (read from config)
        int32_t talkerHiddenSize{};        //!< Talker hidden dimension (read from config)
        int32_t talkerVocabSize{};         //!< Talker vocabulary size (read from config)
        int32_t codePredictorHiddenSize{}; //!< CodePredictor hidden dimension (read from CodePredictor config)
        int32_t codebookSize{};            //!< Codebook vocabulary size per layer (read from config or hardcoded)
        int32_t maxSeqLen{};               //!< Maximum input sequence length from thinker (read from config)

        // TTS special tokens (from thinker vocab, projected through text_projection)
        int32_t ttsPadTokenId{}; //!< TTS pad token (151671)
        int32_t ttsBosTokenId{}; //!< TTS begin-of-sequence (151672)
        int32_t ttsEosTokenId{}; //!< TTS end-of-sequence (151673)

        // Codec special tokens (from talker vocab, used directly)
        int32_t codecNothinkId{};  //!< Codec no-think control token (2155)
        int32_t codecThinkId{};    //!< Codec think control token (2154)
        int32_t codecThinkBosId{}; //!< Codec think begin-of-sequence (2156)
        int32_t codecThinkEosId{}; //!< Codec think end-of-sequence (2157)
        int32_t codecPadId{};      //!< Codec padding token (2148)
        int32_t codecBosId{};      //!< Codec begin-of-sequence (2149)
        int32_t codecEosId{};      //!< Codec end-of-sequence

        // Speaker configuration (read from config)
        int32_t defaultSpeakerId{}; //!< Default speaker ID (e.g., 2301 for f245)
        std::unordered_map<std::string, int32_t> languageIdMap;
    };

    // ========== Configuration and Initialization ==========

    /*!
     * @brief Validate and fill configuration from talker config file
     * @param talkerEngineDir Directory containing talker engine files
     * @return True on success, false on failure
     */
    bool validateAndFillConfig(std::string const& talkerEngineDir);

    /*!
     * @brief Initialize Talker and CodePredictor engine runners
     * @param talkerEngineDir Directory containing talker engine files
     * @param codePredictorEngineDir Directory containing code predictor engine files
     * @return True on success, false on failure
     */
    bool initializeEngineRunners(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir);
    bool loadCodePredictorConfig(std::string const& codePredictorEngineDir);

    /*!
     * @brief Load CodePredictor lm_head weights and small_to_mtp_projection
     * @param codePredictorEngineDir Directory containing code predictor engine files
     * @return True on success, false on failure
     */
    bool loadCodePredictorWeights(std::string const& codePredictorEngineDir);

    /*!
     * @brief Allocate device buffers for Talker pipeline
     * @return True on success, false on failure
     */
    bool allocateBuffer();

    TalkerConfig mTalkerConfig{};                           //!< Talker configuration
    std::unordered_map<std::string, int32_t> mSpeakerIdMap; //!< Speaker name to ID mapping

    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;      //!< Tokenizer for text-to-token-ID conversion
    RuntimeOptions mRuntimeOptions{};
    std::unique_ptr<LLMEngineRunner> mTalkerLLMRunner;     //!< Talker LLM engine runner
    std::unique_ptr<LLMEngineRunner> mCodePredictorRunner; //!< CodePredictor engine runner
    std::unique_ptr<Qwen3TTSCodePredictorEngine>
        mQwen3TTSCodePredictorEngine; //!< Optional Qwen3-TTS native CodePredictor engine
    bool mUseQwen3TTSCodePredictorEngine{false}; //!< Whether the Qwen3-TTS native CodePredictor engine is enabled
    std::filesystem::path mQwen3TTSCodePredictorEnginePath;
    std::unique_ptr<Qwen3TTSTalkerEngine> mQwen3TTSTalkerEngine; //!< Explicit-KV Qwen3-TTS Talker engine
    bool mUseHostTextProjection{false};

    LLMEngineRunnerConfig mTalkerLLMConfig;     //!< Talker LLM configuration
    LLMEngineRunnerConfig mCodePredictorConfig; //!< CodePredictor configuration

    //! Shared GPU execution context memory for Talker and CodePredictor (kUSER_MANAGED).
    rt::Tensor mSharedExecContextMemory;

    // cuBLAS handle removed — GEMM is now via CuTe DSL compiled kernels (CuteDslGemmRunner).

    // Projects from thinker (embedding) space to talker input space
    rt::Tensor mTextFC1Weight; //!< FC1 weight [2048, 2048] FP16 column-major
    rt::Tensor mTextFC1Bias;   //!< FC1 bias [2048] FP16
    rt::Tensor mTextFC2Weight; //!< FC2 weight [2048, 2048] FP16 column-major
    rt::Tensor mTextFC2Bias;   //!< FC2 bias [2048] FP16

    // Projects from Talker space (2048) to CodePredictor space (1024)
    rt::Tensor mSmallToMtpWeight; //!< Linear weight [1024, 2048] FP16
    rt::Tensor mSmallToMtpBias;   //!< Linear bias [1024] FP16

    // ========== Embedding Tables ==========
    rt::Tensor mTextEmbeddingTable; //!< Text embedding table [thinkerVocabSize, thinkerHiddenSize] (for standalone TTS)
    rt::Tensor mTextEmbeddingScale; //!< Optional per-group dequant scales [vocab, hidden/128] when text embedding is FP8
    bool mTextEmbeddingHasScale{false}; //!< True when mTextEmbeddingScale is populated (FP8 text embedding)
    std::vector<int32_t>
        mTextTokenIdToPrunedRow; //!< Optional original thinker token ID -> pruned text embedding row mapping
    bool mUsePrunedTextEmbedding{false};
    rt::Tensor mTalkerEmbeddingTable; //!< Talker LLM embedding table [vocabSize, hiddenSize]
    std::vector<rt::Tensor>
        mCodePredictorEmbeddingTables; //!< CodePredictor embedding tables (15 layers) [codebookSize, hiddenSize]

    // CodePredictor LM Heads (bound as input tensors via setLMHeadWeights)
    std::vector<rt::Tensor>
        mCodePredictorLmHeadWeights; //!< CodePredictor lm_head weights (15 layers) [vocabSize, hiddenSize]

    // TTS special token embeddings (initialized from thinker embedding table)
    // Initialized in constructor from Thinker embedding table
    rt::Tensor mTtsPadEmbed; //!< TTS pad embedding [talkerHiddenSize] FP16
    rt::Tensor mTtsBosEmbed; //!< TTS bos embedding [talkerHiddenSize] FP16
    rt::Tensor mTtsEosEmbed; //!< TTS eos embedding [talkerHiddenSize] FP16

    int32_t mTrailingTextLen{0}; //!< Number of text tokens used as residual addends after prefill

    // Workspace tensors
    rt::Tensor mThinkerEmbedBuffer; //!< Pre-allocated text embedding output [maxSeqLen, thinkerHiddenSize] FP16
    rt::Tensor mGpuTokenIdsBuffer;  //!< Pre-allocated token IDs upload buffer [1, maxSeqLen] INT32
    rt::Tensor mMLPWorkspace;       //!< Workspace for MLP intermediate results [maxTokens, 2048] FP16
    rt::Tensor mProjectedBuffer;    //!< Buffer for projected tokens [maxTokens, 1024] FP16
    rt::Tensor mTalkerInputEmbeds;  //!< Final talker input embeddings [seqLen, 1024] FP16
    rt::Tensor mSpeakerEmbedding;   //!< Optional raw speaker embedding [talkerHiddenSize] FP16
    rt::Tensor mSamplingWorkspace;  //!< Workspace for sampling operations

    // Talker LLM workspace
    rt::Tensor mTalkerLogits;            //!< Talker LLM output logits FP32 [1, vocabSize]
    rt::Tensor mTalkerSelectedIndices;   //!< Selected token indices [1, 1]
    rt::Tensor mHostSelectedTokenIds;    //!< Host tensor for selected tokens [1]
    rt::Tensor mHostTalkerContextLength; //!< Host tensor for context length [1]
    rt::Tensor mSeenCodecTokensBuf;      //!< GPU buffer of previously sampled codec tokens [maxAudioLength] INT32

    // CodePredictor workspace

    rt::Tensor mCodePredictorLogits; //!< CodePredictor output logits FP32 [1, codebookSize]
    //! Workaround for LLMEngineRunner's cudaGraph capture limitation. Will be replaced with a better design.
    std::vector<rt::Tensor> mCodePredictorLogitsPerHead;
    bool mCodePredictorGraphsCaptured{false}; //!< Whether CodePredictor CUDA graphs were captured
    rt::Tensor mCodePredictorSelectedIndices; //!< Selected code indices [1, 1]
    rt::Tensor mCodePredictorPrefillInput;    //!< Prefill input buffer [1, 2, codePredictorHiddenSize]
    rt::Tensor mCodePredictorCodecIds;        //!< Codec token IDs buffer [1, 1]
    rt::Tensor
        mCodePredictorCodecEmbed; //!< Projected codec embed [1, 1, codePredictorHiddenSize] (CodePredictor input)
    rt::Tensor mRawCodecEmbed;    //!< Raw codec embed [1, 1, talkerHiddenSize] (before small_to_mtp_projection)
    rt::Tensor
        mSmallToMtpProjectedHidden;  //!< Projected talker hidden [1, codePredictorHiddenSize] (for prefill slot 0)
    rt::Tensor mResidualEmbedBuffer; //!< Residual embedding buffer [1, 1, talkerHiddenSize] (feeds Talker decoder)
    rt::Tensor mHostSelectedCodeIds; //!< Host tensor for selected codes [1]
    rt::Tensor mHostCodePredictorContextLength; //!< Host tensor for CodePredictor context length [1]

    // Talker decoding step buffers
    rt::Tensor mTalkerDecodingIds;   //!< Talker decoding token IDs [1, 1]
    rt::Tensor mTalkerDecodingEmbed; //!< Talker decoding embedding [1, 1, hiddenSize]

    // KVCache reset helper (used in both Talker and CodePredictor prefill)
    rt::Tensor mHostReuseKVCacheLengths; //!< Host tensor for KVCache reset [1]

    // Generation loop workspace
    rt::Tensor mTalkerHiddenStatesBuffer;        //!< Buffer for Talker hidden states (all layers)
    rt::Tensor mCodePredictorHiddenStatesBuffer; //!< Buffer for CodePredictor hidden states (all layers)
    rt::Tensor mTalkerLastHidden; //!< Buffer for extracted Talker last hidden state [1, talkerHiddenSize]
    nvinfer1::DataType mTalkerInputEmbedsDataType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mResidualEmbedDataType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mTalkerHiddenStatesDataType{nvinfer1::DataType::kHALF};
    // [Phase B C1] mCodecHiddensBuffer was moved out of the runtime-global state to
    // per-request scope to avoid cross-request `cudaMemsetAsync` races at N>1. The
    // buffer is now a local in `handleAudioGeneration` (passed by reference into
    // CP/residual helpers). C2 will fold it back into `TalkerSlot::codecHiddensBuffer`
    // once full slot-plumbing lands.
    std::vector<float> mHostTalkerEmbeddingTable;
    std::vector<float> mHostCodePredictorEmbeddingTables;
    std::vector<float> mHostTextFC1Weight;
    std::vector<float> mHostTextFC1Bias;
    std::vector<float> mHostTextFC2Weight;
    std::vector<float> mHostTextFC2Bias;
    std::vector<float> mHostProjectedBuffer;
    std::vector<float> mHostTtsPadEmbed;
    std::vector<float> mHostTtsBosEmbed;
    std::vector<float> mHostTtsEosEmbed;

    cudaStream_t mStream{nullptr};                 //!< CUDA stream for operations
    metrics::MultimodalMetrics mMultimodalMetrics; //!< Performance metrics for Talker pipeline

    /*!
     * @brief Perform MLP projection from thinker embed to talker input space (non-streaming)
     *
     * Builds the Qwen3-TTS Talker prefill buffer. The exported model expects a
     * fixed 9-row prefill: the first raw text token is in the prefill and the
     * remaining text tokens are injected during residual feedback.
     *
     * @param thinkerEmbed Embedded token sequence [seqLen, thinkerHiddenSize]
     * @param languageId Codec language token ID
     * @param output Projected talker input embeddings [9, talkerHiddenSize]
     * @param outputSeqLen fixed prefill length
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool projectToTalkerInput(rt::Tensor const& thinkerEmbed, int32_t languageId, int32_t speakerId,
        std::vector<float> const& speakerEmbedding, rt::Tensor& output, int64_t& outputSeqLen,
        TalkerLocal& tlocal, cudaStream_t stream);
    bool projectToTalkerInputHost(rt::Tensor const& thinkerEmbed, int32_t languageId, int32_t speakerId,
        std::vector<float> const& speakerEmbedding, rt::Tensor& output, int64_t& outputSeqLen,
        TalkerLocal& tlocal, cudaStream_t stream);

    //! Embed token IDs, run MLP projection, and reshape buffers ready for Talker prefill.
    //! Populates tlocal.talkerInputEmbeds and tlocal.talkerHiddenStatesBuffer as side effects.
    //! \param[out] outSeqLen  seqLen + 2 (non-streaming prefill length)
    bool prepareTalkerInput(std::vector<int32_t> const& textTokenIds, TalkerGenerationRequest const& request,
        int64_t& outSeqLen, TalkerLocal& tlocal, cudaStream_t stream);

    /*!
     * @brief Execute CodePredictor prefill step using CUDA Graph
     *
     * Performs prefill inference for one codebook layer using pre-captured CUDA Graph.
     * The graph already has the correct lm_head bound.
     *
     * @param codecTokenEmbeds Codec token embeddings [1, 2, hiddenSize] — concat([past_hidden, embed(code_0)])
     * @param generationStep Which lm_head/graph to use (0-14)
     * @param outputLogits Output logits [1, seqLen, codebookSize] (engine output)
     * @param outputHiddenStates Output hidden states for residual connection
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool executeCodePredictorPrefillStep(rt::Tensor const& codecTokenEmbeds, int32_t generationStep,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*!
     * @brief Execute CodePredictor decoding step using CUDA Graph
     *
     * Performs single-step decoding for one codebook layer using pre-captured CUDA Graph.
     * The graph already has the correct lm_head bound.
     *
     * @param tokenId Current code token ID
     * @param embeddingTableIndex Which embedding table to use (0-14)
     * @param generationStep Which lm_head/graph to use (0-14)
     * @param outputLogits Output logits [1, 1, codebookSize] (engine output)
     * @param outputHiddenStates Output hidden states for next residual connection
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool executeCodePredictorDecodingStep(int32_t tokenId, int32_t embeddingTableIndex, int32_t generationStep,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, rt::Tensor& codecHiddensBuffer,
        CodePredictorLocal& cplocal, cudaStream_t stream);

    /*!
     * @brief Load Talker weights from safetensors files
     *
     * Loads text_projection MLP weights, text embedding table, and Talker embedding table.
     *
     * @param weightsDir Directory containing weight files
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream);
    bool loadTextTokenMap(std::filesystem::path const& weightsDir);
    int32_t mapTextTokenId(int32_t tokenId, char const* context) const;
    std::vector<int32_t> mapTextTokenIds(std::vector<int32_t> const& tokenIds, char const* context) const;
};

} // namespace rt
} // namespace trt_edgellm
