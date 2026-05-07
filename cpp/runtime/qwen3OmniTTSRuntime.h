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
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
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
constexpr int32_t kQwen3TTSActiveCodePredictorGroups = 15; //!< Residual groups used by native Qwen3-TTS CP engine
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
        CodePredictorBackend codePredictorBackend{CodePredictorBackend::kAuto};
        TextProjectionMode textProjectionMode{TextProjectionMode::kAuto};
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

private:
    class Qwen3TTSCodePredictorEngine;
    class Qwen3TTSTalkerEngine;

    // ========== Internal Methods ==========

    void initializeTTSEmbeddings(cudaStream_t stream);

    bool executeTalkerPrefillStep(
        rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    bool executeTalkerDecodingStep(
        rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    bool runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
        SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes, cudaStream_t stream);

    bool computeResidualConnection(
        std::vector<int32_t> const& codes, rt::Tensor& outputResidual, int32_t frameIdx, cudaStream_t stream);
    bool computeResidualConnectionHost(
        std::vector<int32_t> const& codes, rt::Tensor& outputResidual, int32_t frameIdx, cudaStream_t stream);

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
    rt::Tensor
        mCodecHiddensBuffer; //!< Buffer for codec hiddens [1, 16, talkerHiddenSize] (Talker's space, for residual)
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
    bool projectToTalkerInput(rt::Tensor const& thinkerEmbed, int32_t languageId, rt::Tensor& output,
        int64_t& outputSeqLen, cudaStream_t stream);
    bool projectToTalkerInputHost(rt::Tensor const& thinkerEmbed, int32_t languageId, rt::Tensor& output,
        int64_t& outputSeqLen, cudaStream_t stream);

    //! Embed token IDs, run MLP projection, and reshape buffers ready for Talker prefill.
    //! Populates mTalkerInputEmbeds and mTalkerHiddenStatesBuffer as side effects.
    //! \param[out] outSeqLen  seqLen + 2 (non-streaming prefill length)
    bool prepareTalkerInput(std::vector<int32_t> const& textTokenIds, TalkerGenerationRequest const& request,
        int64_t& outSeqLen, cudaStream_t stream);

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
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

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
};

} // namespace rt
} // namespace trt_edgellm
