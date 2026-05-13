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

#include "multimodalRunner.h"
#include "runtime/audioUtils.h"
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Qwen3-Omni audio encoder
struct AudioConfig
{
    int32_t melBins{128};          //!< Number of mel-frequency bins
    int32_t audioFeatureDim{2560}; //!< Audio feature dimension (output embedding size)
    int32_t nWindow{100};          //!< Window size for audio chunking
    int32_t nWindowInfer{100};     //!< Inference window size

    // Audio special tokens (from tokenizer_config.json)
    int32_t audioTokenId{151675};    //!< <|audio_pad|> token ID
    int32_t audioBosTokenId{151669}; //!< <|audio_start|> token ID
    int32_t audioEosTokenId{151670}; //!< <|audio_end|> token ID
    float mropeTheta{0.0F};          //!< Multi-dimensional RoPE theta (0 = no MRope)
    int32_t mropeSectionH{20};       //!< MRoPE section: frequency pairs for height (default Qwen3-Omni)
    int32_t mropeSectionW{20};       //!< MRoPE section: frequency pairs for width (default Qwen3-Omni)
};

//! \brief Runner for Qwen3-Omni audio encoder
//!
//! This class handles audio preprocessing and encoder inference for Qwen3-Omni model.
class Qwen3OmniAudioRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for Qwen3OmniAudioRunner
    //! \param[in] engineDir Directory containing the audio encoder engine
    //! \param[in] stream CUDA stream for execution
    //! \throws std::runtime_error if engine loading fails or configuration is invalid
    Qwen3OmniAudioRunner(std::string const& engineDir, cudaStream_t stream);

    ~Qwen3OmniAudioRunner() noexcept = default;

    //! \brief Preprocess multimodal input including audio and text
    //! \param[in] request LLM generation request containing audio and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[in,out] ropeRotaryCosSinDevice RoPE rotary position encoding cache
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream,
        bool imageOnly = false) override;

    //! \brief Run inference on the audio encoder
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    bool infer(cudaStream_t stream) override;

    //! \brief Validate and load configuration from JSON file
    //! \param[in] engineDir Path to engine directory
    //! \return True if configuration is valid and loaded successfully, false otherwise
    bool validateAndFillConfig(std::string const& engineDir) override;

    //! \brief Allocate buffers for inference
    //! \param[in] stream CUDA stream for execution
    //! \return True if allocation succeeded, false otherwise
    bool allocateBuffer(cudaStream_t stream) override;

    //! \brief Get audio embeddings from encoder output
    //! \return Reference to audio embedding tensor
    rt::Tensor& getOutputEmbedding() override;

    //! \brief Initialize sequential MRope cache for system prompt KVCache saving
    //! \details For audio-only MRope models (e.g. Qwen3-ASR), initialize sequential MRope cache
    //!          since no vision runner will fill it. When a vision runner is present, this is a no-op
    //!          because QwenViTRunner::preprocessSystemPrompt handles MRope initialization.
    //! \param[in] systemPrompt System prompt text
    //! \param[in] tokenizer Tokenizer instance
    //! \param[in,out] ropeRotaryCosSinDevice RoPE cache tensor
    //! \param[in] stream CUDA stream
    //! \return True on success, false on failure
    bool preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer const* tokenizer,
        rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

    //! \brief Encode a single mel-spectrogram chunk to audio embeddings (streaming entry point).
    //! \details Thin public wrapper over the per-audio encoder forward path. Does NOT touch
    //!          text tokens or MRope state — callers (streaming worker / runtime) drive those
    //!          separately. The one-shot `preprocess` path is preserved bit-exact and unchanged.
    //! \param[in]  mel          AudioData describing the mel chunk (must carry
    //!                          `melSpectrogramPath`/`melSpectrogramFormat`; PCM is M4 scope).
    //! \param[out] outEmbedding Caller-allocated GPU half-precision tensor. Reshaped on
    //!                          success to [audio_tokens_this_chunk, audioFeatureDim] and
    //!                          populated. Caller must size the underlying allocation big
    //!                          enough — the function only reshapes within it.
    //! \param[in]  stream       CUDA stream for execution.
    //! \return True on success, false otherwise.
    bool encodeMelChunk(
        rt::audioUtils::AudioData const& mel, rt::Tensor& outEmbedding, cudaStream_t stream);

    //! \brief Initialize MRope cos/sin cache once per streaming ASR session.
    //! \details Session-scoped variant of `initializeSequentialMRopeCache`. Sized once at
    //!          session start so per-chunk `encodeMelChunk` calls do not need to touch the
    //!          MRope cache. Streaming batch is always 1 (single sequence per ASR session).
    //!          Position-id layout matches the one-shot path (T=H=W sequential).
    //! \param[in]      maxAudioTokens          Upper bound on cumulative audio tokens for this
    //!                                         session (informational/bound; the underlying
    //!                                         cache spans the full `maxPositionEmbeddings`
    //!                                         of the rope tensor, same as the one-shot path).
    //! \param[in,out]  ropeRotaryCosSinDevice  RoPE cache tensor (e.g.
    //!                                         `LLMEngineRunner::getRopeCosSinCacheTensor()`).
    //! \param[in]      stream                  CUDA stream.
    //! \return True on success, false otherwise.
    bool initializeMRopeForSession(
        int32_t maxAudioTokens, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

private:
    //! \brief Preprocess audio buffers and run encoder inference
    //! \param[in] audioBuffers Input audio data with mel-spectrogram paths or waveforms
    //! \param[out] audioTokenLengths Output token lengths for each audio clip
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing and inference succeeded, false otherwise
    bool preprocessAudio(std::vector<rt::audioUtils::AudioData> const& audioBuffers,
        std::vector<int64_t>& audioTokenLengths, cudaStream_t stream);

    //! \brief Encode a single mel chunk to `mAudioEmbedding` via the encoder TRT context.
    //! \details Shared body for both `preprocessAudio` (one-shot, looped per buffer) and
    //!          `encodeMelChunk` (streaming, single buffer). Does only the forward pass —
    //!          no text/MRope side effects. On success, `mAudioEmbedding` is reshaped to
    //!          [totalAudioTokens, audioFeatureDim] and `outTokens` carries that count.
    //! \param[in]  audio      Single audio data (mel-spectrogram path).
    //! \param[out] outTokens  Number of audio tokens produced for this chunk.
    //! \param[in]  stream     CUDA stream.
    //! \return True on success, false otherwise.
    bool encodeOneAudioImpl(
        rt::audioUtils::AudioData const& audio, int64_t& outTokens, cudaStream_t stream);

    //! \brief Tokenize text and insert audio tokens
    //! \param[in] request LLM generation request
    //! \param[out] batchInputIds Batched input IDs after tokenization and audio token insertion
    //! \param[in] audioTokenLengths Token lengths for each audio clip
    //! \param[in] tokenizer Tokenizer for text encoding
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& audioTokenLengths, tokenizer::Tokenizer const* tokenizer);

    //! \brief Load pre-computed mel-spectrogram from file
    //! \param[in] filePath Path to .npy or .raw file
    //! \param[in] format File format: "npy" or "raw"
    //! \param[out] melSpectrogram Output tensor [1, mel_bins, time_steps]
    //! \param[in] stream CUDA stream for execution
    //! \return True on success, false otherwise
    bool loadMelSpectrogramFromFile(
        std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream);

    //! \brief Initialize MRope cos/sin cache with sequential position IDs (T=H=W=[0,1,2,...])
    //! \details Used for audio+text only models (e.g. Qwen3-ASR) where all 3 MRope dimensions
    //!          use identical sequential positions. Skipped when mConfig.mropeTheta == 0.
    //! \param[in] activeBatchSize Number of active sequences in the batch
    //! \param[in,out] ropeRotaryCosSinDevice RoPE cache tensor to fill
    //! \param[in] stream CUDA stream for execution
    //! \return True if initialization succeeded or was skipped, false on failure
    bool initializeSequentialMRopeCache(
        int64_t activeBatchSize, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream);

    AudioConfig mConfig{};                                      //!< Audio encoder configuration
    std::unique_ptr<nvinfer1::ICudaEngine> mAudioEngine;        //!< Audio encoder TensorRT engine
    std::unique_ptr<nvinfer1::IExecutionContext> mAudioContext; //!< Audio encoder execution context
    rt::Tensor mMelSpectrogram{};                               //!< [batch, mel_bins, time] Mel-spectrogram input
    rt::Tensor mPaddedFeature{};      //!< [num_chunks, mel_bins, max_chunk_len] Padded audio chunks
    rt::Tensor mPaddedMaskAfterCNN{}; //!< [num_chunks, max_len_after_cnn] Mask for valid tokens
    rt::Tensor mPaddedMaskIndices{};  //!< [num_valid_elements, 2] Nonzero indices from mask
    rt::Tensor mAudioAttentionMask{}; //!< [num_attention_elems, num_attention_elems] Block-diagonal attention mask
    rt::Tensor mAfterCNNLens{};       //!< [batch_size] Length of each audio after CNN (CPU tensor)
    rt::Tensor mAudioEmbedding{};     //!< [num_audio_tokens, hidden_dim] Audio encoder output
};

} // namespace rt
} // namespace trt_edgellm