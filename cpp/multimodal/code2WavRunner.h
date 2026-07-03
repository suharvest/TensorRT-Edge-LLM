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
#include "common/trtUtils.h"
#include "runtime/audioUtils.h"
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Qwen3-Omni Code2Wav vocoder
//! All values initialized to 0 and must be read from config.json.
//! Do NOT use hardcoded defaults that may not match the actual model.
struct Code2WavConfig
{
    // Model architecture (from config.json)
    int32_t numQuantizers{0}; //!< Number of RVQ quantizer layers
    int32_t codebookSize{0};  //!< Codebook vocabulary size per layer
    int32_t hiddenSize{0};    //!< Code2Wav hidden dimension
    int32_t decoderDim{0};    //!< Decoder base dimension
    int32_t upsampleRate{0};  //!< Total upsampling rate (codes → samples)

    // Audio output parameters
    int32_t sampleRate{24000}; //!< Output audio sample rate (24kHz for Qwen3-Omni)

    // Chunked processing parameters (from builder_config or defaults)
    int32_t chunkSize{300};      //!< Chunk size for long sequences (in codec frames)
    int32_t leftContextSize{25}; //!< Overlap size to avoid boundary artifacts
};

//! \brief Runner for Qwen3-Omni Code2Wav vocoder
//!
//! This class handles Code2Wav vocoder inference, converting RVQ codec codes to audio waveform.
//! It follows the same pattern as AudioRunner and QwenViTRunner for consistency.
class Code2WavRunner
{
public:
    //! \brief Constructor for Code2WavRunner
    //! \param[in] engineDir Directory containing the Code2Wav engine
    //! \param[in] stream CUDA stream for execution
    Code2WavRunner(std::string const& engineDir, cudaStream_t stream);

    //! \brief Construct a Code2WavRunner that SHARES (borrows) an already-deserialized engine.
    //!
    //! Reuses a live, read-only ICudaEngine owned by another Code2WavRunner (e.g. slot 0 of a
    //! TTS slot pool) instead of re-deserializing code2wav.engine, removing the per-slot copy
    //! of the engine weights. The borrowed engine is NOT owned: the owner must outlive this
    //! instance. The execution context, activation memory, and I/O buffers are allocated fresh
    //! per instance, so concurrent slots never contend on mutable state.
    //!
    //! \param[in] borrowedEngine Non-owning pointer to a live engine (from getEnginePtr(); non-null)
    //! \param[in] engineDir Directory with the non-engine assets (config.json) — must match the owner
    //! \param[in] stream CUDA stream for execution
    Code2WavRunner(nvinfer1::ICudaEngine* borrowedEngine, std::string const& engineDir, cudaStream_t stream);

    //! \brief Borrowed (non-owning) engine pointer, for sharing weights across slot-pool instances.
    //! \note Read-only shared weights; callers must NOT delete it and must keep the owner alive.
    nvinfer1::ICudaEngine* getEnginePtr() const noexcept
    {
        return mCode2WavEngine ? mCode2WavEngine.get() : mBorrowedEngine;
    }

    ~Code2WavRunner() noexcept = default;

    //! \brief Generate audio waveform from RVQ codes (single sample)
    //! \param[in] codes RVQ codec codes [numQuantizers][seqLen]
    //! \param[out] outputAudio Output audio waveform data
    //! \param[in] stream CUDA stream for execution
    //! \return True if generation succeeded, false otherwise
    bool generateWaveform(
        std::vector<std::vector<int32_t>> const& codes, rt::audioUtils::AudioData& outputAudio, cudaStream_t stream);

    //! \brief Get Code2Wav configuration
    //! \return Reference to Code2Wav config
    Code2WavConfig const& getConfig() const
    {
        return mConfig;
    }

    //! \brief Get expected waveform length for given code length
    //! \param[in] codeLen Number of codec frames
    //! \return Expected waveform length in samples
    int64_t getExpectedWaveformLength(int64_t codeLen) const
    {
        return codeLen * mConfig.upsampleRate;
    }

private:
    //! \brief Run inference on Code2Wav vocoder
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    bool infer(cudaStream_t stream);

    //! \brief Validate and load configuration from JSON file
    //! \param[in] engineDir Path to engine directory
    //! \return True if configuration is valid and loaded successfully, false otherwise
    bool validateAndFillConfig(std::string const& engineDir);

    //! \brief Allocate buffers for inference
    //! \return True if allocation succeeded, false otherwise
    bool allocateBuffer();

    //! \brief Prepare codes for inference (reshape and copy to device)
    //! \param[in] codes Input RVQ codes [numQuantizers][seqLen]
    //! \param[in] stream CUDA stream
    //! \return True on success, false otherwise
    bool prepareCodes(std::vector<std::vector<int32_t>> const& codes, cudaStream_t stream);

    //! \brief Run chunked inference for long sequences
    //! \param[in] codes Input RVQ codes [numQuantizers][seqLen]
    //! \param[out] outputWaveform Output waveform tensor
    //! \param[in] stream CUDA stream
    //! \return True on success, false otherwise
    bool runChunkedInference(
        std::vector<std::vector<int32_t>> const& codes, rt::Tensor& outputWaveform, cudaStream_t stream);

    //! Resolve the active engine pointer (owned mCode2WavEngine, or borrowed mBorrowedEngine).
    nvinfer1::ICudaEngine* enginePtr() const noexcept
    {
        return mCode2WavEngine ? mCode2WavEngine.get() : mBorrowedEngine;
    }

    Code2WavConfig mConfig{};                                      //!< Code2Wav vocoder configuration
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;                  //!< TensorRT runtime (null in borrowed mode)
    std::unique_ptr<nvinfer1::ICudaEngine> mCode2WavEngine;        //!< Code2Wav TensorRT engine (null in borrowed mode)
    nvinfer1::ICudaEngine* mBorrowedEngine{nullptr};               //!< Non-owning engine (borrowed/shared mode)
    std::unique_ptr<nvinfer1::IExecutionContext> mCode2WavContext; //!< Per-instance execution context (never shared)
    rt::Tensor mInputCodesDevice{};                                //!< [1, numQuantizers, seqLen] Input codes on GPU
    rt::Tensor mOutputWaveform{};                                  //!< [1, 1, waveformLen] Output waveform
    nvinfer1::DataType mWaveformDtype{nvinfer1::DataType::kHALF};  //!< Engine's actual waveform output dtype
    rt::Tensor mInputCodesHost{};                                  //!< Host buffer for code preparation
};

} // namespace rt
} // namespace trt_edgellm
