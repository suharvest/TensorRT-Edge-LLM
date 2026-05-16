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
#include "runtime/audioUtils.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct StatefulCode2WavConfig
{
    int32_t numQuantizers{0};
    int32_t upsampleRate{0};
    int32_t sampleRate{24000};
};

//! Generic runner for a stateful Code2Wav TensorRT engine.
//!
//! Expected binding contract:
//! - input:  "codes" [1, num_quantizers, chunk_frames]
//! - output: "waveform" [1, 1, output_samples]
//! - optional input: "position_offset" scalar or [1]
//! - state pairs use "<name>_in" and "<name>_out"
class StatefulCode2WavRunner
{
public:
    StatefulCode2WavRunner(std::string const& engineDir, cudaStream_t stream);
    ~StatefulCode2WavRunner() noexcept = default;

    StatefulCode2WavConfig const& getConfig() const
    {
        return mConfig;
    }

    int64_t getMaxCodeLen() const noexcept
    {
        return mMaxCodeLen;
    }

    void reset(cudaStream_t stream);

    //! \brief Generate the full waveform from RVQ codes using chunked stateful inference.
    //! \param[in] codes RVQ codec codes [numQuantizers][seqLen]
    //! \param[out] outputAudio Output audio waveform data
    //! \param[in] stream CUDA stream for execution
    //! \return True on success, false otherwise
    bool generateWaveform(
        std::vector<std::vector<int32_t>> const& codes, rt::audioUtils::AudioData& outputAudio, cudaStream_t stream);

    bool generateChunk(std::vector<std::vector<int32_t>> const& codes, bool isFinal,
        rt::audioUtils::AudioData& outputAudio, cudaStream_t stream);

private:
    struct StateBinding
    {
        std::string inputName;
        std::string outputName;
        rt::Tensor read;
        rt::Tensor write;
    };

    bool validateAndFillConfig(std::string const& engineDir);
    bool allocateBuffers(cudaStream_t stream);
    bool prepareCodes(std::vector<std::vector<int32_t>> const& codes, cudaStream_t stream);
    bool prepareScalarInputs(bool isFinal, cudaStream_t stream);
    bool infer(cudaStream_t stream);
    void swapStateBuffers();

    static bool endsWith(std::string const& value, std::string const& suffix);
    static std::string replaceSuffix(std::string const& value, std::string const& oldSuffix, std::string const& newSuffix);
    static rt::Coords dimsToCoords(nvinfer1::Dims const& dims);

    StatefulCode2WavConfig mConfig{};
    int64_t mPositionOffset{0};
    int64_t mMaxCodeLen{0};

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    rt::Tensor mInputCodesDevice{};
    rt::Tensor mInputCodesHost{};
    rt::Tensor mOutputWaveformDevice{};
    rt::Tensor mOutputWaveformHost{};
    rt::Tensor mPositionOffsetHost{};
    rt::Tensor mPositionOffsetDevice{};
    rt::Tensor mIsFinalHost{};
    rt::Tensor mIsFinalDevice{};

    std::vector<StateBinding> mStates;
    bool mHasPositionOffset{false};
    bool mHasIsFinal{false};
};

} // namespace rt
} // namespace trt_edgellm
