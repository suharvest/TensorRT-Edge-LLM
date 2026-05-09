/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

    void reset(cudaStream_t stream);

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
