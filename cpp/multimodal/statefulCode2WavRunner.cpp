/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "statefulCode2WavRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/mmapReader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

bool StatefulCode2WavRunner::endsWith(std::string const& value, std::string const& suffix)
{
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string StatefulCode2WavRunner::replaceSuffix(
    std::string const& value, std::string const& oldSuffix, std::string const& newSuffix)
{
    return value.substr(0, value.size() - oldSuffix.size()) + newSuffix;
}

rt::Coords StatefulCode2WavRunner::dimsToCoords(nvinfer1::Dims const& dims)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            throw std::runtime_error("Stateful Code2Wav state/output tensor has unresolved dynamic dimension");
        }
        shape.push_back(dims.d[i]);
    }
    return rt::Coords(shape);
}

StatefulCode2WavRunner::StatefulCode2WavRunner(std::string const& engineDir, cudaStream_t stream)
{
    if (!validateAndFillConfig(engineDir))
    {
        throw std::runtime_error("Failed to validate stateful Code2Wav config");
    }

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    std::string const enginePath = engineDir + "/code2wav_stateful.engine";
    if (!std::filesystem::exists(enginePath))
    {
        throw std::runtime_error("Stateful Code2Wav engine not found at " + enginePath);
    }

    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    if (!mEngine)
    {
        throw std::runtime_error("Failed to deserialize stateful Code2Wav engine");
    }
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        throw std::runtime_error("Failed to create stateful Code2Wav execution context");
    }
    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        throw std::runtime_error("Failed to set stateful Code2Wav optimization profile");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!allocateBuffers(stream))
    {
        throw std::runtime_error("Failed to allocate stateful Code2Wav buffers");
    }
    reset(stream);
    LOG_INFO("Stateful Code2Wav runner initialized successfully");
}

bool StatefulCode2WavRunner::validateAndFillConfig(std::string const& engineDir)
{
    std::string const configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open stateful Code2Wav config file: %s", configPath.c_str());
        return false;
    }

    Json jsonConfig;
    try
    {
        jsonConfig = Json::parse(configFileStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse stateful Code2Wav config file: %s", e.what());
        return false;
    }

    Json const& cfg = jsonConfig.contains("code2wav_config") ? jsonConfig["code2wav_config"] : jsonConfig;
    if (!cfg.contains("num_quantizers"))
    {
        LOG_ERROR("num_quantizers not found in stateful Code2Wav config");
        return false;
    }
    mConfig.numQuantizers = cfg["num_quantizers"].get<int32_t>();

    int64_t rate = 1;
    if (cfg.contains("upsample_rates"))
    {
        for (auto const& r : cfg["upsample_rates"])
        {
            rate *= r.get<int64_t>();
        }
    }
    if (cfg.contains("upsampling_ratios"))
    {
        for (auto const& r : cfg["upsampling_ratios"])
        {
            rate *= r.get<int64_t>();
        }
    }
    if (rate <= 1)
    {
        LOG_ERROR("Failed to calculate stateful Code2Wav upsample rate");
        return false;
    }
    mConfig.upsampleRate = math::cast<int32_t>(rate);
    mConfig.sampleRate = cfg.value("sample_rate", 24000);
    return true;
}

bool StatefulCode2WavRunner::allocateBuffers(cudaStream_t stream)
{
    nvinfer1::Dims const codesShapeMax
        = mEngine->getProfileShape(binding_names::kCode2WavCodes, 0, nvinfer1::OptProfileSelector::kMAX);
    int64_t const maxCodeLen = codesShapeMax.d[2];
    int64_t const maxWaveformLen = maxCodeLen * mConfig.upsampleRate;

    mInputCodesDevice
        = rt::Tensor({1, mConfig.numQuantizers, maxCodeLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    mInputCodesHost
        = rt::Tensor({1, mConfig.numQuantizers, maxCodeLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64);
    mOutputWaveformDevice = rt::Tensor({1, 1, maxWaveformLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mOutputWaveformHost = rt::Tensor({1, maxWaveformLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT);

    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        std::string const name = mEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode const mode = mEngine->getTensorIOMode(name.c_str());
        if (mode == nvinfer1::TensorIOMode::kINPUT && name == "position_offset")
        {
            nvinfer1::DataType const dtype = mEngine->getTensorDataType(name.c_str());
            mPositionOffsetHost = rt::Tensor({1}, rt::DeviceType::kCPU, dtype);
            mPositionOffsetDevice = rt::Tensor({1}, rt::DeviceType::kGPU, dtype);
            mHasPositionOffset = true;
        }
        else if (mode == nvinfer1::TensorIOMode::kINPUT && name == "is_final")
        {
            nvinfer1::DataType const dtype = mEngine->getTensorDataType(name.c_str());
            mIsFinalHost = rt::Tensor({1}, rt::DeviceType::kCPU, dtype);
            mIsFinalDevice = rt::Tensor({1}, rt::DeviceType::kGPU, dtype);
            mHasIsFinal = true;
        }
    }

    std::unordered_map<std::string, std::string> outputNames;
    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        std::string const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputNames.emplace(name, name);
        }
    }

    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        std::string const inputName = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(inputName.c_str()) != nvinfer1::TensorIOMode::kINPUT || !endsWith(inputName, "_in"))
        {
            continue;
        }
        std::string const outputName = replaceSuffix(inputName, "_in", "_out");
        if (outputNames.find(outputName) == outputNames.end())
        {
            LOG_WARNING("Ignoring stateful Code2Wav input state without matching output: %s", inputName.c_str());
            continue;
        }

        nvinfer1::Dims stateDims = mEngine->getTensorShape(inputName.c_str());
        if (std::any_of(stateDims.d, stateDims.d + stateDims.nbDims, [](int32_t dim) { return dim < 0; }))
        {
            stateDims = mEngine->getProfileShape(inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        }
        nvinfer1::DataType const dtype = mEngine->getTensorDataType(inputName.c_str());
        StateBinding binding;
        binding.inputName = inputName;
        binding.outputName = outputName;
        binding.read = rt::Tensor(dimsToCoords(stateDims), rt::DeviceType::kGPU, dtype);
        binding.write = rt::Tensor(dimsToCoords(stateDims), rt::DeviceType::kGPU, dtype);
        mStates.push_back(std::move(binding));
    }

    bool ok = true;
    ok &= mContext->setTensorAddress(binding_names::kCode2WavCodes, mInputCodesDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kCode2WavWaveform, mOutputWaveformDevice.rawPointer());
    if (mHasPositionOffset)
    {
        ok &= mContext->setTensorAddress("position_offset", mPositionOffsetDevice.rawPointer());
    }
    if (mHasIsFinal)
    {
        ok &= mContext->setTensorAddress("is_final", mIsFinalDevice.rawPointer());
    }
    for (auto& state : mStates)
    {
        ok &= mContext->setTensorAddress(state.inputName.c_str(), state.read.rawPointer());
        ok &= mContext->setTensorAddress(state.outputName.c_str(), state.write.rawPointer());
    }
    if (!ok)
    {
        LOG_ERROR("Failed to set stateful Code2Wav tensor addresses");
        return false;
    }

    LOG_INFO("Stateful Code2Wav buffers allocated: maxCodeLen=%ld maxWaveformLen=%ld states=%zu", maxCodeLen,
        maxWaveformLen, mStates.size());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

void StatefulCode2WavRunner::reset(cudaStream_t stream)
{
    mPositionOffset = 0;
    for (auto& state : mStates)
    {
        CUDA_CHECK(cudaMemsetAsync(state.read.rawPointer(), 0, state.read.getMemoryCapacity(), stream));
        CUDA_CHECK(cudaMemsetAsync(state.write.rawPointer(), 0, state.write.getMemoryCapacity(), stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool StatefulCode2WavRunner::prepareCodes(std::vector<std::vector<int32_t>> const& codes, cudaStream_t stream)
{
    if (codes.empty() || codes[0].empty())
    {
        LOG_ERROR("Empty stateful Code2Wav codes provided");
        return false;
    }
    int64_t const numLayers = math::cast<int64_t>(codes.size());
    int64_t const seqLen = math::cast<int64_t>(codes[0].size());
    if (numLayers != mConfig.numQuantizers)
    {
        LOG_ERROR("Expected %d quantizer layers, got %ld", mConfig.numQuantizers, numLayers);
        return false;
    }
    for (size_t layer = 1; layer < codes.size(); ++layer)
    {
        if (math::cast<int64_t>(codes[layer].size()) != seqLen)
        {
            LOG_ERROR("Inconsistent stateful Code2Wav code lengths");
            return false;
        }
    }
    if (!mInputCodesDevice.reshape({1, numLayers, seqLen}))
    {
        LOG_ERROR("Failed to reshape stateful Code2Wav input codes");
        return false;
    }
    int64_t* const hostData = static_cast<int64_t*>(mInputCodesHost.rawPointer());
    for (int64_t layer = 0; layer < numLayers; ++layer)
    {
        for (int64_t t = 0; t < seqLen; ++t)
        {
            hostData[layer * seqLen + t] = math::cast<int64_t>(codes[layer][t]);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mInputCodesDevice.rawPointer(), hostData,
        math::cast<size_t>(numLayers * seqLen) * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    return true;
}

bool StatefulCode2WavRunner::prepareScalarInputs(bool isFinal, cudaStream_t stream)
{
    if (mHasPositionOffset)
    {
        nvinfer1::DataType const dtype = mPositionOffsetHost.getDataType();
        if (dtype == nvinfer1::DataType::kINT64)
        {
            *static_cast<int64_t*>(mPositionOffsetHost.rawPointer()) = mPositionOffset;
        }
        else if (dtype == nvinfer1::DataType::kINT32)
        {
            *static_cast<int32_t*>(mPositionOffsetHost.rawPointer()) = math::cast<int32_t>(mPositionOffset);
        }
        else
        {
            LOG_ERROR("Unsupported position_offset dtype for stateful Code2Wav");
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(mPositionOffsetDevice.rawPointer(), mPositionOffsetHost.rawPointer(),
            mPositionOffsetHost.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    }
    if (mHasIsFinal)
    {
        nvinfer1::DataType const dtype = mIsFinalHost.getDataType();
        if (dtype == nvinfer1::DataType::kINT32)
        {
            *static_cast<int32_t*>(mIsFinalHost.rawPointer()) = isFinal ? 1 : 0;
        }
        else if (dtype == nvinfer1::DataType::kBOOL)
        {
            *static_cast<bool*>(mIsFinalHost.rawPointer()) = isFinal;
        }
        else
        {
            LOG_ERROR("Unsupported is_final dtype for stateful Code2Wav");
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(mIsFinalDevice.rawPointer(), mIsFinalHost.rawPointer(), mIsFinalHost.getMemoryCapacity(),
            cudaMemcpyHostToDevice, stream));
    }
    return true;
}

bool StatefulCode2WavRunner::infer(cudaStream_t stream)
{
    if (!mContext->setInputShape(binding_names::kCode2WavCodes, mInputCodesDevice.getTRTDims()))
    {
        LOG_ERROR("Failed to set stateful Code2Wav codes shape");
        return false;
    }
    for (auto& state : mStates)
    {
        if (!mContext->setInputShape(state.inputName.c_str(), state.read.getTRTDims()))
        {
            LOG_ERROR("Failed to set stateful Code2Wav state shape: %s", state.inputName.c_str());
            return false;
        }
    }
    if (mHasPositionOffset && !mContext->setInputShape("position_offset", mPositionOffsetDevice.getTRTDims()))
    {
        LOG_ERROR("Failed to set stateful Code2Wav position_offset shape");
        return false;
    }
    if (mHasIsFinal && !mContext->setInputShape("is_final", mIsFinalDevice.getTRTDims()))
    {
        LOG_ERROR("Failed to set stateful Code2Wav is_final shape");
        return false;
    }
    if (!mContext->enqueueV3(stream))
    {
        LOG_ERROR("Stateful Code2Wav inference failed");
        return false;
    }
    return true;
}

void StatefulCode2WavRunner::swapStateBuffers()
{
    for (auto& state : mStates)
    {
        std::swap(state.read, state.write);
        bool ok = true;
        ok &= mContext->setTensorAddress(state.inputName.c_str(), state.read.rawPointer());
        ok &= mContext->setTensorAddress(state.outputName.c_str(), state.write.rawPointer());
        if (!ok)
        {
            throw std::runtime_error("Failed to rebind stateful Code2Wav state buffers");
        }
    }
}

bool StatefulCode2WavRunner::generateChunk(std::vector<std::vector<int32_t>> const& codes, bool isFinal,
    rt::audioUtils::AudioData& outputAudio, cudaStream_t stream)
{
    if (!prepareCodes(codes, stream) || !prepareScalarInputs(isFinal, stream) || !infer(stream))
    {
        return false;
    }

    nvinfer1::Dims waveformDims = mContext->getTensorShape(binding_names::kCode2WavWaveform);
    int64_t waveformLen = 0;
    if (waveformDims.nbDims >= 3 && waveformDims.d[2] > 0)
    {
        waveformLen = waveformDims.d[2];
    }
    else
    {
        waveformLen = math::cast<int64_t>(codes[0].size()) * mConfig.upsampleRate;
    }
    if (waveformLen < 0 || waveformLen > mOutputWaveformHost.getShape()[1])
    {
        LOG_ERROR("Invalid stateful Code2Wav waveform length: %ld", waveformLen);
        return false;
    }

    outputAudio.waveform
        = std::make_shared<rt::Tensor>(rt::Tensor({1, waveformLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT));
    CUDA_CHECK(cudaMemcpyAsync(outputAudio.waveform->rawPointer(), mOutputWaveformDevice.rawPointer(),
        math::cast<size_t>(waveformLen) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mPositionOffset += math::cast<int64_t>(codes[0].size());
    swapStateBuffers();

    outputAudio.sampleRate = mConfig.sampleRate;
    outputAudio.numChannels = 1;
    outputAudio.hasWaveform = true;
    return true;
}

} // namespace rt
} // namespace trt_edgellm
