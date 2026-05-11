/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/*!
 * @brief TensorRT plugin for dense weight-only W8A16 Linear.
 *
 * Inputs:
 *   0. activation: HALF, [M, K] or [B, S, K]
 *   1. weight: INT8, [K, N]
 *   2. scales: HALF, [N] for scale_mode=0
 *
 * Output:
 *   0. activation @ dequant(weight): HALF, [M, N] or [B, S, N]
 */
class W8A16LinearPlugin : public nvinfer1::IPluginV3,
                          public nvinfer1::IPluginV3OneCore,
                          public nvinfer1::IPluginV3OneBuild,
                          public nvinfer1::IPluginV3OneRuntime
{
public:
    W8A16LinearPlugin(std::string const& name, int32_t N, int32_t K, int32_t scaleMode, int32_t groupSize, int32_t weightLayout);
    W8A16LinearPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    W8A16LinearPlugin() = delete;
    W8A16LinearPlugin(W8A16LinearPlugin const&) = delete;
    ~W8A16LinearPlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;

    int32_t mGemmN{};
    int32_t mGemmK{};
    int32_t mScaleMode{};
    int32_t mGroupSize{};
    int32_t mWeightLayout{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class W8A16LinearPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    W8A16LinearPluginCreator();
    ~W8A16LinearPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm

