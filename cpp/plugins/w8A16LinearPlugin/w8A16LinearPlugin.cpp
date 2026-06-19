/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "w8A16LinearPlugin.h"
#include "kernels/w8A16LinearKernels/w8A16Linear.h"

#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cuda_fp16.h>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kW8A16_LINEAR_PLUGIN_VERSION{"1"};
constexpr char const* kW8A16_LINEAR_PLUGIN_NAME{"W8A16LinearPlugin"};

int32_t getLastDim(Dims const& dims)
{
    return dims.nbDims > 0 ? dims.d[dims.nbDims - 1] : -1;
}

bool hasSupportedActivationRank(Dims const& dims)
{
    return dims.nbDims == 2 || dims.nbDims == 3;
}

struct W8A16ProfileStats
{
    uint64_t calls{0};
    double totalMs{0.0};
    float minMs{std::numeric_limits<float>::max()};
    float maxMs{0.0F};
};

std::mutex& w8A16ProfileMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::map<std::tuple<int32_t, int32_t, int32_t>, W8A16ProfileStats>& w8A16ProfileStatsMap()
{
    static std::map<std::tuple<int32_t, int32_t, int32_t>, W8A16ProfileStats> stats;
    return stats;
}

bool w8A16ProfileEnabled()
{
    static bool enabled = [] {
        char const* value = std::getenv("EDGE_LLM_W8A16_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void printW8A16ProfileSummary()
{
    if (!w8A16ProfileEnabled())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(w8A16ProfileMutex());
    auto const& stats = w8A16ProfileStatsMap();
    if (stats.empty())
    {
        return;
    }
    std::cerr << "[W8A16_PROFILE] summary shapes=" << stats.size() << std::endl;
    for (auto const& item : stats)
    {
        auto const& key = item.first;
        auto const& value = item.second;
        double avg = value.calls == 0 ? 0.0 : value.totalMs / static_cast<double>(value.calls);
        std::cerr << "[W8A16_PROFILE] M=" << std::get<0>(key) << " K=" << std::get<1>(key)
                  << " N=" << std::get<2>(key) << " calls=" << value.calls << " avg_ms=" << avg
                  << " total_ms=" << value.totalMs << " min_ms=" << value.minMs << " max_ms=" << value.maxMs
                  << std::endl;
    }
}

void registerW8A16ProfilePrinter()
{
    static bool registered = [] {
        if (w8A16ProfileEnabled())
        {
            std::atexit(printW8A16ProfileSummary);
        }
        return true;
    }();
    (void) registered;
}

void updateW8A16Profile(int32_t M, int32_t K, int32_t N, float elapsedMs)
{
    bool shouldPrint = false;
    {
        std::lock_guard<std::mutex> lock(w8A16ProfileMutex());
        auto& stats = w8A16ProfileStatsMap()[std::make_tuple(M, K, N)];
        stats.calls += 1;
        stats.totalMs += static_cast<double>(elapsedMs);
        stats.minMs = std::min(stats.minMs, elapsedMs);
        stats.maxMs = std::max(stats.maxMs, elapsedMs);

        uint64_t totalCalls = 0;
        for (auto const& item : w8A16ProfileStatsMap())
        {
            totalCalls += item.second.calls;
        }
        shouldPrint = (totalCalls % 1000U) == 0U;
    }
    if (shouldPrint)
    {
        printW8A16ProfileSummary();
    }
}

} // namespace

PluginFieldCollection W8A16LinearPluginCreator::mFieldCollection{};
std::vector<PluginField> W8A16LinearPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(W8A16LinearPluginCreator);

W8A16LinearPlugin::W8A16LinearPlugin(
    std::string const& name, int32_t N, int32_t K, int32_t scaleMode, int32_t groupSize, int32_t weightLayout)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mScaleMode(scaleMode)
    , mGroupSize(groupSize)
    , mWeightLayout(weightLayout)
{
}

W8A16LinearPlugin::W8A16LinearPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "gemm_n")
        {
            mGemmN = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "scale_mode")
        {
            mScaleMode = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "group_size")
        {
            mGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "weight_layout")
        {
            mWeightLayout = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
}

W8A16LinearPlugin::~W8A16LinearPlugin() {}

IPluginCapability* W8A16LinearPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
}

IPluginV3* W8A16LinearPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new W8A16LinearPlugin(mLayerName, mGemmN, mGemmK, mScaleMode, mGroupSize, mWeightLayout);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
}

char const* W8A16LinearPlugin::getPluginName() const noexcept
{
    return kW8A16_LINEAR_PLUGIN_NAME;
}

char const* W8A16LinearPlugin::getPluginVersion() const noexcept
{
    return kW8A16_LINEAR_PLUGIN_VERSION;
}

char const* W8A16LinearPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void W8A16LinearPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t W8A16LinearPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t W8A16LinearPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == 1);
        assert(nbInputs == 3);
        outputTypes[0] = inputTypes[0];
        return 0;
    }
    catch (std::exception const&)
    {
        return -1;
    }
}

int32_t W8A16LinearPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbInputs == 3);
        assert(nbOutputs == 1);
        outputs[0].nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < inputs[0].nbDims; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mGemmN);
        return 0;
    }
    catch (std::exception const&)
    {
        return -1;
    }
}

bool W8A16LinearPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == 3 && nbOutputs == 1);
        assert(pos < nbInputs + nbOutputs);
        auto const& desc = inOut[pos].desc;

        switch (pos)
        {
        case 0:
            return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR
                && hasSupportedActivationRank(desc.dims) && getLastDim(desc.dims) == mGemmK;
        case 1:
            return desc.type == DataType::kINT8 && desc.format == PluginFormat::kLINEAR && desc.dims.nbDims == 2
                && ((mWeightLayout == 1 && desc.dims.d[0] == mGemmN && desc.dims.d[1] == mGemmK)
                    || (mWeightLayout != 1 && desc.dims.d[0] == mGemmK && desc.dims.d[1] == mGemmN));
        case 2:
            return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR && desc.dims.nbDims == 1
                && desc.dims.d[0] == mGemmN;
        case 3:
            return desc.type == inOut[0].desc.type && desc.format == PluginFormat::kLINEAR
                && desc.dims.nbDims == inOut[0].desc.dims.nbDims && getLastDim(desc.dims) == mGemmN;
        default: return false;
        }
    }
    catch (std::exception const&)
    {
        return false;
    }
}

int32_t W8A16LinearPlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

size_t W8A16LinearPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t W8A16LinearPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        auto const& inputDims = inputDesc[0].dims;
        int32_t M = 1;
        for (int32_t i = 0; i < inputDims.nbDims - 1; ++i)
        {
            M *= inputDims.d[i];
        }

        auto const* inputPtr = reinterpret_cast<half const*>(inputs[0]);
        auto const* weightPtr = reinterpret_cast<int8_t const*>(inputs[1]);
        auto const* scalesPtr = reinterpret_cast<half const*>(inputs[2]);
        auto* outputPtr = reinterpret_cast<half*>(outputs[0]);

        if (w8A16ProfileEnabled())
        {
            registerW8A16ProfilePrinter();
            cudaEvent_t start{};
            cudaEvent_t stop{};
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, stream);
            trt_edgellm::kernel::w8a16_linear_forward(inputPtr, weightPtr, scalesPtr, outputPtr, M, mGemmN, mGemmK,
                static_cast<trt_edgellm::kernel::W8A16ScaleMode>(mScaleMode), mGroupSize, mWeightLayout, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float elapsedMs{0.0F};
            cudaEventElapsedTime(&elapsedMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            updateW8A16Profile(M, mGemmK, mGemmN, elapsedMs);
        }
        else
        {
            trt_edgellm::kernel::w8a16_linear_forward(inputPtr, weightPtr, scalesPtr, outputPtr, M, mGemmN, mGemmK,
                static_cast<trt_edgellm::kernel::W8A16ScaleMode>(mScaleMode), mGroupSize, mWeightLayout, stream);
        }
        return 0;
    }
    catch (std::exception const&)
    {
        return -1;
    }
}

int32_t W8A16LinearPlugin::onShapeChange(
    PluginTensorDesc const* /* in */, int32_t /* nbInputs */, PluginTensorDesc const* /* out */, int32_t /* nbOutputs */)
    noexcept
{
    return 0;
}

IPluginV3* W8A16LinearPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* W8A16LinearPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("scale_mode", &mScaleMode, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("group_size", &mGroupSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("weight_layout", &mWeightLayout, PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

W8A16LinearPluginCreator::W8A16LinearPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scale_mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight_layout", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* W8A16LinearPluginCreator::getPluginName() const noexcept
{
    return kW8A16_LINEAR_PLUGIN_NAME;
}

char const* W8A16LinearPluginCreator::getPluginVersion() const noexcept
{
    return kW8A16_LINEAR_PLUGIN_VERSION;
}

PluginFieldCollection const* W8A16LinearPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* W8A16LinearPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void W8A16LinearPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

IPluginV3* W8A16LinearPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new W8A16LinearPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm

