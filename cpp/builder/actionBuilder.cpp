/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "actionBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

ActionBuilder::ActionBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, ActionBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool ActionBuilder::build()
{
    auto pluginHandles = loadEdgellmPluginLib();

    if (!parseConfig())
    {
        LOG_ERROR("Failed to parse action expert config from %s", mOnnxDir.string().c_str());
        return false;
    }

    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    std::string onnxPath = (mOnnxDir / "model.onnx").string();
    auto parser = parseOnnxModel(builder.get(), network, onnxPath);
    if (!parser)
    {
        LOG_ERROR("Failed to parse ONNX model from %s", onnxPath.c_str());
        return false;
    }

    LOG_DEBUG("%s", printNetworkInfo(network.get(), "Action").c_str());

    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        return false;
    }

    if (!setupActionOptimizationProfile(*builder.get(), *config.get(), *network.get()))
    {
        LOG_ERROR("Failed to setup action optimization profile");
        return false;
    }

    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving Action engine.", mEngineDir.string().c_str());
    }

    std::string engineFilePath = (mEngineDir / "action.engine").string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        LOG_ERROR("Failed to build and serialize engine to %s", engineFilePath.c_str());
        return false;
    }

    if (!copyConfig())
    {
        LOG_ERROR("Failed to copy config to engine directory");
        return false;
    }

    return true;
}

bool ActionBuilder::parseConfig()
{
    std::string configPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(configPath, mModelConfig))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    return true;
}

bool ActionBuilder::setupActionOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* profile = builder.createOptimizationProfile();

    bool setShapeValuesStatus{true};
    setShapeValuesStatus
        &= profile->setShapeValues("seq_length", nvinfer1::OptProfileSelector::kMIN, &mBuilderConfig.minSeqLen, 1);
    setShapeValuesStatus
        &= profile->setShapeValues("seq_length", nvinfer1::OptProfileSelector::kOPT, &mBuilderConfig.optSeqLen, 1);
    setShapeValuesStatus
        &= profile->setShapeValues("seq_length", nvinfer1::OptProfileSelector::kMAX, &mBuilderConfig.maxSeqLen, 1);
    if (!setShapeValuesStatus)
    {
        LOG_ERROR("Failed to set optimization profile for input \"seq_length\"");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(profile, "action_profile", &network).c_str());
    config.addOptimizationProfile(profile);

    return true;
}

bool ActionBuilder::copyConfig()
{
    if (!saveConfigWithBuilderInfo(mEngineDir, mModelConfig, mBuilderConfig.toJson()))
    {
        LOG_ERROR("Failed to save config to engine directory");
        return false;
    }
    return true;
}

} // namespace builder
} // namespace trt_edgellm
