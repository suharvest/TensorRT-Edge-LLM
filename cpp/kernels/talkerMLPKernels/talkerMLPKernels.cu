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

#include "talkerMLPKernels.h"

#ifdef CUTE_DSL_GEMM_ENABLED
#include "cuteDslGemmRunner.h"
#endif

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

class CudaDeviceGuard
{
public:
    explicit CudaDeviceGuard(void const* devicePtr)
    {
        check::check(devicePtr != nullptr, "cuBLAS fallback device pointer must not be null");

        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, devicePtr));
        check::check(attributes.type == cudaMemoryTypeDevice, "cuBLAS fallback tensors must be CUDA device memory");
        mDevice = attributes.device;
        check::check(mDevice >= 0, "Failed to determine cuBLAS fallback tensor device");

        CUDA_CHECK(cudaGetDevice(&mPreviousDevice));
        if (mPreviousDevice != mDevice)
        {
            CUDA_CHECK(cudaSetDevice(mDevice));
            mRestore = true;
        }
    }

    ~CudaDeviceGuard()
    {
        if (mRestore)
        {
            auto const status = cudaSetDevice(mPreviousDevice);
            if (status != cudaSuccess)
            {
                LOG_WARNING("Failed to restore CUDA device %d: %s", mPreviousDevice, cudaGetErrorString(status));
            }
        }
    }

    CudaDeviceGuard(CudaDeviceGuard const&) = delete;
    CudaDeviceGuard& operator=(CudaDeviceGuard const&) = delete;
    CudaDeviceGuard(CudaDeviceGuard&&) = delete;
    CudaDeviceGuard& operator=(CudaDeviceGuard&&) = delete;

private:
    int32_t mPreviousDevice{-1};
    int32_t mDevice{-1};
    bool mRestore{false};
};

class CublasHandle
{
public:
    CublasHandle(cudaStream_t stream, void const* devicePtr)
        : mDeviceGuard(devicePtr)
    {
        auto status = cublasCreate(&mHandle);
        check::check(status == CUBLAS_STATUS_SUCCESS, "Failed to create cuBLAS handle");

        // The caller must pass a stream that belongs to the same device as the tensors.
        status = cublasSetStream(mHandle, stream);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            static_cast<void>(cublasDestroy(mHandle));
            mHandle = nullptr;
        }
        check::check(status == CUBLAS_STATUS_SUCCESS, "Failed to set cuBLAS stream");
    }

    ~CublasHandle()
    {
        if (mHandle != nullptr)
        {
            auto const status = cublasDestroy(mHandle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                LOG_WARNING("Failed to destroy cuBLAS handle: %d", static_cast<int32_t>(status));
            }
        }
    }

    CublasHandle(CublasHandle const&) = delete;
    CublasHandle& operator=(CublasHandle const&) = delete;
    CublasHandle(CublasHandle&&) = delete;
    CublasHandle& operator=(CublasHandle&&) = delete;

    cublasHandle_t get() const
    {
        return mHandle;
    }

private:
    CudaDeviceGuard mDeviceGuard;
    cublasHandle_t mHandle{nullptr};
};

//! \brief SiLU activation for FP16
//! \param x Input value
//! \return silu(x) = x / (1 + exp(-x))
__device__ __forceinline__ half silu(half x)
{
    float fx = __half2float(x);
    return __float2half(fx / (1.0f + __expf(-fx)));
}

//! \brief Fused bias addition and SiLU activation kernel (vectorized)
//!
//! Each block processes one token, threads within block handle different dimensions.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//! Requires hiddenDim to be a multiple of 8 and pointers to be 16-byte aligned.
//!
//! \param[in,out] data Token data with shape [numTokens, hiddenDim] (FP16)
//! \param[in] bias Bias vector with shape [hiddenDim] (FP16)
//! \param[in] numTokens Number of tokens
//! \param[in] hiddenDim Hidden dimension size (must be multiple of 8)
__global__ void biasAndSiLUKernelVectorized(
    half* __restrict__ data, half const* __restrict__ bias, int64_t numTokens, int64_t hiddenDim)
{
    constexpr int32_t kVEC_SIZE = 8; // sizeof(uint4) / sizeof(half)

    // Each block handles one token
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    half* tokenData = data + tokenIdx * hiddenDim;

    using vec_t = uint4;
    int64_t const numVecs = hiddenDim / kVEC_SIZE;

    for (int64_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(tokenData)[i];
        vec_t biasVec = reinterpret_cast<vec_t const*>(bias)[i];

        half* dataPtr = reinterpret_cast<half*>(&dataVec);
        half const* biasPtr = reinterpret_cast<half const*>(&biasVec);

#pragma unroll
        for (int32_t j = 0; j < kVEC_SIZE; ++j)
        {
            dataPtr[j] = silu(__hadd(dataPtr[j], biasPtr[j]));
        }

        reinterpret_cast<vec_t*>(tokenData)[i] = dataVec;
    }
}

//! \brief Vectorized bias addition kernel using half2
//!
//! Uses 2D grid to avoid expensive modulo operation.
//! blockIdx.x handles dimension chunks, blockIdx.y handles tokens.
//!
//! \param[in,out] data Token data with shape [numTokens, hiddenDim] (FP16)
//! \param[in] bias Bias vector with shape [hiddenDim] (FP16)
//! \param[in] numTokens Number of tokens
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void addBiasKernelVectorized(
    half* __restrict__ data, half const* __restrict__ bias, int32_t numTokens, int32_t hiddenDim)
{
    using vec_t = uint4;

    int32_t const tokenIdx = blockIdx.y;
    int32_t const vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    if (tokenIdx >= numTokens || vecIdx >= numVecs)
    {
        return;
    }

    half* tokenData = data + static_cast<int64_t>(tokenIdx) * hiddenDim;

    vec_t dataVec = reinterpret_cast<vec_t const*>(tokenData)[vecIdx];
    vec_t biasVec = reinterpret_cast<vec_t const*>(bias)[vecIdx];

    half2* dataPtr = reinterpret_cast<half2*>(&dataVec);
    half2 const* biasPtr = reinterpret_cast<half2 const*>(&biasVec);

#pragma unroll
    for (int32_t i = 0; i < VEC_SIZE / 2; ++i)
    {
        dataPtr[i] = __hadd2(dataPtr[i], biasPtr[i]);
    }

    reinterpret_cast<vec_t*>(tokenData)[vecIdx] = dataVec;
}

//! \brief Vectorized gather kernel
//!
//! Each block handles one output row, threads cooperate to copy hiddenDim elements.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//!
//! \param[in] source Source tensor [srcNumTokens, hiddenDim] (FP16)
//! \param[in] indices Indices to gather [numIndices] (INT32)
//! \param[out] output Output tensor [numIndices, hiddenDim] (FP16)
//! \param[in] numIndices Number of rows to gather
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void gatherKernelVectorized(half const* __restrict__ source, int32_t const* __restrict__ indices,
    half* __restrict__ output, int32_t numIndices, int32_t hiddenDim)
{
    // Each block handles one output row
    int32_t const outIdx = blockIdx.x;
    if (outIdx >= numIndices)
    {
        return;
    }

    // Get source row index
    int32_t const srcIdx = indices[outIdx];

    half const* srcRow = source + static_cast<int64_t>(srcIdx) * hiddenDim;
    half* dstRow = output + static_cast<int64_t>(outIdx) * hiddenDim;

    // Vectorized processing (8 FP16 = 16 bytes = uint4)
    using vec_t = uint4;

    int32_t const numVecs = hiddenDim / VEC_SIZE;

    // Vectorized copy
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(srcRow)[i];
        reinterpret_cast<vec_t*>(dstRow)[i] = dataVec;
    }

    // Handle remainder
    int32_t const remainderStart = numVecs * VEC_SIZE;
    for (int32_t i = remainderStart + threadIdx.x; i < hiddenDim; i += blockDim.x)
    {
        dstRow[i] = srcRow[i];
    }
}

//! \brief Vectorized scatter kernel
//!
//! Each block handles one source row, threads cooperate to copy hiddenDim elements.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//!
//! \param[in] source Source tensor [numIndices, hiddenDim] (FP16)
//! \param[in] indices Indices to scatter to [numIndices] (INT32)
//! \param[out] output Output tensor [dstNumTokens, hiddenDim] (FP16)
//! \param[in] numIndices Number of rows to scatter
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void scatterKernelVectorized(half const* __restrict__ source, int32_t const* __restrict__ indices,
    half* __restrict__ output, int32_t numIndices, int32_t hiddenDim)
{
    // Each block handles one source row
    int32_t const srcIdx = blockIdx.x;
    if (srcIdx >= numIndices)
    {
        return;
    }

    // Get destination row index
    int32_t const dstIdx = indices[srcIdx];

    half const* srcRow = source + static_cast<int64_t>(srcIdx) * hiddenDim;
    half* dstRow = output + static_cast<int64_t>(dstIdx) * hiddenDim;

    // Vectorized processing (8 FP16 = 16 bytes = uint4)
    using vec_t = uint4;

    int32_t const numVecs = hiddenDim / VEC_SIZE;

    // Vectorized copy
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(srcRow)[i];
        reinterpret_cast<vec_t*>(dstRow)[i] = dataVec;
    }

    // Handle remainder
    int32_t const remainderStart = numVecs * VEC_SIZE;
    for (int32_t i = remainderStart + threadIdx.x; i < hiddenDim; i += blockDim.x)
    {
        dstRow[i] = srcRow[i];
    }
}

// Internal host function wrappers for kernel launches (not exposed in header)

void invokeBiasAndSiLU(rt::Tensor& data, rt::Tensor const& bias, cudaStream_t stream)
{
    check::check(data.getDataType() == nvinfer1::DataType::kHALF, "Data tensor must be FP16");
    check::check(bias.getDataType() == nvinfer1::DataType::kHALF, "Bias tensor must be FP16");
    check::check(data.getShape().getNumDims() == 2, "Data tensor must be 2D [numTokens, hiddenDim]");
    check::check(bias.getShape().getNumDims() == 1, "Bias tensor must be 1D [hiddenDim]");
    check::check(data.getShape()[1] == bias.getShape()[0], "Hidden dimension mismatch");
    check::check(data.getShape()[1] % 8 == 0, "hiddenDim must be a multiple of 8 for vectorized access");
    check::check(reinterpret_cast<uintptr_t>(data.rawPointer()) % 16 == 0, "Data pointer must be 16-byte aligned");
    check::check(reinterpret_cast<uintptr_t>(bias.rawPointer()) % 16 == 0, "Bias pointer must be 16-byte aligned");

    int64_t const numTokens = data.getShape()[0];
    int64_t const hiddenDim = data.getShape()[1];

    dim3 const grid(numTokens);
    dim3 const block(256);

    biasAndSiLUKernelVectorized<<<grid, block, 0, stream>>>(
        static_cast<half*>(data.rawPointer()), static_cast<half const*>(bias.rawPointer()), numTokens, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeAddBias(rt::Tensor& data, rt::Tensor const& bias, cudaStream_t stream)
{
    check::check(data.getDataType() == nvinfer1::DataType::kHALF, "Data tensor must be FP16");
    check::check(bias.getDataType() == nvinfer1::DataType::kHALF, "Bias tensor must be FP16");
    check::check(data.getShape().getNumDims() == 2, "Data tensor must be 2D [numTokens, hiddenDim]");
    check::check(bias.getShape().getNumDims() == 1, "Bias tensor must be 1D [hiddenDim]");
    check::check(data.getShape()[1] == bias.getShape()[0], "Hidden dimension mismatch");
    check::check(data.getShape()[1] % 8 == 0, "Hidden dimension must be divisible by 8 for vectorization");
    check::check(reinterpret_cast<uintptr_t>(data.rawPointer()) % 16 == 0, "Data pointer must be 16-byte aligned");
    check::check(reinterpret_cast<uintptr_t>(bias.rawPointer()) % 16 == 0, "Bias pointer must be 16-byte aligned");

    int32_t const numTokens = static_cast<int32_t>(data.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(data.getShape()[1]);

    constexpr int32_t VEC_SIZE = 8;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    dim3 const block(256);
    dim3 const grid((numVecs + block.x - 1) / block.x, numTokens);

    addBiasKernelVectorized<VEC_SIZE><<<grid, block, 0, stream>>>(
        static_cast<half*>(data.rawPointer()), static_cast<half const*>(bias.rawPointer()), numTokens, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace

void invokeTalkerMLP(rt::Tensor const& input, rt::Tensor const& fc1Weight, rt::Tensor const& fc1Bias,
    rt::Tensor const& fc2Weight, rt::Tensor const& fc2Bias, rt::Tensor& output, rt::Tensor& workspace,
    cudaStream_t stream)
{
    auto inputShape = input.getShape();
    auto outputShape = output.getShape();
    auto workspaceShape = workspace.getShape();
    auto fc1WeightShape = fc1Weight.getShape();
    auto fc2WeightShape = fc2Weight.getShape();

    if (input.getDataType() != nvinfer1::DataType::kHALF || fc1Weight.getDataType() != nvinfer1::DataType::kHALF
        || fc1Bias.getDataType() != nvinfer1::DataType::kHALF || fc2Weight.getDataType() != nvinfer1::DataType::kHALF
        || fc2Bias.getDataType() != nvinfer1::DataType::kHALF || output.getDataType() != nvinfer1::DataType::kHALF
        || workspace.getDataType() != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("All tensors must be FP16");
        return;
    }

    if (inputShape.getNumDims() != 2 || outputShape.getNumDims() != 2 || workspaceShape.getNumDims() != 2)
    {
        LOG_ERROR("Tensors must be 2D");
        return;
    }

    int32_t const numTokens = static_cast<int32_t>(inputShape[0]);
    int32_t const inputDim = static_cast<int32_t>(inputShape[1]);
    int32_t const hiddenDim = static_cast<int32_t>(fc1WeightShape[0]);
    int32_t const outputDim = static_cast<int32_t>(fc2WeightShape[0]);

    if (fc1WeightShape[1] != inputDim)
    {
        LOG_ERROR("FC1 weight shape mismatch: expected [*, %d], got [%ld, %ld]", inputDim, fc1WeightShape[0],
            fc1WeightShape[1]);
        return;
    }

    if (fc2WeightShape[1] != hiddenDim)
    {
        LOG_ERROR("FC2 weight shape mismatch: expected [*, %d], got [%ld, %ld]", hiddenDim, fc2WeightShape[0],
            fc2WeightShape[1]);
        return;
    }

#ifdef CUTE_DSL_GEMM_ENABLED
    // FC1: workspace = SiLU(input @ fc1Weight^T + fc1Bias)
    if (!CuteDslGemmRunner::runBiasSiLU(input.rawPointer(), fc1Weight.rawPointer(), workspace.rawPointer(),
            fc1Bias.rawPointer(), numTokens, hiddenDim, inputDim, stream))
    {
        LOG_ERROR("FC1 GEMM+bias+SiLU failed");
        return;
    }

    // FC2: output = workspace @ fc2Weight^T + fc2Bias
    if (!CuteDslGemmRunner::runBias(workspace.rawPointer(), fc2Weight.rawPointer(), output.rawPointer(),
            fc2Bias.rawPointer(), numTokens, outputDim, hiddenDim, stream))
    {
        LOG_ERROR("FC2 GEMM+bias failed");
        return;
    }
#else
    CublasHandle handle(stream, input.rawPointer());
    half const alpha = __float2half(1.0F);
    half const beta = __float2half(0.0F);

    // FC1: workspace = input @ fc1Weight^T
    auto status = cublasGemmEx(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, hiddenDim, numTokens, inputDim, &alpha,
        fc1Weight.rawPointer(), CUDA_R_16F, inputDim, input.rawPointer(), CUDA_R_16F, inputDim, &beta,
        workspace.rawPointer(), CUDA_R_16F, hiddenDim, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        LOG_ERROR("FC1 cuBLAS GEMM failed with status %d", static_cast<int32_t>(status));
        return;
    }

    biasAndSiLUKernelVectorized<<<numTokens, 256, 0, stream>>>(static_cast<half*>(workspace.rawPointer()),
        static_cast<half const*>(fc1Bias.rawPointer()), numTokens, hiddenDim);
    CUDA_CHECK(cudaPeekAtLastError());

    // FC2: output = workspace @ fc2Weight^T
    status = cublasGemmEx(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, outputDim, numTokens, hiddenDim, &alpha,
        fc2Weight.rawPointer(), CUDA_R_16F, hiddenDim, workspace.rawPointer(), CUDA_R_16F, hiddenDim, &beta,
        output.rawPointer(), CUDA_R_16F, outputDim, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        LOG_ERROR("FC2 cuBLAS GEMM failed with status %d", static_cast<int32_t>(status));
        return;
    }

    dim3 const block(128);
    dim3 const grid((outputDim / 8 + block.x - 1) / block.x, numTokens);
    addBiasKernelVectorized<<<grid, block, 0, stream>>>(static_cast<half*>(output.rawPointer()),
        static_cast<half const*>(fc2Bias.rawPointer()), numTokens, outputDim);
    CUDA_CHECK(cudaPeekAtLastError());
#endif
}

void invokeLinearLayer(
    rt::Tensor const& input, rt::Tensor const& weight, rt::Tensor const& bias, rt::Tensor& output, cudaStream_t stream)
{
    auto inputShape = input.getShape();
    auto outputShape = output.getShape();
    auto weightShape = weight.getShape();

    if (input.getDataType() != nvinfer1::DataType::kHALF || weight.getDataType() != nvinfer1::DataType::kHALF
        || bias.getDataType() != nvinfer1::DataType::kHALF || output.getDataType() != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("All tensors must be FP16");
        return;
    }

    if (inputShape.getNumDims() != 2 || outputShape.getNumDims() != 2 || weightShape.getNumDims() != 2)
    {
        LOG_ERROR("Tensors must be 2D [N, dim]");
        return;
    }

    int32_t const numTokens = static_cast<int32_t>(inputShape[0]);
    int32_t const inputDim = static_cast<int32_t>(inputShape[1]);
    int32_t const outputDim = static_cast<int32_t>(weightShape[0]);

    if (weightShape[1] != inputDim)
    {
        LOG_ERROR("Weight shape mismatch: expected [*, %d], got [%ld, %ld]", inputDim, weightShape[0], weightShape[1]);
        return;
    }

#ifdef CUTE_DSL_GEMM_ENABLED
    // output = input @ weight^T + bias
    if (!CuteDslGemmRunner::runBias(input.rawPointer(), weight.rawPointer(), output.rawPointer(), bias.rawPointer(),
            numTokens, outputDim, inputDim, stream))
    {
        LOG_ERROR("Linear layer GEMM+bias failed");
        return;
    }
#else
    CublasHandle handle(stream, input.rawPointer());
    half const alpha = __float2half(1.0F);
    half const beta = __float2half(0.0F);

    auto const status = cublasGemmEx(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, outputDim, numTokens, inputDim, &alpha,
        weight.rawPointer(), CUDA_R_16F, inputDim, input.rawPointer(), CUDA_R_16F, inputDim, &beta,
        output.rawPointer(), CUDA_R_16F, outputDim, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        LOG_ERROR("Linear layer cuBLAS GEMM failed with status %d", static_cast<int32_t>(status));
        return;
    }

    dim3 const block(128);
    dim3 const grid((outputDim / 8 + block.x - 1) / block.x, numTokens);
    addBiasKernelVectorized<<<grid, block, 0, stream>>>(static_cast<half*>(output.rawPointer()),
        static_cast<half const*>(bias.rawPointer()), numTokens, outputDim);
    CUDA_CHECK(cudaPeekAtLastError());
#endif
}

void invokeGather(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream)
{
    check::check(source.getDataType() == nvinfer1::DataType::kHALF, "Source tensor must be FP16");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT32, "Indices tensor must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto const srcDims = source.getShape();
    int32_t const numIndices = static_cast<int32_t>(indices.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(srcDims[srcDims.getNumDims() - 1]);

    if (numIndices == 0)
    {
        return;
    }

    dim3 const grid(numIndices);
    dim3 const block(256);

    gatherKernelVectorized<8><<<grid, block, 0, stream>>>(source.dataPointer<half>(), indices.dataPointer<int32_t>(),
        static_cast<half*>(output.rawPointer()), numIndices, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeScatter(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream)
{
    check::check(source.getDataType() == nvinfer1::DataType::kHALF, "Source tensor must be FP16");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT32, "Indices tensor must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto const srcDims = source.getShape();
    int32_t const numIndices = static_cast<int32_t>(indices.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(srcDims[srcDims.getNumDims() - 1]);

    if (numIndices == 0)
    {
        return;
    }

    dim3 const grid(numIndices);
    dim3 const block(256);

    scatterKernelVectorized<8><<<grid, block, 0, stream>>>(source.dataPointer<half>(), indices.dataPointer<int32_t>(),
        static_cast<half*>(output.rawPointer()), numIndices, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

//! \brief Non-streaming fused assistant preamble construction kernel
//!
//! Each block handles one output row (blockIdx.x). Total rows = 8 + textLen + 2.
//!
//! Row definitions:
//!   0-2:        copy projected[0-2]
//!   3:          ttsPad + embTable[codecNothinkId]
//!   4:          ttsPad + embTable[codecThinkBosId]
//!   5:          ttsPad + embTable[codecThinkEosId]
//!   6:          ttsPad + embTable[speakerId]
//!   7:          ttsBos + embTable[codecPadId]
//!   8..8+N-1:   projected[3+i] + embTable[codecPadId]  (i = rowIdx-8)
//!   8+N:        ttsEos + embTable[codecPadId]
//!   8+N+1:      ttsPad + embTable[codecBosId]
template <int32_t VEC_SIZE = 8>
__global__ void assistantPreambleKernel(half const* __restrict__ projected, half const* __restrict__ ttsPadEmbed,
    half const* __restrict__ ttsBosEmbed, half const* __restrict__ ttsEosEmbed, half const* __restrict__ embTable,
    int32_t codecNothinkId, int32_t codecThinkBosId, int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId,
    int32_t codecBosId, int32_t textLen, int32_t hiddenDim, half* __restrict__ output)
{
    constexpr int32_t kFixedPrefixLen = 8; // rows 0-7
    int32_t const rowIdx = blockIdx.x;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    using vec_t = uint4;

    half const* srcA;
    half const* srcB = nullptr;
    half* const dstRow = output + static_cast<int64_t>(rowIdx) * hiddenDim;

    if (rowIdx < kFixedPrefixLen)
    {
        switch (rowIdx)
        {
        case 0: srcA = projected; break;
        case 1: srcA = projected + hiddenDim; break;
        case 2: srcA = projected + 2 * hiddenDim; break;
        case 3:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(codecNothinkId) * hiddenDim;
            break;
        case 4:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(codecThinkBosId) * hiddenDim;
            break;
        case 5:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(codecThinkEosId) * hiddenDim;
            break;
        case 6:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(speakerId) * hiddenDim;
            break;
        default: // rowIdx == 7
            srcA = ttsBosEmbed;
            srcB = embTable + static_cast<int64_t>(codecPadId) * hiddenDim;
            break;
        }
    }
    else if (rowIdx < kFixedPrefixLen + textLen)
    {
        // Text token rows: projected[3 + (rowIdx-8)] + embTable[codecPadId]
        int32_t const textIdx = rowIdx - kFixedPrefixLen;
        srcA = projected + static_cast<int64_t>(3 + textIdx) * hiddenDim;
        srcB = embTable + static_cast<int64_t>(codecPadId) * hiddenDim;
    }
    else if (rowIdx == kFixedPrefixLen + textLen)
    {
        // ttsEos + embTable[codecPadId]
        srcA = ttsEosEmbed;
        srcB = embTable + static_cast<int64_t>(codecPadId) * hiddenDim;
    }
    else
    {
        // ttsPad + embTable[codecBosId]
        srcA = ttsPadEmbed;
        srcB = embTable + static_cast<int64_t>(codecBosId) * hiddenDim;
    }

    // Vectorized copy-and-optionally-add
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t va = reinterpret_cast<vec_t const*>(srcA)[i];
        if (srcB != nullptr)
        {
            vec_t vb = reinterpret_cast<vec_t const*>(srcB)[i];
            half2* aPtr = reinterpret_cast<half2*>(&va);
            half2 const* bPtr = reinterpret_cast<half2 const*>(&vb);
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE / 2; ++j)
            {
                aPtr[j] = __hadd2(aPtr[j], bPtr[j]);
            }
        }
        reinterpret_cast<vec_t*>(dstRow)[i] = va;
    }
}

void invokeAssistantPreamble(rt::Tensor const& projected, rt::Tensor const& ttsPadEmbed, rt::Tensor const& ttsBosEmbed,
    rt::Tensor const& ttsEosEmbed, rt::Tensor const& talkerEmbTable, int32_t codecNothinkId, int32_t codecThinkBosId,
    int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId, int32_t codecBosId, int32_t textLen,
    rt::Tensor& output, cudaStream_t stream)
{
    constexpr int32_t kVecSize = 8;

    int32_t const hiddenDim = static_cast<int32_t>(projected.getShape()[1]);
    int32_t const numVecs = hiddenDim / kVecSize;
    // totalRows = 8 fixed prefix + textLen text rows + 2 suffix rows
    int32_t const totalRows = 8 + textLen + 2;

    // 128 threads covers H=1024 with VEC_SIZE=8 in one pass
    dim3 const block(std::min(numVecs, 128));
    dim3 const grid(totalRows);

    half const* projPtr = projected.dataPointer<half>();
    half const* padPtr = ttsPadEmbed.dataPointer<half>();
    half const* bosPtr = ttsBosEmbed.dataPointer<half>();
    half const* eosPtr = ttsEosEmbed.dataPointer<half>();
    half const* embPtr = talkerEmbTable.dataPointer<half>();
    half* outPtr = static_cast<half*>(output.rawPointer());

    assistantPreambleKernel<kVecSize><<<grid, block, 0, stream>>>(projPtr, padPtr, bosPtr, eosPtr, embPtr,
        codecNothinkId, codecThinkBosId, codecThinkEosId, speakerId, codecPadId, codecBosId, textLen, hiddenDim,
        outPtr);
    CUDA_CHECK(cudaPeekAtLastError());
}

//! Fused residual connection kernel
//!
//! Computes: output[j] = embed0[code0,j] + embed15[code15,j] + addend[j] + sum_{k=1}^{14}(codecHiddens[k,j])
//! Single block of (hiddenDim/VEC_SIZE) threads; FP32 accumulators for precision.
template <int32_t VEC_SIZE = 8>
__global__ void residualConnectionKernel(half const* __restrict__ codecHiddens, half const* __restrict__ embTable0,
    half const* __restrict__ embTable15, int32_t code0, int32_t code15, half const* __restrict__ addend,
    int32_t hiddenDim, half* __restrict__ output)
{
    using vec_t = uint4;
    int32_t const vecIdx = static_cast<int32_t>(threadIdx.x);
    int32_t const numVecs = hiddenDim / VEC_SIZE;
    if (vecIdx >= numVecs)
    {
        return;
    }

    float acc[VEC_SIZE];

    // embed(code0) from Talker embedding table
    {
        vec_t v = reinterpret_cast<vec_t const*>(embTable0 + static_cast<int64_t>(code0) * hiddenDim)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] = __half2float(p[i]);
        }
    }
    // embed(code15) from CodePredictor embedding table
    {
        vec_t v = reinterpret_cast<vec_t const*>(embTable15 + static_cast<int64_t>(code15) * hiddenDim)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] += __half2float(p[i]);
        }
    }
    // addend: trailing_text_hidden[generationStep] or tts_pad_embed
    {
        vec_t v = reinterpret_cast<vec_t const*>(addend)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] += __half2float(p[i]);
        }
    }
    // sum codecHiddens rows 1..14 (row 0 and 15 replaced by direct embedding lookups above)
    for (int32_t k = 1; k <= 14; ++k)
    {
        vec_t v = reinterpret_cast<vec_t const*>(codecHiddens + static_cast<int64_t>(k) * hiddenDim)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] += __half2float(p[i]);
        }
    }

    // store as FP16
    vec_t result;
    half* rp = reinterpret_cast<half*>(&result);
#pragma unroll
    for (int32_t i = 0; i < VEC_SIZE; ++i)
    {
        rp[i] = __float2half(acc[i]);
    }
    reinterpret_cast<vec_t*>(output)[vecIdx] = result;
}

void invokeResidualConnection(rt::Tensor const& codecHiddens, rt::Tensor const& embTable0, rt::Tensor const& embTable15,
    int32_t code0, int32_t code15, half const* addend, rt::Tensor& output, cudaStream_t stream)
{
    int32_t const hiddenDim = static_cast<int32_t>(embTable0.getShape()[1]);
    constexpr int32_t kVecSize = 8;
    int32_t const numVecs = hiddenDim / kVecSize;

    // codecHiddens has shape [1, 16, H]: skip leading batch dim
    half const* codecPtr = codecHiddens.dataPointer<half>();

    residualConnectionKernel<kVecSize><<<1, numVecs, 0, stream>>>(codecPtr, embTable0.dataPointer<half>(),
        embTable15.dataPointer<half>(), code0, code15, addend, hiddenDim, static_cast<half*>(output.rawPointer()));
    CUDA_CHECK(cudaPeekAtLastError());
}

// Each thread handles one work item. The total work is (suppressCount + numSeenTokens).
// Threads [0, suppressCount) do suppression; threads [suppressCount, suppressCount+numSeenTokens) apply penalty.
// The two ranges never overlap: suppression covers [suppressStart, suppressEnd) which is always in the
// upper special-token region, while seenTokens are sampled from the normal codec range below suppressStart.
__global__ void talkerLogitAdjustKernel(float* logits, int32_t suppressStart, int32_t suppressCount, int32_t codecEosId,
    int32_t const* seenTokens, int32_t numSeenTokens, float repetitionPenalty)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x) * blockDim.x + static_cast<int32_t>(threadIdx.x);

    if (idx < suppressCount)
    {
        int32_t const tokenId = suppressStart + idx;
        if (tokenId != codecEosId)
        {
            logits[tokenId] = -INFINITY;
        }
    }
    else if (idx < suppressCount + numSeenTokens)
    {
        int32_t const tokenId = seenTokens[idx - suppressCount];
        float const logit = logits[tokenId];
        logits[tokenId] = (logit >= 0.0f) ? logit / repetitionPenalty : logit * repetitionPenalty;
    }
}

void invokeTalkerLogitAdjust(rt::Tensor const& seenTokens, rt::Tensor& logits, int32_t suppressStart,
    int32_t suppressEnd, int32_t codecEosId, int32_t numSeenTokens, float repetitionPenalty, cudaStream_t stream)
{
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT, "Logits tensor must be FP32");
    check::check(seenTokens.getDataType() == nvinfer1::DataType::kINT32, "seenTokens tensor must be INT32");

    int32_t const suppressCount = suppressEnd - suppressStart;
    int32_t const totalWork = suppressCount + numSeenTokens;
    if (totalWork <= 0)
    {
        return;
    }

    constexpr int32_t kBlockSize = 128;
    int32_t const gridSize = (totalWork + kBlockSize - 1) / kBlockSize;
    float* logitsPtr = static_cast<float*>(logits.rawPointer());

    talkerLogitAdjustKernel<<<gridSize, kBlockSize, 0, stream>>>(logitsPtr, suppressStart, suppressCount, codecEosId,
        seenTokens.dataPointer<int32_t>(), numSeenTokens, repetitionPenalty);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace kernel
} // namespace trt_edgellm
