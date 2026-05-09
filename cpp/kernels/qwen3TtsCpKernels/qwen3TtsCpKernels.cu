#include "qwen3TtsCpKernels.h"
#include "common/checkMacros.h"

#include <algorithm>
#include <cfloat>

namespace trt_edgellm::kernel
{
namespace
{

__global__ void qwen3TtsCpArgmaxKernel(
    float const* __restrict__ logits, int32_t codebookSize, int32_t group, int32_t* __restrict__ selectedTokens)
{
    extern __shared__ unsigned char sharedRaw[];
    auto* values = reinterpret_cast<float*>(sharedRaw);
    auto* indices = reinterpret_cast<int32_t*>(values + blockDim.x);

    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    float bestValue = -FLT_MAX;
    int32_t bestIndex = 0;
    float const* groupLogits = logits + static_cast<size_t>(group) * codebookSize;
    for (int32_t idx = tid; idx < codebookSize; idx += static_cast<int32_t>(blockDim.x))
    {
        float const value = groupLogits[idx];
        if (value > bestValue || (value == bestValue && idx < bestIndex))
        {
            bestValue = value;
            bestIndex = idx;
        }
    }
    values[tid] = bestValue;
    indices[tid] = bestIndex;
    __syncthreads();

    for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            float const otherValue = values[tid + stride];
            int32_t const otherIndex = indices[tid + stride];
            if (otherValue > values[tid] || (otherValue == values[tid] && otherIndex < indices[tid]))
            {
                values[tid] = otherValue;
                indices[tid] = otherIndex;
            }
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        selectedTokens[group] = indices[0];
    }
}

__global__ void qwen3TtsCpGatherEmbeddingKernel(float const* __restrict__ embeddingTable, int32_t codebookSize,
    int32_t hiddenSize, int32_t group, int32_t const* __restrict__ selectedTokens, float* __restrict__ outputEmbedding)
{
    int32_t token = selectedTokens[group];
    token = token < 0 ? 0 : token;
    token = token >= codebookSize ? codebookSize - 1 : token;
    float const* src = embeddingTable + (static_cast<size_t>(group) * codebookSize + token) * hiddenSize;
    for (int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x); idx < hiddenSize;
         idx += static_cast<int32_t>(blockDim.x * gridDim.x))
    {
        outputEmbedding[idx] = src[idx];
    }
}

} // namespace

void qwen3TtsCpArgmax(
    float const* logits, int32_t codebookSize, int32_t group, int32_t* selectedTokens, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    size_t const sharedBytes = static_cast<size_t>(kThreads) * (sizeof(float) + sizeof(int32_t));
    qwen3TtsCpArgmaxKernel<<<1, kThreads, sharedBytes, stream>>>(logits, codebookSize, group, selectedTokens);
    CUDA_CHECK(cudaGetLastError());
}

void qwen3TtsCpGatherEmbedding(float const* embeddingTable, int32_t codebookSize, int32_t hiddenSize, int32_t group,
    int32_t const* selectedTokens, float* outputEmbedding, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int32_t const blocks = std::min(16, (hiddenSize + kThreads - 1) / kThreads);
    qwen3TtsCpGatherEmbeddingKernel<<<blocks, kThreads, 0, stream>>>(
        embeddingTable, codebookSize, hiddenSize, group, selectedTokens, outputEmbedding);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm::kernel
