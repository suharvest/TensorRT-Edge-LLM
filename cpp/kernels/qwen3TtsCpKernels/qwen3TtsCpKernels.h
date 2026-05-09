#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace trt_edgellm::kernel
{

void qwen3TtsCpArgmax(
    float const* logits, int32_t codebookSize, int32_t group, int32_t* selectedTokens, cudaStream_t stream);

void qwen3TtsCpGatherEmbedding(float const* embeddingTable, int32_t codebookSize, int32_t hiddenSize, int32_t group,
    int32_t const* selectedTokens, float* outputEmbedding, cudaStream_t stream);

} // namespace trt_edgellm::kernel
