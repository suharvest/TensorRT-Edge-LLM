# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Qwen3-Omni Inference Example Note: cuBLAS is loaded at runtime via
# dlopen/dlsym in TTSRuntime, no compile-time linking needed.
add_executable(qwen3_tts_inference qwen3_tts_inference.cpp)

target_include_directories(
  qwen3_tts_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                              ${CMAKE_SOURCE_DIR}/examples/utils)

target_link_libraries(qwen3_tts_inference PRIVATE edgellmCore exampleUtils
                                                  commonLibraryExt)

# CuTe-DSL cudart shim: provides weak cudaLibrary*/cudaKernelSetAttributeForDevice
# (CUDA 12.0-12.6 omit these exports). edgellmCore links it only as a static lib so
# the symbols don't propagate to this exe — link it directly, AFTER edgellmCore.
if(TARGET trt_edgellm_cutedsl_cudart_shim)
  target_link_libraries(qwen3_tts_inference PRIVATE trt_edgellm_cutedsl_cudart_shim)
  # Force the weak shim defs into the link regardless of archive order (ld is single-pass;
  # libcutedsl references these but is ordered after the shim).
  target_link_options(qwen3_tts_inference PRIVATE
    -Wl,-u,cudaLibraryLoadData -Wl,-u,cudaLibraryUnload
    -Wl,-u,cudaLibraryGetKernel -Wl,-u,cudaKernelSetAttributeForDevice)
endif()

add_cross_build_link_options(qwen3_tts_inference)

# Set output directory
set_target_properties(
  qwen3_tts_inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                 ${CMAKE_BINARY_DIR}/examples/omni)
