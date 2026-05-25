# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Qwen3-Omni Inference Example Note: cuBLAS is loaded at runtime via
# dlopen/dlsym in TTSRuntime, no compile-time linking needed.
add_executable(qwen3_tts_inference qwen3_tts_inference.cpp)
add_executable(qwen3_tts_streaming_worker qwen3_tts_streaming_worker.cpp)

target_include_directories(
  qwen3_tts_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                              ${CMAKE_SOURCE_DIR}/examples/utils)
target_include_directories(
  qwen3_tts_streaming_worker PRIVATE ${COMMON_INCLUDE_DIRS}
                                     ${CMAKE_SOURCE_DIR}/examples/utils)

target_link_libraries(qwen3_tts_inference PRIVATE edgellmCore exampleUtils
                                                  commonLibraryExt)
target_link_libraries(qwen3_tts_streaming_worker PRIVATE edgellmCore exampleUtils
                                                         commonLibraryExt)

add_cross_build_link_options(qwen3_tts_inference)
add_cross_build_link_options(qwen3_tts_streaming_worker)

# Set output directory
set_target_properties(
  qwen3_tts_inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                 ${CMAKE_BINARY_DIR}/examples/omni)
set_target_properties(
  qwen3_tts_streaming_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/examples/omni)
