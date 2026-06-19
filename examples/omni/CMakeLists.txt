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

add_cross_build_link_options(qwen3_tts_inference)

# Set output directory
set_target_properties(
  qwen3_tts_inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                 ${CMAKE_BINARY_DIR}/examples/omni)

# Qwen3-TTS standalone streaming worker (JSON-line stdin/stdout, slot-pool N>1).
# Reuses the same edgellmCore TTS runtime; cuBLAS is loaded at runtime via dlopen in the runtime.
add_executable(qwen3_tts_streaming_worker qwen3_tts_streaming_worker.cpp)

target_include_directories(
  qwen3_tts_streaming_worker PRIVATE ${COMMON_INCLUDE_DIRS}
                                     ${CMAKE_SOURCE_DIR}/examples/utils)

target_link_libraries(qwen3_tts_streaming_worker PRIVATE edgellmCore exampleUtils
                                                         commonLibraryExt)

add_cross_build_link_options(qwen3_tts_streaming_worker)

set_target_properties(
  qwen3_tts_streaming_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/examples/omni)
