# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

# Build executables for chat, benchmark and accuracy mode
add_executable(llm_build llm_build.cpp)
target_link_libraries(llm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt
                                        edgellmBuilder)
target_include_directories(llm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_build)

add_executable(llm_inference llm_inference.cpp)
target_link_libraries(llm_inference PRIVATE edgellmCore exampleUtils
                                            commonLibraryExt)
target_include_directories(
  llm_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(llm_inference)

add_executable(llm_stream llm_stream.cpp)
target_link_libraries(llm_stream PRIVATE edgellmCore exampleUtils
                                         commonLibraryExt)
target_include_directories(
  llm_stream PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(llm_stream)

add_executable(llm_bench llm_bench.cpp)
target_link_libraries(llm_bench PRIVATE commonLibraryExt edgellmCore
                                        exampleUtils)
target_include_directories(llm_bench PRIVATE ${COMMON_INCLUDE_DIRS}
                                             ${CMAKE_SOURCE_DIR}/examples/utils)
target_compile_options(llm_bench PRIVATE -Wno-psabi)
add_cross_build_link_options(llm_bench)

# Streaming-ASR M1 acceptance test (design doc §12).
# Drives the new appendPrefillEmbeds public API through one-shot vs split-chunk
# paths on the real thinker engine and compares logits element-wise.
add_executable(spike_m1_append_prefill_embeds spike_m1_append_prefill_embeds.cpp)
target_link_libraries(spike_m1_append_prefill_embeds PRIVATE edgellmCore
                                                             exampleUtils
                                                             commonLibraryExt)
target_include_directories(
  spike_m1_append_prefill_embeds PRIVATE ${COMMON_INCLUDE_DIRS}
                                          ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(spike_m1_append_prefill_embeds)

# Streaming-ASR M2 acceptance test (design doc §12 milestone 2): session
# lifecycle (beginAsrSession/endAsrSession idempotence) + capacity caps
# (KV-overflow + per-chunk-overflow refusal).
add_executable(spike_m2_session_lifecycle spike_m2_session_lifecycle.cpp)
target_link_libraries(spike_m2_session_lifecycle PRIVATE edgellmCore
                                                          exampleUtils
                                                          commonLibraryExt)
target_include_directories(
  spike_m2_session_lifecycle PRIVATE ${COMMON_INCLUDE_DIRS}
                                      ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(spike_m2_session_lifecycle)

# Streaming-ASR M3.5 acceptance test (design doc §13b): audio runner split
# correctness — encodeMelChunk on full mel vs three chunk-aligned splits
# must produce bit-exact identical audio embeddings.
add_executable(spike_m35_audio_runner_split spike_m35_audio_runner_split.cpp)
target_link_libraries(spike_m35_audio_runner_split PRIVATE edgellmCore
                                                            exampleUtils
                                                            commonLibraryExt)
target_include_directories(
  spike_m35_audio_runner_split PRIVATE ${COMMON_INCLUDE_DIRS}
                                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(spike_m35_audio_runner_split)
