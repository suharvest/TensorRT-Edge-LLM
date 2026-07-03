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

# Qwen3-TTS standalone streaming worker (JSON-line stdin/stdout, slot-pool).
# v0.9.0 re-port: uses the runtime's native streamingChunkFrames/onChunkReady
# streaming API; Code2Wav vocoding happens worker-side.
add_executable(qwen3_tts_streaming_worker qwen3_tts_streaming_worker.cpp)

target_include_directories(
  qwen3_tts_streaming_worker PRIVATE ${COMMON_INCLUDE_DIRS}
                                     ${CMAKE_SOURCE_DIR}/examples/utils)

target_link_libraries(qwen3_tts_streaming_worker PRIVATE edgellmCore
                                                         exampleUtils commonLibraryExt)

if(TARGET trt_edgellm_cutedsl_cudart_shim)
  target_link_libraries(qwen3_tts_streaming_worker
                        PRIVATE trt_edgellm_cutedsl_cudart_shim)
  target_link_options(
    qwen3_tts_streaming_worker PRIVATE -Wl,-u,cudaLibraryLoadData
    -Wl,-u,cudaLibraryUnload -Wl,-u,cudaLibraryGetKernel
    -Wl,-u,cudaKernelSetAttributeForDevice)
endif()

add_cross_build_link_options(qwen3_tts_streaming_worker)

set_target_properties(
  qwen3_tts_streaming_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/examples/omni)

# MOSS-TTS-Nano standalone JSON-line worker (re-port from v0.7.x line
# 3c6c263, formerly cpp/workers/moss_tts_nano_worker.cpp built via
# build_moss_worker.sh). Needs onnxruntime (voice-clone codec-encode path)
# and SentencePiece (tokenizer); target is skipped when either is missing
# so the default build never breaks.
set(MOSS_ORT_ROOT
    "$ENV{ORT_ROOT}"
    CACHE PATH "onnxruntime install root for moss_tts_nano_worker")
find_path(
  MOSS_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${MOSS_ORT_ROOT}/include /usr/local/onnxruntime/include
        $ENV{HOME}/ort-from-container/include
  PATH_SUFFIXES onnxruntime)
find_library(
  MOSS_ORT_LIBRARY onnxruntime
  HINTS ${MOSS_ORT_ROOT}/lib /usr/local/onnxruntime/lib
        $ENV{HOME}/ort-from-container/lib)
find_path(MOSS_SP_INCLUDE_DIR sentencepiece_processor.h)
find_library(MOSS_SP_LIBRARY sentencepiece)

if(MOSS_ORT_INCLUDE_DIR
   AND MOSS_ORT_LIBRARY
   AND MOSS_SP_INCLUDE_DIR
   AND MOSS_SP_LIBRARY)
  add_executable(moss_tts_nano_worker moss_tts_nano_worker.cpp)

  target_include_directories(
    moss_tts_nano_worker
    PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/examples/utils
            ${MOSS_ORT_INCLUDE_DIR} ${MOSS_SP_INCLUDE_DIR})

  target_link_libraries(
    moss_tts_nano_worker PRIVATE edgellmCore exampleUtils commonLibraryExt
                                 ${MOSS_ORT_LIBRARY} ${MOSS_SP_LIBRARY})

  if(TARGET trt_edgellm_cutedsl_cudart_shim)
    target_link_libraries(moss_tts_nano_worker
                          PRIVATE trt_edgellm_cutedsl_cudart_shim)
    target_link_options(
      moss_tts_nano_worker PRIVATE -Wl,-u,cudaLibraryLoadData
      -Wl,-u,cudaLibraryUnload -Wl,-u,cudaLibraryGetKernel
      -Wl,-u,cudaKernelSetAttributeForDevice)
  endif()

  add_cross_build_link_options(moss_tts_nano_worker)

  # rpath so the worker finds libonnxruntime.so next to a non-system install.
  get_filename_component(MOSS_ORT_LIBDIR ${MOSS_ORT_LIBRARY} DIRECTORY)
  set_target_properties(
    moss_tts_nano_worker
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/omni
               BUILD_RPATH ${MOSS_ORT_LIBDIR}
               INSTALL_RPATH ${MOSS_ORT_LIBDIR})
else()
  message(
    STATUS
      "moss_tts_nano_worker skipped (onnxruntime/sentencepiece not found: "
      "ORT inc=${MOSS_ORT_INCLUDE_DIR} lib=${MOSS_ORT_LIBRARY} "
      "SP inc=${MOSS_SP_INCLUDE_DIR} lib=${MOSS_SP_LIBRARY})")
endif()
