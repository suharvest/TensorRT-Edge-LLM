#!/usr/bin/env bash
# Standalone build script for moss_tts_nano_worker on orin-nx.
# Bypasses CMake (which is blocked by pre-existing gdn_decode.h CUDA 12.6 conflict).
# See docs/playbooks/tts-model-edge-port-playbook.md.

set -euo pipefail

EDGELLM_SRC=${EDGELLM_SRC:-/home/harvest/TensorRT-Edge-LLM}
BUILD=${BUILD:-${EDGELLM_SRC}/build}
ORT_ROOT=${ORT_ROOT:-/usr/local/onnxruntime}     # adjust per machine
SP_ROOT=${SP_ROOT:-/usr}                          # SentencePiece headers + libsentencepiece.so
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
OUT=${OUT:-/tmp/moss_tts_nano_worker}

# Required pre-built object files (must exist in the build tree).
# Note: logger.h is header-only (inline gLogger at logger.h:313); no logger.cpp.o.
KERNEL_OBJ="${BUILD}/cpp/CMakeFiles/edgellmCore.dir/kernels/kvCacheUtilKernels/mossLinearKvKernels.cu.o"
RUNTIME_OBJ="${BUILD}/cpp/CMakeFiles/edgellmCore.dir/runtime/mossTtsNanoRuntime.cpp.o"
STRINGUTILS_OBJ="${BUILD}/cpp/CMakeFiles/edgellmCore.dir/common/stringUtils.cpp.o"

# If runtime source is newer than the compiled .o, rebuild it. Otherwise repeated
# changes to mossTtsNanoRuntime.cpp would be silently ignored by the link step
# (as happened during the 2026-05 KV dtype ABI fix).
RUNTIME_SRC="${EDGELLM_SRC}/cpp/runtime/mossTtsNanoRuntime.cpp"
if [[ -f "${RUNTIME_SRC}" ]]; then
  if [[ ! -f "${RUNTIME_OBJ}" ]] || [[ "${RUNTIME_SRC}" -nt "${RUNTIME_OBJ}" ]]; then
    echo "[build] rebuilding ${RUNTIME_OBJ} (source newer than obj)..."
    (cd "${BUILD}/cpp" && make -j2 runtime/mossTtsNanoRuntime.o)
  fi
fi

for f in "${KERNEL_OBJ}" "${RUNTIME_OBJ}" "${STRINGUTILS_OBJ}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f" >&2
    echo "Run 'make -j2 runtime/mossTtsNanoRuntime.o' from build/cpp first" >&2
    exit 1
  fi
done

echo "[build] compile worker main..."
${CUDA_ROOT}/bin/nvcc -std=c++17 -O2 -arch=sm_87 \
    -I "${EDGELLM_SRC}/cpp" \
    -I "${EDGELLM_SRC}/3rdParty/nlohmannJson/include" \
    -I "${ORT_ROOT}/include" \
    -I "${SP_ROOT}/include" \
    -c "${EDGELLM_SRC}/cpp/workers/moss_tts_nano_worker.cpp" \
    -o /tmp/moss_tts_nano_worker.o

echo "[build] link..."
${CUDA_ROOT}/bin/nvcc -std=c++17 -arch=sm_87 \
    /tmp/moss_tts_nano_worker.o \
    "${KERNEL_OBJ}" "${RUNTIME_OBJ}" "${STRINGUTILS_OBJ}" \
    -L "${ORT_ROOT}/lib" \
    -L "${SP_ROOT}/lib" \
    -L "${SP_ROOT}/lib/aarch64-linux-gnu" \
    -L /usr/lib/aarch64-linux-gnu \
    -lnvinfer -lcudart -lcuda -lonnxruntime -lsentencepiece \
    -lstdc++fs -lpthread -ldl \
    -Xlinker "-rpath=${ORT_ROOT}/lib" \
    -Xlinker "-rpath=\$ORIGIN/../lib" \
    -o "${OUT}"

echo "[build] OK -> ${OUT}"
ls -lh "${OUT}"
