#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Run smoke inference or benchmark with the Qwen3-4B-AWQ engine on Orin.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/harvest/TensorRT-Edge-LLM}"
WORKSPACE="${WORKSPACE:-/home/harvest/edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-AWQ}"
ENGINE_DIR="${ENGINE_DIR:-${WORKSPACE}/${MODEL_NAME}/engines-3072}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
PROMPT="${PROMPT:-用一句话介绍 TensorRT Edge-LLM。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUTPUT_FILE="${OUTPUT_FILE:-${WORKSPACE}/output_qwen3_3072_cn.json}"
INPUT_FILE="${INPUT_FILE:-${WORKSPACE}/input_qwen3_cn.json}"

info() { echo "[INFO] $*"; }
err() { echo "[ERROR] $*" >&2; }

write_input() {
    python3 - "$INPUT_FILE" "$PROMPT" "$MAX_TOKENS" <<'PY'
import json
import sys

path, prompt, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = {
    "batch_size": 1,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_generate_length": max_tokens,
    "requests": [{"messages": [{"role": "user", "content": prompt}]}],
}
with open(path, "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
}

run_inference() {
    ${DOCKER_CMD} run --rm --runtime=nvidia \
        -v "${REPO_ROOT}:/repo" \
        -v "${WORKSPACE}:/workspace" \
        "${CONTAINER_IMAGE}" bash -lc "
set -e
cd /repo/build_container
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:\${LD_LIBRARY_PATH:-}
./examples/llm/llm_inference \
  --engineDir /workspace/${MODEL_NAME}/$(basename "${ENGINE_DIR}") \
  --inputFile /workspace/$(basename "${INPUT_FILE}") \
  --outputFile /workspace/$(basename "${OUTPUT_FILE}")
"
}

show_output() {
    python3 - "$OUTPUT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
for response in data.get("responses", []):
    text = response.get("output_text", "")
    print(text)
PY
}

run_prefill_bench() {
    ${DOCKER_CMD} run --rm --runtime=nvidia \
        -v "${REPO_ROOT}:/repo" \
        -v "${WORKSPACE}:/workspace" \
        "${CONTAINER_IMAGE}" bash -lc "
set -e
cd /repo/build_container
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:\${LD_LIBRARY_PATH:-}
for L in 128 1024 2048 3072; do
  echo BENCH_PREFILL_\${L}
  ./examples/llm/llm_bench \
    --engineDir /workspace/${MODEL_NAME}/$(basename "${ENGINE_DIR}") \
    --mode prefill \
    --inputLen \${L} \
    --iterations 3 \
    --warmup 1 \
    --noProfile
done
"
}

run_decode_bench() {
    ${DOCKER_CMD} run --rm --runtime=nvidia \
        -v "${REPO_ROOT}:/repo" \
        -v "${WORKSPACE}:/workspace" \
        "${CONTAINER_IMAGE}" bash -lc "
set -e
cd /repo/build_container
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:\${LD_LIBRARY_PATH:-}
./examples/llm/llm_bench \
  --engineDir /workspace/${MODEL_NAME}/$(basename "${ENGINE_DIR}") \
  --mode decode \
  --pastKVLen 2048 \
  --osl 100 \
  --iterations 5 \
  --warmup 2 \
  --noProfile \
  --useCudaGraph
"
}

usage() {
    echo "Usage: $0 [--test | --bench]"
}

case "${1:---test}" in
    --test)
        [ -f "${ENGINE_DIR}/llm.engine" ] || {
            err "Missing engine: ${ENGINE_DIR}/llm.engine"
            exit 1
        }
        write_input
        run_inference
        show_output
        ;;
    --bench)
        run_prefill_bench
        run_decode_bench
        ;;
    --help|-h)
        usage
        ;;
    *)
        err "Unknown flag: $1"
        usage
        exit 1
        ;;
esac
