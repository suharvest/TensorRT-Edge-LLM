#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Build the Qwen3-4B-AWQ TensorRT engine on Orin from an existing ONNX export.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/harvest/TensorRT-Edge-LLM}"
WORKSPACE="${WORKSPACE:-/home/harvest/edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-AWQ}"
ONNX_DIR="${ONNX_DIR:-${WORKSPACE}/${MODEL_NAME}/onnx/llm}"
ENGINE_DIR="${ENGINE_DIR:-${WORKSPACE}/${MODEL_NAME}/engines-3072}"
TEMPLATE_FILE="${TEMPLATE_FILE:-${REPO_ROOT}/docs/deploy-container/qwen_processed_chat_template.json}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-3072}"
MAX_KV_CACHE_CAPACITY="${MAX_KV_CACHE_CAPACITY:-4096}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"

info() { echo "[INFO] $*"; }
err() { echo "[ERROR] $*" >&2; }

check_inputs() {
    [ -f "${ONNX_DIR}/model.onnx" ] || {
        err "Missing ONNX model: ${ONNX_DIR}/model.onnx"
        exit 1
    }
    [ -f "${ONNX_DIR}/model.onnx.data" ] || {
        err "Missing ONNX weights: ${ONNX_DIR}/model.onnx.data"
        exit 1
    }
    [ -f "${ONNX_DIR}/embedding.safetensors" ] || {
        err "Missing embedding: ${ONNX_DIR}/embedding.safetensors"
        exit 1
    }
    [ -f "${TEMPLATE_FILE}" ] || {
        err "Missing chat template: ${TEMPLATE_FILE}"
        exit 1
    }
}

build_engine() {
    info "Building engine into ${ENGINE_DIR}"
    ${DOCKER_CMD} run --rm --runtime=nvidia \
        -v "${REPO_ROOT}:/repo" \
        -v "${WORKSPACE}:/workspace" \
        "${CONTAINER_IMAGE}" bash -lc "
set -e
cd /repo/build_container
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:\${LD_LIBRARY_PATH:-}
./examples/llm/llm_build \
  --onnxDir /workspace/${MODEL_NAME}/onnx/llm \
  --engineDir /workspace/${MODEL_NAME}/$(basename "${ENGINE_DIR}") \
  --maxBatchSize ${MAX_BATCH_SIZE} \
  --maxInputLen ${MAX_INPUT_LEN} \
  --maxKVCacheCapacity ${MAX_KV_CACHE_CAPACITY} || true
test -f /workspace/${MODEL_NAME}/$(basename "${ENGINE_DIR}")/llm.engine
"
}

fixup_engine_dir() {
    info "Copying runtime auxiliary files into ${ENGINE_DIR}"
    cp "${ONNX_DIR}/embedding.safetensors" "${ENGINE_DIR}/embedding.safetensors" || sudo cp "${ONNX_DIR}/embedding.safetensors" "${ENGINE_DIR}/embedding.safetensors"
    cp "${TEMPLATE_FILE}" "${ENGINE_DIR}/processed_chat_template.json" || sudo cp "${TEMPLATE_FILE}" "${ENGINE_DIR}/processed_chat_template.json"
    cp "${ONNX_DIR}/tokenizer.json" "${ENGINE_DIR}/tokenizer.json" || sudo cp "${ONNX_DIR}/tokenizer.json" "${ENGINE_DIR}/tokenizer.json"
    cp "${ONNX_DIR}/tokenizer_config.json" "${ENGINE_DIR}/tokenizer_config.json" || sudo cp "${ONNX_DIR}/tokenizer_config.json" "${ENGINE_DIR}/tokenizer_config.json"
}

print_summary() {
    info "Engine directory contents:"
    find "${ENGINE_DIR}" -maxdepth 1 -type f -printf "%f %s\n" | sort
    du -sh "${ENGINE_DIR}"
    df -h "${WORKSPACE}"
}

echo "=============================================="
echo " Qwen3-4B AWQ Orin Engine Build"
echo "=============================================="
echo " REPO_ROOT:             ${REPO_ROOT}"
echo " ONNX_DIR:              ${ONNX_DIR}"
echo " ENGINE_DIR:            ${ENGINE_DIR}"
echo " MAX_INPUT_LEN:         ${MAX_INPUT_LEN}"
echo " MAX_KV_CACHE_CAPACITY: ${MAX_KV_CACHE_CAPACITY}"
echo " CONTAINER_IMAGE:       ${CONTAINER_IMAGE}"
echo " DOCKER_CMD:            ${DOCKER_CMD}"
echo "=============================================="

check_inputs
mkdir -p "${ENGINE_DIR}"
build_engine
fixup_engine_dir
print_summary
