#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Download Qwen/Qwen3-4B-AWQ and export LLM-only ONNX on an x86/WSL host.

set -euo pipefail

EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"
WORKSPACE="${WORKSPACE:-$HOME/edgellm-workspace}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-4B-AWQ}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-AWQ}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE}/${MODEL_NAME}}"
ONNX_DIR="${ONNX_DIR:-${CHECKPOINT_DIR}/onnx}"
PYTHON="${PYTHON:-${EDGELLM_ROOT}/.venv/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

if [ ! -x "${PYTHON}" ]; then
    PYTHON="${PYTHON_FALLBACK:-python3}"
fi

info() { echo "[INFO] $*"; }
err() { echo "[ERROR] $*" >&2; }

check_env() {
    [ -d "${EDGELLM_ROOT}" ] || {
        err "EDGELLM_ROOT does not exist: ${EDGELLM_ROOT}"
        exit 1
    }
    "${PYTHON}" -c "import torch; import huggingface_hub; import onnxscript; import safetensors" || {
        err "Python environment is missing export dependencies."
        err "Use the project .venv or install torch, huggingface_hub, onnxscript, safetensors, transformers."
        exit 1
    }
    "${PYTHON}" -c "import inspect, torch; assert 'dynamic_shapes' in str(inspect.signature(torch.onnx.export))" || {
        err "PyTorch is too old for llm_loader ONNX export; torch.onnx.export lacks dynamic_shapes."
        exit 1
    }
}

download_checkpoint() {
    if [ "${SKIP_DOWNLOAD}" = "1" ]; then
        info "Skipping checkpoint download."
        return
    fi
    if [ -f "${CHECKPOINT_DIR}/model.safetensors" ]; then
        info "Checkpoint already exists: ${CHECKPOINT_DIR}"
        return
    fi

    mkdir -p "${CHECKPOINT_DIR}"
    info "Downloading ${MODEL_REPO} to ${CHECKPOINT_DIR}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    HF_ENDPOINT="${HF_ENDPOINT}" "${PYTHON}" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${MODEL_REPO}', local_dir='${CHECKPOINT_DIR}')
"
}

export_onnx() {
    if [ "${SKIP_EXPORT}" = "1" ]; then
        info "Skipping ONNX export."
        return
    fi
    if [ -f "${ONNX_DIR}/llm/model.onnx" ] && [ -f "${ONNX_DIR}/llm/model.onnx.data" ]; then
        info "ONNX already exists: ${ONNX_DIR}/llm"
        return
    fi

    info "Exporting LLM-only ONNX to ${ONNX_DIR}"
    cd "${EDGELLM_ROOT}"
    export PYTHONPATH="${EDGELLM_ROOT}:${EDGELLM_ROOT}/experimental:${PYTHONPATH:-}"
    "${PYTHON}" -m llm_loader.export_all_cli "${CHECKPOINT_DIR}" "${ONNX_DIR}" --skip-visual
}

print_summary() {
    info "Export complete."
    du -sh "${CHECKPOINT_DIR}" "${ONNX_DIR}/llm" 2>/dev/null || true
    find "${ONNX_DIR}/llm" -maxdepth 1 -type f -printf "%f %s\n" 2>/dev/null | sort || true
}

echo "=============================================="
echo " Qwen3-4B AWQ ONNX Export"
echo "=============================================="
echo " EDGELLM_ROOT:   ${EDGELLM_ROOT}"
echo " MODEL_REPO:     ${MODEL_REPO}"
echo " CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo " ONNX_DIR:       ${ONNX_DIR}"
echo " PYTHON:         ${PYTHON}"
echo " HF_ENDPOINT:    ${HF_ENDPOINT}"
echo "=============================================="

check_env
download_checkpoint
export_onnx
print_summary
