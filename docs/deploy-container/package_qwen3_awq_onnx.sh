#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Package the exported Qwen3-4B-AWQ LLM ONNX directory for transfer to Orin.

set -euo pipefail

WORKSPACE="${WORKSPACE:-${HOME}/edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-AWQ}"
ONNX_DIR="${ONNX_DIR:-${WORKSPACE}/${MODEL_NAME}/onnx/llm}"
TAR_FILE="${TAR_FILE:-${WORKSPACE}/qwen3-4b-awq-onnx-llm.tar}"

info() { echo "[INFO] $*"; }
err() { echo "[ERROR] $*" >&2; }

check_inputs() {
    [ -d "${ONNX_DIR}" ] || {
        err "Missing ONNX directory: ${ONNX_DIR}"
        exit 1
    }
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
}

check_inputs
mkdir -p "$(dirname "${TAR_FILE}")"

info "Packaging ${ONNX_DIR}"
tar cf "${TAR_FILE}" -C "${ONNX_DIR}" .
info "Wrote ${TAR_FILE}"
du -sh "${TAR_FILE}"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${TAR_FILE}" | tee "${TAR_FILE}.sha256"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${TAR_FILE}" | tee "${TAR_FILE}.sha256"
else
    info "sha256sum/shasum not found; checksum skipped"
fi

cat <<EOF

Transfer this tarball to the Orin host, then extract it into the target ONNX
directory. Example:

  scp ${TAR_FILE} orin:/home/harvest/edgellm-workspace/

  ssh orin '
    mkdir -p /home/harvest/edgellm-workspace/${MODEL_NAME}/onnx/llm
    tar xf /home/harvest/edgellm-workspace/$(basename "${TAR_FILE}") \\
      -C /home/harvest/edgellm-workspace/${MODEL_NAME}/onnx/llm
  '
EOF
