#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Start the experimental OpenAI-compatible HTTP server for Qwen3-4B-AWQ.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/harvest/TensorRT-Edge-LLM}"
WORKSPACE="${WORKSPACE:-/home/harvest/edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-AWQ}"
ENGINE_DIR="${ENGINE_DIR:-${WORKSPACE}/${MODEL_NAME}/engines-3072}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-AWQ}"
INSTALL_SERVER_DEPS="${INSTALL_SERVER_DEPS:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
DETACH="${DETACH:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-awq-http}"
ENABLE_PROFILING="${ENABLE_PROFILING:-1}"

info() { echo "[INFO] $*"; }
err() { echo "[ERROR] $*" >&2; }

[ -f "${ENGINE_DIR}/llm.engine" ] || {
    err "Missing engine: ${ENGINE_DIR}/llm.engine"
    exit 1
}
[ -n "$(find "${REPO_ROOT}/build_container/pybind" -maxdepth 1 -name '*_edgellm_runtime*.so' -print -quit 2>/dev/null)" ] || {
    err "Missing Python runtime binding under ${REPO_ROOT}/build_container/pybind"
    err "Run docs/deploy-container/build_server_bindings_on_orin.sh first."
    exit 1
}

info "Starting HTTP server on ${HOST}:${PORT}"
info "Engine: ${ENGINE_DIR}"

pip_env_args=()
if [ -n "${PIP_INDEX_URL}" ]; then
    pip_env_args+=(-e "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi
if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then
    pip_env_args+=(-e "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}")
fi

docker_args=(run --runtime=nvidia
    -p "${PORT}:${PORT}"
    -v "${REPO_ROOT}:/repo"
    -v "${WORKSPACE}:/workspace"
    "${pip_env_args[@]}")

if [ "${DETACH}" = "1" ]; then
    docker_args+=(-d --name "${CONTAINER_NAME}")
else
    docker_args+=(--rm)
fi

docker_args+=("${CONTAINER_IMAGE}" bash -lc "
set -e
cd /repo
if [ '${INSTALL_SERVER_DEPS}' = '1' ]; then
  python3 -m pip install --user fastapi uvicorn
fi
python3 - <<'PY'
import fastapi
import uvicorn
PY
export PYTHONPATH=/repo:/repo/build_container/pybind:\${PYTHONPATH:-}
export EDGELLM_PYBIND_DIR=/repo/build_container/pybind
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:\${LD_LIBRARY_PATH:-}
profiling_flag=''
if [ '${ENABLE_PROFILING}' = '1' ]; then
  profiling_flag='--enable-profiling'
fi
python3 -m experimental.server \
  --engine-dir /workspace/${MODEL_NAME}/$(basename "${ENGINE_DIR}") \
  --served-model-name '${SERVED_MODEL_NAME}' \
  --host ${HOST} \
  --port ${PORT} \
  \${profiling_flag}
")

${DOCKER_CMD} "${docker_args[@]}"
