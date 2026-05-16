#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Build the Python runtime binding required by experimental/server on Orin.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/harvest/TensorRT-Edge-LLM}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
BUILD_DIR="${BUILD_DIR:-/repo/build_container}"
JOBS="${JOBS:-$(nproc)}"
INSTALL_PYBIND11="${INSTALL_PYBIND11:-0}"
PYBIND11_DIR_HOST="${PYBIND11_DIR_HOST:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"

info() { echo "[INFO] $*"; }

if [ -z "${PYBIND11_DIR_HOST}" ]; then
    PYBIND11_DIR_HOST="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || true)"
fi

pybind_mount_args=()
if [ -n "${PYBIND11_DIR_HOST}" ] && [ -d "${PYBIND11_DIR_HOST}" ]; then
    info "Using host pybind11 CMake package: ${PYBIND11_DIR_HOST}"
    pybind_mount_args=(-v "${PYBIND11_DIR_HOST}:${PYBIND11_DIR_HOST}:ro")
    if [ -d /usr/include/pybind11 ]; then
        pybind_mount_args+=(-v /usr/include/pybind11:/usr/include/pybind11:ro)
    fi
    if [ -d /usr/lib/python3/dist-packages/pybind11/include ]; then
        pybind_mount_args+=(
            -v /usr/lib/python3/dist-packages/pybind11/include:/usr/lib/python3/dist-packages/pybind11/include:ro)
    fi
fi

pip_env_args=()
if [ -n "${PIP_INDEX_URL}" ]; then
    pip_env_args+=(-e "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi
if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then
    pip_env_args+=(-e "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}")
fi

info "Building Python bindings in ${REPO_ROOT}/build_container"
${DOCKER_CMD} run --rm --runtime=nvidia \
    -v "${REPO_ROOT}:/repo" \
    "${pybind_mount_args[@]}" \
    "${pip_env_args[@]}" \
    "${CONTAINER_IMAGE}" bash -lc "
set -e
if [ -n '${PYBIND11_DIR_HOST}' ] && [ -d '${PYBIND11_DIR_HOST}' ]; then
  PYBIND11_DIR='${PYBIND11_DIR_HOST}'
elif [ '${INSTALL_PYBIND11}' = '1' ]; then
  python3 -m pip install --user pybind11
  PYBIND11_DIR=\$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
else
  python3 - <<'PY'
import pybind11
PY
  PYBIND11_DIR=\$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
fi
cd ${BUILD_DIR}
cmake .. \
  -DTRT_PACKAGE_DIR=/usr \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR=\${PYBIND11_DIR}
make -j${JOBS} _edgellm_runtime
ls -l ${BUILD_DIR}/pybind/*_edgellm_runtime*.so
"

cat <<EOF

Python binding built. If pybind11 is missing in the container, rerun with:

  INSTALL_PYBIND11=1 docs/deploy-container/build_server_bindings_on_orin.sh

If the container cannot access pip but the Orin host has pybind11 installed,
the script automatically mounts the host pybind11 CMake package into the
container.

If pip is needed and the default Jetson/NVIDIA indexes are slow or unavailable,
provide a mirror:

  INSTALL_PYBIND11=1 \\
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \\
  docs/deploy-container/build_server_bindings_on_orin.sh
EOF
