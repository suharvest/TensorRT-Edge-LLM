#!/usr/bin/env bash
# v071 LLM server launcher for Orin NX
#
# 用法：
#   MODEL=qwen3-4b bash launch.sh                  # 默认，稳定，~3GB RAM
#   MODEL=qwen3.5-4b-4k bash launch.sh             # ⚠️ 见下方 KNOWN ISSUE
#   MODEL=qwen3.5-4b-8k DST=3 VTS=4 bash launch.sh # 同上
#
# 环境变量：
#   MODEL                  qwen3-4b | qwen3.5-4b-4k | qwen3.5-4b-8k (默认 qwen3-4b)
#   DST                    spec_decode draft_step (default 4，Qwen3.5 only)
#   VTS                    spec_decode verify_tree_size (default 4)
#   DTK                    spec_decode draft_top_k (default 1)
#   PORT                   server port (default 8100)
#   WEIGHT_STREAMING_BUDGET 可选，AWQ engine 上 TRT 只暴露 ~2MB streamable，
#                          对 RAM 节省**几乎无效** (见 Task #22)。仅 forward-compat 保留。
#
# ============================================================
# KNOWN ISSUE — Qwen3.5-4B-AWQ Orin NX 16GB OOM (2026-05-25)
# ============================================================
# Qwen3.5-4B-AWQ (GDN hybrid: 24 mamba + 8 attention 层) 在 Orin NX 16GB
# 上单 LLM server 占用 12.9-13.5 GB RAM (4k context, 关 MTP 仍 12.9 GB)，
# 导致 ASR + TTS + LLM 三件套同驻必 OOM。
#
# 已穷尽软件优化：
#   - maxInputLen 8k→4k: 仅省 1GB
#   - 关 MTP draft: 仅省 500MB
#   - TRT Weight Streaming: AWQ weight 在 plugin/constant buffer，
#     不在 TRT streamable 池 (streamable=2MB)，无效果
#   - mmap embedding / FP8 KV / reduced vocab 全做完估计仅再省 ~1.5GB
#
# 根因 (推测)：TRT-LLM v0.7.1 对 GDN mamba layer 的 workspace pool
# 按 worst-case pre-allocate，没有 size-elastic 路径。
#
# 临时缓解：默认用 Qwen3-4B (纯 attention，~3GB)。
# 长期解：等 NVIDIA 上游 fix mamba workspace，或换 AGX 32GB / 拆远程 LLM。
# 详见 /tmp/qwen35-orin-nx-oom-report.md
# ============================================================

set -e
cd /home/harvest/spike-v071-nx/server

MODEL="${MODEL:-qwen3-4b}"
PORT="${PORT:-8100}"

case "$MODEL" in
  qwen3-4b)
    ENGINE_DIR=/home/harvest/edgellm-workspace/Qwen3-4B-AWQ/engines-8192
    SERVED_NAME=qwen3-4b-awq
    SPEC_ARGS=""  # Qwen3 老版无 MTP
    EXPECTED_RAM_GB=3
    ;;
  qwen3.5-4b-4k)
    echo "⚠️  Qwen3.5-4B on Orin NX uses ~13GB RAM, will OOM with ASR+TTS. See KNOWN ISSUE in this script."
    ENGINE_DIR=/home/harvest/edgellm-workspace/qwen35-4b-awq/engines-mtp8-4k/base
    SERVED_NAME=qwen3.5-4b-awq
    SPEC_ARGS="--spec-decode-engine-dir $ENGINE_DIR --draft-top-k ${DTK:-1} --draft-step ${DST:-3} --verify-tree-size ${VTS:-4}"
    EXPECTED_RAM_GB=13
    ;;
  qwen3.5-4b-8k)
    echo "⚠️  Qwen3.5-4B on Orin NX uses ~14GB RAM, will OOM with ASR+TTS. See KNOWN ISSUE in this script."
    ENGINE_DIR=/home/harvest/edgellm-workspace/qwen35-4b-awq/engines-mtp8-8k/base
    SERVED_NAME=qwen3.5-4b-awq
    SPEC_ARGS="--spec-decode-engine-dir $ENGINE_DIR --draft-top-k ${DTK:-1} --draft-step ${DST:-3} --verify-tree-size ${VTS:-4}"
    EXPECTED_RAM_GB=14
    ;;
  *)
    echo "Unknown MODEL: $MODEL"
    echo "Use one of: qwen3-4b | qwen3.5-4b-4k | qwen3.5-4b-8k"
    exit 1
    ;;
esac

pkill -f experimental.server 2>/dev/null || true
sleep 2
rm -f server.log

export EDGELLM_PLUGIN_PATH=/home/harvest/spike-v071-nx/build/libNvInfer_edgellm_plugin.so
export EDGELLM_PYBIND_DIR=/home/harvest/spike-v071-nx/server/experimental/pybind/build
export PYTHONPATH=/home/harvest/spike-v071-nx/server
if [ -n "${WEIGHT_STREAMING_BUDGET}" ]; then
  export EDGELLM_WEIGHT_STREAMING_BUDGET="${WEIGHT_STREAMING_BUDGET}"
fi

echo "Launching: MODEL=$MODEL  ENGINE_DIR=$ENGINE_DIR  PORT=$PORT  (expected ~${EXPECTED_RAM_GB}GB RAM)"

setsid nohup .venv/bin/python -m experimental.server \
  --engine-dir "$ENGINE_DIR" \
  --served-model-name "$SERVED_NAME" \
  --enable-profiling \
  --port "$PORT" --host 0.0.0.0 \
  $SPEC_ARGS \
  >> server.log 2>&1 < /dev/null &
echo $! > server.pid
sleep 1
echo "PID=$(cat server.pid)"
