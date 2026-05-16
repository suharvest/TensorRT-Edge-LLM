# Qwen3-4B AWQ on Jetson Orin NX

This directory contains the validated high-performance service path for
`Qwen/Qwen3-4B-AWQ` on Jetson Orin NX with TensorRT Edge-LLM.

The flow is intentionally split across the x86/WSL export host and the Orin
runtime host:

1. Download the official HuggingFace checkpoint and export LLM-only ONNX.
2. Package the ONNX directory for transfer.
3. Build the TensorRT engine on Orin.
4. Run smoke inference and benchmarks.
5. Build the Python runtime binding.
6. Start the OpenAI-compatible HTTP server.

## Validated Target

| Component | Value |
|------|------|
| Model | `Qwen/Qwen3-4B-AWQ` |
| Device | Jetson Orin NX 16GB |
| JetPack / L4T | JetPack 6.2 / R36.4.x |
| Container | `docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0` |
| Engine | batch 1, `maxInputLen=3072`, `maxKVCacheCapacity=4096` |
| Runtime mode | LLM-only, no visual encoder exported or loaded |

## Scripts

| Step | Script | Host |
|------|--------|------|
| Export ONNX | `export_qwen3_awq_on_wsl.sh` | x86/WSL |
| Package ONNX | `package_qwen3_awq_onnx.sh` | x86/WSL |
| Build engine | `build_qwen3_awq_on_orin.sh` | Orin |
| Smoke test / bench | `run_qwen3_awq_inference.sh` | Orin |
| Build Python binding | `build_server_bindings_on_orin.sh` | Orin |
| Start HTTP service | `serve_qwen3_awq_http.sh` | Orin |

## Export On x86/WSL

```bash
cd ~/TensorRT-Edge-LLM

EDGELLM_ROOT=$PWD \
WORKSPACE=~/edgellm-workspace \
HF_ENDPOINT=https://hf-mirror.com \
  docs/deploy-container/export_qwen3_awq_on_wsl.sh

WORKSPACE=~/edgellm-workspace \
  docs/deploy-container/package_qwen3_awq_onnx.sh
```

Transfer the generated tarball with any site-local mechanism:

```bash
scp ~/edgellm-workspace/qwen3-4b-awq-onnx-llm.tar \
  orin:/home/harvest/edgellm-workspace/
```

## Build And Run On Orin

```bash
mkdir -p /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm
tar xf /home/harvest/edgellm-workspace/qwen3-4b-awq-onnx-llm.tar \
  -C /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm

cd /home/harvest/TensorRT-Edge-LLM

docs/deploy-container/build_qwen3_awq_on_orin.sh
docs/deploy-container/run_qwen3_awq_inference.sh --test
docs/deploy-container/build_server_bindings_on_orin.sh
docs/deploy-container/serve_qwen3_awq_http.sh
```

If Docker requires root on the Orin host:

```bash
DOCKER_CMD="sudo docker" docs/deploy-container/build_qwen3_awq_on_orin.sh
DOCKER_CMD="sudo docker" docs/deploy-container/run_qwen3_awq_inference.sh --test
DOCKER_CMD="sudo docker" docs/deploy-container/build_server_bindings_on_orin.sh
DOCKER_CMD="sudo docker" docs/deploy-container/serve_qwen3_awq_http.sh
```

## HTTP Service

The service uses the patched `experimental.server` entrypoint and loads an
existing engine directly:

```bash
DETACH=1 \
CONTAINER_NAME=qwen3-awq-http \
INSTALL_SERVER_DEPS=1 \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  docs/deploy-container/serve_qwen3_awq_http.sh
```

Health check:

```bash
curl -s http://<orin-ip>:8000/health
```

Chat completion:

```bash
curl -s http://<orin-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "用一句话介绍 TensorRT Edge-LLM。"}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "return_cache_metrics": true
  }'
```

Warm a reusable system prompt cache:

```bash
curl -s http://<orin-ip>:8000/v1/cache/system_prompt \
  -H "Content-Type: application/json" \
  -d '{"system_prompt":"You are a concise assistant."}'
```

Inspect runtime counters:

```bash
curl -s http://<orin-ip>:8000/metrics
```

## Validated Output

Prompt:

```text
用一句话介绍 TensorRT Edge-LLM。
```

Observed output from `engines-3072`:

```text
TensorRT Edge-LLM 是一个结合了NVIDIA TensorRT和LLM（大语言模型）的边缘侧推理框架，旨在在边缘设备上高效部署和加速大语言模型的推理性能。
```

See `STATUS.md` for the measured prefill/decode numbers and current caveats.
