# Qwen3-4B AWQ on Orin NX - Deploy Status

## Environment

| Component | Value |
|------|------|
| Device | Jetson Orin NX 16GB (`orin-nx`, `100.82.225.102`) |
| JetPack / L4T | JetPack 6.2 / R36.4.x |
| Container | `docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0` |
| TensorRT in container | 10.4.x |
| Build directory | `/home/harvest/TensorRT-Edge-LLM/build_container` |
| Checkpoint | `Qwen/Qwen3-4B-AWQ` |
| Workspace | `/home/harvest/edgellm-workspace/Qwen3-4B-AWQ` |
| Active engine | `engines-3072/llm.engine` |
| Runtime plugin | `/repo/build_container/libNvInfer_edgellm_plugin.so` |

## Current Result

`Qwen/Qwen3-4B-AWQ` is usable on Orin NX with TensorRT-Edge-LLM. The validated
engine supports batch 1, `maxInputLen=3072`, and `maxKVCacheCapacity=4096`, so
it covers a context length greater than 2k.

The engine was built from LLM-only ONNX export. No visual encoder was exported
or loaded.

## Real Generation Data

Input:

```json
{
  "batch_size": 1,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_generate_length": 128,
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "用一句话介绍 TensorRT Edge-LLM。"}
      ]
    }
  ]
}
```

Output from `engines-3072`:

```text
TensorRT Edge-LLM 是一个结合了NVIDIA TensorRT和LLM（大语言模型）的边缘侧推理框架，旨在在边缘设备上高效部署和加速大语言模型的推理性能。
```

## Performance Data

Commands were run on `orin-nx` in the container with:

```bash
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:$LD_LIBRARY_PATH
```

### Engine Build

| Engine | Build config | Result |
|------|------|------|
| `engines` | `maxInputLen=2048`, `maxKVCacheCapacity=4096` | Engine generated; auxiliary files had to be copied manually |
| `engines-3072` | `maxInputLen=3072`, `maxKVCacheCapacity=4096` | Engine generated; auxiliary files had to be copied manually |

TensorRT build stats for `engines-3072`:

| Metric | Value |
|------|------|
| Engine size | 2544 MiB |
| Total weights memory | 2651890688 bytes |
| Activation memory at 3072 prefill profile | 217056768 bytes |
| Engine generation time | 57.7264 s |
| Peak TRT GPU allocator usage during build | 3453 MiB |
| Peak CPU usage during build/serialization | 7473 MiB |

### Prefill

`llm_bench --mode prefill --iterations 3 --warmup 1 --noProfile`

| Input length | E2E time | Throughput |
|------|------:|------:|
| 128 | 100.9758 ms | 1267.6 tok/s |
| 1024 | 733.2195 ms | 1396.6 tok/s |
| 2048 | 1533.5579 ms | 1335.5 tok/s |
| 3072 | 2420.1692 ms | 1269.3 tok/s |

### Decode

`llm_bench --mode decode --pastKVLen 2048 --osl 100 --iterations 5 --warmup 2 --noProfile`

| Mode | E2E time for 99 steps | Per-token latency | Throughput |
|------|------:|------:|------:|
| Without CUDA graph | 3461.3477 ms | 34.963 ms | 28.6 tok/s |
| With `--useCudaGraph` | 3299.6565 ms | 33.330 ms | 30.0 tok/s |

Runtime logs during real inference showed the 3072 engine loading successfully
with `maxSupportedInputLength=3072`, `maxKVCacheCapacity=4096`, and the FP16
embedding table `[151936, 2560]`.

## Notes

- `llm_build` generated a valid `llm.engine`, but returned failure because
  `processed_chat_template.json` was missing from the ONNX directory and was not
  copied into the engine directory.
- `embedding.safetensors` also needed to be copied manually into the engine
  directory for runtime loading.
- The current workaround is to copy:

```bash
cp onnx/llm/embedding.safetensors engines-3072/embedding.safetensors
cp <qwen-chat-template>/processed_chat_template.json \
  engines-3072/processed_chat_template.json
```

## Validated Deployment Path

The reproduction scripts are checked in under `docs/deploy-container/`:

| Step | Script | Where |
|------|--------|-------|
| Download + LLM-only ONNX export | `export_qwen3_awq_on_wsl.sh` | x86/WSL |
| Package ONNX for transfer | `package_qwen3_awq_onnx.sh` | x86/WSL |
| Build TensorRT engine + copy auxiliary files | `build_qwen3_awq_on_orin.sh` | Orin |
| Smoke inference / benchmark | `run_qwen3_awq_inference.sh` | Orin |
| Build Python runtime binding | `build_server_bindings_on_orin.sh` | Orin |
| Start OpenAI-compatible HTTP server | `serve_qwen3_awq_http.sh` | Orin |

Default flow:

```bash
# On the x86/WSL export host.
docs/deploy-container/export_qwen3_awq_on_wsl.sh
docs/deploy-container/package_qwen3_awq_onnx.sh

# Transfer the generated tarball to Orin with scp, rsync, USB storage, or any
# site-local transfer mechanism.
scp ~/edgellm-workspace/qwen3-4b-awq-onnx-llm.tar \
  orin:/home/harvest/edgellm-workspace/

# On Orin.
mkdir -p /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm
tar xf /home/harvest/edgellm-workspace/qwen3-4b-awq-onnx-llm.tar \
  -C /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm

cd /home/harvest/TensorRT-Edge-LLM
docs/deploy-container/build_qwen3_awq_on_orin.sh
docs/deploy-container/run_qwen3_awq_inference.sh --test
docs/deploy-container/build_server_bindings_on_orin.sh
docs/deploy-container/serve_qwen3_awq_http.sh
```

If Docker requires root on the Orin host, run the Orin scripts with:

```bash
DOCKER_CMD="sudo docker" docs/deploy-container/build_qwen3_awq_on_orin.sh
DOCKER_CMD="sudo docker" docs/deploy-container/run_qwen3_awq_inference.sh --test
DOCKER_CMD="sudo docker" docs/deploy-container/build_server_bindings_on_orin.sh
DOCKER_CMD="sudo docker" docs/deploy-container/serve_qwen3_awq_http.sh
```

If Python dependencies need to be installed inside the container and the default
indexes are slow or unavailable, pass a pip mirror:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  INSTALL_SERVER_DEPS=1 \
  docs/deploy-container/serve_qwen3_awq_http.sh
```

## Current Invocation

The currently validated engine can be called directly on `orin-nx` with:

```bash
cat > /home/harvest/edgellm-workspace/input_qwen3_cn.json <<'JSON'
{
  "batch_size": 1,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_generate_length": 128,
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "用一句话介绍 TensorRT Edge-LLM。"}
      ]
    }
  ]
}
JSON

docker run --rm --runtime=nvidia \
  -v /home/harvest/TensorRT-Edge-LLM:/repo \
  -v /home/harvest/edgellm-workspace:/workspace \
  docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0 bash -lc '
cd /repo/build_container
export EDGELLM_PLUGIN_PATH=/repo/build_container/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/repo/build_container:$LD_LIBRARY_PATH
./examples/llm/llm_inference \
  --engineDir /workspace/Qwen3-4B-AWQ/engines-3072 \
  --inputFile /workspace/input_qwen3_cn.json \
  --outputFile /workspace/output_qwen3_3072_cn.json'

cat /home/harvest/edgellm-workspace/output_qwen3_3072_cn.json
```

The expected output should contain a normal Chinese answer similar to:

```text
TensorRT Edge-LLM 是一个结合了NVIDIA TensorRT和LLM（大语言模型）的边缘侧推理框架，旨在在边缘设备上高效部署和加速大语言模型的推理性能。
```

## WSL Export Details

The Jetson container has PyTorch 2.4.0, which is too old for the current
`llm_loader` ONNX export path because `torch.onnx.export` does not accept the
`dynamic_shapes` argument. Export was therefore done on `wsl2-local`, where the
project virtual environment had a newer CPU PyTorch.

WSL also had stale proxy variables, so the working command explicitly cleared
them before using the HuggingFace mirror:

```bash
cd ~/TensorRT-Edge-LLM
mkdir -p ~/edgellm-workspace

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c '
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-4B-AWQ",
    local_dir="/home/harve/edgellm-workspace/Qwen3-4B-AWQ",
)'

export PYTHONPATH=$PWD:$PWD/experimental:$PYTHONPATH
.venv/bin/python -m llm_loader.export_all_cli \
  /home/harve/edgellm-workspace/Qwen3-4B-AWQ \
  /home/harve/edgellm-workspace/Qwen3-4B-AWQ/onnx \
  --skip-visual
```

The exported `onnx/llm` directory contained:

```text
config.json
embedding.safetensors
model.onnx
model.onnx.data
tokenizer.json
tokenizer_config.json
```

It did not contain `processed_chat_template.json`, which is why the auxiliary
template copy is required before runtime.

## Transfer Details

Directory transfer can be unreliable when the destination directory is
root-owned or the transfer tool handles directory contents differently. The
portable path is to package `onnx/llm` into one tar file, transfer that file,
then extract it on Orin NX:

```bash
docs/deploy-container/package_qwen3_awq_onnx.sh
scp ~/edgellm-workspace/qwen3-4b-awq-onnx-llm.tar \
  orin:/home/harvest/edgellm-workspace/

# On Orin NX.
mkdir -p /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm
tar xf /home/harvest/edgellm-workspace/qwen3-4b-awq-onnx-llm.tar \
  -C /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm
```

## Auxiliary File Fixup

`build_qwen3_awq_on_orin.sh` now performs this fixup automatically after
`llm_build`:

```bash
cp /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/onnx/llm/embedding.safetensors \
  /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/engines-3072/embedding.safetensors

cp /home/harvest/TensorRT-Edge-LLM/docs/deploy-container/qwen_processed_chat_template.json \
  /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/engines-3072/processed_chat_template.json

sudo chown -R harvest:harvest \
  /home/harvest/edgellm-workspace/Qwen3-4B-AWQ/engines-3072
```

The copied chat template is standard Qwen ChatML:

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

For clean reproduction, the template is now shipped as
`docs/deploy-container/qwen_processed_chat_template.json`.

## Memory Data

`tegrastats` during a real 512-token max generation request on `engines-3072`:

| Stage | Unified RAM |
|------|------:|
| Before request | 3783 / 15656 MB |
| Engine/runtime loading | 5718-5766 / 15656 MB |
| Generating | peak 5995 / 15656 MB |

Service deployment should budget roughly 6 GiB unified memory for one loaded
instance. `llm_bench` can show 7.8-7.9 GiB because it loads/manages extra engine
state and is not representative of steady service residency.

## Scripted Reproduction Status

The model/runtime path is now scripted through export, package, engine build,
smoke inference, Python binding build, and HTTP server launch. Remaining work
before a production-style service:

- add a systemd unit or Docker entrypoint for boot-time service management.

## HTTP Service Status

The repository already contains an experimental OpenAI-compatible HTTP server
under `experimental/server` with:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

It is designed to keep `LLMRuntime` resident in process and supports streaming
SSE responses. This is the right base for an external service; a separate
project is not required for the first deployable version.

The server was validated on `orin-nx` with the existing `engines-3072` engine:

```bash
DETACH=1 \
CONTAINER_NAME=qwen3-awq-http-test \
INSTALL_SERVER_DEPS=1 \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  docs/deploy-container/serve_qwen3_awq_http.sh
```

Health check:

```json
{"status":"healthy","model":"/workspace/Qwen3-4B-AWQ/engines-3072","speculative_decoding":false}
```

Metrics endpoint after service start:

```json
{"profiling_enabled":true,"prefill":{"reused_tokens":0,"computed_tokens":0}}
```

OpenAI-compatible chat completion returned:

```json
{
  "id": "chatcmpl-a815dfd75ac5",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "TensorRT Edge-LLM 是一个结合了NVIDIA TensorRT和LLM（大语言模型）的边缘侧推理框架，旨在在边缘设备上高效部署和加速大语言模型的推理性能。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"completion_tokens": 45}
}
```

System-prompt KV cache was also validated through the HTTP layer:

1. First chat request used `save_system_prompt_kv_cache=true` and
   `return_cache_metrics=true`.
2. Second chat request used the exact same system prompt and did not request
   saving.
3. The second response reported:

```json
{"cache_metrics":{"prefill":{"reused_tokens":28,"computed_tokens":20}}}
```

The explicit warmup endpoint was validated with a formatted Qwen ChatML system
prompt:

```json
{"object":"cache.system_prompt","cached":true,"lora_weights_name":""}
```

The subsequent chat request reported:

```json
{"cache_metrics":{"prefill":{"reused_tokens":23,"computed_tokens":22}}}
```

This confirms the service supports Edge-LLM's current in-memory system-prompt
prefix KV cache. It does not yet implement arbitrary multi-turn conversation
prefix caching; only exact repeated system prompt prefixes are reused.

The warmup endpoint also supports raw `system_prompt` and `messages` input. In
that mode the service formats the prompt with the engine directory's
`processed_chat_template.json`, so callers do not need to hand-build Qwen
ChatML. This raw `system_prompt` mode was validated on `orin-nx`:

```json
{"object":"cache.system_prompt","cached":true,"lora_weights_name":""}
```

The subsequent chat request with the same system prompt reported:

```json
{"cache_metrics":{"prefill":{"reused_tokens":23,"computed_tokens":22}}}
```

Arbitrary multi-turn text prefix cache was validated after rebuilding
`_edgellm_runtime` with preformatted request support. The first request used
`save_prefix_cache=true` and cached this prefix:

```text
system -> user -> assistant
```

The second request kept the same prefix and changed only the final user
message, with `prefix_cache=true`. It returned:

```json
{"cache_metrics":{"prefill":{"reused_tokens":76,"computed_tokens":20}}}
```

This verifies that the service can cache and reuse more than just the leading
system prompt. Current prefix-cache mode is text-only and in-memory for the
server process lifetime.

Current deployment requirements on `orin-nx`:

- Python binding must exist under `build_container/pybind`; this was generated
  with `build_server_bindings_on_orin.sh` and produced
  `_edgellm_runtime.cpython-310-aarch64-linux-gnu.so`;
- server dependencies `fastapi` and `uvicorn` must be installed in the runtime
  Python environment; use an image that includes them or start with
  `INSTALL_SERVER_DEPS=1`;
- the current engine directory must include the fixed
  `embedding.safetensors` and `processed_chat_template.json`.

Current server launch:

```bash
docs/deploy-container/build_server_bindings_on_orin.sh
docs/deploy-container/serve_qwen3_awq_http.sh

curl -sN http://localhost:8000/v1/chat/completions \
  -H Content-Type:application/json \
  -d '{"model":"Qwen/Qwen3-4B-AWQ","messages":[{"role":"system","content":"你是一个运行在边缘设备上的中文对话机器人。回答要简洁、准确，并且优先说明可执行步骤。"},{"role":"user","content":"用一句话介绍 TensorRT Edge-LLM。"}],"max_tokens":128,"save_system_prompt_kv_cache":true,"return_cache_metrics":true}'
```
