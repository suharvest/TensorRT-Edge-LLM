# [TensorRT-Edge-LLM v0.7.1] Qwen3.5-4B-AWQ uses 13GB RAM on Orin NX 16GB, blocks ASR+TTS+LLM co-residency

## TL;DR

`tensorrt-edge-llm` v0.7.1 (commit `5136119`, plus customvoice product branch) loads `Qwen3.5-4B-AWQ` with **12.9–13.5 GB** RAM on Jetson Orin NX 16GB at `maxInputLen=4096` context, which prevents the typical ASR + TTS + LLM edge stack from co-residing on the device. The same hardware comfortably hosts `Qwen3-4B-AWQ` (pure attention) at **~3 GB**, indicating the cost is specific to the GDN hybrid mamba/attention architecture as currently exposed by the runtime, not an inherent property of mamba models.

After exhausting common configuration knobs, only ~1.5 GB of further savings looks achievable in software (mmap embedding sidecar + reduced vocab). Weight streaming, the canonical TRT lever, is a no-op here because AWQ weights live in plugin/constant buffers and aren't in the streamable pool.

## Environment

| Item | Value |
|---|---|
| Device | NVIDIA Jetson Orin NX 16GB (sm_87, JetPack 6.2) |
| CUDA | 12.6.68 |
| TensorRT | 10.3.0.30 |
| TensorRT-Edge-LLM | v0.7.1 + customvoice product layer migration |
| Model | `harvestsu/Qwen3.5-4B-AWQ-TensorRT-EdgeLLM-engine` (private, ONNX-MTP branch) |
| Engine build | `--maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 4096 --specBase --maxVerifyTreeSize 8` |
| Engine size | `eagle_base.engine` 2.1 GB + `eagle_draft.engine` 374 MB + `embedding.safetensors` 1.27 GB |

## Architecture

Qwen3.5-4B-AWQ uses NVIDIA's Gated DeltaNet hybrid:
- 32 hidden layers: **24 mamba/SSM + 8 attention**
- hidden_size 2560, num_attn_heads 16, head_dim 256, kv_heads 8
- vocab_size 248320 (Qwen3.5 expanded vocab)

## Reproduction

```bash
# Launch server (custom branch v071/customvoice-product HEAD cb51c14, with optional patches reverted)
EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so \
python -m experimental.server \
  --engine-dir /path/to/engines-mtp8-4k/base \
  --served-model-name qwen3.5-4b-awq \
  --port 8100 --host 0.0.0.0

# Observe RAM after engine load (no requests yet):
$ free -m
               total        used        free   ...
Mem:           15656       12947         385   ...
```

Same hardware, same toolchain, with `Qwen3-4B-AWQ` (pure attention 32 layers, vocab 152064, similar AWQ INT4 + FP16 setup):

```
Mem:            ~3000          ...
```

Roughly **4× the RAM footprint** for a same-parameter-count model when the runtime sees a hybrid mamba/attention layout.

## What we tried (and why it didn't help)

| Lever | Expected | Measured | Notes |
|---|---|---|---|
| Lower `maxInputLen` 8192 → 4096 | Linear with seq | −1.0 GB | Workspace pool clearly not pure-seq-linear |
| Disable MTP draft engine | −400-500 MB | −500 MB | Draft is minor |
| TRT Weight Streaming (`setWeightStreamingBudgetV2`) | −2.5–3.7 GB | **~0** | AWQ weights are in plugin/constant buffers, not in `getStreamableWeightsSize()` (reports only 2.1 MB streamable for base, 92 KB for draft). All four budget values (`unset`, `min`, `1g`, `off`) produced identical RAM / tok-s, variance ≤ 1%. |

TRT log line at engine load:
```
[INFO] [TensorRT] [MemUsageChange] Init cuBLAS/cuBLASLt: ... now: CPU 79, GPU 11891 (MiB)
```

→ **~12 GB is pre-allocated** in the GPU pool right after cuBLAS init, before any inference. The shared execution context manager later reports its own 2 GB allocation on top.

## Hypothesis

The runtime / engine builder appears to size the **mamba/SSM scan and state buffers pessimistically** for every hidden layer at engine load time, regardless of actual prefill batch shape. With 24 mamba layers at hidden_size 2560 + parallel-scan workspace + Conv state caches, the worst-case pool dominates the 12 GB.

By contrast, pure-attention models like Qwen3-4B only allocate KV cache (a few hundred MB at 4k context for 8 KV heads) plus regular attention scratch, both of which scale much more gracefully.

If this is correct, the fix is in how the mamba/GDN layers' workspace pool is sized — potentially making it elastic with the batched prefill shape rather than fixed at `maxInputLen` worst-case.

## Public references suggesting this is suboptimal

- NVIDIA's own Nemotron-H (92% Mamba-2 + 8% attention) is documented as **3× faster than Llama-3.1 at matched accuracy** thanks to mamba's constant-memory recurrent state. The expected memory advantage is not present in our v0.7.1 deployment of Qwen3.5.
- Community guidance for Qwen3.5-4B states the Q4-quantized model "needs only ~2.5 GB" disk size and fits "4–6 GB GPUs" — far from the 12+ GB we measure at runtime.

## Ask

1. Are mamba/GDN layer workspace allocations meant to be elastic at runtime in v0.7.1, or are they intentionally pre-sized at `maxInputLen`? If the latter, can a future release expose a knob to clamp this pool independently of `maxInputLen` (e.g., a `mambaPrefillMaxSeq`)?
2. Can AWQ weights for hybrid models be migrated into the regular TRT streamable weight pool, so `setWeightStreamingBudgetV2` becomes effective for AWQ engines?
3. Any pointers on profiling the 12 GB pool composition would also be very welcome (e.g., per-layer or per-buffer breakdown) — `--verbose` only shows aggregate `MemUsageChange` lines.

## Reproducer artifacts

- Engine: `harvestsu/Qwen3.5-4B-AWQ-TensorRT-EdgeLLM-engine` (will share access if needed)
- Server log + `free -m` snapshots: available on request

Thanks for the great work on the framework!
