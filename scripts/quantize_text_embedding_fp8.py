#!/usr/bin/env python3
"""Quantize the Qwen3-TTS `text_embedding` table to FP8 (e4m3) + per-group FP32 scales,
for the runtime's native FP8 embedding-lookup path (memory-only: ~half the table size, no speed/quality cost).

Output: a single safetensors with TWO tensors:
  - "text_embedding"        : FP8 e4m3 table,   shape [vocab, hidden]            (serialized as F8_E4M3)
  - "text_embedding_scales" : FP32 per-group scales, shape [vocab, hidden/block] (F32)

The runtime (cpp/runtime/qwen3OmniTTSRuntime.cpp loader) classifies these by dtype (no names from
loadSafetensors) and the kernel (cpp/kernels/embeddingKernels/embeddingKernels.cu Fp8EmbeddingLoader)
dequantizes as:   fp16_val = float(fp8_table[v, h]) * scales[v, h // blockSize]

Kernel constraints: blockSize must be a multiple of 16 (vecSize for __nv_fp8_e4m3 128-bit loads) and divide hidden.
torch.float8_e4m3fn matches NVIDIA __nv_fp8_e4m3 (e4m3, finite, max magnitude 448). Usage:
    python quantize_text_embedding_fp8.py text_embedding.safetensors text_embedding.fp8.safetensors [--block 128]
"""
import argparse, torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--key", default="text_embedding")
ap.add_argument("--block", type=int, default=128)
a = ap.parse_args()

t = load_file(a.inp)
key = a.key if a.key in t else next(iter(t))
w = t[key].to(torch.float32)  # [vocab, hidden]
assert w.dim() == 2, f"expected 2D table, got {tuple(w.shape)}"
vocab, hidden = w.shape
bs = a.block
assert hidden % bs == 0, f"hidden {hidden} not divisible by block {bs}"
assert bs % 16 == 0, f"block {bs} must be a multiple of 16 (kernel vecSize)"
ngroups = hidden // bs
E4M3_MAX = 448.0

wg = w.view(vocab, ngroups, bs)
amax = wg.abs().amax(dim=2, keepdim=True)                                   # [vocab, ngroups, 1]
scale = (amax / E4M3_MAX).clamp(min=1e-8)                                   # fp32 per group
q = (wg / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)         # round to nearest e4m3
table_fp8 = q.view(vocab, hidden).contiguous()
scales = scale.view(vocab, ngroups).to(torch.float32).contiguous()

# sanity: dequant error matching the kernel's math
dq = (table_fp8.to(torch.float32).view(vocab, ngroups, bs) * scale).view(vocab, hidden)
mse = torch.mean((w - dq) ** 2).item()
print(f"key={key} shape=({vocab},{hidden}) block={bs} ngroups={ngroups} dq_mse={mse:.3e}")
print(f"fp8 table dtype={table_fp8.dtype} shape={tuple(table_fp8.shape)} "
      f"(~{table_fp8.numel()/2**20:.0f}MiB) | scales dtype={scales.dtype} shape={tuple(scales.shape)}")

save_file({"text_embedding": table_fp8, "text_embedding_scales": scales}, a.out)
print("wrote", a.out)
