#!/usr/bin/env python
"""STAGE 2: assemble the int4 talker checkpoint + run the native talker ONNX export.

Takes stage-1's _hf_unified W4A16 backbone, re-prefixes keys back to talker.*,
merges the untouched fp16 TTS sidecars (text_embedding, text_projection) from the
original base checkpoint, copies the original qwen3_tts config.json +
hf_quant_config.json, then calls the EXISTING _export_talker so ModelConfig
detects W4A16_AWQ -> make_linear builds ModelOptAWQPrepackedLinear ->
export_onnx emits Int4GroupwiseGemmPlugin nodes.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def reprefix_unified_to_talker(uni_sd):
    """model.* (HF) backbone -> talker.* (export) keys, preserving int4 buffers."""
    out = {}
    for k, v in uni_sd.items():
        if k == "model.embed_tokens.weight":
            out["talker.model.codec_embedding.weight"] = v
        elif k == "lm_head.weight":
            out["talker.codec_head.weight"] = v
        elif k == "model.norm.weight":
            out["talker.model.norm.weight"] = v
        elif k.startswith("model.layers."):
            # keeps .weight / .weight_scale / .pre_quant_scale suffixes
            out["talker." + k] = v
        else:
            out["talker." + k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_model_dir", required=True)
    ap.add_argument("--unified_dir", required=True)
    ap.add_argument("--stage2_ckpt", required=True)   # assembled int4 talker ckpt
    ap.add_argument("--onnx_out", required=True)       # final ONNX dir
    args = ap.parse_args()

    os.makedirs(args.stage2_ckpt, exist_ok=True)

    # ---- merge weights ---------------------------------------------------
    uni = load_file(os.path.join(args.unified_dir, "model.safetensors"), device="cpu")
    talker_sd = reprefix_unified_to_talker(uni)

    # carry untouched fp16 TTS sidecars from the original base checkpoint
    orig = load_file(os.path.join(args.orig_model_dir, "model.safetensors"), device="cpu")
    sidecar_keys = [
        "talker.model.text_embedding.weight",
        "talker.text_projection.linear_fc1.weight",
        "talker.text_projection.linear_fc1.bias",
        "talker.text_projection.linear_fc2.weight",
        "talker.text_projection.linear_fc2.bias",
    ]
    for k in sidecar_keys:
        if k in orig:
            talker_sd[k] = orig[k]
        else:
            raise SystemExit(f"missing sidecar in orig ckpt: {k}")

    save_file(talker_sd, os.path.join(args.stage2_ckpt, "model.safetensors"))
    print(f"[stage2] wrote {len(talker_sd)} tensors -> {args.stage2_ckpt}/model.safetensors",
          flush=True)

    # ---- config.json (original qwen3_tts) + hf_quant_config.json ---------
    shutil.copy2(os.path.join(args.orig_model_dir, "config.json"),
                 os.path.join(args.stage2_ckpt, "config.json"))
    shutil.copy2(os.path.join(args.unified_dir, "hf_quant_config.json"),
                 os.path.join(args.stage2_ckpt, "hf_quant_config.json"))
    # copy tokenizer / aux files if present (harmless, used by export sidecar copy)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "chat_template.jinja", "generation_config.json",
               "preprocessor_config.json"):
        src = os.path.join(args.orig_model_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.stage2_ckpt, fn))
    print("[stage2] stage2_ckpt files:", sorted(os.listdir(args.stage2_ckpt)), flush=True)

    # sanity: ModelConfig must detect W4A16
    from tensorrt_edgellm.config import ModelConfig
    mc = ModelConfig.from_pretrained(args.stage2_ckpt)
    print(f"[stage2] ModelConfig.quant.quant_type = {mc.quant.quant_type} "
          f"group_size={mc.quant.group_size} excluded={mc.quant.excluded}", flush=True)

    # ---- run the native talker ONNX export -------------------------------
    from tensorrt_edgellm.scripts.export import _export_talker
    print(f"[stage2] _export_talker -> {args.onnx_out}", flush=True)
    _export_talker(args.stage2_ckpt, args.onnx_out, "qwen3_tts")
    print("[stage2] export done. files:", sorted(os.listdir(args.onnx_out)), flush=True)


if __name__ == "__main__":
    main()
