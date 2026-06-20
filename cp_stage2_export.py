#!/usr/bin/env python
"""STAGE 2 (CodePredictor): assemble int4 CP checkpoint + run native CP ONNX export.

Re-prefixes cp-stage1's _hf_unified_cp W4A16 backbone back to
talker.code_predictor.*, merges the untouched fp16 CP sidecars (15 codec
embedding tables, 15 RVQ lm_heads, small_to_mtp_projection) from the original
base ckpt, copies root config.json + hf_quant_config.json, then runs the CP
ONNX export.

IMPORTANT FIX vs the stock _export_code_predictor: that function builds
ModelConfig from a *temp dir* containing ONLY the CP sub-config + symlinked
safetensors -> it never sees hf_quant_config.json, so it always builds the CP
as FP16 and the loader silently skips all int4 (.weight_scale/.pre_quant_scale/
packed .weight) tensors -> q_proj ends up wrong-shaped -> reshape failure.
We inline the export and drop hf_quant_config.json into the temp dir so
ModelConfig detects W4A16_AWQ and CodePredictorCausalLM builds int4 Linears.
"""
import argparse, json, os, shutil, sys, tempfile
import torch
from safetensors.torch import load_file, save_file

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)


def reprefix_unified_to_cp(uni_sd):
    out = {}
    for k, v in uni_sd.items():
        if k in ("model.embed_tokens.weight", "lm_head.weight"):
            continue  # dummies added only to build vanilla Qwen3
        if k == "model.norm.weight":
            out["talker.code_predictor.model.norm.weight"] = v
        elif k.startswith("model.layers."):
            out["talker.code_predictor.model.layers." + k[len("model.layers."):]] = v
        else:
            out["talker.code_predictor." + k] = v
    return out


def assemble(orig_model_dir, unified_dir, stage2_ckpt):
    os.makedirs(stage2_ckpt, exist_ok=True)
    uni = load_file(os.path.join(unified_dir, "model.safetensors"), device="cpu")
    cp_sd = reprefix_unified_to_cp(uni)
    n_backbone = len(cp_sd)
    print(f"[cp-stage2] re-prefixed {n_backbone} backbone tensors (int4 + fp norms/down_proj)", flush=True)

    orig = load_file(os.path.join(orig_model_dir, "model.safetensors"), device="cpu")
    n_emb = n_head = 0
    for i in range(15):
        ek = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        hk = f"talker.code_predictor.lm_head.{i}.weight"
        if ek not in orig or hk not in orig:
            raise SystemExit(f"missing CP sidecar: {ek if ek not in orig else hk}")
        cp_sd[ek] = orig[ek]; n_emb += 1
        cp_sd[hk] = orig[hk]; n_head += 1
    for pk in ("talker.code_predictor.small_to_mtp_projection.weight",
               "talker.code_predictor.small_to_mtp_projection.bias"):
        if pk in orig:
            cp_sd[pk] = orig[pk]
            print(f"[cp-stage2] carried {pk}", flush=True)

    save_file(cp_sd, os.path.join(stage2_ckpt, "model.safetensors"))
    print(f"[cp-stage2] wrote {len(cp_sd)} tensors "
          f"(backbone {n_backbone} + {n_emb} codec_emb + {n_head} lm_head) "
          f"-> {stage2_ckpt}/model.safetensors", flush=True)

    shutil.copy2(os.path.join(orig_model_dir, "config.json"),
                 os.path.join(stage2_ckpt, "config.json"))
    shutil.copy2(os.path.join(unified_dir, "hf_quant_config.json"),
                 os.path.join(stage2_ckpt, "hf_quant_config.json"))
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "generation_config.json", "preprocessor_config.json"):
        src = os.path.join(orig_model_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(stage2_ckpt, fn))
    print("[cp-stage2] stage2_ckpt files:", sorted(os.listdir(stage2_ckpt)), flush=True)


def export_cp_int4(model_dir, cp_out_dir):
    """Inlined _export_code_predictor with hf_quant_config.json staged into tmp_dir
    so ModelConfig detects W4A16_AWQ and builds int4 CP Linears."""
    from tensorrt_edgellm.scripts.export import (
        _load_config, _extract_code_predictor_weights, _CP_RUNTIME_MODEL_TYPE)
    from tensorrt_edgellm.config import ModelConfig
    from tensorrt_edgellm.models.qwen3_tts import (
        CodePredictorCausalLM, apply_code_predictor_mlp_war)
    from tensorrt_edgellm.checkpoint.loader import load_weights
    from tensorrt_edgellm.onnx.export import export_onnx

    os.makedirs(cp_out_dir, exist_ok=True)
    output_path = os.path.join(cp_out_dir, "model.onnx")

    root_config = _load_config(model_dir)
    talker_cfg = root_config.get("talker_config", {})
    cp_cfg = talker_cfg.get("code_predictor_config", {})
    assert cp_cfg.get("hidden_size"), "code_predictor_config not found"

    with tempfile.TemporaryDirectory() as tmp_dir:
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors") or fname.endswith(".safetensors.index.json"):
                dst = os.path.join(tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(model_dir, fname), dst)
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(cp_cfg, f)
        # THE FIX: stage hf_quant_config.json so ModelConfig detects W4A16_AWQ
        hqc = os.path.join(model_dir, "hf_quant_config.json")
        if os.path.isfile(hqc):
            shutil.copy2(hqc, os.path.join(tmp_dir, "hf_quant_config.json"))
            print("[cp-stage2] staged hf_quant_config.json into CP tmp_dir", flush=True)
        config = ModelConfig.from_pretrained(tmp_dir)

    config.model_type = _CP_RUNTIME_MODEL_TYPE.get("qwen3_tts", "qwen3_tts_code_predictor")
    print(f"[cp-stage2] CP ModelConfig: quant_type={config.quant.quant_type} "
          f"group_size={config.quant.group_size} "
          f"n_excluded={len(config.quant.excluded)} excluded[:3]={config.quant.excluded[:3]}",
          flush=True)

    model = CodePredictorCausalLM(config)
    model.to("cpu")
    load_weights(model, model_dir, device="cpu", key_prefix="talker.code_predictor.")
    apply_code_predictor_mlp_war(model)

    print(f"[cp-stage2] export_onnx -> {output_path}", flush=True)
    export_onnx(model, output_path, model_dir=model_dir)

    _extract_code_predictor_weights(model_dir, cp_out_dir, talker_cfg)

    cfg_path = os.path.join(cp_out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["use_embeddings_input"] = True
        cfg["num_code_groups"] = talker_cfg.get("num_code_groups", 16)
        cfg["num_deepstack_features"] = 0
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print("[cp-stage2] CP export done. files:", sorted(os.listdir(cp_out_dir)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_model_dir", required=True)
    ap.add_argument("--unified_dir", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--onnx_out", required=True)
    args = ap.parse_args()
    assemble(args.orig_model_dir, args.unified_dir, args.stage2_ckpt)
    export_cp_int4(args.stage2_ckpt, args.onnx_out)


if __name__ == "__main__":
    main()
