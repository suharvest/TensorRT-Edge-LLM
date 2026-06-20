#!/usr/bin/env python
"""STAGE 1 (CodePredictor): int4-AWQ (W4A16) quantize the CP transformer backbone.

The CodePredictor (CP) is a small 5-layer Qwen3 decoder (hidden=1024,
intermediate=3072, vocab=2048). It predicts the 15 residual RVQ codec
sub-codes per acoustic frame -> ~15 backbone forwards/frame, the dominant
per-frame cost.

Mirror of the talker stage1, adapted:
  * Rebuild the CP backbone as a vanilla Qwen3ForCausalLM from
    talker_config.code_predictor_config; remap
    talker.code_predictor.model.layers.* -> model.layers.*
    talker.code_predictor.model.norm.weight -> model.norm.weight.
  * The CP has NO single embed_tokens (it takes inputs_embeds directly) and
    NO single lm_head (15 dynamic RVQ heads, ONNX inputs). We give the vanilla
    model dummy embed_tokens/lm_head purely to construct it; neither is
    quantized (lm_head excluded; embeddings aren't Linear).
  * Calibration: feed RANDOM fp16 inputs_embeds [B, seq, 1024] straight into
    the backbone (AWQ is weight-only -> we only need representative activation
    magnitudes through the body Linears; no codec embeddings needed).
  * EXCLUDE: *lm_head*  AND  *down_proj*  from quantization.
      - lm_head: codec heads -> kept fp16 (codec-token fidelity), analogous to talker.
      - down_proj: CP down_proj is deliberately FP32 in the export
        (silu*up intermediates reach ~[-39,72]); int4 there would break codec-EOS.
  * Export _hf_unified W4A16 backbone for STAGE 2 reassembly.
"""
import argparse, copy, json, os, time
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM

from tensorrt_edgellm.quantization.quantization_configs import build_quant_config


def remap_cp_to_hf(src_sd):
    """talker.code_predictor.* backbone keys -> vanilla Qwen3ForCausalLM keys.

    Only the transformer body + final norm are carried. codec_embedding.*,
    lm_head.*, small_to_mtp_projection are CP sidecars (NOT backbone) and are
    left untouched (re-merged fp16 in stage2)."""
    PREFIX = "talker.code_predictor."
    out = {}
    for k, v in src_sd.items():
        if not k.startswith(PREFIX):
            continue
        sub = k[len(PREFIX):]
        if sub.startswith("model.codec_embedding."):
            continue   # sidecar
        if sub.startswith("lm_head."):
            continue   # sidecar (dynamic RVQ heads)
        if sub.startswith("small_to_mtp_projection"):
            continue   # sidecar
        if sub == "model.norm.weight":
            out["model.norm.weight"] = v
        elif sub.startswith("model.layers."):
            out["model.layers." + sub[len("model.layers."):]] = v
        # nothing else expected in the backbone
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Build vanilla Qwen3 from code_predictor_config ----------------
    cfg_full = json.load(open(os.path.join(args.model_dir, "config.json")))
    tc = cfg_full["talker_config"]
    cp = tc["code_predictor_config"]
    hf_cfg = Qwen3Config(
        hidden_size=cp["hidden_size"],
        num_hidden_layers=cp["num_hidden_layers"],
        num_attention_heads=cp["num_attention_heads"],
        num_key_value_heads=cp["num_key_value_heads"],
        head_dim=cp["head_dim"],
        intermediate_size=cp["intermediate_size"],
        vocab_size=cp["vocab_size"],
        hidden_act=cp.get("hidden_act", "silu"),
        rms_norm_eps=cp.get("rms_norm_eps", 1e-6),
        rope_theta=cp.get("rope_theta", 1000000),
        max_position_embeddings=cp.get("max_position_embeddings", 65536),
        attention_bias=cp.get("attention_bias", False),
        tie_word_embeddings=False,
        torch_dtype="float16",
    )
    print(f"[cp-stage1] Qwen3 cfg: H={hf_cfg.hidden_size} L={hf_cfg.num_hidden_layers} "
          f"inter={hf_cfg.intermediate_size} vocab={hf_cfg.vocab_size}", flush=True)
    model = Qwen3ForCausalLM(hf_cfg).to(torch.float16)

    # ---- 2. Load + remap CP backbone weights ------------------------------
    src_sd = load_file(os.path.join(args.model_dir, "model.safetensors"), device="cpu")
    hf_sd = remap_cp_to_hf(src_sd)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    # embed_tokens.weight + lm_head.weight are dummies (not in CP) -> expected missing
    missing = [m for m in missing
               if ("rotary" not in m and "inv_freq" not in m
                   and m != "model.embed_tokens.weight" and m != "lm_head.weight")]
    print(f"[cp-stage1] loaded {len(hf_sd)} backbone tensors; "
          f"residual-missing={missing[:8]}({len(missing)}) "
          f"unexpected={unexpected[:8]}({len(unexpected)})", flush=True)
    if missing or unexpected:
        raise SystemExit(f"CP backbone state_dict mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(dev).eval()

    # ---- 3. Calibration: random fp16 inputs_embeds straight into backbone --
    torch.manual_seed(0)
    # match real CP hidden-state scale (residual-stream RMS ~ O(1)); std=1 is a
    # safe representative magnitude for weight-only AWQ smoothing.
    data = torch.randn(args.num_samples, args.seq_len, hf_cfg.hidden_size,
                       dtype=torch.float16)
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=False)
    nan = {"n": 0}

    def forward_loop(m):
        for (batch,) in tqdm(loader, desc="Calibrating-CP"):
            batch = batch.to(dev)
            with torch.no_grad():
                # drive the backbone via inputs_embeds (CP has no token ids)
                out = m.model(inputs_embeds=batch, use_cache=False)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            if not torch.isfinite(h).all():
                nan["n"] += 1

    # ---- 4. quant cfg: int4_awq + EXCLUDE down_proj (and lm_head default) --
    quant_cfg = copy.deepcopy(build_quant_config("int4_awq"))
    # CP down_proj is FP32-critical for codec-EOS -> never int4 it.
    quant_cfg["quant_cfg"]["*down_proj*"] = {"enable": False}
    qc = quant_cfg["quant_cfg"]
    print(f"[cp-stage1] algo={quant_cfg.get('algorithm')}", flush=True)
    print(f"[cp-stage1] exclusions: lm_head={qc.get('*lm_head*')} "
          f"down_proj={qc.get('*down_proj*')} input_q={qc.get('*input_quantizer')}", flush=True)
    print("[cp-stage1] mtq.quantize(int4_awq, down_proj excluded)...", flush=True)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    print(f"[cp-stage1] non-finite calib batches: {nan['n']}", flush=True)

    # which Linears actually got a weight_quantizer w/ amax (=quantized)?
    q_layers, fp_layers = [], []
    for n, mod in model.named_modules():
        wq = getattr(mod, "weight_quantizer", None)
        if wq is not None:
            if getattr(wq, "is_enabled", False) or getattr(wq, "_enabled", False):
                q_layers.append(n)
            else:
                fp_layers.append(n)
    print(f"[cp-stage1] quantized Linears ({len(q_layers)}): "
          f"{[x for x in q_layers][:8]} ...", flush=True)
    print(f"[cp-stage1] kept-fp Linears ({len(fp_layers)}): {fp_layers[:8]}", flush=True)

    bad = 0
    for n, p in model.named_parameters():
        if p.is_floating_point() and not torch.isfinite(p).all():
            bad += 1; print(f"[cp-stage1] NaN/Inf in {n}", flush=True)
    print(f"[cp-stage1] params with NaN/Inf after quant: {bad}", flush=True)

    # ---- 5. Export unified HF checkpoint -----------------------------------
    tmp_hf = os.path.join(args.output_dir, "_hf_unified_cp")
    os.makedirs(tmp_hf, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=tmp_hf)
    print(f"[cp-stage1] export_hf_checkpoint -> {tmp_hf}", flush=True)
    print("[cp-stage1] files:", sorted(os.listdir(tmp_hf)), flush=True)
    hqc = os.path.join(tmp_hf, "hf_quant_config.json")
    if os.path.exists(hqc):
        print("[cp-stage1] hf_quant_config.json:",
              json.dumps(json.load(open(hqc)), ensure_ascii=False)[:400], flush=True)
    print(f"[cp-stage1] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
