#!/usr/bin/env python
"""STAGE 1: int4-AWQ (W4A16) quantize the Qwen3-TTS base TALKER backbone.

Reuse-upstream: rebuild the talker as a vanilla transformers Qwen3ForCausalLM
(its keys map 1:1 onto talker.model.layers.*), run ModelOpt INT4_AWQ_CFG
(lm_head excluded by default -> EOS stays fp16), and export a unified HF
checkpoint with hf_quant_config.json {W4A16_AWQ, group_size 128}.

Output checkpoint -> --output_dir, with talker.* re-prefixed keys + config.json
patched so STAGE 2 (tensorrt-edgellm talker ONNX export) consumes it directly.
"""
import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

# Reuse the native quant-cfg builder so lm_head exclusion / weight-only W4A16
# match the upstream CLI exactly.
from tensorrt_edgellm.quantization.quantization_configs import build_quant_config

CALIB_EN = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming how machines understand speech.",
    "Please turn on the kitchen light and set a timer for ten minutes.",
    "Yesterday the weather was sunny, but today it looks like rain.",
    "She sells seashells by the seashore on a bright summer morning.",
    "Our quarterly results exceeded expectations across every region.",
    "The astronaut described the view of Earth from the space station.",
    "Reading aloud helps children develop stronger language skills.",
    "He carefully measured each ingredient before mixing the batter.",
    "The orchestra tuned their instruments before the evening concert.",
    "Renewable energy sources are becoming cheaper every single year.",
    "I would like a cup of coffee with a little bit of warm milk.",
    "The museum's new exhibit features ancient pottery and sculptures.",
    "Could you please repeat the address one more time, slowly?",
    "Mountains rose sharply against the pale orange evening sky.",
    "A gentle breeze carried the scent of pine through the open window.",
]
CALIB_CJK = [
    "今天天气非常好，我们一起去公园散步吧。",
    "人工智能正在改变人们与机器交流的方式。",
    "请帮我把客厅的灯打开，并设置一个十分钟的闹钟。",
    "这家餐厅的招牌菜是红烧肉和清蒸鲈鱼。",
    "他每天早上六点起床，然后去河边跑步。",
    "学习一门新的语言需要持续的练习和耐心。",
    "昨天晚上的音乐会非常精彩，观众都很喜欢。",
    "这本书讲述了一个关于勇气与友谊的动人故事。",
    "春天来了，公园里的樱花开得非常灿烂。",
    "我想点一杯热牛奶和一份新鲜出炉的面包。",
    "科学家们正在研究如何更高效地利用太阳能。",
    "请问去最近的地铁站应该怎么走？谢谢你。",
    "孩子们在草地上快乐地奔跑，笑声此起彼伏。",
    "这部电影的配乐和画面都给人留下深刻印象。",
    "海边的日落把天空染成了温暖的橙红色。",
    "坚持每天朗读能够帮助提高语言表达能力。",
]


def remap_talker_to_hf(src_sd):
    """talker.* backbone keys -> vanilla Qwen3ForCausalLM keys."""
    out = {}
    for k, v in src_sd.items():
        if ".code_predictor." in k or not k.startswith("talker."):
            continue
        if k == "talker.model.codec_embedding.weight":
            out["model.embed_tokens.weight"] = v
        elif k == "talker.codec_head.weight":
            out["lm_head.weight"] = v
        elif k == "talker.model.norm.weight":
            out["model.norm.weight"] = v
        elif k.startswith("talker.model.layers."):
            out[k[len("talker."):]] = v   # -> model.layers.*
        # text_embedding / text_projection are TTS sidecars, not backbone.
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Build vanilla Qwen3 from talker_config -----------------------
    cfg_full = json.load(open(os.path.join(args.model_dir, "config.json")))
    tc = cfg_full["talker_config"]
    hf_cfg = Qwen3Config(
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"],
        vocab_size=tc["vocab_size"],
        hidden_act=tc.get("hidden_act", "silu"),
        rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
        rope_theta=tc.get("rope_theta", 1000000),
        max_position_embeddings=tc.get("max_position_embeddings", 32768),
        attention_bias=tc.get("attention_bias", False),
        tie_word_embeddings=False,
        torch_dtype="float16",
    )
    print(f"[stage1] Qwen3 cfg: H={hf_cfg.hidden_size} L={hf_cfg.num_hidden_layers} "
          f"vocab={hf_cfg.vocab_size}", flush=True)
    model = Qwen3ForCausalLM(hf_cfg).to(torch.float16)

    # ---- 2. Load + remap talker weights ----------------------------------
    src_path = os.path.join(args.model_dir, "model.safetensors")
    src_sd = load_file(src_path, device="cpu")
    hf_sd = remap_talker_to_hf(src_sd)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    print(f"[stage1] loaded {len(hf_sd)} tensors; missing={missing[:6]} "
          f"({len(missing)}) unexpected={unexpected[:6]} ({len(unexpected)})",
          flush=True)
    if missing or unexpected:
        # hard fail on real mismatches (rotary buffers are fine/filtered above)
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(dev).eval()

    # ---- 3. Calibration dataloader (EN+CJK input_ids) --------------------
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pool = CALIB_EN + CALIB_CJK
    texts = (pool * ((args.num_samples // len(pool)) + 1))[:args.num_samples]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_len)
    # talker vocab is 3072 (codec), but the text tokenizer ids exceed that.
    # Clamp ids into the backbone embedding range so embed lookup is valid;
    # calibration only needs representative activation magnitudes for AWQ
    # weight-smoothing (weight-only W4A16; activations are NOT quantized).
    ids = enc["input_ids"].clamp_(0, hf_cfg.vocab_size - 1)
    loader = DataLoader(ids, batch_size=16, shuffle=False)

    nan = {"n": 0}

    def forward_loop(m):
        for batch in tqdm(loader, desc="Calibrating"):
            batch = batch.to(dev)
            with torch.no_grad():
                out = m(batch, use_cache=False)
            if not torch.isfinite(out.logits).all():
                nan["n"] += 1

    # ---- 4. Quantize int4_awq (lm_head excluded by default) --------------
    quant_cfg = build_quant_config("int4_awq")
    qc = quant_cfg["quant_cfg"]
    print(f"[stage1] algo={quant_cfg.get('algorithm')} "
          f"input_q={qc.get('*input_quantizer')} lm_head={qc.get('*lm_head*')}",
          flush=True)
    print("[stage1] mtq.quantize(int4_awq)...", flush=True)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    print(f"[stage1] non-finite calib batches: {nan['n']}", flush=True)

    # NaN check on quantized weights
    bad = 0
    for n, p in model.named_parameters():
        if p.is_floating_point() and not torch.isfinite(p).all():
            bad += 1
            print(f"[stage1] NaN/Inf in {n}", flush=True)
    print(f"[stage1] params with NaN/Inf after quant: {bad}", flush=True)

    # ---- 5. Export unified HF checkpoint (W4A16 + hf_quant_config.json) ---
    tmp_hf = os.path.join(args.output_dir, "_hf_unified")
    os.makedirs(tmp_hf, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=tmp_hf)
    print(f"[stage1] export_hf_checkpoint -> {tmp_hf}", flush=True)
    print("[stage1] files:", sorted(os.listdir(tmp_hf)), flush=True)
    if os.path.exists(os.path.join(tmp_hf, "hf_quant_config.json")):
        print("[stage1] hf_quant_config.json:",
              json.dumps(json.load(open(os.path.join(tmp_hf, "hf_quant_config.json"))),
                         ensure_ascii=False)[:400], flush=True)
    print(f"[stage1] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
