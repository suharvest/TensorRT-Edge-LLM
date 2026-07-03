# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-Omni Thinker + Talker NVFP4 quantization driver.

End-to-end NVFP4 quantization for both the Thinker text-MoE subgraph and
the Talker text-MoE subgraph of Qwen3-Omni-30B-A3B-Instruct. Produces two
standalone HF checkpoints in <output_dir>:

  thinker/   model_type=qwen3_omni_moe_text    (H=2048, 128 experts top-8)
  talker/    model_type=qwen3_omni_moe_talker  (H=1024, 128 experts top-6 + shared expert)

Audio encoder, visual encoder, code2wav vocoder, Talker projections, and
the talker code_predictor/output heads stay in FP16 and are exported
separately by ``experimental.llm_loader.export_all_cli``.

Calibration is **joint and multimodal**: a single ``mtq.quantize(model, ...)``
call observes both Thinker and Talker quantizers in the same forward loop.
Per sample:

  1. Thinker(audio / image / text) with ``output_hidden_states=True`` --
     Thinker quantizers observe true FP16 activations (modelopt's
     observe-only mode does not apply QDQ during calibration).
  2. ``thinker_hidden = hidden_states[accept_hidden_layer]`` and
     ``thinker_embed = hidden_states[0]`` (= inputs_embeds) feed
     ``talker.hidden_projection`` / ``talker.text_projection`` to produce
     realistic Talker ``inputs_embeds`` (same data flow as
     ``buildTalkerPrefillFromSegments`` in the C++ runtime).
  3. Talker(inputs_embeds=..., talker_input_ids=thinker_input_ids,
     attention_mask=...) -- Talker quantizers observe activations.

This mirrors the dense Qwen3-Omni quantization convention used by
``tensorrt_edgellm.quantization.llm_quantization.quantize_llm`` /
``omni_quantization.omni_multimodal_calib_loop``.

Usage::

    tensorrt-edgellm-quantize qwen3-omni \\
        --model_dir Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --output_dir ./out_qwen3_omni_nvfp4 \\
        --kv_cache_quantization fp8 \\
        --num_samples 64
"""

import json
import os
import time
from typing import Optional, Tuple

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .quantization_configs import build_quant_config
# Reuse Thinker-side helpers from the dedicated thinker driver.
from .qwen3_omni_thinker import _extract_and_save_thinker_text

# ---------------------------------------------------------------------------
# Quant config — Thinker + Talker text path
# ---------------------------------------------------------------------------


def _build_full_model_quant_cfg(lm_head_quantization: Optional[str],
                                kv_cache_quantization: Optional[str]) -> dict:
    """Build a NVFP4 quant_cfg for the full ``Qwen3OmniMoeForConditionalGeneration``.

    Used by a single joint ``mtq.quantize(model, cfg, forward_loop=...)`` call
    that calibrates both Thinker and Talker subgraphs in one multimodal pass
    (mirrors the dense Qwen3-Omni convention in
    ``tensorrt_edgellm.quantization.llm_quantization.quantize_llm``).  Because
    the recipe is applied to the full model, all module paths in the
    ``disable_globs`` carry the ``thinker.`` / ``talker.`` prefix; this avoids
    cross-contamination between submodels that bare wildcards would risk.

    Disables: visual encoder, audio encoder, code2wav, talker sidecar
    projections/code-predictor/output heads, all MoE routers and
    shared-expert gates.
    """
    cfg = build_quant_config("nvfp4", lm_head_quantization,
                             kv_cache_quantization)

    # Optional AWQ-Lite calibration: rebalances per-channel weight scales
    # against activation outliers. modelopt's NVFP4_DEFAULT_CFG and
    # NVFP4_AWQ_LITE_CFG share an identical ``quant_cfg`` (per-block FP4
    # weight + FP8 scale); they differ only in ``algorithm`` -- ``"max"``
    # runs plain RTN absmax calibration, ``"awq_lite"`` runs an additional
    # per-channel AWQ scale search. Output checkpoint format is identical;
    # the same NVFP4 kernels consume either.
    #
    # Opted in via env var to keep RTN as the default. ``QWEN3_OMNI_AWQ_ALPHA_STEP``
    # defaults to 0.25 (5 alpha values across [0, 1]) because modelopt's
    # default of 0.1 (11 alpha values) is prohibitively slow on this MoE
    # (~10 s per multimodal sample -> ~18 h for 500 samples * 11 alphas).
    # AWQ typically lands best_alpha in [0.3, 0.7], so a 0.25 step still
    # resolves the optimum.
    if os.environ.get("QWEN3_OMNI_USE_AWQ_LITE") == "1":
        alpha_step = float(os.environ.get("QWEN3_OMNI_AWQ_ALPHA_STEP", "0.25"))
        cfg["algorithm"] = {"method": "awq_lite", "alpha_step": alpha_step}
        print(
            f"[Omni quant] AWQ-Lite enabled: method=awq_lite, "
            f"alpha_step={alpha_step}",
            flush=True)

    disable_globs = [
        # ---- Thinker non-LLM submodules (exported separately, FP16) ----
        "*thinker.visual.*",  # Visual encoder (handled by visual.py path)
        "*thinker.audio_tower.*",  # Whisper-style audio encoder
        "*thinker.lm_head*",  # Final LM head — FP16 for logit fidelity
        # ---- Talker non-LLM submodules (FP16) ----
        "*talker.code_predictor.*",  # CodePredictor head + transformer
        "*talker.code2wav.*",  # Code2Wav vocoder (not actually attached
        #   to the talker subtree in HF but kept
        #   here defensively for any wrappers)
        "*talker.hidden_projection*",  # Thinker hidden -> talker-space MLP
        "*talker.text_projection*",  # Thinker embed  -> talker-space MLP
        "*talker.codec_head*",  # Codec output projection (FP16: feeds
        #   Code2Wav; FP4 noise here materially
        #   hurts audio quality)
        # ---- MoE routers and shared-expert gates (FP16 for top-k stability) ----
        # The router is a tiny [H × num_experts] linear; its output drives
        # discrete top-k selection so even small quant noise can flip experts
        # and propagate large output errors.
        "*mlp.gate.*",
        # ``shared_expert_gate`` is a 1-output Linear producing a sigmoid
        # mixer scalar; quantizing it directly biases per-token shared-expert
        # contribution.
        "*shared_expert_gate.*",
        # ``shared_expert`` FFN runs at NVFP4: a layer-wise SNR audit showed
        # its contribution to downstream noise is negligible (Δcos < 2e-4
        # at the late layers), so the quantization-sensitivity budget is
        # better spent on the routed experts. Uncomment the three globs
        # below to put it back at FP16 if a future model proves more
        # sensitive.
        # "*shared_expert.gate_proj.*",
        # "*shared_expert.up_proj.*",
        # "*shared_expert.down_proj.*",
    ]
    # modelopt's ``quant_cfg`` is an ordered list of rule dicts; append a
    # disable rule per glob at the end so it overrides the earlier
    # ``*weight_quantizer`` / ``*input_quantizer`` enables. A ``dict``
    # ``quant_cfg`` (``{pattern: cfg}``) is handled too.
    qc = cfg["quant_cfg"]
    for g in disable_globs:
        if isinstance(qc, list):
            qc.append({"quantizer_name": g, "enable": False})
        else:
            qc[g] = {"enable": False}
    return cfg


# ---------------------------------------------------------------------------
# Multimodal calibration for Talker
# ---------------------------------------------------------------------------


class _OmniMultimodalCalibDataset:
    """Mixed-modality calibration dataset for Qwen3-Omni Talker.

    Pre-encodes each sample with the HF processor so calibration just
    iterates ready-to-forward dicts.
    """

    def __init__(self, processor, audio_data, image_data, text_data):
        self.processor = processor
        self.samples = []
        for item in audio_data:
            self.samples.append({"type": "audio", "raw": item})
        for item in image_data:
            self.samples.append({"type": "image", "raw": item})
        for text in text_data:
            self.samples.append({"type": "text", "raw": text})

    def __len__(self):
        return len(self.samples)

    def _process_audio(self, raw):
        import io

        import soundfile as sf
        audio_item = raw["audio"]
        audio_bytes = audio_item.get("bytes")
        if audio_bytes is not None:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio, sr = sf.read(audio_item["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "audio",
                    "audio": "placeholder"
                },
                {
                    "type": "text",
                    "text": "Describe what you hear."
                },
            ],
        }]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _process_image(self, raw):
        from PIL import Image
        images = [
            v.convert("RGB") for k, v in raw.items()
            if "image" in k and isinstance(v, Image.Image)
        ]
        if not images:
            return self._process_text("Describe the scene.")
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": "Describe what you see."
                },
            ],
        }]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text,
                                images=images[:1],
                                return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _process_text(self, content):
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kind = sample["type"]
        if kind == "audio":
            return self._process_audio(sample["raw"])
        if kind == "image":
            return self._process_image(sample["raw"])
        return self._process_text(sample["raw"])


def _build_multimodal_calib_dataset(processor, num_audio: int, num_image: int,
                                    num_text: int):
    """Build the multimodal calibration dataset (audio + image + text).

    Setting any ``num_*`` to 0 skips that modality.
    """
    from datasets import load_dataset
    audio_data = []
    if num_audio > 0:
        from datasets import Audio
        print(f"[Omni calib] Loading audio (LibriSpeech), n={num_audio}")
        audio_stream = load_dataset("openslr/librispeech_asr",
                                    "clean",
                                    split="test",
                                    streaming=True)
        audio_stream = audio_stream.cast_column("audio", Audio(decode=False))
        audio_data = list(audio_stream.take(num_audio))

    image_data = []
    if num_image > 0:
        print(f"[Omni calib] Loading images (MMMU), n={num_image}")
        image_dataset = load_dataset("lmms-lab/MMMU", split="dev")
        image_data = list(
            image_dataset.select(range(min(num_image, len(image_dataset)))))

    text_data = []
    if num_text > 0:
        print(f"[Omni calib] Loading text (cnn_dailymail), n={num_text}")
        text_ds = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_data = text_ds["article"][:num_text]

    return _OmniMultimodalCalibDataset(processor, audio_data, image_data,
                                       text_data)


def _calib_full_multimodal(model, calib_dataset,
                           accept_hidden_layer: int) -> None:
    """Joint multimodal calibration forward loop for the full Omni-MoE model.

    Runs one Thinker forward and one Talker forward per sample so that
    BOTH submodels' quantizers observe activations in a single calibration
    pass.  Mirrors the dense Qwen3-Omni convention in
    :func:`tensorrt_edgellm.quantization.omni_quantization.omni_multimodal_calib_loop`.

    During calibration, modelopt's freshly-inserted quantizers are in
    observe-only mode (collecting amax statistics; no QDQ applied), so the
    Thinker forward returns true FP16 hidden states.  Those hidden states
    are projected through Talker's text/hidden projection MLPs (the actual
    runtime data flow handled by ``buildTalkerPrefillFromSegments`` in
    ``qwen3OmniTTSRuntime``), giving Talker quantizers the same input
    distribution they will see at inference.

    Args:
        model: Top-level ``Qwen3OmniMoeForConditionalGeneration``.
        calib_dataset: Mixed audio/image/text dataset built by
            :func:`_build_multimodal_calib_dataset`.
        accept_hidden_layer: Which Thinker layer feeds Talker's
            ``hidden_projection`` for multimodal token positions.  Must be
            the same value the C++ runtime reads from talker config
            (``mTalkerConfig.acceptHiddenLayer``) and the same value the
            Thinker ONNX export emits at ``last_pre_norm_hidden_states``.
    """
    device = next(model.parameters()).device
    thinker_config = model.config.thinker_config
    multimodal_token_ids = [
        getattr(thinker_config, field, -1)
        for field in ("audio_token_id", "image_token_id", "video_token_id")
    ]
    talker_dtype = next(model.talker.parameters()).dtype
    tc = model.talker.config.text_config

    skipped = 0
    for i in tqdm(range(len(calib_dataset)),
                  desc="Calibrating Omni (multimodal: thinker+talker)"):
        try:
            data = calib_dataset[i]
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"[Omni calib] Skipping sample {i}: {e}")
            continue
        data = {
            k:
            v.unsqueeze(0).to(
                device,
                dtype=model.thinker.dtype if v.is_floating_point() else None)
            for k, v in data.items()
        }
        with torch.no_grad():
            thinker_out = model.thinker(**data, output_hidden_states=True)
            all_hidden = thinker_out.hidden_states
            if all_hidden is None or len(all_hidden) <= accept_hidden_layer:
                raise RuntimeError(
                    f"Thinker returned "
                    f"{0 if all_hidden is None else len(all_hidden)} "
                    f"hidden-state tensors; "
                    f"accept_hidden_layer={accept_hidden_layer}")
            thinker_hidden = all_hidden[accept_hidden_layer]
            thinker_embed = all_hidden[0]

            input_ids = data["input_ids"]
            mm_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in multimodal_token_ids:
                if token_id >= 0:
                    mm_mask |= (input_ids == token_id)

            seq_len = thinker_hidden.shape[1]
            inputs_embeds = torch.empty(1,
                                        seq_len,
                                        tc.hidden_size,
                                        dtype=talker_dtype,
                                        device=device)
            if mm_mask.any():
                inputs_embeds[mm_mask] = model.talker.hidden_projection(
                    thinker_hidden[mm_mask].to(talker_dtype))
            text_mask = ~mm_mask
            if text_mask.any():
                inputs_embeds[text_mask] = model.talker.text_projection(
                    thinker_embed[text_mask].to(talker_dtype))

            # ``talker_input_ids`` reuses the Thinker's input_ids verbatim,
            # matching HF's reference flow (modeling_qwen3_omni_moe.py:
            # ``_get_talker_user_parts``).  Talker only consumes this tensor
            # for 3D RoPE position bookkeeping (``get_rope_index`` checks
            # equality against ``audio_token_id`` / ``image_token_id`` etc.)
            # and never does an embedding lookup on it, so the IDs can stay
            # in Thinker's full vocab range even though it exceeds Talker's
            # codec vocab size.
            talker_ids = input_ids
            attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
            model.talker(inputs_embeds=inputs_embeds,
                         attention_mask=attn_mask,
                         talker_input_ids=talker_ids)

    if skipped:
        print(f"[Omni calib] Skipped {skipped}/{len(calib_dataset)} samples "
              f"due to processor errors")


# ---------------------------------------------------------------------------
# Standalone Talker checkpoint extraction
# ---------------------------------------------------------------------------


def _write_standalone_talker_config(talker_text_dict: dict, talker_dict: dict,
                                    root_dict: dict, output_dir: str) -> None:
    """Drop a standalone ``config.json`` with ``model_type=qwen3_omni_moe_talker``.

    The Talker config is the union of ``talker_config.text_config`` (the
    main MoE backbone) and the talker-level fields needed by downstream
    consumers (codec ids, accept_hidden_layer, thinker_hidden_size, etc.).
    """
    cfg = dict(talker_text_dict)
    talker_fields = (
        "accept_hidden_layer",
        "audio_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "codec_bos_id",
        "codec_eos_token_id",
        "codec_language_id",
        "codec_nothink_id",
        "codec_pad_id",
        "codec_think_bos_id",
        "codec_think_eos_id",
        "codec_think_id",
        "num_code_groups",
        "speaker_id",
        "thinker_hidden_size",
        "position_id_per_seconds",
        "seconds_per_chunk",
        "spatial_merge_size",
        "output_router_logits",
    )
    for k in talker_fields:
        if k in talker_dict and k not in cfg:
            cfg[k] = talker_dict[k]
    for k in ("tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id",
              "user_token_id", "assistant_token_id", "system_token_id",
              "im_start_token_id", "im_end_token_id"):
        if k in root_dict and k not in cfg:
            cfg[k] = root_dict[k]
    if "text_vocab_size" not in cfg:
        thinker_cfg = root_dict.get("thinker_config", {}) or {}
        thinker_text = thinker_cfg.get("text_config", {}) or {}
        text_vocab_size = thinker_text.get("vocab_size") or thinker_cfg.get(
            "vocab_size")
        if text_vocab_size is not None:
            cfg["text_vocab_size"] = text_vocab_size
    speaker_map = cfg.get("speaker_id")
    if isinstance(speaker_map, dict) and speaker_map:
        cfg.setdefault("default_speaker_id", next(iter(speaker_map.values())))
        cfg.setdefault("available_speakers", list(speaker_map.keys()))
    cfg["model_type"] = "qwen3_omni_moe_talker"
    cfg["architectures"] = ["Qwen3OmniMoeTalkerCausalLM"]
    # Talker is a text-only MoE without visual deepstack. Explicit 0 prevents
    # the llm_loader substring fallback from misclassifying ``qwen3_omni_moe_talker``
    # as deepstack=3 (the substring match on "qwen3_omni" would otherwise hit),
    # which would bake 3 dangling deepstack input ports into the engine and
    # SIGSEGV in executePrefillStep when the caller passes an empty deepstack vector.
    cfg["num_deepstack_features"] = 0
    out_path = os.path.join(output_dir, "config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _extract_and_save_talker(model_dir: str, full_export_dir: str,
                             output_dir: str) -> None:
    """Extract the Talker text MoE backbone (``model.layers.*`` + norms +
    embed_tokens) from a full-talker NVFP4 export.

    Drops the ``code_predictor`` / projection side modules and normalizes the
    HF Talker text-backbone keys. The exported keys end up at:

        model.layers.{i}.mlp.experts._experts.{j}.{gate,up,down}_proj.*
        model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.*
        model.layers.{i}.self_attn.{q,k,v,o}_proj.*
        model.layers.{i}.{input,post_attention}_layernorm.weight
        model.codec_embedding.weight
        model.norm.weight
        codec_head.weight
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)

    SKIP_TOKENS = ("code_predictor.", "audio_tower.", "visual.", "code2wav.",
                   "text_projection.", "hidden_projection.")

    keep_state: dict = {}
    for fname in sorted(os.listdir(full_export_dir)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(full_export_dir, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(s in key for s in SKIP_TOKENS):
                    continue
                if key.startswith("talker.model."):
                    new_key = key[len("talker."):]
                elif key.startswith("talker.codec_head."):
                    new_key = key[len("talker."):]
                elif key.startswith("model.") and "code_predictor" not in key:
                    new_key = key
                elif key.startswith("codec_head."):
                    new_key = key
                else:
                    continue
                keep_state[new_key] = f.get_tensor(key)
    if not keep_state:
        raise RuntimeError(f"No Talker text-backbone tensors found under "
                           f"{full_export_dir}")
    print(f"[extract-talker] keeping {len(keep_state)} tensors")
    save_file(keep_state, os.path.join(output_dir, "model.safetensors"))

    # Standalone config: text_config + a handful of talker-level fields.
    full_cfg = AutoConfig.from_pretrained(model_dir,
                                          trust_remote_code=True).to_dict()
    talker_cfg = full_cfg["talker_config"]
    text_cfg = talker_cfg["text_config"]
    _write_standalone_talker_config(text_cfg, talker_cfg, full_cfg, output_dir)

    # hf_quant_config.json — keep only patterns relevant to the talker
    # backbone, rewrite ``talker.model.`` -> ``model.``.
    src_hf = os.path.join(full_export_dir, "hf_quant_config.json")
    if os.path.isfile(src_hf):
        with open(src_hf) as f:
            hfq = json.load(f)

        def _rewrite_pattern(p: str) -> Optional[str]:
            if any(skip in p for skip in ("audio_tower", "visual", "code2wav",
                                          "code_predictor", "text_projection",
                                          "hidden_projection")):
                return None
            if "thinker" in p:
                return None
            return p.replace("talker.model.",
                             "model.").replace("talker.codec_head.",
                                               "codec_head.")

        for list_key in ("exclude_modules", ):
            if list_key in hfq and isinstance(hfq[list_key], list):
                hfq[list_key] = [
                    p for p in (_rewrite_pattern(x) for x in hfq[list_key])
                    if p is not None
                ]
        with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
            json.dump(hfq, f, indent=2)

    # chat_template.json. HF Qwen3-Omni stores its Jinja chat template in a
    # dedicated file (not in tokenizer_config.json), so a plain
    # ``tokenizer.save_pretrained`` would lose it and downstream
    # ``apply_chat_template`` would fall back to the generic
    # ``User:/Assistant:`` template.
    src_tpl = os.path.join(model_dir, "chat_template.json")
    if os.path.isfile(src_tpl):
        import shutil
        shutil.copyfile(src_tpl, os.path.join(output_dir,
                                              "chat_template.json"))


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def quantize_qwen3_omni(
    model_dir: str,
    output_dir: str,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    dataset: str = "cnn_dailymail",
    num_samples: int = 64,
    max_length: int = 64,
    talker_num_audio: int = 150,
    talker_num_image: int = 150,
    talker_num_text: int = 200,
    talker_accept_hidden_layer: Optional[int] = None,
    keep_full_export: bool = False,
) -> Tuple[str, str]:
    """Quantize Thinker + Talker text MoE to NVFP4 and save both standalone.

    Calibration is **joint and multimodal**: a single ``mtq.quantize(model,
    ...)`` call is made on the full ``Qwen3OmniMoeForConditionalGeneration``,
    with a forward loop that for each sample runs

        Thinker(multimodal input)  -- quantizers observe Thinker activations
          → hidden_projection / text_projection
          → Talker(projected inputs_embeds)  -- quantizers observe Talker
                                                activations

    in a single pass.  Both submodels' amax statistics come from the SAME
    realistic multimodal distribution, matching what
    ``tensorrt_edgellm.quantization.llm_quantization.quantize_llm`` does
    for the dense model.

    Notes:
      * ``num_samples`` / ``max_length`` / ``dataset`` are kept in the
        signature for backwards-compatible CLI but are now unused (the
        Thinker no longer runs a separate text-only calibration pass).
      * ``talker_accept_hidden_layer`` defaults to the value baked into
        ``config.talker_config.accept_hidden_layer``; the same integer is
        embedded into both the standalone Thinker config (so the ONNX
        export emits the correct hidden-states layer) and the standalone
        Talker config (so the C++ runtime fetches the right tensor).

    Returns (thinker_dir, talker_dir).
    """
    del num_samples, max_length, dataset  # Joint multimodal calib supersedes
    t0 = time.time()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    # 1. Load multimodal model.
    from transformers import Qwen3OmniMoeForConditionalGeneration
    print(f"[load] {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch_dtype,
        trust_remote_code=True).to(device).eval()
    if talker_accept_hidden_layer is None or talker_accept_hidden_layer < 0:
        talker_accept_hidden_layer = int(
            getattr(model.config.talker_config, "accept_hidden_layer"))
    print(f"[calib] accept_hidden_layer={talker_accept_hidden_layer}")

    # 2. Fused-Parameter MoE experts need no pre-quantization patching.
    #    On transformers >= 5.0 the Thinker/Talker experts are stored fused
    #    (``gate_up_proj`` / ``down_proj`` 3-D ``nn.Parameter``); ModelOpt 0.44
    #    quantizes and exports them natively via its on-the-fly fused-expert
    #    plugins, emitting standard per-expert NVFP4 keys
    #    (``experts.{j}.{gate,up,down}_proj.{weight,weight_scale,...}``).

    # 3. Joint multimodal calibration in a single ``mtq.quantize`` call on
    #    the full model.  The forward loop runs Thinker → projection →
    #    Talker per sample; during calibration all quantizers are in
    #    observe-only mode, so the Thinker output is genuine FP16 and the
    #    Talker sees the same input distribution it will encounter at
    #    inference time.
    if not is_quantized(model.thinker) or not is_quantized(model.talker):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True)
        calib_ds = _build_multimodal_calib_dataset(processor,
                                                   num_audio=talker_num_audio,
                                                   num_image=talker_num_image,
                                                   num_text=talker_num_text)
        print(f"[calib] multimodal dataset: {len(calib_ds)} samples "
              f"(joint thinker + talker calibration)")

        cfg = _build_full_model_quant_cfg(lm_head_quantization,
                                          kv_cache_quantization)
        mtq.quantize(
            model,
            cfg,
            forward_loop=lambda m: _calib_full_multimodal(
                model, calib_ds, talker_accept_hidden_layer),
        )
        mtq.print_quant_summary(model.thinker)
        mtq.print_quant_summary(model.talker)
    print(f"[quant] {time.time() - t0:.1f}s")

    # 3c. Save modelopt state for PyTorch QDQ verification.
    # Persists the full quantizer state (all amax / scale / on/off flags) so
    # the v4_joint-equivalent quant ckpt can be reloaded into an FP16 HF model
    # via ``mto.restore``, after which forward runs in QDQ emulation mode --
    # numerically equivalent to hardware NVFP4 to a few ULP.  This lets us
    # validate the quant ckpt independent of the TRT export / CuTeDSL kernel
    # path (see ``verify_quant_ckpt.py``).
    import modelopt.torch.opt as mto
    os.makedirs(output_dir, exist_ok=True)
    mto_state_path = os.path.join(output_dir, "modelopt_state.pt")
    mto.save(model, mto_state_path)
    print(f"[mto] saved modelopt state to {mto_state_path}")

    # 4. Export Thinker subgraph.
    thinker_dir = os.path.join(output_dir, "thinker")
    os.makedirs(thinker_dir, exist_ok=True)
    full_thinker = thinker_dir + (".keep" if keep_full_export else ".staging")
    os.makedirs(full_thinker, exist_ok=True)
    _export_submodel(model, "thinker", full_thinker)
    _extract_and_save_thinker_text(model_dir, full_thinker, thinker_dir)
    tokenizer.save_pretrained(thinker_dir)
    if not keep_full_export:
        import shutil
        shutil.rmtree(full_thinker, ignore_errors=True)

    # 5. Export Talker subgraph.
    talker_dir = os.path.join(output_dir, "talker")
    os.makedirs(talker_dir, exist_ok=True)
    full_talker = talker_dir + (".keep" if keep_full_export else ".staging")
    os.makedirs(full_talker, exist_ok=True)
    _export_submodel(model, "talker", full_talker)
    _extract_and_save_talker(model_dir, full_talker, talker_dir)
    tokenizer.save_pretrained(talker_dir)
    if not keep_full_export:
        import shutil
        shutil.rmtree(full_talker, ignore_errors=True)

    print(f"[done] thinker={thinker_dir}  talker={talker_dir}  "
          f"(total {time.time() - t0:.1f}s)")
    return thinker_dir, talker_dir


# ---------------------------------------------------------------------------
# Submodule export helper
# ---------------------------------------------------------------------------


def _export_submodel(model, which: str, full_dir: str) -> None:
    """``export_hf_checkpoint`` either the Thinker or Talker submodule.

    Detaches the unused sibling encoders, forces ``architectures``, and
    monkey-patches ``modelopt.torch.export.model_utils.is_multimodal_model``
    to False so ModelOpt's resmooth dummy-forward walks the plain CausalLM
    path. For Talker, also wraps ``forward`` with default
    ``inputs_embeds`` / ``attention_mask`` / ``talker_input_ids`` so the
    dummy walk does not crash on un-derivable tensors.
    """
    sub = getattr(model, which)
    saved_attrs: dict = {}
    saved_cfgs: dict = {}
    saved_forward = None
    saved_tied_keys = None
    _tied_keys_was_patched = False

    if which == "thinker":
        if sub.config.architectures is None:
            sub.config.architectures = ["Qwen3MoeForCausalLM"]
        for attr in ("audio_tower", "visual"):
            if hasattr(sub, attr) and getattr(sub, attr) is not None:
                saved_attrs[attr] = getattr(sub, attr)
                setattr(sub, attr, None)
        for cfg_attr in ("vision_config", "audio_config"):
            if hasattr(sub.config, cfg_attr) and getattr(sub.config,
                                                         cfg_attr) is not None:
                saved_cfgs[cfg_attr] = getattr(sub.config, cfg_attr)
                delattr(sub.config, cfg_attr)
    elif which == "talker":
        if sub.config.architectures is None:
            sub.config.architectures = ["Qwen3MoeForCausalLM"]

        # HF ``Qwen3OmniMoeTalkerForConditionalGeneration`` declares
        #   _tied_weights_keys = {"codec_head": "model.codec_embedding.weight"}
        # but the runtime does NOT actually tie them: ``tie_weights()`` early-
        # returns because ``config.tie_word_embeddings == False``, and the
        # source HF checkpoint stores two independent BF16 tensors with
        # 99.9% of elements differing (max_abs_diff ~ 8.1). ModelOpt's
        # ``export_hf_checkpoint`` trusts the declarative metadata, treats
        # codec_head as a redundant tied copy, and drops codec_head.weight
        # from the exported safetensors -- leaving a randomly-initialised
        # codec output projection at inference and producing garbled audio.
        # Strip the codec_head entry from the tied-keys metadata for the
        # duration of the export so modelopt writes codec_head.weight
        # explicitly. Restored in the ``finally`` block below.
        orig_tied = getattr(sub, "_tied_weights_keys", None)
        if isinstance(orig_tied, dict):
            saved_tied_keys = orig_tied
            sub._tied_weights_keys = {
                k: v
                for k, v in orig_tied.items() if "codec_head" not in k
            }
            _tied_keys_was_patched = True
        elif isinstance(orig_tied, (list, tuple)):
            saved_tied_keys = orig_tied
            sub._tied_weights_keys = type(orig_tied)(
                k for k in orig_tied if "codec_head" not in str(k))
            _tied_keys_was_patched = True

        # ModelOpt's resmooth path calls ``model(fake_input_ids)`` with every
        # other arg None, but the Talker top-forward never derives
        # ``inputs_embeds`` from ``input_ids`` (the runtime always feeds
        # ``inputs_embeds`` directly from Thinker hidden states) and dereferences
        # ``attention_mask`` / ``talker_input_ids`` unguarded. Wrap forward to
        # synthesize the missing tensors so the dummy walk reaches every
        # quantizable linear.
        saved_forward = sub.forward
        _t_hidden_size = sub.config.text_config.hidden_size
        _t_device = next(sub.parameters()).device
        _t_dtype = next(sub.parameters()).dtype

        def _talker_forward_with_defaults(*args, **kwargs):
            if args and "input_ids" not in kwargs:
                kwargs["input_ids"] = args[0]
                args = args[1:]
            input_ids = kwargs.pop("input_ids", None)
            if input_ids is not None:
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                bsz, seq = input_ids.shape
                if kwargs.get("inputs_embeds", None) is None:
                    kwargs["inputs_embeds"] = torch.zeros(bsz,
                                                          seq,
                                                          _t_hidden_size,
                                                          dtype=_t_dtype,
                                                          device=_t_device)
                if kwargs.get("attention_mask", None) is None:
                    kwargs["attention_mask"] = torch.ones(bsz,
                                                          seq,
                                                          dtype=torch.long,
                                                          device=_t_device)
                if kwargs.get("talker_input_ids", None) is None:
                    kwargs["talker_input_ids"] = input_ids
            return saved_forward(*args, **kwargs)

        sub.forward = _talker_forward_with_defaults
    else:
        raise ValueError(f"unknown submodule: {which}")

    from modelopt.torch.export import model_utils as _mu
    _orig_is_mm = _mu.is_multimodal_model
    _mu.is_multimodal_model = lambda *a, **kw: False
    try:
        with torch.inference_mode():
            export_hf_checkpoint(sub, export_dir=full_dir)
    finally:
        _mu.is_multimodal_model = _orig_is_mm
        for attr, val in saved_attrs.items():
            setattr(sub, attr, val)
        for cfg_attr, val in saved_cfgs.items():
            setattr(sub.config, cfg_attr, val)
        if saved_forward is not None:
            sub.forward = saved_forward
        if _tied_keys_was_patched:
            sub._tied_weights_keys = saved_tied_keys
    print(f"[export-{which}] {full_dir}")
