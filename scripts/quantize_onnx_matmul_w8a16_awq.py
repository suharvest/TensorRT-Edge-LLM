#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWQ-aware ONNX rewriter to W8A16LinearPlugin (minimal-fix flavor).

Why this exists
---------------
The naive ``quantize_onnx_matmul_w8a16.py`` uses per-output max-abs of the
ORIGINAL weight as the INT8 scale. For Qwen3-TTS CustomVoice this destroys
audio quality (chunks runs go to 142, hallucinated). AWQ supplies two pieces
of calibrated information that fix it:

  1. ``pre_quant_scale`` (s_in, shape [K])   — per-input-channel smoothing so
     outlier columns get attenuated before quantization. Modelopt stores it
     in ``<linear>.input_quantizer._pre_quant_scale``.
  2. ``amax`` (shape [N * num_groups, 1])     — modelopt-calibrated block max
     of the smoothed weight, stored at ``<linear>.weight_quantizer._amax``.

This script consumes the **minimal-fix FP16 ONNX** (the one that produces
known-good audio in FP16 at orin-nx) and a modelopt ``talker_quant_state.pt``
saved after ``mtq.quantize``. It walks MatMul nodes in topological order,
assigns each to a HF layer/op slot (the network's deterministic per-layer
order is q, k, v, o, gate, up, down for 28 layers = 196 MatMuls; the 197th
MatMul is ``codec_head`` and is left untouched), and rewrites the chain:

  Before:  act ── MatMul(W_fp16[K,N]) ── y
  After:   act ── Mul(1/s_in[K]) ── W8A16LinearPlugin(act, qweight[K,N], scales[N]) ── y

where:
  * ``W' = W * s_in[None, :]`` (smoothed weight in HF [N,K] layout)
  * ``s_o = max_g(amax[o, g]) / 127``     (per-output INT8 scale, FP16)
  * ``qweight[K,N] = clip(round(W'_torch / s_o[:,None]), -128, 127).T``

The output ONNX is structurally identical (W8A16LinearPlugin attrs / inputs /
domain / scale layout) to the naive PTQ rewriter's output — so the worker
plugin dispatch path is unchanged. Only the weight values + scale values +
the extra ``Mul`` per layer differ.

Inputs
------
    --input        minimal-fix FP16 model.onnx (with .data sidecar)
    --quant-state  talker_quant_state.pt  (dict with model_state_dict[...])

Output
------
    --output       new model.onnx (+ .data sidecar if --external-data)

The 197th MatMul (codec_head) is skipped automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


SCALE_MODE_PER_OUTPUT = 0
W8A16_DOMAIN_DEFAULT = "trt"

# Per-layer op order observed in the minimal-fix export (verified by shape
# pattern q(K,2K) k(K,K) v(K,K) o(2K,K) gate(K,3K) up(K,3K) down(3K,K) for
# hidden=1024 across 28 layers).
LAYER_OP_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
OPS_PER_LAYER = len(LAYER_OP_ORDER)


def _ensure_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == domain:
            opset.version = max(opset.version, version)
            return
    model.opset_import.append(helper.make_opsetid(domain, version))


def _load_quant_state(path: Path) -> dict[str, np.ndarray]:
    """Load modelopt save and return a flat ``{layer.op : (amax, pqs, group_size)}``-ish dict."""
    import torch

    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if hasattr(v, "numpy"):
            out[k] = v.detach().numpy()
    return out


def _collect_layer_quant(state: dict[str, np.ndarray], num_layers: int):
    """Return a list of ``num_layers`` dicts, each ``{op_name: (W_hf, amax, pqs)}``.

    Raises if anything is missing — we want a hard fail, not silent fallback to
    naive max-abs.
    """
    layers = []
    for li in range(num_layers):
        ops = {}
        for op in LAYER_OP_ORDER:
            if op in ("q_proj", "k_proj", "v_proj", "o_proj"):
                prefix = f"model.layers.{li}.self_attn.{op}"
            else:
                prefix = f"model.layers.{li}.mlp.{op}"
            w_key = f"{prefix}.weight"
            amax_key = f"{prefix}.weight_quantizer._amax"
            pqs_key = f"{prefix}.input_quantizer._pre_quant_scale"
            if w_key not in state or amax_key not in state or pqs_key not in state:
                raise KeyError(f"missing quant tensors for layer {li} op {op}: "
                               f"have_w={w_key in state} have_amax={amax_key in state} "
                               f"have_pqs={pqs_key in state}")
            ops[op] = (state[w_key], state[amax_key], state[pqs_key])
        layers.append(ops)
    return layers


def _compute_per_output_int8(weight_hf: np.ndarray, amax: np.ndarray, pqs: np.ndarray):
    """Return (qweight_kn_int8, per_output_scale_fp16).

    weight_hf:  [N, K]  (HF Linear convention)
    amax:       [N * num_groups, 1]  modelopt block amax of SMOOTHED weight
    pqs:        [K]    AWQ per-input-channel smoothing scale
    """
    if weight_hf.ndim != 2:
        raise ValueError(f"weight must be 2D, got {weight_hf.shape}")
    out_dim, in_dim = int(weight_hf.shape[0]), int(weight_hf.shape[1])
    if pqs.shape != (in_dim,):
        raise ValueError(f"pqs shape {pqs.shape} vs in_dim {in_dim}")
    amax = amax.reshape(-1)
    if amax.size % out_dim != 0:
        raise ValueError(f"amax size {amax.size} not divisible by out_dim {out_dim}")
    num_groups = amax.size // out_dim
    if in_dim % num_groups != 0:
        raise ValueError(f"in_dim {in_dim} not divisible by num_groups {num_groups}")
    amax_2d = amax.reshape(out_dim, num_groups).astype(np.float32)
    # collapse per-block scales to per-output by taking max — over-estimates dynamic
    # range so all blocks remain representable with int8 per-output quant.
    per_output_amax = amax_2d.max(axis=1)
    # AWQ stores amax of the smoothed weight already. For int8 symmetric:
    #     scale = amax / 127, qmax = 127
    per_output_scale = per_output_amax / 127.0
    per_output_scale = np.where(per_output_scale > 0, per_output_scale, 1.0).astype(np.float32)

    # Smoothed weight: W' = W * pqs[None, :]  (HF [N,K]).
    weight_smooth = weight_hf.astype(np.float32) * pqs.astype(np.float32)[None, :]
    qweight_nk = np.clip(np.rint(weight_smooth / per_output_scale[:, None]), -128, 127).astype(np.int8)
    qweight_kn = qweight_nk.T  # [K, N] — matches kernel expectation
    return qweight_kn, per_output_scale.astype(np.float16)


def rewrite_model(
    model: onnx.ModelProto,
    *,
    layer_quant: list[dict],
    domain: str,
    cast_plugin_inputs_to_fp16: bool,
    cast_plugin_outputs_to_fp32: bool,
    plan_dump_path: Path | None,
) -> dict:
    graph = model.graph
    initializers = {ini.name: ini for ini in graph.initializer}
    initializer_use_counts: dict[str, int] = {n: 0 for n in initializers}
    for n in graph.node:
        for inp in n.input:
            if inp in initializer_use_counts:
                initializer_use_counts[inp] += 1

    mm_nodes_in_order = [n for n in graph.node if n.op_type == "MatMul"]
    total_mm = len(mm_nodes_in_order)
    quant_mm = len(layer_quant) * OPS_PER_LAYER
    if total_mm < quant_mm:
        raise RuntimeError(
            f"ONNX has only {total_mm} MatMul nodes but quant state demands {quant_mm} ({len(layer_quant)} layers × {OPS_PER_LAYER}). "
            "Check that the ONNX is the minimal-fix FP16 schema."
        )
    # Build position -> (layer_idx, op_name)
    position_to_slot: dict[int, tuple[int, str]] = {}
    expected_shapes = []
    for li in range(len(layer_quant)):
        for oi, op in enumerate(LAYER_OP_ORDER):
            position_to_slot[li * OPS_PER_LAYER + oi] = (li, op)
            w_hf = layer_quant[li][op][0]
            # ONNX weight is W.T, shape [K, N] = [in, out]
            expected_shapes.append((int(w_hf.shape[1]), int(w_hf.shape[0])))

    # Sanity check shapes against initializers at each position.
    mismatch = []
    for pos, mm in enumerate(mm_nodes_in_order[:quant_mm]):
        w_init = mm.input[1]
        if w_init not in initializers:
            mismatch.append((pos, "weight not initializer", mm.name, w_init))
            continue
        shape = tuple(int(d) for d in initializers[w_init].dims)
        if shape != expected_shapes[pos]:
            mismatch.append((pos, "shape mismatch", mm.name, shape, expected_shapes[pos]))
    if mismatch:
        for row in mismatch[:10]:
            print("[MISMATCH]", row)
        raise RuntimeError(f"{len(mismatch)} MatMul slots disagree on weight shape. Aborting.")

    plan_dump = []
    new_inits = []
    rewritten_nodes: list[onnx.NodeProto] = []
    converted_count = 0
    skipped_count = 0
    mm_seen = 0
    converted_weight_names: set[str] = set()

    for node in graph.node:
        if node.op_type != "MatMul":
            rewritten_nodes.append(node)
            continue
        pos = mm_seen
        mm_seen += 1
        if pos >= quant_mm:
            # Beyond covered range (e.g. codec_head) — keep as-is.
            skipped_count += 1
            rewritten_nodes.append(node)
            continue
        layer_idx, op = position_to_slot[pos]
        w_init_name = node.input[1]
        w_hf, amax, pqs = layer_quant[layer_idx][op]
        qweight_kn, scales_fp16 = _compute_per_output_int8(w_hf, amax, pqs)
        N = int(w_hf.shape[0])
        K = int(w_hf.shape[1])

        # Names
        safe_name = (node.name or node.output[0] or f"mm_{pos}").replace("/", "_").replace(":", "_").lstrip("_")
        qweight_name = f"{safe_name}_w8a16_qweight"
        scales_name = f"{safe_name}_w8a16_scales"
        inv_pqs_name = f"{safe_name}_w8a16_inv_pqs"
        new_inits.append(numpy_helper.from_array(qweight_kn, name=qweight_name))
        new_inits.append(numpy_helper.from_array(scales_fp16, name=scales_name))
        # inv_pqs in FP16 to match activation dtype in the talker stack
        inv_pqs_np = (1.0 / pqs.astype(np.float32)).astype(np.float16)
        new_inits.append(numpy_helper.from_array(inv_pqs_np, name=inv_pqs_name))

        activation_in = node.input[0]
        original_output = node.output[0]

        # Insert Mul(activation, inv_pqs) → smoothed_activation
        smoothed_act_name = f"{safe_name}_w8a16_smoothed_act"
        rewritten_nodes.append(
            helper.make_node(
                "Mul",
                [activation_in, inv_pqs_name],
                [smoothed_act_name],
                name=f"{safe_name}_w8a16_pqs_mul",
            )
        )
        plugin_act = smoothed_act_name

        if cast_plugin_inputs_to_fp16:
            cast_in_name = f"{safe_name}_w8a16_act_fp16"
            rewritten_nodes.append(
                helper.make_node(
                    "Cast",
                    [plugin_act],
                    [cast_in_name],
                    name=f"{safe_name}_w8a16_cast_act_fp16",
                    to=onnx.TensorProto.FLOAT16,
                )
            )
            plugin_act = cast_in_name

        plugin_out = original_output if not cast_plugin_outputs_to_fp32 else f"{safe_name}_w8a16_out_fp16"

        # Rewrite the MatMul node into W8A16LinearPlugin in-place
        del node.input[:]
        node.input.extend([plugin_act, qweight_name, scales_name])
        del node.output[:]
        node.output.extend([plugin_out])
        node.op_type = "W8A16LinearPlugin"
        node.domain = domain
        del node.attribute[:]
        node.attribute.extend([
            helper.make_attribute("gemm_n", N),
            helper.make_attribute("gemm_k", K),
            helper.make_attribute("scale_mode", SCALE_MODE_PER_OUTPUT),
            helper.make_attribute("group_size", 0),
        ])
        rewritten_nodes.append(node)
        if cast_plugin_outputs_to_fp32:
            rewritten_nodes.append(
                helper.make_node(
                    "Cast",
                    [plugin_out],
                    [original_output],
                    name=f"{safe_name}_w8a16_cast_out_fp32",
                    to=onnx.TensorProto.FLOAT,
                )
            )

        converted_weight_names.add(w_init_name)
        converted_count += 1
        plan_dump.append({
            "pos": pos, "layer": layer_idx, "op": op,
            "matmul_name": node.name or node.output[0],
            "weight_init": w_init_name,
            "N": N, "K": K,
            "scale_min": float(scales_fp16.min()), "scale_max": float(scales_fp16.max()),
            "qweight_min": int(qweight_kn.min()), "qweight_max": int(qweight_kn.max()),
        })

    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    # Drop FP weight initializers that are no longer referenced.
    referenced = {inp for n in graph.node for inp in n.input}
    kept_inits = [ini for ini in graph.initializer
                  if (ini.name not in converted_weight_names) or (ini.name in referenced)]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    graph.initializer.extend(new_inits)

    _ensure_opset(model, domain, 1)

    if plan_dump_path:
        plan_dump_path.write_text(json.dumps(plan_dump, indent=2))

    int8_bytes = sum(int(np.prod(t.dims)) for t in new_inits if t.name.endswith("_w8a16_qweight"))
    scale_elems = sum(int(np.prod(t.dims)) for t in new_inits if t.name.endswith("_w8a16_scales"))
    return {
        "matmul_total": total_mm,
        "matmul_converted": converted_count,
        "matmul_skipped": skipped_count,
        "int8_weight_elements": int8_bytes,
        "fp16_scale_elements": scale_elems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="minimal-fix FP16 model.onnx")
    parser.add_argument("--output", required=True, type=Path, help="output W8A16 model.onnx")
    parser.add_argument("--quant-state", required=True, type=Path,
                        help="modelopt talker_quant_state.pt")
    parser.add_argument("--num-layers", type=int, default=28, help="num talker decoder layers (default 28)")
    parser.add_argument("--domain", default=W8A16_DOMAIN_DEFAULT)
    parser.add_argument("--cast-plugin-inputs-to-fp16", action="store_true")
    parser.add_argument("--cast-plugin-outputs-to-fp32", action="store_true")
    parser.add_argument("--external-data", action="store_true")
    parser.add_argument("--external-data-file", default=None)
    parser.add_argument("--size-threshold", type=int, default=1024)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan-dump", type=Path, default=None,
                        help="Optional path to write a JSON dump describing each conversion slot")
    args = parser.parse_args()

    print(f"[awq-rewrite] loading quant state: {args.quant_state}")
    state = _load_quant_state(args.quant_state)
    print(f"[awq-rewrite] state has {len(state)} tensors")
    layer_quant = _collect_layer_quant(state, args.num_layers)
    print(f"[awq-rewrite] collected quant for {len(layer_quant)} layers × {OPS_PER_LAYER} ops = {len(layer_quant)*OPS_PER_LAYER}")

    print(f"[awq-rewrite] loading ONNX: {args.input}")
    model = onnx.load_model(args.input, load_external_data=True)

    summary = rewrite_model(
        model,
        layer_quant=layer_quant,
        domain=args.domain,
        cast_plugin_inputs_to_fp16=args.cast_plugin_inputs_to_fp16,
        cast_plugin_outputs_to_fp32=args.cast_plugin_outputs_to_fp32,
        plan_dump_path=args.plan_dump,
    )

    if args.check:
        onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.external_data:
        data_file = args.external_data_file or f"{args.output.name}.data"
        for ini in model.graph.initializer:
            if ini.data_location == onnx.TensorProto.EXTERNAL:
                ini.data_location = onnx.TensorProto.DEFAULT
                del ini.external_data[:]
        onnx.save_model(
            model,
            args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=args.size_threshold,
        )
    else:
        onnx.save_model(model, args.output)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
