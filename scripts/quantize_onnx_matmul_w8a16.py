#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite constant-weight ONNX MatMul nodes to W8A16LinearPlugin.

The converter is intentionally model-agnostic. It looks for ``MatMul`` nodes
whose second input is a 2D constant weight ``[K, N]``, quantizes that weight to
per-output INT8, and rewrites the node to:

    W8A16LinearPlugin(activation, qweight[int8 K,N], scales[fp16 N])

This does not use TensorRT INT8 calibration; activations remain in the graph's
chosen floating-point type. For the first kernel implementation, TensorRT must
select FP16 tensors for the plugin input/output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


SCALE_MODE_PER_OUTPUT = 0


def _compile_optional(pattern: str | None) -> re.Pattern[str] | None:
    return re.compile(pattern) if pattern else None


def _matches(pattern: re.Pattern[str] | None, text: str) -> bool:
    return bool(pattern and pattern.search(text))


def quantize_per_output_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``weight[K, N]`` with one scale per output channel."""
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {weight.shape}")

    weight_fp32 = weight.astype(np.float32, copy=False)
    amax = np.max(np.abs(weight_fp32), axis=0)
    scales = amax / 127.0
    scales[scales == 0.0] = 1.0
    qweight = np.clip(np.rint(weight_fp32 / scales[None, :]), -128, 127).astype(np.int8)
    return qweight, scales.astype(np.float16)


def _initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _node_name(node: onnx.NodeProto, fallback_idx: int) -> str:
    if node.name:
        return node.name
    if node.output:
        return node.output[0]
    return f"matmul_{fallback_idx}"


def _ensure_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == domain:
            opset.version = max(opset.version, version)
            return
    model.opset_import.append(helper.make_opsetid(domain, version))


def rewrite_model(
    model: onnx.ModelProto,
    *,
    include: re.Pattern[str] | None,
    exclude: re.Pattern[str] | None,
    min_elements: int,
    domain: str,
    keep_original_weights: bool,
    cast_plugin_inputs_to_fp16: bool,
    cast_plugin_outputs_to_fp32: bool,
) -> dict[str, int]:
    graph = model.graph
    initializers = _initializer_map(graph)
    initializer_use_counts: dict[str, int] = {name: 0 for name in initializers}
    for node in graph.node:
        for input_name in node.input:
            if input_name in initializer_use_counts:
                initializer_use_counts[input_name] += 1
    converted_weight_counts: dict[str, int] = {}
    new_initializers: list[onnx.TensorProto] = []
    converted = 0
    skipped = 0
    converted_elements = 0
    original_weight_bytes = 0
    rewritten_nodes: list[onnx.NodeProto] = []

    for idx, node in enumerate(graph.node):
        if node.op_type != "MatMul" or len(node.input) != 2:
            rewritten_nodes.append(node)
            continue

        name = _node_name(node, idx)
        if include and not _matches(include, name):
            skipped += 1
            rewritten_nodes.append(node)
            continue
        if _matches(exclude, name):
            skipped += 1
            rewritten_nodes.append(node)
            continue

        activation_name = node.input[0]
        weight_name = node.input[1]
        if weight_name not in initializers:
            skipped += 1
            rewritten_nodes.append(node)
            continue

        weight_tensor = initializers[weight_name]
        if len(weight_tensor.dims) != 2:
            skipped += 1
            rewritten_nodes.append(node)
            continue

        k, n = [int(dim) for dim in weight_tensor.dims]
        elements = k * n
        if elements < min_elements:
            skipped += 1
            rewritten_nodes.append(node)
            continue

        weight = numpy_helper.to_array(weight_tensor)
        qweight, scales = quantize_per_output_int8(weight)

        safe_name = name.replace("/", "_").replace(":", "_")
        qweight_name = f"{safe_name}_w8a16_qweight"
        scales_name = f"{safe_name}_w8a16_scales"
        new_initializers.append(numpy_helper.from_array(qweight, name=qweight_name))
        new_initializers.append(numpy_helper.from_array(scales, name=scales_name))

        original_output_name = node.output[0]
        if cast_plugin_inputs_to_fp16:
            activation_fp16_name = f"{safe_name}_w8a16_activation_fp16"
            rewritten_nodes.append(
                helper.make_node(
                    "Cast",
                    [activation_name],
                    [activation_fp16_name],
                    name=f"{safe_name}_w8a16_cast_activation_fp16",
                    to=onnx.TensorProto.FLOAT16,
                )
            )
            activation_name = activation_fp16_name

        if cast_plugin_outputs_to_fp32:
            plugin_output_name = f"{safe_name}_w8a16_output_fp16"
        else:
            plugin_output_name = original_output_name

        del node.input[:]
        node.input.extend([activation_name, qweight_name, scales_name])
        del node.output[:]
        node.output.extend([plugin_output_name])

        node.op_type = "W8A16LinearPlugin"
        node.domain = domain
        del node.attribute[:]
        node.attribute.extend(
            [
                helper.make_attribute("gemm_n", n),
                helper.make_attribute("gemm_k", k),
                helper.make_attribute("scale_mode", SCALE_MODE_PER_OUTPUT),
                helper.make_attribute("group_size", 0),
            ]
        )
        rewritten_nodes.append(node)

        if cast_plugin_outputs_to_fp32:
            rewritten_nodes.append(
                helper.make_node(
                    "Cast",
                    [plugin_output_name],
                    [original_output_name],
                    name=f"{safe_name}_w8a16_cast_output_fp32",
                    to=onnx.TensorProto.FLOAT,
                )
            )

        converted_weight_counts[weight_name] = converted_weight_counts.get(weight_name, 0) + 1
        converted += 1
        converted_elements += elements
        original_weight_bytes += weight.nbytes

    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    if not keep_original_weights and converted_weight_counts:
        removable_weight_names = {
            name
            for name, converted_count in converted_weight_counts.items()
            if converted_count >= initializer_use_counts.get(name, 0)
        }
        kept_initializers = [
            initializer for initializer in graph.initializer if initializer.name not in removable_weight_names
        ]
        del graph.initializer[:]
        graph.initializer.extend(kept_initializers)
    graph.initializer.extend(new_initializers)

    if domain:
        _ensure_opset(model, domain, 1)

    return {
        "converted_matmuls": converted,
        "skipped_matmuls": skipped,
        "converted_weight_elements": converted_elements,
        "original_weight_bytes": original_weight_bytes,
        "fp16_weight_bytes": converted_elements * 2,
        "int8_weight_bytes": converted_elements,
        "per_output_scale_bytes": converted * 0,  # filled below
        "estimated_weight_savings_bytes": converted_elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX model")
    parser.add_argument("--include", help="Only convert MatMul node names matching this regex")
    parser.add_argument("--exclude", help="Skip MatMul node names matching this regex")
    parser.add_argument("--min-elements", type=int, default=0, help="Skip weights smaller than this many elements")
    parser.add_argument("--domain", default="trt", help="Custom op domain for plugin nodes")
    parser.add_argument("--keep-original-weights", action="store_true", help="Keep replaced FP weights in the ONNX file")
    parser.add_argument(
        "--cast-plugin-inputs-to-fp16",
        action="store_true",
        help="Insert Cast nodes before each W8A16 plugin so FLOAT graphs can keep FP32 public tensors",
    )
    parser.add_argument(
        "--cast-plugin-outputs-to-fp32",
        action="store_true",
        help="Insert Cast nodes after each W8A16 plugin so downstream FLOAT graph consumers remain unchanged",
    )
    parser.add_argument("--external-data", action="store_true", help="Save tensors as external data")
    parser.add_argument("--external-data-file", default=None, help="External tensor data filename")
    parser.add_argument("--size-threshold", type=int, default=1024, help="External data size threshold")
    parser.add_argument("--check", action="store_true", help="Run ONNX checker after rewriting")
    args = parser.parse_args()

    model = onnx.load_model(args.input, load_external_data=True)
    summary = rewrite_model(
        model,
        include=_compile_optional(args.include),
        exclude=_compile_optional(args.exclude),
        min_elements=args.min_elements,
        domain=args.domain,
        keep_original_weights=args.keep_original_weights,
        cast_plugin_inputs_to_fp16=args.cast_plugin_inputs_to_fp16,
        cast_plugin_outputs_to_fp32=args.cast_plugin_outputs_to_fp32,
    )

    scale_elems = 0
    for initializer in model.graph.initializer:
        if initializer.name.endswith("_w8a16_scales"):
            scale_elems += math.prod(initializer.dims)
    summary["per_output_scale_bytes"] = scale_elems * 2
    summary["estimated_weight_savings_bytes"] = summary["original_weight_bytes"] - (
        summary["int8_weight_bytes"] + summary["per_output_scale_bytes"]
    )

    if args.check:
        onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.external_data:
        data_file = args.external_data_file or f"{args.output.name}.data"
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
