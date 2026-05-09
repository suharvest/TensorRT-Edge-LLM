#!/usr/bin/env python3
"""Create a runner-compatible Qwen3-TTS vocoder50 ONNX from vocoder100.

The existing vocoder100 wrapper exposes:
  codes [1,16,T] -> waveform [1,1,192000]

For the streaming path we normally vocode 25 new frames plus 25 frames of
left context. This script keeps the proven vocoder graph but exposes only the
first 50 codec frames worth of samples:
  waveform [1,1,96000]

The TensorRT profile must also use max codes=50 so Code2WavRunner allocates
matching input/output buffers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


def _shape_value_info(name: str, dims: list[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source vocoder100 ONNX dir")
    parser.add_argument("--out", required=True, help="Output vocoder50 ONNX dir")
    parser.add_argument("--samples", type=int, default=96_000)
    parser.add_argument("--max-code-len", type=int, default=50)
    parser.add_argument("--opt-code-len", type=int, default=25)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model_path = src / "model.onnx"
    config_path = src / "config.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    model = onnx.load(model_path, load_external_data=True)
    graph = model.graph

    if len(graph.output) != 1 or graph.output[0].name != "waveform":
        raise RuntimeError(f"Expected single waveform output, got {[o.name for o in graph.output]}")

    old_output = "waveform"
    full_output = "waveform_full_192000"
    for node in reversed(graph.node):
        for idx, output in enumerate(node.output):
            if output == old_output:
                node.output[idx] = full_output
                break
        else:
            continue
        break
    else:
        raise RuntimeError("Could not find node producing waveform")

    starts_name = "vocoder50_slice_starts"
    ends_name = "vocoder50_slice_ends"
    axes_name = "vocoder50_slice_axes"
    steps_name = "vocoder50_slice_steps"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0], dtype=np.int64), starts_name),
            numpy_helper.from_array(np.array([args.samples], dtype=np.int64), ends_name),
            numpy_helper.from_array(np.array([2], dtype=np.int64), axes_name),
            numpy_helper.from_array(np.array([1], dtype=np.int64), steps_name),
        ]
    )
    graph.node.append(
        helper.make_node(
            "Slice",
            [full_output, starts_name, ends_name, axes_name, steps_name],
            [old_output],
            name="vocoder50_waveform_slice",
        )
    )

    graph.output.clear()
    graph.output.append(_shape_value_info(old_output, ["batch", 1, args.samples]))

    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        out / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="onnx_model.data",
        size_threshold=1024,
        convert_attribute=False,
    )

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    builder = cfg.setdefault("builder_config", {})
    builder["min_code_len"] = 1
    builder["opt_code_len"] = args.opt_code_len
    builder["max_code_len"] = args.max_code_len
    builder["min_time_steps"] = min(builder.get("min_time_steps", 100), args.samples)
    builder["max_time_steps"] = args.samples // 16
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {out / 'model.onnx'}")
    print(f"Output waveform samples: {args.samples}")
    print(f"Profile max code len: {args.max_code_len}")


if __name__ == "__main__":
    main()
