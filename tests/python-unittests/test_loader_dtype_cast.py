# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for ``tensorrt_edgellm/checkpoint/loader.py`` — ``_set_tensor``
dtype-cast semantics.

FP32 checkpoint sources (e.g. SparkTTS) must be cast to the destination's
declared half-precision dtype, exactly like BF16/FP16 sources. Before the
fix, an FP32 source was raw-assigned, silently replacing an FP16Linear weight
with a float32 Parameter and leaving the model in a mixed-dtype state.
Deliberately-FP32 destinations (quant scales) must stay FP32. CPU-only.
"""

import os
import sys

import pytest

# Load the package from the repository root without installing it.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch
    from torch import nn

    from tensorrt_edgellm.checkpoint.loader import _set_tensor
    from tensorrt_edgellm.models.linear import FP16Linear
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"torch / tensorrt_edgellm not importable: {exc}",
                allow_module_level=True)


class _TinyModel(nn.Module):
    """Minimal module tree exercising the loader destinations."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = FP16Linear(4, 8, bias=True)
        # Deliberately-FP32 buffer, mirroring quant scales such as
        # W4A16Linear.weights_scaling_factor / FP8Linear.input_scale.
        self.register_buffer("input_scale", torch.ones(1,
                                                       dtype=torch.float32))


def test_fp32_source_cast_to_fp16_destination():
    """FP32 checkpoint weight loaded into FP16Linear ends up float16."""
    model = _TinyModel()
    src = torch.randn(8, 4, dtype=torch.float32)

    assert _set_tensor(model, "proj.weight", src)

    assert model.proj.weight.dtype == torch.float16
    torch.testing.assert_close(model.proj.weight.float(),
                               src,
                               rtol=1e-3,
                               atol=1e-3)


def test_fp32_source_cast_to_fp16_bias():
    model = _TinyModel()
    src = torch.randn(8, dtype=torch.float32)

    assert _set_tensor(model, "proj.bias", src)

    assert model.proj.bias.dtype == torch.float16


def test_bf16_source_still_cast_to_fp16_destination():
    """Regression: the pre-existing BF16 -> destination dtype cast holds."""
    model = _TinyModel()
    src = torch.randn(8, 4, dtype=torch.bfloat16)

    assert _set_tensor(model, "proj.weight", src)

    assert model.proj.weight.dtype == torch.float16


def test_fp32_source_kept_for_declared_fp32_buffer():
    """Quant-scale-style FP32 destinations must NOT be downcast."""
    model = _TinyModel()
    src = torch.full((1, ), 0.125, dtype=torch.float32)

    assert _set_tensor(model, "input_scale", src)

    assert model.input_scale.dtype == torch.float32
    torch.testing.assert_close(model.input_scale, src)
