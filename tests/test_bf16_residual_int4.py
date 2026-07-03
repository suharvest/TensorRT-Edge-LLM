# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Wiring tests for the mixed-precision + INT4-AWQ overflow-safe path.

Covers the opt-in dtype routing that lets an INT4-AWQ backbone coexist with the
BF16-residual / FP16-attention mixed-precision graph, with the MLP down_proj
emitting BF16 output (so its overflow-prone activation never lands in FP16):

  * ``config.bf16_residual_active`` truth table
  * ``make_linear`` routes down_proj -> AWQLinear(output_bf16=True); gate/up stay
    FP16; behaviour is byte-identical without the opt-in flag
  * ``int4_groupwise_gemm`` op returns BF16 only when output_dtype='bfloat16'
  * ``build_quant_config(exclude_attention=True)`` disables q/k/v/o quantizers
    (requires modelopt; skipped otherwise)

These are pure-Python / CPU; no GPU, no ONNX build.
"""
import pytest
import torch

from tensorrt_edgellm.config import (ModelConfig, QuantConfig, QUANT_FP8,
                                     QUANT_INT4_AWQ)
from tensorrt_edgellm.models import linear as L
from tensorrt_edgellm.models.ops import int4_groupwise_gemm


def _cfg(bf16_residual, quant_type, with_quant):
    return ModelConfig(
        model_type="qwen2",
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
        vocab_size=166000,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        quant=QuantConfig(quant_type=quant_type, group_size=128),
        bf16_residual=bf16_residual,
        bf16_residual_with_quant=with_quant,
    )


@pytest.mark.parametrize(
    "mp,qtype,flag,expected",
    [
        (True, "float16", False, True),   # unquantized + mp
        (True, QUANT_INT4_AWQ, True, True),    # int4 + mp + opt-in
        (True, QUANT_INT4_AWQ, False, False),  # int4 + mp, no flag -> legacy
        (True, QUANT_FP8, True, False),        # fp8 needs fp16 input
        (False, QUANT_INT4_AWQ, True, False),  # no mp
    ],
)
def test_bf16_residual_active(mp, qtype, flag, expected):
    assert _cfg(mp, qtype, flag).bf16_residual_active is expected


def test_make_linear_down_proj_bf16_output():
    cfg = _cfg(True, QUANT_INT4_AWQ, True)
    down = L.make_linear(cfg, 4864, 896, module_name="layers.0.mlp.down_proj")
    gate = L.make_linear(cfg, 896, 4864, module_name="layers.0.mlp.gate_proj")
    up = L.make_linear(cfg, 896, 4864, module_name="layers.0.mlp.up_proj")
    assert isinstance(down, L.AWQLinear)
    assert down.output_bf16 is True
    assert gate.output_bf16 is False  # FP16 for the SwiGLU mul
    assert up.output_bf16 is False


def test_make_linear_legacy_int4_unchanged():
    # INT4 without the opt-in flag: down_proj stays FP16-output (byte-identical).
    cfg = _cfg(True, QUANT_INT4_AWQ, False)
    down = L.make_linear(cfg, 4864, 896, module_name="layers.0.mlp.down_proj")
    assert isinstance(down, L.AWQLinear)
    assert down.output_bf16 is False


def test_int4_op_output_dtype():
    x = torch.zeros(1, 4, 896, dtype=torch.float16)
    qw = torch.zeros(896 // 2, 896, dtype=torch.int8)
    sc = torch.ones(896 // 128, 896, dtype=torch.float16)
    assert int4_groupwise_gemm(x, qw, sc, 896, 896, 128).dtype == torch.float16
    assert int4_groupwise_gemm(
        x, qw, sc, 896, 896, 128,
        output_dtype="bfloat16").dtype == torch.bfloat16


def test_build_quant_config_exclude_attention():
    pytest.importorskip("modelopt")
    from tensorrt_edgellm.quantization.quantization_configs import \
        build_quant_config
    qcfg = build_quant_config("int4_awq", exclude_attention=True)["quant_cfg"]
    disabled = {
        k
        for k, v in qcfg.items()
        if isinstance(v, dict) and v.get("enable") is False
    }
    for proj in ("*q_proj*", "*k_proj*", "*v_proj*", "*o_proj*"):
        assert proj in disabled
    # Default recipe must NOT disable attention (byte-identical legacy path).
    qcfg_def = build_quant_config("int4_awq")["quant_cfg"]
    disabled_def = {
        k
        for k, v in qcfg_def.items()
        if isinstance(v, dict) and v.get("enable") is False
    }
    assert "*q_proj*" not in disabled_def
