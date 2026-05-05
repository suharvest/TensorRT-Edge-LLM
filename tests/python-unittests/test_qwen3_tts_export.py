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
"""
Regression tests for Qwen3-TTS LLM export.

Usage:
    python3 -m pytest tests/python-unittests/test_qwen3_tts_export.py -v --noconftest
"""

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file


class _DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"


class _FakeQwen3TTSConfig:

    @classmethod
    def from_pretrained(cls, model_dir):
        return cls()

    def to_dict(self):
        return {"model_type": "qwen3_tts"}


class _FakeCodePredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.lm_head = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        nn.init.constant_(self.lm_head[0].weight, -1.0)


class _FakeTalker(nn.Module):

    def __init__(self):
        super().__init__()
        self.code_predictor = _FakeCodePredictor()


class _FakeQwen3TTSForConditionalGeneration(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.talker = _FakeTalker()


def _install_fake_qwen_tts(monkeypatch):
    qwen_tts = types.ModuleType("qwen_tts")
    core = types.ModuleType("qwen_tts.core")
    models = types.ModuleType("qwen_tts.core.models")
    models.Qwen3TTSConfig = _FakeQwen3TTSConfig
    models.Qwen3TTSForConditionalGeneration = _FakeQwen3TTSForConditionalGeneration

    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts)
    monkeypatch.setitem(sys.modules, "qwen_tts.core", core)
    monkeypatch.setitem(sys.modules, "qwen_tts.core.models", models)


def test_qwen3_tts_loader_uses_checkpoint_for_code_predictor(monkeypatch, tmp_path):
    from tensorrt_edgellm.llm_models import model_utils

    _install_fake_qwen_tts(monkeypatch)
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr(model_utils.AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_utils, "_is_qwen3_tts_model", lambda model_dir: True)

    model_dir = tmp_path / "qwen3_tts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}), encoding="utf-8")

    expected_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    save_file({"talker.code_predictor.lm_head.0.weight": expected_weight}, model_dir / "model.safetensors")

    model, _, _ = model_utils.load_hf_model(str(model_dir), dtype="fp16", device="cpu")

    actual_weight = model.talker.code_predictor.lm_head[0].weight.detach()
    assert actual_weight.dtype == torch.float16
    assert torch.equal(actual_weight, expected_weight)


def test_qwen3_tts_loader_uses_sharded_checkpoint_for_code_predictor(monkeypatch, tmp_path):
    from tensorrt_edgellm.llm_models import model_utils

    _install_fake_qwen_tts(monkeypatch)
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr(model_utils.AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_utils, "_is_qwen3_tts_model", lambda model_dir: True)

    model_dir = tmp_path / "qwen3_tts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}), encoding="utf-8")

    expected_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float16)
    shard_name = "model-00001-of-00001.safetensors"
    save_file({"talker.code_predictor.lm_head.0.weight": expected_weight}, model_dir / shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": expected_weight.numel() * expected_weight.element_size()},
            "weight_map": {"talker.code_predictor.lm_head.0.weight": shard_name},
        }),
        encoding="utf-8",
    )

    model, _, _ = model_utils.load_hf_model(str(model_dir), dtype="fp16", device="cpu")

    actual_weight = model.talker.code_predictor.lm_head[0].weight.detach()
    assert actual_weight.dtype == torch.float16
    assert torch.equal(actual_weight, expected_weight)


def test_qwen3_tts_loader_rejects_incomplete_checkpoint(monkeypatch, tmp_path):
    from tensorrt_edgellm.llm_models import model_utils

    _install_fake_qwen_tts(monkeypatch)
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr(model_utils.AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_utils, "_is_qwen3_tts_model", lambda model_dir: True)

    model_dir = tmp_path / "qwen3_tts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}), encoding="utf-8")
    save_file({}, model_dir / "model.safetensors")

    with pytest.raises(RuntimeError, match="Failed to load Qwen3-TTS checkpoint cleanly"):
        model_utils.load_hf_model(str(model_dir), dtype="fp16", device="cpu")


def test_qwen3_tts_uses_generic_code_predictor_path():
    repo_root = Path(__file__).resolve().parents[2]
    source_files = [
        repo_root / "cpp" / "builder" / "llmBuilder.cpp",
        repo_root / "cpp" / "builder" / "llmBuilder.h",
        repo_root / "cpp" / "runtime" / "qwen3OmniTTSRuntime.cpp",
        repo_root / "cpp" / "runtime" / "qwen3OmniTTSRuntime.h",
        repo_root / "tensorrt_edgellm" / "llm_models" / "models" / "qwen3_omni_talker.py",
    ]
    forbidden = (
        "SpecialCodePredictor",
        "Qwen3TTSCodePredictorNativeExport",
        "export_qwen3_tts_native",
        "original_code_predictor",
        "qwen3_tts_cp.onnx",
        "qwen3_tts_cp.engine",
    )

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{needle} should not appear in {path.relative_to(repo_root)}"
