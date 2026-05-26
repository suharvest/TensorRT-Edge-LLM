# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /v1/warmup (P8).

Verifies that the endpoint runs ``handle_request`` exactly once with
``max_generate_length`` forced to 1, regardless of caller input.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_llm():
    llm = MagicMock()
    llm.model_dir = "/fake/model"
    llm.has_draft_model = False
    llm._model_id = "fake"

    # _rt namespace: LLMGenerationRequest + Request constructors used by both
    # /v1/chat/completions and /v1/warmup. We capture the constructed
    # LLMGenerationRequest so the test can assert on max_generate_length.
    captured = {}

    class _LLMGenReq:
        def __init__(self):
            self.requests = []
            self.image_buffers = []
            self.temperature = None
            self.top_p = None
            self.top_k = None
            self.max_generate_length = None
            self.apply_chat_template = None
            self.add_generation_prompt = None
            self.enable_thinking = None
            self.save_system_prompt_kv_cache = None
            self.lora_weights_name = None
            self.disable_spec_decode = None
            captured["req"] = self

    class _Req:
        def __init__(self, messages=None):
            self.messages = messages
            self.image_buffers = []

    rt = MagicMock()
    rt.LLMGenerationRequest = _LLMGenReq
    rt.Request = _Req
    llm._rt = rt

    # handle_request returns a response with one empty output.
    response = MagicMock()
    response.output_ids = [[]]
    response.output_texts = [""]
    response.prompt_token_count = 7
    llm._runtime.handle_request.return_value = response

    llm._captured = captured  # type: ignore[attr-defined]
    return llm


def _client(fake_llm):
    from fastapi.testclient import TestClient
    from experimental.server.api_server import _create_app
    return TestClient(_create_app(fake_llm))


def _patch_engine_helpers(monkeypatch):
    """Replace engine helpers that need a real C++ runtime."""
    from experimental.server import engine as _engine
    monkeypatch.setattr(_engine, "_convert_messages_to_cpp", lambda rt, m: m)
    monkeypatch.setattr(_engine, "_load_image_buffers", lambda rt, m: [])


def test_warmup_forces_max_tokens_1(fake_llm, monkeypatch):
    _patch_engine_helpers(monkeypatch)
    client = _client(fake_llm)
    resp = client.post(
        "/v1/warmup",
        json={
            "messages": [
                {"role": "system", "content": "you are an arm."},
                {"role": "user", "content": ""},
            ],
            # Caller tries to override; server MUST ignore.
            "max_tokens": 999,
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["warmed"] is True
    assert data["object"] == "warmup"
    assert "took_ms" in data
    assert data["generated_tokens"] == 0
    # The actual TRT request: max_generate_length must be 1.
    captured = fake_llm._captured["req"]
    assert captured.max_generate_length == 1
    # And handle_request was called exactly once.
    assert fake_llm._runtime.handle_request.call_count == 1


def test_warmup_with_tools(fake_llm, monkeypatch):
    _patch_engine_helpers(monkeypatch)
    client = _client(fake_llm)
    tools = [{
        "type": "function",
        "function": {
            "name": "wave_hand",
            "description": "wave the arm",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    resp = client.post(
        "/v1/warmup",
        json={
            "messages": [
                {"role": "system", "content": "you are an arm."},
                {"role": "user", "content": ""},
            ],
            "tools": tools,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["has_tools"] is True
    assert data["warmed"] is True


def test_warmup_missing_messages_400(fake_llm, monkeypatch):
    _patch_engine_helpers(monkeypatch)
    client = _client(fake_llm)
    resp = client.post("/v1/warmup", json={})
    assert resp.status_code == 400
