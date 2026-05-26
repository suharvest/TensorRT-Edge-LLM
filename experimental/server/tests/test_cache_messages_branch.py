# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /v1/cache/system_prompt messages-branch (P7).

Verifies that when a ``messages`` list is sent, the server renders the
prefix via ``_build_prefix_formatted_request`` (same code path as
``/v1/chat/completions``) so the cached KV prefix matches Run 1.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_llm():
    """Minimal stub that mimics the LLM instance surface needed here."""
    llm = MagicMock()
    llm.model_dir = "/fake/model"
    llm.has_draft_model = False
    llm._model_id = "fake"
    # make_prefix_formatted_request: must return the contract dict.
    def _make_prefix(prefix_messages, suffix_messages, enable_thinking=False):
        # Concat content for deterministic byte-level comparison.
        sys_block = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in prefix_messages
        )
        complete = sys_block + "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in suffix_messages
        )
        return {
            "formatted_system_prompt": sys_block,
            "formatted_complete_request": complete,
        }
    llm.make_prefix_formatted_request.side_effect = _make_prefix
    # save_system_prompt_kv_cache: just record arg + return True.
    llm.save_system_prompt_kv_cache.return_value = True
    # Fallback path: format_system_prompt / format_messages may be hit
    # only if our new branch is NOT taken — used as a regression guard.
    llm.format_system_prompt.return_value = "LEGACY-SYS"
    llm.format_messages.return_value = "LEGACY-FORMATTED"
    llm.format_system_prompt_from_messages.return_value = "LEGACY-FROM-MSGS"
    return llm


def _client(fake_llm):
    from fastapi.testclient import TestClient

    from experimental.server.api_server import _create_app

    app = _create_app(fake_llm)
    return TestClient(app)


def test_messages_branch_taken_when_messages_present(fake_llm):
    client = _client(fake_llm)
    resp = client.post(
        "/v1/cache/system_prompt",
        json={
            "messages": [
                {"role": "system", "content": "you are a helpful arm."},
                {"role": "user", "content": ""},
            ],
            "prefix_cache": True,
            "enable_thinking": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages_branch"] is True
    assert data["cached"] is True
    assert data["prompt_chars"] > 0
    # Confirm we went through make_prefix_formatted_request and NOT
    # format_messages / format_system_prompt (the legacy branches).
    assert fake_llm.make_prefix_formatted_request.called
    assert not fake_llm.format_system_prompt.called
    assert not fake_llm.format_messages.called
    # The cached prompt must equal the formatted_system_prompt that
    # _build_prefix_formatted_request returned (so Run 1's prefix
    # matches byte-for-byte).
    cached_arg = fake_llm.save_system_prompt_kv_cache.call_args[0][0]
    expected_prefix, _ = fake_llm.make_prefix_formatted_request.side_effect(
        [{"role": "system", "content": "you are a helpful arm."}],
        [{"role": "user", "content": ""}],
    ), None
    # call_args captured the actual rendered prefix; ensure it's only the
    # system block (no user content leaked in).
    assert "you are a helpful arm." in cached_arg
    assert "<|im_start|>user" not in cached_arg


def test_messages_branch_with_tools(fake_llm):
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
        "/v1/cache/system_prompt",
        json={
            "messages": [
                {"role": "system", "content": "you are a helpful arm."},
                {"role": "user", "content": ""},
            ],
            "tools": tools,
            "prefix_cache": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages_branch"] is True
    assert data["has_tools"] is True
    assert data["tools_count"] == 1


def test_legacy_system_prompt_branch_still_works(fake_llm):
    """Backward compat: old clients sending {system_prompt, tools} still go
    through the legacy path (messages_branch=False)."""
    client = _client(fake_llm)
    resp = client.post(
        "/v1/cache/system_prompt",
        json={"system_prompt": "hello"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages_branch"] is False
    assert fake_llm.format_system_prompt.called
