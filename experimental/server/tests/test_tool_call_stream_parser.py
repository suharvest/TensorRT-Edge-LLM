# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming tool_call name early-emit parser.

These tests target ``_ToolCallStreamParser`` directly without touching
the FastAPI / TRT-LLM stack, so they can run on any host with plain
Python.
"""

from __future__ import annotations

import json

from experimental.server.api_server import _ToolCallStreamParser


def _feed_chunks(parser: _ToolCallStreamParser, chunks):
    emits = []
    for c in chunks:
        emits.extend(parser.feed(c))
    return emits


def _name_of(emit):
    return emit["tool_calls"][0]["function"]["name"]


def _index_of(emit):
    return emit["tool_calls"][0]["index"]


def test_single_tool_call_name_emitted_early():
    """Name should emit as soon as its closing quote is seen,
    long before </tool_call> arrives."""
    p = _ToolCallStreamParser()
    # Split a canonical tool_call across many chunks; ensure name
    # surfaces in an emit BEFORE the arguments JSON is finished.
    chunks = [
        '<tool_',
        'call>\n{"',
        'name"',
        ': "wave',
        '_hand"',
        ', "arguments"',
        ': {"reps": 3}',
        '}\n</tool_call>',
    ]
    emits = []
    saw_name_before_close = False
    consumed = ""
    for c in chunks:
        emits_now = p.feed(c)
        if emits_now and "</tool_call>" not in consumed + c:
            saw_name_before_close = True
        emits.extend(emits_now)
        consumed += c

    assert saw_name_before_close, "name delta should fire before stream end"
    assert len(emits) == 1
    assert _name_of(emits[0]) == "wave_hand"
    assert _index_of(emits[0]) == 0
    # Arguments must be empty in the partial — canonical end-of-stream
    # delta fills them.
    assert emits[0]["tool_calls"][0]["function"]["arguments"] == ""


def test_name_not_first_key():
    """Parser must locate ``name`` by key, not position."""
    p = _ToolCallStreamParser()
    payload = '<tool_call>{"arguments": {"x": 1}, "name": "go_home"}</tool_call>'
    emits = _feed_chunks(p, list(payload))  # one char at a time
    names = [_name_of(e) for e in emits]
    assert "go_home" in names


def test_multiple_tool_calls_increment_index():
    p = _ToolCallStreamParser()
    payload = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        ' some chatter '
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    # Chunk in 7-char windows to exercise boundary splits.
    chunks = [payload[i:i + 7] for i in range(0, len(payload), 7)]
    emits = _feed_chunks(p, chunks)
    assert [_name_of(e) for e in emits] == ["a", "b"]
    assert [_index_of(e) for e in emits] == [0, 1]


def test_plain_text_yields_nothing():
    p = _ToolCallStreamParser()
    emits = _feed_chunks(p, ["hello ", "world, no tools here."])
    assert emits == []


def test_open_tag_split_across_chunks():
    """Pathological split: ``<`` and ``tool_call>`` arrive separately."""
    p = _ToolCallStreamParser()
    emits = _feed_chunks(p, [
        "<",
        "tool_call>",
        '{"name":"',
        'pick"',
        ',"arguments":{}}',
        "</tool_call>",
    ])
    assert len(emits) == 1
    assert _name_of(emits[0]) == "pick"


def test_escaped_quote_in_name_unescapes():
    """JSON escape sequences in the name must be honored."""
    p = _ToolCallStreamParser()
    payload = '<tool_call>{"name": "say_\\"hi\\"", "arguments": {}}</tool_call>'
    emits = _feed_chunks(p, [payload])
    assert len(emits) == 1
    assert _name_of(emits[0]) == 'say_"hi"'


def test_bare_json_fallback_awq_mode():
    """AWQ/quantized models often drop the <tool_call> XML wrapper and
    emit raw JSON. Early-emit must still trigger on this shape."""
    p = _ToolCallStreamParser()
    chunks = [
        '{"',
        'name":',
        ' "wave_hand"',
        ', "arguments": {}',
        '}',
    ]
    emits = []
    name_emitted_before_args = False
    seen_args = False
    for c in chunks:
        if 'arguments' in c:
            seen_args = True
        emits_now = p.feed(c)
        if emits_now and not seen_args:
            name_emitted_before_args = True
        emits.extend(emits_now)
    assert name_emitted_before_args, (
        "name should emit before arguments JSON arrives")
    assert len(emits) == 1
    assert _name_of(emits[0]) == "wave_hand"


def test_tools_disabled_path_does_not_engage():
    """Parser is opt-in — callers gated on tools_enabled. Sanity: feeding
    a parser without any tool tag yields no emits and buffer stays bounded.
    """
    p = _ToolCallStreamParser()
    for _ in range(1000):
        p.feed("just some text. ")
    # buffer must remain small (bounded by tag-prefix lookahead)
    assert len(p._buf) < 64


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
