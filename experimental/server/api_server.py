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
OpenAI-compatible HTTP server for TensorRT Edge-LLM.

Endpoints:
    GET  /health                  - Health check
    GET  /v1/models               - List available models
    POST /v1/chat/completions     - Chat completion (OpenAI-compatible)

Usage (standalone)::

    python -m experimental.server \\
        --model Qwen/Qwen3-1.7B --port 8000

    python -m experimental.server \\
        --engine-dir /path/to/llm_engine --port 8000

Usage (from LLM object)::

    from experimental.server import LLM
    llm = LLM(model="Qwen/Qwen3-1.7B")
    llm.serve(port=8000)
"""

import argparse
import asyncio
import copy
import json
import logging
import queue as _queue
import re
import threading
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("edgellm.api_server")

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
IM_END_TOKEN = "<|im_end|>"

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)


def _split_reasoning_and_content(text: str):
    """Split model output into (reasoning_content, content) around <think> tags."""
    think_open = text.find(THINK_OPEN_TAG)
    think_close = text.find(THINK_CLOSE_TAG)
    if think_open != -1 and think_close != -1 and think_close > think_open:
        reasoning = text[think_open + len(THINK_OPEN_TAG):think_close].strip()
        content = text[think_close + len(THINK_CLOSE_TAG):].strip()
        return reasoning, content or None
    return None, text.strip() if text.strip() else None


# ---------------------------------------------------------------------------
# Tool-calling (Qwen3-format) helpers
#
# The compiled C++ runtime applies a static prefix/suffix-per-role template
# (``processed_chat_template.json``) which has no notion of OpenAI ``tools``.
# To support tool-calling without modifying the C++ runtime, we render the
# tool schema into a Qwen3-style "# Tools" system block in Python and inject
# it as the leading system message before the request reaches the runtime.
# On the response side we parse ``<tool_call>{...}</tool_call>`` blocks out
# of the generated text and emit them as OpenAI ``tool_calls`` entries.
# ---------------------------------------------------------------------------


def _render_tools_system_block(tools: List[Dict[str, Any]],
                               existing_system: str = "") -> str:
    """Build the Qwen3 tool-system prompt text.

    Mirrors the upstream Qwen3 chat_template.jinja ``{% if tools %}`` branch.
    """
    parts: List[str] = []
    if existing_system:
        parts.append(existing_system + "\n\n")
    parts.append(
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query."
        "\n\nYou are provided with function signatures within "
        "<tools></tools> XML tags:\n<tools>")
    for tool in tools:
        parts.append("\n" + json.dumps(tool, ensure_ascii=False))
    parts.append(
        "\n</tools>\n\nFor each function call, return a json object with "
        "function name and arguments within <tool_call></tool_call> XML "
        "tags:\n<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>")
    return "".join(parts)


def _flatten_message_for_runtime(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI tool-call / tool-result messages into plain text.

    The C++ runtime only understands ``system|user|assistant`` roles with
    string-or-multimodal content. We rewrite:
        - ``role=tool``                 -> user with ``<tool_response>...``
        - ``assistant`` w/ tool_calls   -> assistant with ``<tool_call>...``
    """
    role = msg.get("role")
    if role == "tool":
        out = {
            "role": "user",
            "content": ("<tool_response>\n" + str(msg.get("content", "")) +
                        "\n</tool_response>"),
        }
        return out
    if role == "assistant" and msg.get("tool_calls"):
        text_parts: List[str] = []
        content = msg.get("content") or ""
        if isinstance(content, str) and content:
            text_parts.append(content)
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if isinstance(args, str):
                args_str = args
            else:
                args_str = json.dumps(args, ensure_ascii=False)
            text_parts.append(
                f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": "
                f"{args_str}}}\n</tool_call>")
        return {"role": "assistant", "content": "\n".join(text_parts)}
    return msg


def _inject_tools_and_normalize(
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Return a new messages list with tool schema injected and tool/tool_call
    messages flattened so the C++ runtime can handle them.
    """
    flat = [_flatten_message_for_runtime(copy.deepcopy(m)) for m in messages]
    if not tools:
        return flat

    existing_system = ""
    if flat and flat[0].get("role") == "system":
        sys_content = flat[0].get("content", "")
        if isinstance(sys_content, str):
            existing_system = sys_content
        elif isinstance(sys_content, list):
            existing_system = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in sys_content)
        rendered = _render_tools_system_block(tools, existing_system)
        flat[0] = {"role": "system", "content": rendered}
    else:
        rendered = _render_tools_system_block(tools, "")
        flat = [{"role": "system", "content": rendered}] + flat
    return flat


def _tool_call_from_json_obj(obj: Any, idx: int) -> Optional[Dict[str, Any]]:
    """Build an OpenAI ``tool_calls`` entry from a raw {name,arguments} dict."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "arguments" not in obj and "parameters" not in obj:
        return None
    args_obj = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args_obj, str):
        arguments_str = args_obj
    else:
        arguments_str = json.dumps(args_obj, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "index": idx,
        "function": {
            "name": name,
            "arguments": arguments_str,
        },
    }


def _extract_tool_calls(text: str):
    """Pull tool-call payloads out of model output.

    Handles both the canonical Qwen3 ``<tool_call>{...}</tool_call>`` wrapping
    *and* the common AWQ/quantized failure mode where the model drops the XML
    tags and emits the raw ``{"name": ..., "arguments": ...}`` JSON. The
    fallback is conservative: it only triggers when the stripped text starts
    with ``{`` and parses to a dict with both ``name`` and ``arguments``.

    Returns ``(tool_calls_list, remaining_text)`` where ``tool_calls_list`` is
    a list of OpenAI-shaped ``tool_calls`` dicts, and ``remaining_text`` is
    the text with the tool-call payload stripped (used as ``content``).
    """
    tool_calls: List[Dict[str, Any]] = []
    cleaned_parts: List[str] = []
    last_end = 0
    for match in _TOOL_CALL_RE.finditer(text):
        cleaned_parts.append(text[last_end:match.start()])
        last_end = match.end()
        body = match.group(1).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            obj = None
        tc = _tool_call_from_json_obj(obj, len(tool_calls))
        if tc is not None:
            tool_calls.append(tc)
        else:
            # Preserve raw text on malformed tool_call body.
            cleaned_parts.append(match.group(0))
    cleaned_parts.append(text[last_end:])
    cleaned = "".join(cleaned_parts).strip()

    # Fallback: model dropped the <tool_call> XML wrapper but emitted the
    # bare JSON payload. Only attempt when no XML-form calls were found and
    # the entire cleaned text is one JSON object that looks like a call.
    if not tool_calls and cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            obj = None
        tc = _tool_call_from_json_obj(obj, 0)
        if tc is not None:
            tool_calls.append(tc)
            cleaned = ""

    return tool_calls, cleaned


def _create_app(llm_instance):
    """Create a FastAPI app backed by the given LLM instance."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for the server. "
                           "Install: pip install fastapi uvicorn") from exc

    app = FastAPI(
        title="TensorRT Edge-LLM Server",
        version="0.1.0",
        description=
        "OpenAI-compatible inference server powered by TensorRT Edge-LLM",
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": llm_instance.model_dir,
            "speculative_decoding": llm_instance.has_draft_model,
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object":
            "list",
            "data": [{
                "id": llm_instance._model_id,
                "object": "model",
                "owned_by": "tensorrt-edgellm",
            }],
        }

    @app.get("/metrics")
    def metrics():
        return {
            "profiling_enabled": llm_instance.get_profiling_enabled(),
            "prefill": llm_instance.get_prefill_metrics(),
        }

    @app.post("/v1/cache/system_prompt")
    def cache_system_prompt(body: Dict[str, Any]):
        # Optional ``tools`` (OpenAI schema). When provided alongside a
        # ``system_prompt`` (or messages), render them into the cached prefix
        # so that subsequent /v1/chat/completions requests carrying the same
        # tools list hit the radix-tree prefix cache instead of re-prefilling
        # the ~400 tokens of tool schema every turn.
        tools = body.get("tools") or None
        has_tools = bool(tools)
        if body.get("formatted_system_prompt") or body.get("formatted_prefix"):
            # Caller pre-formatted everything; trust it as-is. Tools must be
            # baked in by caller in this mode.
            prompt = body.get("formatted_system_prompt") or body["formatted_prefix"]
        elif body.get("prompt"):
            prompt = body["prompt"]
        elif body.get("system_prompt") is not None or has_tools:
            try:
                if has_tools:
                    # Build a synthetic [system] messages list with tools
                    # injected, then format the full message (system role
                    # prefix/suffix included) so the cached KV covers
                    # system + tools spec as one contiguous prefix.
                    sys_text = body.get("system_prompt", "") or ""
                    base_messages = [{"role": "system", "content": sys_text}]
                    injected = _inject_tools_and_normalize(base_messages, tools)
                    prompt = llm_instance.format_messages(
                        injected,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                else:
                    prompt = llm_instance.format_system_prompt(
                        body.get("system_prompt", ""))
            except Exception as exc:
                logger.exception("System prompt formatting failed")
                return JSONResponse(status_code=500,
                                    content={"error": str(exc)})
        elif body.get("messages"):
            try:
                messages = body["messages"]
                if has_tools:
                    injected = _inject_tools_and_normalize(messages, tools)
                    # Only cache the leading system block (post-injection)
                    # so the prefix matches what chat_completions emits.
                    prompt = llm_instance.format_messages(
                        [injected[0]] if injected and injected[0].get("role") == "system" else injected,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                else:
                    prompt = llm_instance.format_system_prompt_from_messages(
                        messages)
            except Exception as exc:
                logger.exception("System prompt formatting failed")
                return JSONResponse(status_code=500,
                                    content={"error": str(exc)})
        else:
            prompt = ""
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "error":
                    "formatted_system_prompt, formatted_prefix, prompt, system_prompt, "
                    "tools, or messages with a leading system message is required"
                },
            )
        lora_weights_name = body.get("lora_weights_name",
                                     body.get("lora_name", ""))
        try:
            cached = llm_instance.save_system_prompt_kv_cache(
                prompt,
                lora_weights_name,
            )
        except Exception as exc:
            logger.exception("System prompt cache warmup failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})
        return {
            "object": "cache.system_prompt",
            "cached": cached,
            "lora_weights_name": lora_weights_name,
            "has_tools": has_tools,
            "tools_count": len(tools) if has_tools else 0,
            "prompt_chars": len(prompt),
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: Dict[str, Any], request: Request):
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(status_code=400,
                                content={"error": "messages required"})

        # OpenAI ``tools`` / ``tool_choice`` support. The compiled C++ runtime
        # cannot render the OpenAI tools schema directly (its template is a
        # static prefix/suffix-per-role JSON, no Jinja). We inject a
        # Qwen3-format "# Tools" system block here and parse <tool_call>...
        # </tool_call> back out of the response below.
        tools = body.get("tools") or None
        if tools:
            try:
                messages = _inject_tools_and_normalize(messages, tools)
            except Exception as exc:
                logger.exception("Tool prompt rendering failed")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid tools schema: {exc}"})
        else:
            # Still normalize any tool_calls / tool messages already in
            # history so the runtime never sees an unsupported role.
            messages = [
                _flatten_message_for_runtime(m) for m in messages
            ]

        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        top_k = body.get("top_k", 50)
        max_tokens = body.get("max_tokens", 2048)
        stream = body.get("stream", False)
        enable_thinking = body.get("enable_thinking", False)
        save_system_prompt_kv_cache = body.get(
            "save_system_prompt_kv_cache", body.get("cache_prompt", False))
        save_prefix_cache = body.get("save_prefix_cache", False)
        prefix_cache = body.get("prefix_cache", save_prefix_cache)
        lora_weights_name = body.get("lora_weights_name",
                                     body.get("lora_name", ""))
        disable_spec_decode = body.get("disable_spec_decode", False)
        return_cache_metrics = body.get("return_cache_metrics", False)

        rt = llm_instance._rt
        from .engine import _convert_messages_to_cpp, _load_image_buffers

        try:
            cpp_messages = _convert_messages_to_cpp(rt, messages)
        except (ValueError, KeyError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid messages: {exc}"},
            )

        image_buffers = _load_image_buffers(rt, messages)

        # NOTE: do NOT shadow the FastAPI ``request`` parameter (line 158).
        # The disconnect watcher in _generate_stream_sse needs the original
        # Starlette Request to poll is_disconnected(); rebinding the name
        # here would silently break the watcher.
        trt_request = rt.LLMGenerationRequest()
        req = rt.Request(messages=cpp_messages)
        req.image_buffers = image_buffers
        trt_request.requests = [req]
        trt_request.temperature = temperature
        trt_request.top_p = top_p
        trt_request.top_k = top_k
        trt_request.max_generate_length = max_tokens
        trt_request.apply_chat_template = True
        trt_request.add_generation_prompt = True
        trt_request.enable_thinking = enable_thinking
        trt_request.save_system_prompt_kv_cache = (
            save_system_prompt_kv_cache or save_prefix_cache)
        trt_request.lora_weights_name = lora_weights_name
        trt_request.disable_spec_decode = disable_spec_decode

        formatted_prefix = ""
        formatted_complete = ""
        if prefix_cache:
            try:
                formatted = _build_prefix_formatted_request(
                    llm_instance,
                    body,
                    messages,
                    enable_thinking,
                )
                formatted_prefix = formatted["formatted_system_prompt"]
                formatted_complete = formatted["formatted_complete_request"]
                formatted_request = rt.FormattedRequest()
                formatted_request.formatted_system_prompt = formatted_prefix
                formatted_request.formatted_complete_request = formatted_complete
                trt_request.formatted_requests = [formatted_request]
            except Exception as exc:
                logger.exception("Prefix cache formatting failed")
                return JSONResponse(status_code=400,
                                    content={"error": str(exc)})

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        before_metrics = (
            llm_instance.get_prefill_metrics()
            if return_cache_metrics else None)

        if stream:
            from .engine import SamplingParams

            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                save_system_prompt_kv_cache=(
                    save_system_prompt_kv_cache or save_prefix_cache),
                lora_weights_name=lora_weights_name,
                disable_spec_decode=disable_spec_decode,
                formatted_system_prompt=formatted_prefix,
                formatted_complete_request=formatted_complete,
            )

            return StreamingResponse(
                _generate_stream_sse(
                    request,                # FastAPI Request for disconnect watcher
                    llm_instance,
                    messages,
                    params,
                    response_id,
                    enable_thinking,
                    before_metrics,
                    return_cache_metrics,
                    tools_enabled=bool(tools),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            response = llm_instance._runtime.handle_request(trt_request)
        except Exception as exc:
            logger.exception("Inference failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})

        raw_text = response.output_texts[0] if response.output_texts else ""
        output_text = raw_text.replace(IM_END_TOKEN, "")
        output_ids = response.output_ids[0] if response.output_ids else []
        completion_tokens = len(output_ids)
        after_metrics = (
            llm_instance.get_prefill_metrics()
            if return_cache_metrics else None)

        reasoning, answer = _split_reasoning_and_content(output_text)

        message_body: Dict[str, Any] = {"role": "assistant"}
        if reasoning is not None:
            message_body["reasoning"] = reasoning
        answer_text = (answer if answer is not None else reasoning) or ""

        finish_reason = "stop"
        if tools:
            tcs, cleaned = _extract_tool_calls(answer_text)
            if tcs:
                # Strip ``index`` (only meaningful in streaming deltas) for
                # the non-stream OpenAI shape.
                message_body["tool_calls"] = [
                    {k: v for k, v in tc.items() if k != "index"}
                    for tc in tcs
                ]
                message_body["content"] = cleaned or None
                finish_reason = "tool_calls"
            else:
                message_body["content"] = answer_text
        else:
            message_body["content"] = answer_text

        result = {
            "id":
            response_id,
            "object":
            "chat.completion",
            "choices": [{
                "index": 0,
                "message": message_body,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "completion_tokens": completion_tokens,
            },
        }
        cache_metrics = _cache_metrics_delta(before_metrics, after_metrics)
        if cache_metrics is not None:
            result["cache_metrics"] = cache_metrics
        return result

    return app


def _build_prefix_formatted_request(
    llm_instance,
    body: Dict[str, Any],
    messages,
    enable_thinking: bool,
) -> Dict[str, str]:
    """Build formatted prefix and complete request for prefix-cache mode."""
    if body.get("formatted_prefix") and body.get("formatted_complete_request"):
        return {
            "formatted_system_prompt": body["formatted_prefix"],
            "formatted_complete_request": body["formatted_complete_request"],
        }

    if body.get("prefix_messages") is not None:
        prefix_messages = body["prefix_messages"]
        suffix_messages = messages
    else:
        if len(messages) < 2:
            raise ValueError(
                "prefix_cache requires prefix_messages or at least two messages")
        prefix_messages = messages[:-1]
        suffix_messages = messages[-1:]

    if not isinstance(prefix_messages, list) or not isinstance(
            suffix_messages, list):
        raise ValueError("prefix_messages and messages must be arrays")

    return llm_instance.make_prefix_formatted_request(
        prefix_messages,
        suffix_messages,
        enable_thinking=enable_thinking,
    )


class _ThinkingStateMachine:
    """Tracks <think>...</think> boundaries across streaming deltas."""

    def __init__(self, thinking_enabled: bool):
        self._enabled = thinking_enabled
        self._in_think = False
        self._think_opened = False
        self._buf = ""

    def feed(self, text: str):
        """Yield (field, text) pairs: field is 'reasoning' or 'content'."""
        if not self._enabled:
            yield "content", text
            return

        self._buf += text
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(THINK_OPEN_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_OPEN_TAG):
                        safe = self._buf[:-len(THINK_OPEN_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe and self._think_opened:
                            yield "content", safe
                        elif safe:
                            yield "content", safe
                    break
                if idx > 0 and self._think_opened:
                    yield "content", self._buf[:idx]
                elif idx > 0:
                    yield "content", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_OPEN_TAG):]
                self._in_think = True
                self._think_opened = True
            else:
                idx = self._buf.find(THINK_CLOSE_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_CLOSE_TAG):
                        safe = self._buf[:-len(THINK_CLOSE_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe:
                            yield "reasoning", safe
                    break
                if idx > 0:
                    yield "reasoning", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_CLOSE_TAG):]
                self._in_think = False

    def flush(self):
        """Flush remaining buffer at end of stream."""
        if self._buf:
            field = "reasoning" if self._in_think else "content"
            yield field, self._buf
            self._buf = ""


def _cache_metrics_delta(before_metrics, after_metrics):
    if before_metrics is None or after_metrics is None:
        return None
    return {
        "prefill": {
            "reused_tokens": after_metrics["reused_tokens"]
            - before_metrics["reused_tokens"],
            "computed_tokens": after_metrics["computed_tokens"]
            - before_metrics["computed_tokens"],
        }
    }


async def _generate_stream_sse(request, llm_instance, messages, params,
                               response_id, enable_thinking,
                               before_metrics=None,
                               return_cache_metrics: bool = False,
                               tools_enabled: bool = False):
    """Yield real SSE chunks via StreamChannel streaming.

    Runs the synchronous ``generate_stream`` iteration in a background
    thread and forwards chunks via a thread-safe queue. A concurrent
    asyncio task polls ``request.is_disconnected()``; when the HTTP
    client drops the connection mid-stream (voice-agent barge-in is the
    canonical trigger) the watcher sets a stop flag, the drain thread
    breaks out of the ``for`` loop on the next chunk boundary and calls
    ``gen.close()`` from inside the same thread — which raises
    GeneratorExit at engine.py's ``yield`` and runs the ``finally``:
    ``channel.cancel()`` propagates the cancel into the TRT runtime, the
    C++ worker returns within hundreds of ms, and ``worker.join()``
    completes. Without this, the worker keeps generating tokens on the
    single engine context after every disconnect and the *next*
    /v1/chat/completions request crashes the engine with::

        [TensorRT] Error Code 1: Myelin (Called with an already loaded
        binary graph.)
    """
    yield _sse_chunk(response_id, {"role": "assistant"})

    sm = _ThinkingStateMachine(enable_thinking)
    finish_reason: Optional[str] = None

    gen = llm_instance.generate_stream(messages, params)
    # Unbounded queue: an LLM turn produces at most max_tokens chunks,
    # each ~50 bytes — bounded in practice by the engine's max_seq_len,
    # which is what the length-guard middleware already enforces. A
    # bounded Queue would let drain block on put() after the consumer
    # disconnects, and stop_flag would never be observed in time.
    chunk_q: _queue.Queue = _queue.Queue()
    stop_flag = threading.Event()
    disconnected = False

    def _drain():
        try:
            for delta in gen:
                if stop_flag.is_set():
                    break
                chunk_q.put_nowait(("delta", delta))
                if stop_flag.is_set():
                    break
        except Exception as exc:  # noqa: BLE001
            try:
                chunk_q.put_nowait(("error", exc))
            except _queue.Full:  # pragma: no cover - unbounded queue
                pass
        finally:
            # close() from the same thread as next() — safe (no
            # "generator already executing" race) and guarantees
            # engine.py's finally runs even on the natural exit path.
            try:
                gen.close()
            except Exception:  # pragma: no cover - cleanup must not raise
                pass
            try:
                chunk_q.put_nowait(("done", None))
            except _queue.Full:  # pragma: no cover - unbounded queue
                pass

    drain_thread = threading.Thread(
        target=_drain, daemon=True, name="sse-drain")
    drain_thread.start()

    loop = asyncio.get_running_loop()

    async def _watch_disconnect():
        nonlocal disconnected
        # Poll every 100 ms — well below per-token latency, so the
        # drain thread sees the stop flag within a single chunk
        # boundary after the client drops. Cancel latency after
        # ``stop_flag`` is set is bounded by ``StreamChannel.wait_pop``'s
        # 200 ms timeout in engine.py:752, so total disconnect→TRT-cancel
        # is at worst ~300 ms.
        while not stop_flag.is_set():
            try:
                if await request.is_disconnected():
                    disconnected = True
                    stop_flag.set()
                    logger.info(
                        "client disconnected mid-stream; cancelling LLM "
                        "generation (response_id=%s)", response_id)
                    return
            except asyncio.CancelledError:
                # The outer task got cancelled (normal shutdown path
                # when streaming finishes naturally) — let it propagate
                # so the watcher stops cleanly.
                raise
            except (ConnectionResetError, OSError, RuntimeError):
                # Network-layer errors during is_disconnected() probing
                # are themselves a strong signal the peer is gone.
                # Treat as disconnected rather than spinning.
                disconnected = True
                stop_flag.set()
                logger.info(
                    "client probe raised network error; treating as "
                    "disconnected (response_id=%s)", response_id)
                return
            # Other exceptions (notably AttributeError from wrong-type
            # ``request`` shadowing the FastAPI parameter — see the
            # comment at chat_completions where trt_request is named to
            # NOT collide with the FastAPI parameter) are programming
            # errors we want surfaced, NOT swallowed.
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(_watch_disconnect())

    # When tools are enabled, buffer all assistant ``content`` until the
    # stream finishes, then emit ``tool_calls`` (or content) atomically.
    # Mid-stream <tool_call> XML must not leak to the OpenAI SDK client —
    # it would be interpreted as raw assistant content and the eval would
    # never see a structured tool_call.
    content_buf: List[str] = []

    try:
        while True:
            try:
                kind, val = await loop.run_in_executor(None, chunk_q.get)
            except asyncio.CancelledError:
                stop_flag.set()
                raise
            if kind == "done":
                break
            if kind == "error":
                logger.error("Streaming inference failed: %r", val)
                finish_reason = "error"
                break
            delta = val
            if delta.text:
                for field, text in sm.feed(delta.text):
                    if tools_enabled and field == "content":
                        content_buf.append(text)
                    else:
                        yield _sse_chunk(response_id, {field: text})
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
            if disconnected:
                break
    finally:
        stop_flag.set()
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    if disconnected and finish_reason is None:
        finish_reason = "cancelled"

    # If the client disconnected, downstream send() will fail anyway —
    # still yield the trailing frames so a graceful close (e.g. server
    # shutdown) drains the SSE protocol cleanly.
    for field, text in sm.flush():
        if tools_enabled and field == "content":
            content_buf.append(text)
        else:
            yield _sse_chunk(response_id, {field: text})

    if tools_enabled:
        full_text = "".join(content_buf)
        tcs, cleaned = _extract_tool_calls(full_text)
        if tcs:
            for tc in tcs:
                # Emit each tool_call as a single delta with name + full
                # JSON arguments — the OpenAI SDK accumulates ``arguments``
                # across deltas, so one-shot is fine.
                tc_delta = {
                    "tool_calls": [{
                        "index": tc["index"],
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }]
                }
                yield _sse_chunk(response_id, tc_delta)
            if finish_reason in (None, "stop"):
                finish_reason = "tool_calls"
            if cleaned:
                # Trailing/leading non-tool text (rare). Emit as content so
                # nothing is silently dropped.
                yield _sse_chunk(response_id, {"content": cleaned})
        else:
            if full_text:
                yield _sse_chunk(response_id, {"content": full_text})

    after_metrics = (
        llm_instance.get_prefill_metrics()
        if return_cache_metrics else None)
    cache_metrics = _cache_metrics_delta(before_metrics, after_metrics)
    yield _sse_chunk(
        response_id,
        {},
        finish_reason=finish_reason or "stop",
        cache_metrics=cache_metrics,
    )
    yield "data: [DONE]\n\n"


def _sse_chunk(response_id: str,
               delta: dict,
               finish_reason: Optional[str] = None,
               cache_metrics: Optional[Dict[str, Any]] = None):
    choice: Dict[str, Any] = {"delta": delta, "index": 0}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    payload = {"id": response_id, "choices": [choice]}
    if cache_metrics is not None:
        payload["cache_metrics"] = cache_metrics
    return f"data: {json.dumps(payload)}\n\n"


def run_server(llm_instance, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the OpenAI-compatible server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install: pip install uvicorn") from exc

    app = _create_app(llm_instance)
    logger.info("Starting server on %s:%d ...", host, port)
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="TensorRT Edge-LLM OpenAI-compatible server")
    parser.add_argument(
        "--model",
        default="",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--onnx-dir",
        default="",
        help="Existing LLM ONNX directory; build engine then serve",
    )
    parser.add_argument(
        "--visual-onnx-dir",
        default="",
        help="Existing visual ONNX directory for VLM serving",
    )
    parser.add_argument(
        "--engine-dir",
        default="",
        help="Existing LLM engine directory; load directly",
    )
    parser.add_argument(
        "--visual-engine-dir",
        default="",
        help="Existing visual engine directory for VLM serving",
    )
    parser.add_argument(
        "--served-model-name",
        default="",
        help="Model id returned by /v1/models and chat completions",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=4096,
        help="Max input sequence length",
    )
    parser.add_argument("--max-batch-size",
                        type=int,
                        default=1,
                        help="Max batch size")
    parser.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=8192,
        help="Max KV cache capacity",
    )
    parser.add_argument(
        "--use-trt-native-ops",
        action="store_true",
        default=False,
        help="Use TensorRT native ops instead of custom plugins",
    )
    parser.add_argument(
        "--spec-decode-engine-dir",
        "--eagle-engine-dir",  # deprecated alias, kept for backward compat
        dest="spec_decode_engine_dir",
        default="",
        help="Pre-built speculative decoding engine dir (EAGLE or MTP)",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        default=False,
        help="Enable runtime metrics such as prefill cache reuse counters",
    )
    parser.add_argument("--draft-top-k",
                        type=int,
                        default=10,
                        help="Speculative decoding: tokens per predecessor")
    parser.add_argument("--draft-step",
                        type=int,
                        default=6,
                        help="Speculative decoding: number of draft steps")
    parser.add_argument("--verify-tree-size",
                        type=int,
                        default=60,
                        help="Speculative decoding: verification tree size")
    parser.add_argument(
        "--weight-streaming-budget",
        default=None,
        help=(
            "TensorRT weight streaming budget. Accepts: integer bytes, "
            "'<N>g' GiB, '<N>m' MiB, 'off' (disable streaming, keep all weights "
            "on GPU), 'min'/'-1' (minimum budget = max streaming, smallest "
            "GPU footprint). Has no effect unless the engine was built with "
            "kWEIGHT_STREAMING. Default: unset (full weights resident)."
        ),
    )
    args = parser.parse_args()

    from .engine import LLM

    llm = LLM(
        model=args.model,
        onnx_dir=args.onnx_dir,
        visual_onnx_dir=args.visual_onnx_dir,
        engine_dir=args.engine_dir,
        visual_engine_dir=args.visual_engine_dir,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        use_trt_native_ops=args.use_trt_native_ops,
        eagle_engine_dir=args.spec_decode_engine_dir,
        model_id=args.served_model_name,
        draft_top_k=args.draft_top_k,
        draft_step=args.draft_step,
        verify_tree_size=args.verify_tree_size,
        weight_streaming_budget=args.weight_streaming_budget,
    )
    if args.enable_profiling:
        llm.set_profiling_enabled(True)
    llm.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
