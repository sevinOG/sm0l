"""xAI (Grok) chat via the OpenAI-compatible API. https://api.x.ai/v1"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from .media import attachments_of, attachments_to_b64
from .reasoning import (
    effort_rejected,
    effort_was_hidden,
    pull_reasoning,
    text_of,
    usage_reasoning_tokens,
    with_effort,
)

API_ROOT = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4"
DEFAULT_CTX = 131072
FALLBACK_MODELS = ("grok-4", "grok-3", "grok-3-mini")


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(api_key: str, path: str, payload: dict | None, *, timeout: float, method: str = "POST") -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_ROOT + path,
        data=data,
        headers=_headers(api_key),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI HTTP {e.code}: {err[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"xAI unreachable: {e.reason}") from e
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"xAI returned non-JSON: {body[:200]}") from e


def list_models(api_key: str, timeout: float = 20) -> list[str]:
    if not (api_key or "").strip():
        return []
    data = _request(api_key, "/models", None, timeout=timeout, method="GET")
    names: list[str] = []
    for item in data.get("data") or []:
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        low = mid.lower()
        if any(tok in low for tok in ("embed", "image", "video", "tts", "whisper", "flux")):
            continue
        names.append(mid)
    return names or list(FALLBACK_MODELS)


def account_status(api_key: str, timeout: float = 12) -> tuple[bool, str]:
    """Return (can_call_models, short reason). Does not include the key."""
    if not (api_key or "").strip():
        return False, "no API key"
    try:
        list_models(api_key, timeout=timeout)
        return True, "connected"
    except RuntimeError as e:
        msg = str(e)
        low = msg.lower()
        if "credit" in low or "spending limit" in low or "permission-denied" in low:
            return False, "out of credits or monthly spending limit"
        if "http 401" in low or "invalid" in low or "unauthorized" in low:
            return False, "key rejected"
        if "unreachable" in low or "timed out" in low:
            return False, "unreachable"
        return False, msg[:180]
    except Exception as e:
        return False, str(e)[:180]


def ping(api_key: str, timeout: float = 8) -> bool:
    ok, _reason = account_status(api_key, timeout=timeout)
    return ok


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Turn sm0l transcript messages into OpenAI/xAI chat messages."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role") or "user"
        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(m.get("tool_call_id") or "call"),
                    "content": m.get("content") or "",
                }
            )
            continue
        text = m.get("content") or ""
        content: Any = text
        if role == "user":
            parts: list[dict] = []
            if text:
                parts.append({"type": "text", "text": text})
            for attachment in attachments_of(m):
                encoded = attachments_to_b64([attachment])
                if not encoded:
                    continue
                mime = str(attachment.get("mime") or "image/png").split(";", 1)[0].lower()
                if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                    mime = "image/png"
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded[0]}"},
                    }
                )
            if len(parts) > (1 if text else 0):
                content = parts
        msg: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for i, tc in enumerate(m["tool_calls"]):
                fn = tc.get("function") or tc
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                calls.append(
                    {
                        "id": str(tc.get("id") or f"call_{i}"),
                        "type": "function",
                        "function": {"name": str(fn.get("name") or ""), "arguments": args},
                    }
                )
            msg["tool_calls"] = calls
            if not text:
                msg["content"] = None
        out.append(msg)
    return out


def _usage(data: dict) -> tuple[int, int]:
    usage = data.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _open_chat(api_key: str, body: dict, timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_ROOT + "/chat/completions",
        data=data,
        headers=_headers(api_key),
        method="POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI HTTP {e.code}: {err[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"xAI unreachable: {e.reason}") from e


def _attempt_bodies(api_key: str, bodies: list[dict], timeout: float) -> tuple[int, dict]:
    last: Exception | None = None
    for i, body in enumerate(bodies):
        try:
            return i, _request(api_key, "/chat/completions", body, timeout=timeout)
        except RuntimeError as e:
            last = e
            if not effort_rejected(e) or i == len(bodies) - 1:
                raise
    raise last or RuntimeError("xAI request failed")


def _message_from(choice_message: dict, tools: list[dict] | None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text_of(choice_message.get("content")),
    }
    thought = pull_reasoning(choice_message)
    if thought:
        message["reasoning"] = thought
    if choice_message.get("tool_calls"):
        message["tool_calls"] = choice_message["tool_calls"]
    return message


def chat_once(
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    options: dict | None = None,
    timeout: float = 180,
) -> dict:
    options = options or {}
    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": to_openai_messages(messages),
        "stream": False,
        "temperature": options.get("temperature", 0.2),
    }
    if options.get("num_predict"):
        payload["max_tokens"] = int(options["num_predict"])
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    bodies = with_effort(payload, "xai", options.get("reasoning"))
    index, data = _attempt_bodies(api_key, bodies, timeout)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    prompt_eval, eval_count = _usage(data)
    message = _message_from(msg, tools)
    thought = text_of(message.get("reasoning"))
    hidden = effort_was_hidden("xai", options.get("reasoning"), bodies[index].get("reasoning_effort"))
    if usage_reasoning_tokens(data) > 0 and not thought:
        hidden = True
    return {
        "message": message,
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "used_tools_api": bool(tools),
        "reasoning_hidden": hidden,
    }


def chat_stream(
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    options: dict | None = None,
    timeout: float = 600,
    on_token: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    options = options or {}
    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": to_openai_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": options.get("temperature", 0.2),
    }
    if options.get("num_predict"):
        payload["max_tokens"] = int(options["num_predict"])
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    bodies = with_effort(payload, "xai", options.get("reasoning"))
    resp = None
    used: str | None = None
    last: Exception | None = None
    for i, body in enumerate(bodies):
        try:
            resp = _open_chat(api_key, body, timeout)
            used = body.get("reasoning_effort")
            break
        except RuntimeError as e:
            last = e
            if not effort_rejected(e) or i == len(bodies) - 1:
                raise
    if resp is None:
        raise last or RuntimeError("xAI request failed")
    content = ""
    reasoning = ""
    calls: dict[int, dict[str, str]] = {}
    prompt_eval = 0
    eval_count = 0
    reason_tokens = 0
    try:
        with resp:
            for raw in resp:
                if should_cancel and should_cancel():
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                blob = line[5:].strip()
                if blob == "[DONE]":
                    break
                try:
                    chunk = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                p, e = _usage(chunk)
                if p:
                    prompt_eval = p
                if e:
                    eval_count = e
                reason_tokens = max(reason_tokens, usage_reasoning_tokens(chunk))
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {})
                piece = text_of(delta.get("content"))
                if piece:
                    content += piece
                    if on_token:
                        on_token(piece)
                thought = pull_reasoning(delta)
                if thought:
                    reasoning += thought
                    if on_reasoning:
                        on_reasoning(thought)
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index") or 0)
                    slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI HTTP {e.code}: {err[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"xAI unreachable: {e.reason}") from e
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if calls:
        message["tool_calls"] = [
            {
                "id": slot["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
            }
            for i, slot in sorted(calls.items())
        ]
    hidden = effort_was_hidden("xai", options.get("reasoning"), used)
    if reason_tokens > 0 and not reasoning.strip():
        hidden = True
    return {
        "message": message,
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "used_tools_api": bool(tools),
        "reasoning_hidden": hidden,
    }
