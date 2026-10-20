"""Route a turn to xAI, Meta, or a local Ollama model."""
from __future__ import annotations

from typing import Any, Callable

from .meta_client import chat_once as meta_once
from .meta_client import chat_stream as meta_stream
from .ollama_client import chat_once as ollama_once
from .ollama_client import chat_stream as ollama_stream
from .reasoning import normalize_effort, ollama_think
from .xai_client import DEFAULT_CTX, chat_once as xai_once
from .xai_client import chat_stream as xai_stream


def context_window(provider: str, host: str, model: str, override: int) -> int:
    if provider in ("xai", "meta"):
        native = DEFAULT_CTX
        if override and override > 0:
            return min(override, native)
        return native
    from .ollama_client import effective_num_ctx

    return effective_num_ctx(host, model, override)


def complete(
    provider: str,
    *,
    host: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    options: dict | None = None,
    timeout: float = 180,
) -> dict:
    if provider == "xai":
        if not (api_key or "").strip():
            raise RuntimeError("xAI API key is not set. Open Settings and save a key.")
        return xai_once(api_key, model, messages, tools=tools, options=options, timeout=timeout)
    if provider == "meta":
        if not (api_key or "").strip():
            raise RuntimeError("Meta API key is not set. Open Settings and save a key.")
        return meta_once(api_key, model, messages, tools=tools, options=options, timeout=timeout)
    effort = normalize_effort((options or {}).get("reasoning"))
    return ollama_once(
        host,
        model,
        messages,
        tools=tools,
        options=options,
        think=ollama_think(effort),
        timeout=timeout,
    )


def stream(
    provider: str,
    *,
    host: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    options: dict | None = None,
    timeout: float = 600,
    on_token: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if provider == "xai":
        if not (api_key or "").strip():
            raise RuntimeError("xAI API key is not set. Open Settings and save a key.")
        return xai_stream(
            api_key,
            model,
            messages,
            tools=tools,
            options=options,
            timeout=timeout,
            on_token=on_token,
            on_reasoning=on_reasoning,
            should_cancel=should_cancel,
        )
    if provider == "meta":
        if not (api_key or "").strip():
            raise RuntimeError("Meta API key is not set. Open Settings and save a key.")
        return meta_stream(
            api_key,
            model,
            messages,
            tools=tools,
            options=options,
            timeout=timeout,
            on_token=on_token,
            on_reasoning=on_reasoning,
            should_cancel=should_cancel,
        )
    effort = normalize_effort((options or {}).get("reasoning"))
    return ollama_stream(
        host,
        model,
        messages,
        tools=tools,
        options=options,
        think=ollama_think(effort),
        timeout=timeout,
        on_token=on_token,
        on_reasoning=on_reasoning,
        should_cancel=should_cancel,
    )
