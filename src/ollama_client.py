"""Thin Ollama HTTP client. stdlib only — keeps the freeze small."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator
from urllib.parse import urljoin


def _url(host: str, path: str) -> str:
    base = host.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _request(
    host: str,
    path: str,
    payload: dict | None = None,
    method: str | None = None,
    timeout: float = 30,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        _url(host, path),
        data=data,
        headers=headers,
        method=method or ("POST" if data else "GET"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {host}: {e.reason}") from e


def ping(host: str) -> bool:
    try:
        _request(host, "/api/tags", timeout=3)
        return True
    except Exception:
        return False


def list_models(host: str) -> list[dict]:
    data = _request(host, "/api/tags", timeout=8)
    models = data.get("models") or []
    out = []
    for m in models:
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        details = m.get("details") or {}
        out.append(
            {
                "name": name,
                "size": m.get("size") or 0,
                "param": details.get("parameter_size") or "",
                "family": details.get("family") or "",
            }
        )
    out.sort(key=lambda x: x["name"].lower())
    return out


def show_model(host: str, name: str) -> dict:
    return _request(host, "/api/show", {"name": name}, timeout=20)


def native_context_length(host: str, name: str) -> int:
    """Read the model's own window from /api/show. Fall back to 8192."""
    try:
        info = show_model(host, name)
    except Exception:
        return 8192
    ctx = 0
    model_info = info.get("model_info") or {}
    for key, val in model_info.items():
        if str(key).endswith("context_length"):
            try:
                ctx = max(ctx, int(val))
            except (TypeError, ValueError):
                pass
    params = info.get("parameters") or ""
    if isinstance(params, str):
        for line in params.splitlines():
            if "num_ctx" in line.lower():
                bits = line.replace("=", " ").split()
                for b in reversed(bits):
                    if b.isdigit():
                        ctx = max(ctx, int(b))
                        break
    if ctx <= 0:
        ctx = 8192
    # 8B + huge windows thrash VRAM; cap at 32k unless the native window is smaller
    return min(ctx, 32768) if ctx >= 4096 else ctx


def effective_num_ctx(host: str, model: str, override: int) -> int:
    native = native_context_length(host, model)
    if override and override > 0:
        return min(override, native if native else override)
    return native


def chat_once(
    host: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    options: dict | None = None,
    timeout: float = 180,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {},
    }
    if tools:
        payload["tools"] = tools
    return _request(host, "/api/chat", payload, timeout=timeout)


def chat_stream(
    host: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    options: dict | None = None,
    timeout: float = 600,
    on_token: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """
    Stream a chat turn. Returns the assembled assistant message plus usage:
      {message, prompt_eval_count, eval_count, used_tools_api}
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options or {},
    }
    if tools:
        payload["tools"] = tools

    def _run(with_tools: bool) -> dict:
        body = dict(payload)
        if not with_tools:
            body.pop("tools", None)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            _url(host, "/api/chat"),
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        content = ""
        tool_calls: list[dict] = []
        prompt_eval = 0
        eval_count = 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    if should_cancel and should_cancel():
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    piece = msg.get("content") or ""
                    if piece:
                        content += piece
                        if on_token:
                            on_token(piece)
                    for tc in msg.get("tool_calls") or []:
                        tool_calls.append(tc)
                    if chunk.get("prompt_eval_count"):
                        prompt_eval = int(chunk["prompt_eval_count"])
                    if chunk.get("eval_count"):
                        eval_count = int(chunk["eval_count"])
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {e.code}: {err[:400]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama unreachable at {host}: {e.reason}") from e
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "message": message,
            "prompt_eval_count": prompt_eval,
            "eval_count": eval_count,
            "used_tools_api": with_tools,
        }

    try:
        return _run(bool(tools))
    except RuntimeError as e:
        text = str(e).lower()
        if tools and ("tool" in text or "does not support" in text or "400" in text):
            return _run(False)
        raise


def pull_model(
    host: str,
    name: str,
    on_status: Callable[[str], None] | None = None,
    timeout: float = 3600,
) -> None:
    payload = json.dumps({"name": name, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        _url(host, "/api/pull"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = chunk.get("status") or ""
            completed = chunk.get("completed")
            total = chunk.get("total")
            if completed and total:
                status = f"{status} {completed}/{total}"
            if status and on_status:
                on_status(status)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
