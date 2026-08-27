"""User-turn agent loop. No heartbeat. Designed around 8B tool-calling limits."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from .compact import (
    compact_messages,
    compact_threshold,
    estimate_tokens,
    hard_trim_messages,
    calibrate_cpt,
    message_chars,
    get_cpt,
)
from .ollama_client import chat_stream, effective_num_ctx
from .personality import build_system_prompt
from .tools import SCHEMAS, run_tool

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
MAX_ROUNDS = 8

EventFn = Callable[[str, dict], None]


def _args(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_text_tool_calls(content: str) -> list[dict]:
    calls = []
    for m in TOOL_RE.finditer(content or ""):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool")
        if not name:
            continue
        calls.append(
            {
                "function": {
                    "name": str(name),
                    "arguments": obj.get("arguments") or obj.get("args") or {},
                }
            }
        )
    return calls


def _ensure_system(messages: list[dict], workspace) -> list[dict]:
    prompt = build_system_prompt(workspace)
    primary = {"role": "system", "content": prompt}
    last_memory = None
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            if str(m.get("content") or "").startswith("[compacted memory"):
                last_memory = m
            continue
        rest.append(m)
    kept = [primary]
    if last_memory:
        kept.append(last_memory)
    kept.extend(rest)
    return kept


class Agent:
    def __init__(
        self,
        host: str,
        model: str,
        workspace,
        *,
        num_ctx_override: int = 0,
        temperature: float = 0.2,
        compact_ratio: float = 0.62,
        max_tool_rounds: int = MAX_ROUNDS,
        shell_timeout: int = 60,
        emit: EventFn | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.workspace = workspace
        self.num_ctx_override = num_ctx_override
        self.temperature = temperature
        self.compact_ratio = compact_ratio
        self.max_tool_rounds = max(1, min(int(max_tool_rounds), 12))
        self.shell_timeout = shell_timeout
        self.emit = emit or (lambda *_a, **_k: None)
        self._cancel = False
        self.last_prompt_tokens = 0
        self.last_eval_tokens = 0
        self.num_ctx = 8192
        self.effective_ctx: int | None = None  # calibrated per-run; synced with compact
        self.rounds = 0

    def cancel(self) -> None:
        self._cancel = True

    def _fire(self, kind: str, **data: Any) -> None:
        self.emit(kind, data)

    def run(self, messages: list[dict]) -> list[dict]:
        self._cancel = False
        self.num_ctx = effective_num_ctx(self.host, self.model, self.num_ctx_override)
        # P2-3: expose the same effective ctx to the UI so the bar and the
        # compact threshold agree with what the agent actually uses.
        self.effective_ctx = self.num_ctx
        messages = _ensure_system(list(messages), self.workspace)

        compacted, note = compact_messages(
            self.host,
            self.model,
            messages,
            num_ctx=self.num_ctx,
            ratio=self.compact_ratio,
            on_status=lambda s: self._fire("compact", text=s),
        )
        messages = compacted
        if note:
            self._fire("compact", text=note, estimate=estimate_tokens(messages))

        # P1-2: mid-loop context-pressure handling.
        # Policy: at most ONE extra full LLM compact per run(); afterwards, only
        # cheap hard-trim. This bounds latency and prevents recursive "compact
        # the compact" cost while still preventing unbounded growth between turns.
        mid_loop_full_compacts = 0

        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": min(1024, max(256, self.num_ctx // 16)),
        }

        rounds = 0
        while rounds < self.max_tool_rounds:
            if self._cancel:
                self._fire("error", text="cancelled")
                break
            rounds += 1
            self.rounds = rounds
            self._fire(
                "status",
                text=f"thinking · ctx {self.num_ctx} · round {rounds}/{self.max_tool_rounds}",
                round=rounds,
                max_rounds=self.max_tool_rounds,
            )

            assembled: list[str] = []

            def on_token(piece: str) -> None:
                assembled.append(piece)
                self._fire("token", text=piece)

            try:
                result = chat_stream(
                    self.host,
                    self.model,
                    messages,
                    tools=SCHEMAS,
                    options=options,
                    on_token=on_token,
                    should_cancel=lambda: self._cancel,
                )
            except Exception as e:
                self._fire("error", text=str(e))
                break

            self.last_prompt_tokens = int(result.get("prompt_eval_count") or 0)
            self.last_eval_tokens = int(result.get("eval_count") or 0)
            # prompt_eval_count is for the request payload — calibrate before appending the assistant message.
            if self.last_prompt_tokens > 0:
                calibrate_cpt(message_chars(messages), self.last_prompt_tokens)
                self._fire(
                    "tokens",
                    used=self.last_prompt_tokens,
                    est=estimate_tokens(messages),
                    cpt=get_cpt(),
                )
            msg = result.get("message") or {"role": "assistant", "content": ""}
            content = msg.get("content") or ""
            tool_calls = list(msg.get("tool_calls") or [])
            if not tool_calls:
                tool_calls = parse_text_tool_calls(content)
                if tool_calls:
                    # hide the XML from the user-facing transcript
                    cleaned = TOOL_RE.sub("", content).strip()
                    msg = dict(msg)
                    msg["content"] = cleaned
                    msg["tool_calls"] = tool_calls
                    content = cleaned

            messages.append(msg)

            if not tool_calls:
                self._fire(
                    "done",
                    text=content,
                    complete=True,
                    tokens=self.last_prompt_tokens,
                    estimate=estimate_tokens(messages),
                    num_ctx=self.num_ctx,
                )
                return messages

            # 8B models sometimes talk AND call tools; keep the talk visible
            for tc in tool_calls:
                if self._cancel:
                    break
                fn = tc.get("function") or tc
                name = str(fn.get("name") or "")
                arguments = _args(fn.get("arguments"))
                # P3-1: link tool result to its tool_call by id when present;
                # fall back to uuid so models that require linked ids still work.
                tc_id = tc.get("id") or fn.get("id") or uuid.uuid4().hex
                self._fire("tool_start", name=name, arguments=arguments)
                output = run_tool(
                    name, arguments, self.workspace, shell_timeout=self.shell_timeout
                )
                self._fire("tool_end", name=name, output=output)
                messages.append(
                    {
                        "role": "tool",
                        "content": output,
                        "tool_name": name,
                        "name": name,
                        "tool_call_id": tc_id,
                    }
                )

            # P1-2: after the tool batch, if context is over threshold, compact
            # or hard-trim before the next model call. Bound full LLM compacts
            # to one per run(); fall back to hard_trim_messages otherwise.
            thresh = compact_threshold(self.num_ctx, self.compact_ratio)
            if not self._cancel and estimate_tokens(messages) >= thresh:
                if mid_loop_full_compacts < 1:
                    if mid_loop_full_compacts == 0:
                        # Only need a small status note; full status comes from
                        # compact_messages via its on_status callback.
                        self._fire("compact", text="mid-loop compact")
                    compacted, mid_note = compact_messages(
                        self.host,
                        self.model,
                        messages,
                        num_ctx=self.num_ctx,
                        ratio=self.compact_ratio,
                        on_status=lambda s: self._fire("compact", text=s),
                    )
                    messages = compacted
                    mid_loop_full_compacts += 1
                    if mid_note:
                        self._fire("compact", text=mid_note, estimate=estimate_tokens(messages))
                else:
                    before = estimate_tokens(messages)
                    messages = hard_trim_messages(messages, thresh)
                    if estimate_tokens(messages) < before:
                        self._fire(
                            "compact",
                            text="mid-loop hard-trim",
                            estimate=estimate_tokens(messages),
                        )

        else:
            self._fire(
                "done",
                text="Stopped after max tool rounds (small-model guard). Send another message to continue.",
                complete=False,
                tokens=self.last_prompt_tokens,
                estimate=estimate_tokens(messages),
                num_ctx=self.num_ctx,
            )
        return messages
