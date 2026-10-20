"""User-turn agent loop. No heartbeat. Designed around 8B tool-calling limits."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from .hands import hands_brief, prompt_has_action, window_argument
from .compact import (
    compact_messages,
    compact_threshold,
    estimate_tokens,
    hard_trim_messages,
    calibrate_cpt,
    message_chars,
    get_cpt,
)
from .chat import context_window, stream as chat_stream
from .personality import build_system_prompt
from .reasoning import ReasonTrack, ThinkSplit, chat_predict, finish_call, normalize_effort
from .tools import run_tool, schemas_for, set_local_runtime

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
MAX_ROUNDS = 8
NORMAL_CAP = 12
ROUND_CAP = 24


def round_budget(base: int, *, autorun: bool = False) -> int:
    """Tool rounds for one run. Auto mode gets twice the normal budget."""
    try:
        base = int(base)
    except (TypeError, ValueError):
        base = MAX_ROUNDS
    base = max(1, min(base, ROUND_CAP if autorun else NORMAL_CAP))
    if autorun:
        return min(base * 2, ROUND_CAP)
    return base

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


def _ensure_system(messages: list[dict], workspace, provider: str = "xai") -> list[dict]:
    prompt = build_system_prompt(workspace, provider=provider)
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
        provider: str = "ollama",
        api_key: str = "",
        local_model: str = "",
        xai_model: str = "grok-4",
        reasoning_effort: str = "off",
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.provider = provider if provider in ("xai", "meta", "ollama") else "ollama"
        self.api_key = api_key or ""
        self.local_model = local_model or ""
        self.xai_model = (xai_model or "grok-4").strip()
        self.reasoning_effort = normalize_effort(reasoning_effort)
        self.workspace = workspace
        self.num_ctx_override = num_ctx_override
        self.temperature = temperature
        self.compact_ratio = compact_ratio
        self.max_tool_rounds = max(1, min(int(max_tool_rounds), ROUND_CAP))
        self.shell_timeout = shell_timeout
        self.emit = emit or (lambda *_a, **_k: None)
        self._cancel = False
        self.on_badge = None
        self.button_rect = None
        self.end_requested = None
        self.last_prompt_tokens = 0
        self.last_eval_tokens = 0
        self.num_ctx = 8192
        self.effective_ctx: int | None = None  # calibrated per-run; synced with compact
        self.rounds = 0

    def cancel(self) -> None:
        self._cancel = True

    def _fire(self, kind: str, **data: Any) -> None:
        self.emit(kind, data)

    def _compact_thresh(self) -> int:
        return compact_threshold(self.num_ctx, self.compact_ratio)

    def _compact(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        return compact_messages(
            self.host,
            self.model,
            messages,
            num_ctx=self.num_ctx,
            ratio=self.compact_ratio,
            on_status=lambda s: self._fire("compact", text=s),
            provider=self.provider,
            api_key=self.api_key,
            reasoning=self.reasoning_effort,
            on_reasoning=getattr(self, "_reason", None),
        )

    def run(self, messages: list[dict]) -> list[dict]:
        self._cancel = False
        chat_override = self.num_ctx_override if self.provider in ("xai", "meta") else 0
        self.num_ctx = context_window(self.provider, self.host, self.model, chat_override)
        self.grok_ctx = self.num_ctx
        self.effective_ctx = self.num_ctx
        set_local_runtime(self.host, self.local_model or (self.model if self.provider == "ollama" else ""))
        messages = _ensure_system(list(messages), self.workspace, self.provider)
        tools = schemas_for(self.provider)
        self.reasoning_effort = normalize_effort(self.reasoning_effort)
        self._reason = ReasonTrack(
            lambda text, anchor, call, label: self._fire(
                "reasoning", text=text, anchor=anchor, call=call, label=label
            ),
            f"user-{uuid.uuid4().hex[:8]}",
        )
        # Snapshot before compaction. The latest user words stay the request.
        # The recent chat is what the hands-on planner reads.
        hands_task = ""
        for prior in reversed(messages):
            if prior.get("role") == "user":
                hands_task = str(prior.get("content") or "").strip()
                break
        hands_source = hands_brief(messages)

        compacted, note = self._compact(messages)
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
            "num_predict": chat_predict(self.provider, self.reasoning_effort, self.num_ctx),
            "reasoning": self.reasoning_effort,
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

            anchor = f"round-{uuid.uuid4().hex[:8]}"
            self._reason.anchor = anchor
            self._reason.start("REASONING")
            assembled: list[str] = []
            splitter = ThinkSplit()
            effort_on = self.reasoning_effort != "off"

            def on_token(piece: str, _split: ThinkSplit = splitter, _anchor: str = anchor) -> None:
                if effort_on:
                    content_d, reason_d = _split.feed(piece)
                    if reason_d:
                        self._reason.add(reason_d, provider=self.provider, effort=self.reasoning_effort)
                    piece = content_d
                if not piece:
                    return
                assembled.append(piece)
                self._fire("token", text=piece, anchor=_anchor)

            def on_reasoning(piece: str) -> None:
                self._reason.add(piece, provider=self.provider, effort=self.reasoning_effort)

            try:
                result = chat_stream(
                    self.provider,
                    host=self.host,
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    options=options,
                    on_token=on_token,
                    on_reasoning=on_reasoning,
                    should_cancel=lambda: self._cancel,
                )
            except Exception as e:
                self._fire("error", text=str(e))
                break

            if effort_on:
                tail_c, tail_r = splitter.flush()
                if tail_r:
                    self._reason.add(tail_r, provider=self.provider, effort=self.reasoning_effort)
                if tail_c:
                    assembled.append(tail_c)
                    self._fire("token", text=tail_c, anchor=anchor)

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
            raw_msg = result.get("message") or {"role": "assistant", "content": ""}
            msg = dict(raw_msg)
            content = finish_call(
                self._reason,
                self.provider,
                self.reasoning_effort,
                msg,
                hidden=bool(result.get("reasoning_hidden")),
            )
            if effort_on and splitter.saw_think:
                content = "".join(assembled)
                msg["content"] = content
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
                if name == "hands_on":
                    arguments = dict(arguments)
                    proposed = str(arguments.get("prompt") or "").strip()
                    if hands_task and (not proposed or not prompt_has_action(proposed)):
                        arguments["prompt"] = hands_task
                    arguments["window"] = window_argument(
                        str(arguments.get("window") or ""),
                        f"{hands_task}\n{arguments.get('prompt') or ''}",
                    )
                    arguments["brief"] = hands_source
                # P3-1: link tool result to its tool_call by id when present;
                # fall back to uuid so models that require linked ids still work.
                tc_id = tc.get("id") or fn.get("id") or uuid.uuid4().hex
                shown = dict(arguments)
                shown.pop("brief", None)
                self._fire("tool_start", name=name, arguments=shown)
                look_model = self.model
                look_ctx = self.num_ctx
                output = run_tool(
                    name,
                    arguments,
                    self.workspace,
                    shell_timeout=self.shell_timeout,
                    host=self.host,
                    model=look_model,
                    provider=self.provider,
                    api_key=self.api_key,
                    num_ctx=look_ctx,
                    should_cancel=lambda: self._cancel,
                    on_status=lambda s: (
                        self._fire("status", text=s),
                        self._fire("hands", text=s),
                    ),
                    on_window=lambda hide: self._fire("hands_window", hide=bool(hide)),
                    on_badge=self.on_badge,
                    button_rect=self.button_rect,
                    end_requested=self.end_requested,
                    reasoning=self.reasoning_effort,
                    reason_track=self._reason,
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
            thresh = self._compact_thresh()
            if not self._cancel and estimate_tokens(messages) >= thresh:
                if mid_loop_full_compacts < 1:
                    if mid_loop_full_compacts == 0:
                        # Only need a small status note; full status comes from
                        # compact_messages via its on_status callback.
                        self._fire("compact", text="mid-loop compact")
                    compacted, mid_note = self._compact(messages)
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
