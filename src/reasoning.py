"""Reasoning effort shared by Grok, Muse, and local models.

Off sends the shortest level a provider allows. Muse rejects "none", and several
Grok models do too, so those calls still think a little and the pane stays empty.
A pane line appears only when a call was asked to reason and either returned
text or kept the trace private.
"""
from __future__ import annotations

import re
from typing import Any, Callable

EFFORTS = ("off", "low", "medium", "high")
RESTRICTED = "output restricted"

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_META_EXTRA = {"off": 400, "low": 600, "medium": 1000, "high": 1600}
_OTHER_EXTRA = {"off": 0, "low": 256, "medium": 512, "high": 768}


def normalize_effort(value: str | None) -> str:
    text = str(value or "").strip().lower()
    return text if text in EFFORTS else "off"


def cloud_attempts(provider: str, effort: str | None) -> list[str | None]:
    """reasoning_effort values to try. None omits the field."""
    level = normalize_effort(effort)
    if provider == "meta":
        if level == "off":
            return ["minimal", None]
        return [level, None]
    if level == "off":
        return ["none", "low", None]
    return [level, None]


def cloud_effort(provider: str, effort: str | None) -> str:
    """First reasoning_effort to send. Muse uses minimal when the control is off."""
    first = cloud_attempts(provider, effort)[0]
    if first:
        return first
    return "minimal" if provider == "meta" else "low"


def effort_rejected(exc: BaseException) -> bool:
    """True when the server refused the reasoning field and another value might work."""
    text = str(exc).lower()
    if "reasoning" in text:
        return True
    return "http 400" in text or "http 422" in text


def effort_was_hidden(provider: str, effort: str | None, used: str | None) -> bool:
    """The API refused the requested level, so the trace will not come back."""
    if normalize_effort(effort) == "off":
        return False
    return used != cloud_attempts(provider, effort)[0]


def ollama_think(effort: str | None) -> bool | str:
    level = normalize_effort(effort)
    if level == "off":
        return False
    return level


def model_options(options: dict | None) -> dict:
    """Ollama options. The effort is sent as think, not as an option field."""
    raw = dict(options or {})
    raw.pop("reasoning", None)
    return raw


def display_reasoning(
    provider: str,
    effort: str | None,
    text: str,
    *,
    hidden: bool = False,
) -> str | None:
    """Text for one pane card, or None when the pane should stay quiet."""
    if normalize_effort(effort) == "off":
        return None
    body = (text or "").strip()
    if body:
        return body
    if provider == "meta" or hidden:
        return RESTRICTED
    return None


def predict_budget(base: int, provider: str, effort: str | None, *, cap: int) -> int:
    """Output budget plus room for reasoning, so the answer is not cut off."""
    level = normalize_effort(effort)
    try:
        base_n = int(base)
    except (TypeError, ValueError):
        base_n = 0
    extra = (_META_EXTRA if provider == "meta" else _OTHER_EXTRA)[level]
    return max(1, min(int(cap), base_n + extra))


def chat_predict(provider: str, effort: str | None, num_ctx: int) -> int:
    try:
        ctx = int(num_ctx)
    except (TypeError, ValueError):
        ctx = 0
    base = min(1024, max(256, ctx // 16 if ctx else 256))
    return predict_budget(base, provider, effort, cap=4096)


def hands_predict(provider: str, effort: str | None, *, look: bool) -> int:
    return predict_budget(640 if look else 280, provider, effort, cap=3072 if look else 2048)


def plan_predict(provider: str, effort: str | None) -> int:
    return predict_budget(420, provider, effort, cap=2048)


def compact_predict(provider: str, effort: str | None, num_ctx: int) -> int:
    try:
        ctx = int(num_ctx)
    except (TypeError, ValueError):
        ctx = 0
    base = min(4096, max(600, ctx // 4 if ctx else 600))
    return predict_budget(base, provider, effort, cap=6144)


def text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_of(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "thinking"):
            found = value.get(key)
            if isinstance(found, str):
                return found
        return ""
    return ""


def pull_reasoning(message: dict | None) -> str:
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning", "reasoning_content", "thinking"):
        text = text_of(message.get(key)).strip()
        if text:
            return text
    return ""


def usage_reasoning_tokens(data: dict | None) -> int:
    usage = (data or {}).get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return 0
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if not isinstance(details, dict):
            continue
        try:
            return int(details.get("reasoning_tokens") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def with_effort(payload: dict, provider: str, effort: str | None) -> list[dict]:
    bodies: list[dict] = []
    for level in cloud_attempts(provider, effort):
        body = dict(payload)
        if level:
            body["reasoning_effort"] = level
        bodies.append(body)
    return bodies


def split_think(text: str) -> tuple[str, str]:
    """Split closed <think> blocks out of an answer."""
    raw = text or ""
    parts = [p.strip() for p in _THINK_BLOCK.findall(raw) if p and p.strip()]
    cleaned = _THINK_BLOCK.sub("", raw).strip()
    return cleaned, "\n\n".join(parts)


def _tag_prefix_len(text: str, tag: str) -> int:
    lower = text.lower()
    limit = min(len(tag) - 1, len(lower))
    for n in range(limit, 0, -1):
        if tag.startswith(lower[-n:]):
            return n
    return 0


class ThinkSplit:
    """Pull <think> blocks out of a token stream without eating the answer."""

    def __init__(self) -> None:
        self.hold = ""
        self.inside = False
        self.saw_think = False
        self.reasoning = ""
        self.content = ""

    def feed(self, piece: str) -> tuple[str, str]:
        if not piece:
            return "", ""
        self.hold += piece
        content_out: list[str] = []
        reason_out: list[str] = []
        while self.hold:
            if not self.inside:
                idx = self.hold.lower().find("<think>")
                if idx < 0:
                    keep = _tag_prefix_len(self.hold, "<think>")
                    emit = self.hold[:-keep] if keep else self.hold
                    self.hold = self.hold[-keep:] if keep else ""
                    if emit:
                        content_out.append(emit)
                        self.content += emit
                    break
                self.saw_think = True
                if idx:
                    content_out.append(self.hold[:idx])
                    self.content += self.hold[:idx]
                self.hold = self.hold[idx + len("<think>") :]
                self.inside = True
            else:
                idx = self.hold.lower().find("</think>")
                if idx < 0:
                    keep = _tag_prefix_len(self.hold, "</think>")
                    emit = self.hold[:-keep] if keep else self.hold
                    self.hold = self.hold[-keep:] if keep else ""
                    if emit:
                        reason_out.append(emit)
                        self.reasoning += emit
                    break
                if idx:
                    reason_out.append(self.hold[:idx])
                    self.reasoning += self.hold[:idx]
                self.hold = self.hold[idx + len("</think>") :]
                self.inside = False
        return "".join(content_out), "".join(reason_out)

    def flush(self) -> tuple[str, str]:
        if not self.hold:
            return "", ""
        piece = self.hold
        self.hold = ""
        if self.inside:
            self.reasoning += piece
            return "", piece
        self.content += piece
        return piece, ""


class ReasonTrack:
    """One pane card per model call. The anchor is the related chat message."""

    def __init__(self, emit: Callable[[str, str, str, str], None], anchor: str, label: str = "REASONING"):
        self.emit = emit
        self.anchor = anchor
        self.label = label
        self.call: str | None = None
        self.acc = ""
        self._n = 0

    def start(self, label: str | None = None) -> None:
        if label:
            self.label = label
        self.call = None
        self.acc = ""

    def add(self, piece: str, *, provider: str, effort: str) -> None:
        if not piece or normalize_effort(effort) == "off":
            return
        self.acc += piece
        self._publish(provider, effort, final=False, hidden=False)

    def finish(self, provider: str, effort: str, extra: str = "", *, hidden: bool = False) -> None:
        extra = (extra or "").strip()
        if extra:
            current = self.acc.strip()
            if not current:
                self.acc = extra
            elif extra not in current and current not in extra:
                self.acc = f"{current}\n{extra}"
            elif len(extra) > len(current) and current in extra:
                self.acc = extra
        self._publish(provider, effort, final=True, hidden=hidden)
        self.call = None
        self.acc = ""

    def _publish(self, provider: str, effort: str, *, final: bool, hidden: bool) -> None:
        if normalize_effort(effort) == "off":
            return
        text = self.acc.strip()
        if text:
            shown: str | None = text
        elif final:
            shown = display_reasoning(provider, effort, "", hidden=hidden)
            if not shown:
                return
        else:
            return
        if self.call is None:
            self._n += 1
            self.call = f"{self.anchor}-{self._n}"
        self.emit(shown, self.anchor, self.call, self.label)


def finish_call(
    track: ReasonTrack | None,
    provider: str,
    effort: str | None,
    message: dict | None,
    *,
    hidden: bool = False,
) -> str:
    """Record one call's reasoning and return the answer without think blocks."""
    msg = message if isinstance(message, dict) else {}
    content = text_of(msg.get("content"))
    cleaned, inline = split_think(content)
    extra = pull_reasoning(msg)
    level = normalize_effort(effort)
    if inline and level != "off" and inline not in extra:
        extra = f"{extra}\n{inline}".strip() if extra else inline
    if track is not None:
        track.finish(provider, level, extra, hidden=hidden)
    for key in ("reasoning", "reasoning_content", "thinking"):
        msg.pop(key, None)
    if level != "off" and inline:
        msg["content"] = cleaned
        return cleaned
    return content
