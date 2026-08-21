"""Autocompaction using the selected model, sized to its native context window."""
from __future__ import annotations

import logging
from typing import Callable

from .ollama_client import chat_once

logger = logging.getLogger(__name__)

# 8B models need headroom for the next user turn + a couple of tool results
OUTPUT_RESERVE = 2048
CHARS_PER_TOKEN = 3.6

COMPACT_SYS = (
    "You compress a chat log into a brief for your future self. "
    "Keep: user goals, file paths, commands, decisions, errors, unfinished work. "
    "Drop greetings and repeated tool dumps. Output plain prose, max 280 words. "
    "No tools. No questions."
)


def estimate_tokens(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        n += 8
        content = m.get("content") or ""
        n += int(len(content) / CHARS_PER_TOKEN) + 1
        for tc in m.get("tool_calls") or []:
            n += int(len(str(tc)) / CHARS_PER_TOKEN) + 8
    return n


def compact_threshold(num_ctx: int, ratio: float = 0.62) -> int:
    ctx = max(int(num_ctx or 8192), 2048)
    usable = max(ctx - OUTPUT_RESERVE, ctx // 2)
    return max(int(usable * ratio), 1024)


def _is_toolish(msg: dict) -> bool:
    if msg.get("role") == "tool":
        return True
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        return True
    return False


def split_for_compact(messages: list[dict], keep_user_turns: int = 2) -> tuple[list[dict], list[dict]]:
    """Keep the last N complete user turns (including their tool loops). Never split a tool loop."""
    if len(messages) < 8:
        return [], messages
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_idxs) <= keep_user_turns:
        return [], messages
    cut = user_idxs[-keep_user_turns]
    # Only rewind if we landed *inside* a tool loop (assistant + tool msgs).
    # A user cut point is a real turn boundary — tools *before* it belong to prefix.
    while cut > 0 and _is_toolish(messages[cut]):
        cut -= 1
    prefix, suffix = messages[:cut], messages[cut:]
    if not prefix:
        return [], messages
    return prefix, suffix


def _prefix_text(prefix: list[dict], char_budget: int = 12000) -> str:
    lines: list[str] = []
    for m in prefix:
        role = m.get("role") or "?"
        content = (m.get("content") or "").strip()
        if role == "system":
            continue
        if m.get("tool_calls"):
            names = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or tc
                names.append(str(fn.get("name") or "?"))
            content = (content + "\n" if content else "") + "tools: " + ", ".join(names)
        if not content:
            continue
        if len(content) > 800:
            content = content[:800] + "…"
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) <= char_budget:
        return text
    keep = char_budget // 2
    return text[:keep] + "\n…\n" + text[-keep:]


def compact_messages(
    host: str,
    model: str,
    messages: list[dict],
    *,
    num_ctx: int,
    ratio: float = 0.62,
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[dict], str | None]:
    """
    If over the model's compact threshold, summarize the older turns with the
    same model and splice a memory block back in. Returns (messages, note).
    """
    thresh = compact_threshold(num_ctx, ratio)
    used = estimate_tokens(messages)
    if used < thresh:
        return messages, None

    prefix, suffix = split_for_compact(messages)
    if not prefix:
        # still too big: drop oldest non-system until under threshold
        return _hard_trim(messages, thresh), "hard-trimmed (could not split turns)"

    if on_status:
        on_status(f"Compacting {used} tok → {thresh} tok window ({num_ctx} ctx)…")

    blob = _prefix_text(prefix)
    try:
        result = chat_once(
            host,
            model,
            [
                {"role": "system", "content": COMPACT_SYS},
                {"role": "user", "content": "Compress this log:\n\n" + blob},
            ],
            options={
                "temperature": 0.1,
                "num_ctx": min(num_ctx, 8192),
                "num_predict": 400,
            },
            timeout=120,
        )
        summary = ((result.get("message") or {}).get("content") or "").strip()
    except Exception as e:
        summary = ""
        if on_status:
            on_status(f"Compact model call failed ({e}); dropping old turns")

    if not summary:
        logger.error(f"Empty summary from Ollama: {result}")
        # Don't drop prefix. Keep it as-is to preserve context.
        kept = [m for m in messages if m.get("role") == "system"][:1] + suffix
        kept = _hard_trim(kept, thresh)
        # Enforce minimum retention: keep last 2 user turns + their tools
        kept = _enforce_min_kept(kept, messages)
        return kept, f"dropped old turns (empty compact, retained {len(kept)} msgs)"

    system = [m for m in messages if m.get("role") == "system"][:1]
    memory = {
        "role": "system",
        "content": "[compacted memory — earlier turns]\n" + summary,
    }
    new_msgs = system + [memory] + [m for m in suffix if m.get("role") != "system"]
    new_msgs = _hard_trim(new_msgs, thresh)
    # Enforce minimum retention even after successful compact
    new_msgs = _enforce_min_kept(new_msgs, messages)
    return new_msgs, summary[:240]


def _hard_trim(messages: list[dict], thresh: int) -> list[dict]:
    if estimate_tokens(messages) <= thresh:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    while rest and estimate_tokens(system + rest) > thresh:
        # never drop the latest user message or their tools
        last_idx = len(rest) - 1
        if rest[last_idx].get("role") == "user":
            break
        if rest[last_idx].get("tool_calls"):
            break
        rest.pop(0)
    return system + rest


def _enforce_min_kept(messages: list[dict], original: list[dict]) -> list[dict]:
    """Ensure we keep at least last 2 user turns + their tool loops from original."""
    if len(messages) >= len(original):
        return messages
    # Find last 2 user turn indices in original
    user_idxs = [i for i, m in enumerate(original) if m.get("role") == "user"]
    if len(user_idxs) < 2:
        return messages
    keep_from = user_idxs[-2]
    # Rebuild from original: system + everything from keep_from onward
    result = [m for m in original if m.get("role") == "system"]
    for m in original:
        if original.index(m) >= keep_from:
            result.append(m)
    return result
