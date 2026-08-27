"""Autocompaction using the selected model, sized to its native context window."""
from __future__ import annotations

import logging
from typing import Callable

from .ollama_client import chat_once

logger = logging.getLogger(__name__)

# 8B models need headroom for the next user turn + a couple of tool results
OUTPUT_RESERVE = 2048
CHARS_PER_TOKEN = 4

COMPACT_SYS = (
    "You compress a chat log into a brief for your future self. "
    "Keep: user goals, file paths, commands, decisions, errors, unfinished work. "
    "Drop greetings and repeated tool dumps. Output plain prose, max 280 words. "
    "No tools. No questions."
)

MAX_TRUNCATE_CHARS = 800      # cap for non-system content
MAX_LAST_USER_CHARS = 2000     # P3-3: higher floor for the latest user turn
MIN_SYSTEM_CHARS = 200         # P3-3: never shrink primary system below this
MAX_TRIM_PASSES = 8            # bounded passes to guarantee progress

# P2-1: calibrated chars-per-token. Starts at the default; updated from
# prompt_eval_count after real calls. Clamped to [2, 8] to stay sane.
_DEFAULT_CPT = 4
_CPT_MIN, _CPT_MAX = 2, 8
_cpt: float = _DEFAULT_CPT
_cpt_ema: float = 0.0  # EMA of (chars / prompt_eval_count) across recent calls


def get_cpt() -> float:
    """Return the current chars-per-token calibration."""
    return _cpt


def calibrate_cpt(chars: int, prompt_tokens: int) -> None:
    """
    Update the chars-per-token estimate from a real (chars, prompt_tokens)
    pair. Uses exponential moving average (alpha=0.3) and clamps to [2, 8].
    Safe to call with zero/invalid values — those are ignored.
    """
    global _cpt, _cpt_ema
    if not chars or not prompt_tokens or prompt_tokens <= 0:
        return
    sample = float(chars) / float(prompt_tokens)
    if _cpt_ema <= 0:
        _cpt_ema = sample
    else:
        _cpt_ema = 0.3 * sample + 0.7 * _cpt_ema
    _cpt = max(_CPT_MIN, min(_CPT_MAX, _cpt_ema))


def reset_calibration() -> None:
    """Reset calibration to defaults. Useful on model change."""
    global _cpt, _cpt_ema
    _cpt = _DEFAULT_CPT
    _cpt_ema = 0.0


def estimate_tokens(messages: list[dict]) -> int:
    n = 0
    cpt = _cpt
    for m in messages:
        n += 8
        content = m.get("content") or ""
        n += int(len(content) / cpt) + 1
        for tc in m.get("tool_calls") or []:
            n += int(len(str(tc)) / cpt) + 8
    return n


def message_chars(messages: list[dict]) -> int:
    """Total content + tool_call character length — for calibration."""
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(str(tc))
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


def _primary_system(messages: list[dict]) -> dict | None:
    """Return the *primary* (non-memory) system message, or None."""
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content") or ""
        if str(content).startswith("[compacted memory"):
            continue
        return {"role": "system", "content": content}
    return None


def _previous_memory(messages: list[dict]) -> str | None:
    """Return the body of the most recent [compacted memory] block, or None."""
    for m in reversed(messages):
        if m.get("role") != "system":
            continue
        content = m.get("content") or ""
        if str(content).startswith("[compacted memory"):
            # Strip the header line
            text = content.split("\n", 1)
            return text[1] if len(text) > 1 else None
    return None


def split_for_compact(messages: list[dict], keep_user_turns: int = 2) -> tuple[list[dict], list[dict]]:
    """Keep the last N complete user turns (including their tool loops). Never split a tool loop."""
    if len(messages) < 8:
        return [], messages
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_idxs) <= keep_user_turns:
        return [], messages
    cut = user_idxs[-keep_user_turns]
    while cut > 0 and _is_toolish(messages[cut]):
        cut -= 1
    prefix, suffix = messages[:cut], messages[cut:]
    if not prefix:
        return [], messages
    return prefix, suffix


def _prefix_text(prefix: list[dict], char_budget: int = 12000, prev_memory: str | None = None) -> str:
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
    if prev_memory:
        # P3-2: fold the prior summary into the new compress input so memory
        # accumulates across re-compacts instead of being silently dropped.
        text = f"Earlier summary (fold into new):\n{prev_memory}\n\n---\nNew turns:\n{text}"
    if len(text) <= char_budget:
        return text
    keep = char_budget // 2
    return text[:keep] + "\n…\n" + text[-keep:]


def _last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def hard_trim_messages(messages: list[dict], thresh: int) -> list[dict]:
    """
    Reduce a message list so estimate_tokens(messages) <= thresh, with a clear
    contract (P3-3 — softer order to preserve identity and the active ask):

      1. Drop oldest non-system messages first, but never the last user turn
         nor its trailing tool loop. Drop aggressively until under threshold
         or we run out of droppable messages.
      2. If still over threshold, shrink oversized non-system content
         (tools, memory body, assistant prose) to MAX_TRUNCATE_CHARS.
      3. If still over threshold, shrink the *body* of [compacted memory]
         blocks (keep header) — but only the body, never the marker.
      4. If still over threshold, truncate the last user turn's content, but
         never below MAX_LAST_USER_CHARS (preserve the active ask).
      5. Last resort only: truncate the primary system prompt, but never
         below MIN_SYSTEM_CHARS (preserve identity / runtime rules).

    Bounded — outer loop is capped. Each stage either makes progress or we
    stop. No infinite loops.
    """
    if estimate_tokens(messages) <= thresh:
        return list(messages)

    work = [dict(m) for m in messages]  # shallow copies so content edits stick

    for _ in range(MAX_TRIM_PASSES):
        if estimate_tokens(work) <= thresh:
            return work

        before = estimate_tokens(work)
        last_user = _last_user_index(work)

        # Stage 1: drop oldest non-system messages, repeatedly, until under
        # threshold or there is nothing left to drop.
        dropped = True
        while dropped and estimate_tokens(work) > thresh:
            dropped = False
            non_sys_positions = [i for i, m in enumerate(work) if m.get("role") != "system"]
            if not non_sys_positions:
                break
            first = non_sys_positions[0]
            # Protect the last user turn and its trailing tool loop.
            if first >= last_user:
                break
            work = work[:first] + work[first + 1 :]
            last_user = _last_user_index(work)
            dropped = True
        if estimate_tokens(work) <= thresh:
            return work
        if estimate_tokens(work) < before:
            continue

        # Stage 2: shrink oversized non-system content (tools, assistant prose).
        # P3-3: protect the last user turn — its truncation lives in stage 4.
        any_shrunk = False
        for idx, m in enumerate(work):
            if m.get("role") == "system":
                continue
            if idx == last_user:
                continue
            content = m.get("content") or ""
            if len(content) > MAX_TRUNCATE_CHARS:
                m["content"] = content[:MAX_TRUNCATE_CHARS] + "…"
                any_shrunk = True
        if any_shrunk and estimate_tokens(work) < before:
            continue

        # Stage 3 (P3-3): shrink the body of [compacted memory] blocks, keep header.
        mem_shrunk = False
        for m in work:
            if m.get("role") != "system":
                continue
            content = m.get("content") or ""
            if not str(content).startswith("[compacted memory"):
                continue
            header, _, body = content.partition("\n")
            if len(body) > MAX_TRUNCATE_CHARS:
                m["content"] = header + "\n" + body[:MAX_TRUNCATE_CHARS] + "…"
                mem_shrunk = True
            elif len(body) > 0:
                # Already short — skip
                pass
        if mem_shrunk and estimate_tokens(work) < before:
            continue

        # Stage 4 (P3-3): truncate last user turn, but only down to MAX_LAST_USER_CHARS.
        if last_user >= 0:
            content = work[last_user].get("content") or ""
            if len(content) > MAX_LAST_USER_CHARS:
                work[last_user]["content"] = content[:MAX_LAST_USER_CHARS] + "…"
                if estimate_tokens(work) < before:
                    continue

        # Stage 5 (P3-3): last-resort — shrink primary system prompt, never below MIN_SYSTEM_CHARS.
        for m in work:
            if m.get("role") != "system":
                continue
            content = m.get("content") or ""
            if str(content).startswith("[compacted memory"):
                continue  # never touch compacted memory blocks
            if len(content) > MAX_TRUNCATE_CHARS and len(content) > MIN_SYSTEM_CHARS:
                # Keep at least MIN_SYSTEM_CHARS to preserve identity / runtime rules
                new_len = max(MIN_SYSTEM_CHARS, MAX_TRUNCATE_CHARS)
                if len(content) > new_len:
                    m["content"] = content[:new_len] + "…"
                    if estimate_tokens(work) < before:
                        break
        if estimate_tokens(work) < before:
            continue

        # No progress this iteration — stop to avoid an infinite loop.
        break

    return work


def _hard_trim(messages: list[dict], thresh: int) -> list[dict]:
    """Internal alias; UI/agent should call hard_trim_messages()."""
    return hard_trim_messages(messages, thresh)


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

    Successful compact  -> primary system + one memory system + non-system suffix,
                           then hard-trim only. No full-history rebuild.
    Failed compact      -> primary system + non-system suffix, then hard-trim.
                           Logs safely, never raises.
    """
    thresh = compact_threshold(num_ctx, ratio)
    used = estimate_tokens(messages)
    if used < thresh:
        return messages, None

    # Identify structural pieces from the input up front.
    primary = _primary_system(messages) or {"role": "system", "content": ""}

    prefix, split_suffix = split_for_compact(messages)
    if not prefix:
        # Could not split turns (e.g. too few user turns) — fall through to hard-trim.
        suffix_non_sys = [m for m in messages if m.get("role") != "system"]
        kept = [primary] + suffix_non_sys
        kept = hard_trim_messages(kept, thresh)
        return kept, "hard-trimmed (could not split turns)"

    # Use split_suffix to get the non-system suffix messages
    suffix_non_sys = [m for m in split_suffix if m.get("role") != "system"]

    if on_status:
        on_status(f"Compacting {used} tok → {thresh} tok window ({num_ctx} ctx)…")

    # P2-2: size the compact sub-call to its own window.
    # Use the smaller of the effective ctx and 16k (richer summaries for large windows).
    compact_ctx = min(num_ctx, 16384)
    num_predict = 400
    # Reserve space for system prompt + user prompt overhead (~200 tokens).
    reserve = OUTPUT_RESERVE + 200
    usable = max(compact_ctx - num_predict - reserve, compact_ctx // 4)
    char_budget = max(usable * CHARS_PER_TOKEN, 2000)

    blob = _prefix_text(prefix, char_budget=int(char_budget), prev_memory=_previous_memory(messages))
    result = None
    summary = ""
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
                "num_ctx": compact_ctx,
                "num_predict": num_predict,
            },
            timeout=120,
        )
        summary = ((result.get("message") or {}).get("content") or "").strip()
    except Exception as e:
        summary = ""
        result = None
        if on_status:
            on_status(f"Compact model call failed ({e}); dropping old turns")

    # ----- Empty / failed compact -----
    if not summary:
        # `result` is always defined here (None on exception, dict otherwise).
        logger.error("Empty summary from Ollama: %r", result)
        # Build a clean fallback: primary system + non-system suffix, no full history.
        kept = [primary] + suffix_non_sys
        kept = hard_trim_messages(kept, thresh)
        return kept, f"dropped old turns (empty compact, retained {len(kept)} msgs)"

    # ----- Successful compact -----
    memory = {
        "role": "system",
        "content": "[compacted memory — earlier turns]\n" + summary,
    }
    new_msgs = [primary, memory] + suffix_non_sys
    new_msgs = hard_trim_messages(new_msgs, thresh)
    return new_msgs, summary[:240]