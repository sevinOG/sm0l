"""hands_on: screenshot one window and let the vision model drive it.

A plan is written once from the chat that started the run. That plan is sent
again, unchanged, on every turn. Only the screenshot, the control list, and a
short action log change.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Callable

from .media import save_encoded


def _look(
    provider: str,
    host: str,
    api_key: str,
    model: str,
    messages: list[dict],
    options: dict,
    timeout: float,
    should_cancel: Callable[[], bool] | None,
    reason_track=None,
    reason_label: str = "REASONING",
) -> dict:
    """The active cloud model looks. A local model looks only in local mode."""
    from .chat import stream
    from .reasoning import finish_call, normalize_effort

    effort = normalize_effort((options or {}).get("reasoning"))
    provider_name = provider or "xai"
    if reason_track is not None:
        reason_track.start(reason_label)

    def _on_reason(piece: str) -> None:
        if reason_track is not None and piece:
            reason_track.add(piece, provider=provider_name, effort=effort)

    result = stream(
        provider_name,
        host=host,
        api_key=api_key,
        model=model,
        messages=messages,
        tools=None,
        options=options,
        timeout=timeout,
        on_reasoning=_on_reason if reason_track is not None else None,
        should_cancel=should_cancel,
    )
    msg = result.get("message")
    if isinstance(msg, dict):
        finish_call(
            reason_track,
            provider_name,
            effort,
            msg,
            hidden=bool(result.get("reasoning_hidden")),
        )
    elif reason_track is not None:
        reason_track.finish(provider_name, effort, "", hidden=bool(result.get("reasoning_hidden")))
    return result

MAX_EDGE = 1280
MAX_TURNS_CAP = 24
DEFAULT_TURNS = 12
TYPE_CAP = 2000
PASTE_AT = 40
LOG_KEEP = 8
MAX_CONTROLS = 36
END_LABEL = "end tool rounds"
BADGE_W = 248
BADGE_H = 48

HANDS_SYS = """You control the Windows desktop from screenshots and its control list.
The first user message is a plan. That same plan is repeated at the top of every later user message, unchanged. Follow it from the first unfinished step. Do not invent a different task.

Do one step per reply. A translucent button at the top reads "end tool rounds".
Click it only after the last step of the plan is done. Clicking it earlier is ignored.
Use {"action":"done","text":"..."} only when a step says to describe the screen, or when the plan is finished. The text must be what you actually see or what you finished.

Reply with one JSON object and no other text.
Coordinates are pixels of the attached screenshot. (0,0) is the top-left.
A magenta cross marks the pointer.
Prefer "n" or "target" from the Controls list. Those hit the center of that icon, button, or field.
A raw x,y near a labeled control is moved to that center. Use x,y only for empty desktop, where no label is near the point.

Actions:
{"action":"lock","title":"Notepad"}
{"action":"click","n":1}
{"action":"click","target":"Save"}
{"action":"double_click","n":1}
{"action":"right_click","n":1}
{"action":"click","x":0,"y":0}
{"action":"drag","x":0,"y":0,"x2":0,"y2":0}
{"action":"type","text":"short text"}
{"action":"paste","text":"long text"}
{"action":"key","keys":"win+d"}
{"action":"scroll","n":1,"dy":-3}
{"action":"wait","ms":400}
{"action":"done","text":"what you finished"}

Clicks, drags, scrolls, and paste work after the loop is locked to an app window or to the desktop.
When the screenshot says the desktop is locked, click and right-click it directly. Do not spend a step locking.
Desktop rows labeled Icon are the real icon positions. Open one with a double-click on its name. One click per reply. If it misses, the next reply shifts the point. Do not repeat a click that missed.
A file click counts only when its editor window is in front. Typing into a file, ctrl+a, and ctrl+s wait until that editor is in front. After the Win key, type the app name. Click the document, not the title bar. Alt+F4 is refused until that editor is focused. After the editor closes, the desktop lock is taken again.
Send the input the current step asks for. Do not send the next key until the previous step is confirmed. Use {"action":"done"} only when that step says to describe, or when the plan is finished.
When it is not locked, key and type can still open an app (win, then the app name, then enter). Then lock that app.
Win+D shows the desktop.
type sends real keystrokes. A newline is Enter and a tab is Tab. Text of 40 characters or more is pasted instead.
paste writes the clipboard, sends ctrl+v, then puts the previous clipboard back.
key is for special keys and chords: enter, tab, esc, backspace, delete, space, arrows, home, end, pageup, pagedown, f1-f12, ctrl+a, ctrl+c, alt+tab, win+d, win+e.
scroll dy is wheel notches. Negative scrolls down.
One action per reply. The loop checks the screen before the next step. Finish the requested actions, then click the end tool rounds button. The loop clicks that button only after those actions are confirmed.
"""

PLAN_SYS = """You write the working plan for a small vision model that sees the Windows desktop and sends mouse and keyboard input. It cannot use a shell. It follows your plan literally, one step per screenshot.

Read the Request. Output only a numbered plan, 8 steps maximum.
Each step is one mouse or keyboard action. Do not describe the screen. Do not write a summary.
Do not add a file, app, launch, save, or close that the request did not name. Ignore earlier ideas in the conversation.

Recipes, and only for an action the request already names:
- Show the desktop: press Win+D.
- Open a named desktop icon: press Win+D, then double-click that icon by name.
- Type or save into a file: click the document, then type or save. Do not send those keys while the desktop is locked.
- Close a window: Alt+F4 only after that window is focused.
- Open the Start menu: press the Win key, type the app name, press Enter.
- New text file on the desktop: Win+D, right-click empty desktop, click New, click Text Document, type the file name, press Enter.
- If the user only wants a description of the screen, the plan is two steps: describe the real windows and desktop, then click the end tool rounds button.
- Otherwise every step is an action. Do not emit a describe step. Do not add Notepad, a launch, a save, or a close that the request did not name.

The last step clicks the end tool rounds button only when the request says to end. No earlier step tells it to stop.
Do not mention the shell.
"""

_VK = {
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "backspace": 0x08,
    "bksp": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "space": 0x20,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pgup": 0x21,
    "pagedown": 0x22,
    "pgdn": 0x22,
    "insert": 0x2D,
    "ins": 0x2D,
    "ctrl": 0x11,
    "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "menu": 0x12,
    "win": 0x5B,
    "meta": 0x5B,
    "super": 0x5B,
    "capslock": 0x14,
}
for _i in range(1, 13):
    _VK[f"f{_i}"] = 0x6F + _i

_KEYUP = 0x0002
_UNICODE = 0x0004
_LEFTDOWN = 0x0002
_LEFTUP = 0x0004
_RIGHTDOWN = 0x0008
_RIGHTUP = 0x0010
_MIDDLEDOWN = 0x0020
_MIDDLEUP = 0x0040
_WHEEL = 0x0800
_SRCCOPY = 0x00CC0020
_HALFTONE = 4
_SW_MINIMIZE = 6
_SW_RESTORE = 9
_VK_F12 = 0x7B


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUTUNION),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


@dataclass
class Shot:
    attachment: dict
    vx: int
    vy: int
    vw: int
    vh: int
    iw: int
    ih: int


def image_to_screen(
    x: float,
    y: float,
    *,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
    iw: int,
    ih: int,
) -> tuple[int, int]:
    """Map a screenshot pixel onto the virtual desktop, then clamp."""
    if iw <= 0 or ih <= 0 or vw <= 0 or vh <= 0:
        raise ValueError("bad screen size")
    sx = vx + (float(x) / float(iw)) * vw
    sy = vy + (float(y) / float(ih)) * vh
    sx = min(max(sx, vx), vx + vw - 1)
    sy = min(max(sy, vy), vy + vh - 1)
    return int(round(sx)), int(round(sy))


def parse_keys(spec: str) -> list[str]:
    parts = [p.strip().lower() for p in (spec or "").split("+")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("empty keys")
    return parts


_LOOK_REQUEST = re.compile(
    r"what (?:do |can )?you see|what you see|"
    r"what(?:'s| is) on (?:the |my )?(?:screen|desktop)|"
    r"look at (?:the |my )?(?:screen|desktop)|"
    r"describe (?:the |my )?(?:screen|desktop)|"
    r"tell me what",
    re.I,
)
_HANDS_VERB = re.compile(r"\b(click|type|open|press|drag|scroll|paste|launch|lock)\b", re.I)


def is_look_only(goal: str) -> bool:
    """The user asked for a description, not for the pointer."""
    text = goal or ""
    if _HANDS_VERB.search(text):
        return False
    return _LOOK_REQUEST.search(text) is not None


def reply_aside(text: str) -> str:
    """Words around a JSON action. Empty when the reply is only JSON."""
    raw = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    raw = re.sub(r"```(?:json)?", " ", raw, flags=re.I)
    raw = re.sub(r"\{.*?\}", " ", raw, flags=re.S)
    raw = re.sub(r"\[.*?\]", " ", raw, flags=re.S)
    raw = re.sub(r"\s+", " ", raw).strip(" `\"'")
    if len(raw) < 12:
        return ""
    return raw[:1500]


def _look_lines(description: str, titles: list[str]) -> list[str]:
    lines = [f"1. saw: {description[:1500]}"]
    if titles:
        lines.append("open windows:\n" + "\n".join(f"- {title}" for title in titles))
    return lines


def _look_fallback(titles: list[str]) -> list[str]:
    lines = ["1. the model tried to click instead of describing the screen"]
    if titles:
        lines.append("open windows:\n" + "\n".join(f"- {title}" for title in titles))
    else:
        lines.append("no titled windows were visible")
    return lines


_EMPTY_LOOK = re.compile(
    r"^(what is visible|what you see|what(?:'s| is) visible|the screen|description|n/?a|none)\.?$",
    re.I,
)


def _real_look(text: str) -> str:
    """Drop the placeholder sentences the vision model copies from examples."""
    clean = re.sub(r"\s+", " ", (text or "")).strip(" .\"'")
    if _EMPTY_LOOK.match(clean) or len(clean) < 8:
        return ""
    return clean


def describe_look(content: str, titles: list[str]) -> list[str] | None:
    """Turn a look-only reply into a finished description. None means try once more."""
    aside = _real_look(reply_aside(content))
    try:
        action = parse_action(content)
    except Exception:
        return _look_lines(aside, titles) if aside else None
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    if kind in ("done", "look", "see", "describe"):
        text = _real_look(str(action.get("text") or action.get("reason") or aside or ""))
        return _look_lines(text, titles) if text else None
    if aside:
        return _look_lines(aside, titles)
    return None


def visible_window_titles(limit: int = 12) -> list[str]:
    titles: list[str] = []
    for _hwnd, title in list_top_windows():
        clean = " ".join(title.split())
        if clean and clean not in titles:
            titles.append(clean)
        if len(titles) >= limit:
            break
    return titles


def parse_action(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    raw = (text or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    obj_at = raw.find("{")
    arr_at = raw.find("[")
    if obj_at < 0 and arr_at < 0:
        raise ValueError("no JSON action")
    if arr_at >= 0 and (obj_at < 0 or arr_at < obj_at):
        start = arr_at
    else:
        start = obj_at
    blob = raw[start:]
    try:
        val = json.loads(blob)
    except json.JSONDecodeError:
        closer = "]" if blob.startswith("[") else "}"
        end = blob.rfind(closer)
        if end <= 0:
            raise ValueError("no JSON action") from None
        val = json.loads(blob[: end + 1])
    if isinstance(val, list):
        if not val or not isinstance(val[0], dict):
            raise ValueError("empty action list")
        return val[0]
    if not isinstance(val, dict):
        raise ValueError("action is not an object")
    return val


def hands_brief(messages: list[dict], limit: int = 8) -> str:
    """Recent chat lines that started a hands-on run. Tool output is left out."""
    lines: list[str] = []
    for msg in messages or []:
        role = str(msg.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        text = str(msg.get("content") or "").strip()
        if not text or text.startswith("AUTOMODE ON"):
            continue
        text = re.sub(r"\s+", " ", text)
        if len(text) > 400:
            text = text[:400].rstrip() + "…"
        lines.append(f"{role}: {text}")
    if len(lines) > limit:
        lines = lines[-limit:]
    return "\n".join(lines)


def clean_plan(text: str) -> str:
    """Keep the numbered plan and drop thinking, fences, and extra lines."""
    raw = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    raw = re.sub(r"```(?:json|text|markdown)?", "", raw, flags=re.I)
    lines: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= 8:
            break
    plan = "\n".join(lines).strip()
    if len(plan) < 12:
        return ""
    return plan[:1500]


def fallback_plan(request: str) -> str:
    """Used when the request is only a look, or the planner call fails."""
    if is_look_only(request):
        return (
            "1. Describe the real windows and desktop in the screenshot.\n"
            "2. Click the end tool rounds button."
        )
    return (
        "1. Carry out the original request with the mouse and keyboard, one action at a time.\n"
        "2. If it is about the desktop, press Win+D first.\n"
        "3. Click the end tool rounds button only after the request is finished."
    )


def make_hands_plan(
    host: str,
    model: str,
    request: str,
    brief: str,
    num_ctx: int,
    should_cancel: Callable[[], bool] | None,
    provider: str = "xai",
    api_key: str = "",
    reasoning: str = "off",
    reason_track=None,
) -> str:
    """One text call turns the chat into the plan. A look needs no call."""
    if is_look_only(request):
        return fallback_plan(request)
    user = (
        "Request:\n"
        f"{request}\n\n"
        "Write only the actions this request asks for. Do not add steps from earlier messages."
    )
    from .reasoning import plan_predict

    options: dict[str, Any] = {
        "temperature": 0.1,
        "num_predict": plan_predict(provider, reasoning),
        "reasoning": reasoning,
    }
    if num_ctx and int(num_ctx) > 0:
        options["num_ctx"] = int(num_ctx)
    try:
        result = _look(
            provider,
            host,
            api_key,
            model,
            [
                {"role": "system", "content": PLAN_SYS},
                {"role": "user", "content": user},
            ],
            options,
            90,
            should_cancel,
            reason_track=reason_track,
            reason_label="REASONING · plan",
        )
    except Exception:
        return ""
    if should_cancel and should_cancel():
        return ""
    return clean_plan(((result.get("message") or {}).get("content") or ""))


def focus_text(plan: str, request: str) -> str:
    """The vision loop sees this exact text on every turn."""
    return f"{plan.strip()}\n\nOriginal request: {request.strip()}"


def is_finish_action(action: dict) -> bool:
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    return kind in ("done", "stop", "end", "finish")


def plan_step_count(plan: str) -> int:
    return sum(
        1
        for line in (plan or "").splitlines()
        if re.match(r"^\d+[\.\)]\s+\S", line.strip())
    )


def hands_messages(goal: str, shot_text: str, attachment: dict | None) -> list[dict]:
    """The plan is `goal` on the first user message and again at the top of every screenshot."""
    shot: dict[str, Any] = {"role": "user", "content": f"{goal}\n\n{shot_text}"}
    if attachment:
        shot["attachments"] = [attachment]
    return [
        {"role": "system", "content": HANDS_SYS},
        {"role": "user", "content": goal},
        shot,
    ]


def end_button_place(left: int, top: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Physical screen box for the end button, inside the region about to be captured."""
    width = max(1, int(width))
    height = max(1, int(height))
    if width >= BADGE_W + 16:
        bw = BADGE_W
    elif width > 16:
        bw = width - 16
    else:
        bw = width
    if height >= BADGE_H + 16:
        bh = BADGE_H
    elif height > 8:
        bh = height - 8
    else:
        bh = height
    bx = int(left) + (width - bw) // 2
    inset = 8 if height - bh >= 8 else 0
    by = int(top) + inset
    return bx, by, bw, bh


def image_box(shot: Shot, rect: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    """Map a physical screen rect (left, top, right, bottom) into screenshot pixels."""
    if not rect or shot.iw <= 0 or shot.ih <= 0 or shot.vw <= 0 or shot.vh <= 0:
        return None
    left, top, right, bottom = (int(v) for v in rect)
    if right <= left or bottom <= top:
        return None
    x1 = (float(left) - shot.vx) * shot.iw / float(shot.vw)
    y1 = (float(top) - shot.vy) * shot.ih / float(shot.vh)
    x2 = (float(right) - shot.vx) * shot.iw / float(shot.vw)
    y2 = (float(bottom) - shot.vy) * shot.ih / float(shot.vh)
    if x2 <= 0 or y2 <= 0 or x1 >= shot.iw or y1 >= shot.ih:
        return None
    ix1 = max(0, int(round(x1)))
    iy1 = max(0, int(round(y1)))
    ix2 = min(shot.iw, int(round(x2)))
    iy2 = min(shot.ih, int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2 - ix1, iy2 - iy1


def hits_end_button(
    sx: float,
    sy: float,
    rect: tuple[int, int, int, int] | None,
    pad: int = 8,
) -> bool:
    if not rect:
        return False
    left, top, right, bottom = rect
    return (left - pad) <= sx < (right + pad) and (top - pad) <= sy < (bottom + pad)


def badge_sentence(
    box: tuple[int, int, int, int] | None,
    when: str = "Click it only after the plan is finished.",
) -> str:
    if not box:
        return f'A translucent button at the top reads "end tool rounds". {when}'
    x, y, w, h = box
    return (
        f'A translucent button reading "end tool rounds" covers image pixels '
        f"({x},{y})-({x + w},{y + h}). Center ({x + w // 2},{y + h // 2}). "
        f"{when}"
    )


def _shot_text(
    iw: int,
    ih: int,
    log: list[str],
    *,
    lock_title: str,
    controls_text: str,
    look_only: bool = False,
    windows: str = "",
    badge: str = "",
    shell: bool = False,
    focus_title: str = "",
    editor: bool = False,
) -> str:
    history = "\n".join(log[-LOG_KEEP:]) if log else "No actions yet."
    badge = badge or (
        'A translucent button at the top reads "end tool rounds". Click it to end the run.'
    )
    if look_only:
        listing = f"Open windows:\n{windows}\n" if windows else ""
        return (
            f"Desktop screenshot {iw}x{ih} pixels. The plan is to describe the screenshot.\n"
            f"{listing}"
            f"{badge}\n"
            'Reply with {"action":"done","text":"the real windows and desktop"}.\n'
            "Do not click anything else. Do not lock, type, or scroll.\n"
            f"Recent actions:\n{history}"
        )
    if shell:
        head = (
            "Locked to the desktop. Screenshot "
            f"{iw}x{ih} pixels of the whole screen. (0,0) is the top-left. "
            "Clicks, right-clicks, and drags hit that pixel, including empty desktop. "
            "Icon rows are desktop icons. Opening one double-clicks its center. "
            "The step counts when that file's editor is in front. Typing waits for the editor."
        )
    elif lock_title:
        head = (
            f"Locked to {lock_title!r}. Screenshot {iw}x{ih} pixels. "
            "(0,0) is the top-left."
        )
        if editor:
            head += " Editor document is locked for typing. Click inside the document before keys."
    else:
        head = (
            "Not locked. Clicks, drags, scrolls, and paste are refused until you lock a window. "
            "Win+D shows the desktop before a lock. "
            f"Desktop screenshot is {iw}x{ih} pixels. "
            "Do the next plan step."
        )
    controls = controls_text or "No named controls."
    prefer = (
        "Prefer n or target from Controls. Do the next plan step. "
        "Click the end tool rounds button only when the plan is finished."
        if shell or lock_title
        else "Do the next plan step. The end tool rounds button is only for the last step."
    )
    if focus_title and not look_only:
        head += f" Foreground: {focus_title}."
    return (
        f"{head} A magenta cross is the pointer.\n"
        f"{badge}\n"
        f"Controls:\n{controls}\n"
        f"Recent actions:\n{history}\n"
        f"Reply with one JSON action. {prefer}"
    )


_SKIP_TITLE_WORDS = {
    "untitled",
    "document",
    "window",
    "microsoft",
    "google",
    "file",
    "video",
    "videos",
    "picture",
    "pictures",
    "download",
    "downloads",
    "desktop",
    "folder",
    "program",
    "manager",
    "default",
    "profile",
    "settings",
}

_ROLE = {
    50000: "Button",
    50002: "CheckBox",
    50003: "ComboBox",
    50004: "Edit",
    50005: "Hyperlink",
    50007: "ListItem",
    50011: "MenuItem",
    50013: "RadioButton",
    50015: "Slider",
    50016: "Spinner",
    50019: "TabItem",
    50023: "TreeItem",
    50029: "DataItem",
    50030: "Document",
    50031: "SplitButton",
    50037: "TitleBar",
}
_UNNAMED_OK = {"Button", "CheckBox", "Edit", "MenuItem", "SplitButton"}
_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002


@dataclass
class Control:
    index: int
    role: str
    name: str
    auto_id: str
    left: int
    top: int
    width: int
    height: int


@dataclass
class HandsContext:
    query: str = ""
    lock_hwnd: int = 0
    lock_title: str = ""
    shell: bool = False
    shell_focused: bool = False
    missed_icon: str = ""
    retry_name: str = ""
    retry_index: int = 0
    desktop_cleared: bool = False
    last_target: str = ""
    desktop_hwnd: int = 0
    editor_hwnd: int = 0
    expect_file: str = ""
    held_kind: str = ""
    launched_file: str = ""
    focus_title: str = ""
    input_ready: bool = False
    landed_text: str = ""
    title_before: str = ""
    want_content: bool = False
    pinned: bool = False
    content_at: tuple = ()
    desktop_icons: list = field(default_factory=list)

    @property
    def locked(self) -> bool:
        return self.lock_hwnd != 0


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8),
    ]


def _guid(text: str) -> _GUID:
    parsed = uuid.UUID(text)
    g = _GUID()
    g.Data1 = parsed.time_low
    g.Data2 = parsed.time_mid
    g.Data3 = parsed.time_hi_version
    for i, byte in enumerate(parsed.bytes[8:]):
        g.Data4[i] = byte
    return g


def screen_to_image(sx: float, sy: float, shot: Shot) -> tuple[int, int]:
    if shot.vw <= 0 or shot.vh <= 0:
        raise ValueError("bad screen size")
    ix = (float(sx) - shot.vx) * shot.iw / float(shot.vw)
    iy = (float(sy) - shot.vy) * shot.ih / float(shot.vh)
    return int(round(ix)), int(round(iy))


def choose_window(
    windows: list[tuple[int, str]],
    query: str,
    foreground: int = 0,
) -> tuple[int, str] | None:
    """Pick the visible window whose title best matches `query`."""
    needle = (query or "").strip().casefold()
    if not needle:
        return None
    hits = [(hwnd, title) for hwnd, title in windows if needle in title.casefold()]
    if not hits:
        return None

    def rank(item: tuple[int, str]) -> tuple:
        hwnd, title = item
        folded = title.casefold()
        return (
            0 if foreground and hwnd == foreground else 1,
            0 if folded == needle else 1,
            0 if folded.startswith(needle) else 1,
            len(title),
        )

    hits.sort(key=rank)
    return hits[0]


def infer_window_query(goal: str, windows: list[tuple[int, str]]) -> str:
    """Longest title, or title word, that the user actually typed."""
    goal_l = (goal or "").casefold()
    best = ""
    for _hwnd, title in windows:
        folded = title.casefold().strip()
        if len(folded) >= 4 and folded in goal_l and len(folded) > len(best):
            best = title.strip()
            continue
        for word in re.split(r"[^\w]+", title):
            if len(word) < 4 or word.casefold() in _SKIP_TITLE_WORDS:
                continue
            if word.casefold() in goal_l and len(word) > len(best):
                best = word
    return best


_DESKTOP_QUERIES = {
    "desktop",
    "the desktop",
    "my desktop",
    "windows desktop",
    "shell",
    "desktop shell",
    "shell desktop",
    "progman",
    "program manager",
}


def is_desktop_query(query: str) -> bool:
    """True when the window argument means the desktop shell, not an app title."""
    return (query or "").strip().casefold() in _DESKTOP_QUERIES


def window_argument(proposed: str, task: str) -> str:
    """Keep a passed window title so the run can lock it. desktop and shell stay desktop."""
    text = (proposed or "").strip()
    if not text:
        return ""
    if is_desktop_query(text):
        return "desktop"
    return text


def closed_window_note(title: str) -> str:
    """A named app was not open, so the run starts on the desktop and keeps its plan."""
    name = " ".join((title or "").split()) or "window"
    return f"{name} was not open, started from the desktop"


def find_shell_desktop() -> tuple[int, str] | None:
    """HWND that hosts the desktop icons. Does not focus or click."""
    if sys.platform != "win32":
        return None
    user32 = _user32()
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowExW.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
    ]
    user32.FindWindowExW.restype = wintypes.HWND
    user32.GetDesktopWindow.argtypes = []
    user32.GetDesktopWindow.restype = wintypes.HWND

    def hosts_desktop(parent: int) -> bool:
        if not parent:
            return False
        return bool(user32.FindWindowExW(parent, None, "SHELLDLL_DefView", None))

    progman = int(user32.FindWindowW("Progman", None) or 0)
    if hosts_desktop(progman):
        return progman, "Desktop"
    previous = 0
    for _ in range(64):
        worker = int(user32.FindWindowExW(None, previous or None, "WorkerW", None) or 0)
        if not worker or worker == previous:
            break
        if hosts_desktop(worker):
            return worker, "Desktop"
        previous = worker
    if progman:
        return progman, "Desktop"
    desktop = int(user32.GetDesktopWindow() or 0)
    if desktop:
        return desktop, "Desktop"
    return None


def _window_alive(hwnd: int) -> bool:
    if not hwnd or sys.platform != "win32":
        return False
    user32 = _user32()
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    return bool(user32.IsWindow(hwnd))


def lock_shell(state: HandsContext) -> bool:
    """Lock onto the desktop shell so pointer actions have a target."""
    chosen = find_shell_desktop()
    if not chosen:
        return False
    state.lock_hwnd, state.lock_title = chosen
    state.desktop_hwnd = state.lock_hwnd
    state.shell = True
    state.shell_focused = False
    if state.editor_hwnd and not _window_alive(state.editor_hwnd):
        state.editor_hwnd = 0
    if not is_desktop_query(state.query):
        state.query = "desktop"
    return True


def focus_shell_once(state: HandsContext) -> None:
    """Unused. Pointer and keyboard actions call confirm_focus every time."""
    if not state.shell or state.shell_focused or not state.lock_hwnd:
        return
    state.shell_focused = True
    try:
        focus_window(state.lock_hwnd)
    except Exception:
        pass


def _norm_name(text: str) -> str:
    folded = re.sub(r"[^a-z0-9]+", " ", (text or "").casefold())
    return folded.replace("recycling", "recycle").strip()


def goal_phrase(request: str) -> str:
    """The thing the user asked to open, such as 'recycling bin'."""
    match = re.search(
        r"\b(?:open|launch|start)\s+(?:the\s+|my\s+)?([a-z0-9][a-z0-9 .'-]{1,40})",
        request or "",
        re.I,
    )
    if not match:
        return ""
    phrase = re.split(r"\b(?:then|and)\b", match.group(1), maxsplit=1)[0]
    return phrase.strip(" .")


def wants_open(request: str) -> bool:
    return bool(re.search(r"\b(?:open|launch|start)\b", request or "", re.I))


_TEXT_EXTS = {".txt", ".log", ".md", ".csv", ".ini", ".json", ".py", ".xml", ".cfg", ".text"}


def file_stem(name: str) -> str:
    base = (name or "").strip().replace("\\", "/").split("/")[-1]
    if "." in base and not base.startswith("."):
        base = base.rsplit(".", 1)[0]
    return base


def looks_like_file(name: str) -> bool:
    base = (name or "").strip().replace("\\", "/").split("/")[-1]
    if "." not in base or base.startswith("."):
        return False
    ext = "." + base.rsplit(".", 1)[1].casefold()
    return ext in _TEXT_EXTS or 1 < len(ext) <= 6


def editor_match(name: str, titles: list[str]) -> str:
    """Window title that belongs to this file, such as 'hello_world.txt - Notepad'."""
    stem = _norm_name(file_stem(name))
    full = _norm_name(name)
    if len(stem) < 3 and len(full) < 3:
        return ""
    for title in titles:
        folded = _norm_name(title)
        if len(stem) >= 3 and stem in folded:
            return title
        if len(full) >= 3 and full in folded:
            return title
    return ""


def action_kind(action: dict) -> str:
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    if kind == "rightclick":
        return "right_click"
    if kind == "doubleclick":
        return "double_click"
    return kind


def is_global_chord(spec: str) -> bool:
    """Hotkeys that must still fire when the desktop lock cannot take focus."""
    try:
        tokens = parse_keys(spec)
    except ValueError:
        return False
    if not tokens:
        return False
    if tokens[0] in ("win", "meta", "super"):
        return True
    return tokens in (["alt", "tab"], ["alt", "f4"], ["alt", "shift", "tab"])


def is_edit_chord(spec: str) -> bool:
    if is_global_chord(spec):
        return False
    try:
        tokens = parse_keys(spec)
    except ValueError:
        return False
    editing = {"a", "s", "c", "v", "x", "z", "y"}
    if "ctrl" in tokens or "control" in tokens:
        return any(token in editing for token in tokens)
    return False


def needs_editor(action: dict) -> bool:
    kind = action_kind(action)
    if kind in ("type", "paste"):
        return True
    if kind == "key":
        return is_edit_chord(str(action.get("keys") or action.get("key") or ""))
    return False


def repeats_held(held: str, action: dict) -> bool:
    """An unconfirmed step must be retried. A later plan step is refused."""
    if not held:
        return True
    kind = action_kind(action)
    if held in ("type", "paste") and kind in ("type", "paste", "click", "double_click"):
        return True
    if held == "key" and kind in ("key", "click", "double_click"):
        return True
    if held in ("click", "double_click", "right_click") and kind in (
        "click",
        "double_click",
        "right_click",
    ):
        return True
    return held == kind


def held_from_line(line: str) -> str:
    text = line or ""
    if text.startswith("double_click"):
        return "double_click"
    if text.startswith("right_click"):
        return "right_click"
    if text.startswith("click"):
        return "click"
    if text.startswith("key"):
        return "key"
    if text.startswith("typed") or text.startswith("paste") or "characters:" in text:
        return "type"
    return ""


def matched_title(name: str, titles: list[str]) -> str:
    """Window title that contains this icon or goal name."""
    needle = _norm_name(name)
    if len(needle) < 3:
        return ""
    for title in titles:
        folded = _norm_name(title)
        if needle in folded:
            return title
    return ""


def confirm_opened(
    name: str,
    titles_fn: Callable[[], list[str]],
    tries: int = 4,
    pause: float = 0.2,
) -> str:
    """Poll until a window title matches, or return ''."""
    for attempt in range(max(1, tries)):
        if attempt:
            time.sleep(pause)
        hit = matched_title(name, titles_fn())
        if hit:
            return hit
    return ""


@dataclass
class DesktopIcon:
    name: str
    icon_left: int
    icon_top: int
    icon_width: int
    icon_height: int
    label_left: int = 0
    label_top: int = 0
    label_width: int = 0
    label_height: int = 0

    def icon_center(self) -> tuple[int, int]:
        return (
            int(self.icon_left + self.icon_width / 2),
            int(self.icon_top + self.icon_height / 2),
        )

    def label_center(self) -> tuple[int, int]:
        if self.label_width <= 0 or self.label_height <= 0:
            return self.icon_center()
        return (
            int(self.label_left + self.label_width / 2),
            int(self.label_top + self.label_height / 2),
        )


def find_icon(icons: list[DesktopIcon], name: str) -> DesktopIcon | None:
    needle = _norm_name(name)
    if not needle:
        return None
    exact = [icon for icon in icons if _norm_name(icon.name) == needle]
    if exact:
        return exact[0]
    partial = [icon for icon in icons if needle in _norm_name(icon.name) or _norm_name(icon.name) in needle]
    if len(partial) == 1:
        return partial[0]
    return None


def attempt_points(icon: DesktopIcon | None, x: int, y: int, dpi: int) -> list[tuple[int, int]]:
    """Icon center, then its label, then the other DPI scale. No repeated point."""
    points: list[tuple[int, int]] = []

    def add(px: int, py: int) -> None:
        pt = (int(px), int(py))
        if pt not in points:
            points.append(pt)

    if icon is not None and icon.icon_width > 0 and icon.icon_height > 0:
        add(*icon.icon_center())
        label = icon.label_center()
        if label != icon.icon_center():
            add(*label)
        origin = icon.icon_center()
    else:
        add(x, y)
        origin = (int(x), int(y))
    scale = (dpi or 96) / 96.0
    if abs(scale - 1.0) >= 0.08:
        add(int(round(origin[0] * 96 / (dpi or 96))), int(round(origin[1] * 96 / (dpi or 96))))
        add(int(round(origin[0] * scale)), int(round(origin[1] * scale)))
    return points[:4]


def nearest_labeled(
    controls: list[Control], x: int, y: int, radius: int = 160
) -> Control | None:
    """Closest named control, icon, edit, or button whose center is within radius."""
    best: Control | None = None
    limit = int(radius) * int(radius)
    best_d = limit + 1
    for control in controls:
        if control.width <= 0 or control.height <= 0:
            continue
        named = bool((control.name or control.auto_id or "").strip())
        if not named and control.role not in ("Icon", "Edit", "Document", "Button"):
            continue
        cx = control.left + control.width / 2
        cy = control.top + control.height / 2
        dist = (cx - x) ** 2 + (cy - y) ** 2
        if dist <= limit and dist < best_d:
            best = control
            best_d = dist
    return best


_SHIFTS = ((0, 0), (0, 18), (14, 0), (-14, 0), (0, -18))


def shift_point(x: int, y: int, index: int, dpi: int) -> tuple[int, int]:
    """Try 0 is the point itself. Later tries nudge, then the other DPI scale."""
    index = max(0, int(index))
    if index < len(_SHIFTS):
        dx, dy = _SHIFTS[index]
        return int(x) + dx, int(y) + dy
    dpi = int(dpi or 96)
    scale = (96 / dpi) if ((index - len(_SHIFTS)) % 2 == 0) else (dpi / 96.0)
    return int(round(int(x) * scale)), int(round(int(y) * scale))


def input_control(controls: list[Control]) -> Control | None:
    """Largest edit or document. A named field wins over a blank one."""
    edits = [
        control
        for control in controls
        if control.role in ("Edit", "Document") and control.width >= 40 and control.height >= 16
    ]
    if not edits:
        return None
    named = [control for control in edits if (control.name or "").strip()]
    return max(named or edits, key=lambda control: control.width * control.height)


def is_chrome(control: Control) -> bool:
    if control.role in ("TitleBar", "Window"):
        return True
    return (
        control.height <= 40
        and control.width > 200
        and not (control.name or control.auto_id or "").strip()
    )


def prefers_input(step: str, action: dict) -> bool:
    blob = " ".join(
        [
            step or "",
            str(action.get("action") or ""),
            str(action.get("text") or ""),
            str(action.get("target") or ""),
        ]
    )
    return bool(re.search(r"type|prompt|terminal|console|editor|input|caret", blob, re.I))


def plan_lines(plan: str) -> list[str]:
    return [
        line.strip()
        for line in (plan or "").splitlines()
        if re.match(r"^\d+[\.\)]\s+\S", line.strip())
    ]


def current_step(plan: str, done_actions: int) -> str:
    """The next real step. The end button is last, after the other lines are done."""
    lines = plan_lines(plan)
    if not lines:
        return ""
    work = [line for line in lines if step_kind(line) != "end"]
    if not work:
        return lines[-1]
    if int(done_actions) >= len(work):
        ends = [line for line in lines if step_kind(line) == "end"]
        return ends[-1] if ends else work[-1]
    return work[max(int(done_actions), 0)]


def step_is_describe(step: str) -> bool:
    return bool(re.search(r"\b(describe|what you see|what is visible)\b", step or "", re.I))


def executable_plan(plan: str, request: str) -> str:
    """Drop describe steps unless the user asked for a description."""
    if is_look_only(request):
        return plan
    kept = [
        line
        for line in (plan or "").splitlines()
        if line.strip() and not step_is_describe(line)
    ]
    text = expand_plan("\n".join(kept).strip())
    actionable = [
        line
        for line in plan_lines(text)
        if step_kind(line) not in ("", "end")
    ]
    if actionable:
        return text
    return fallback_plan(request)


def step_kind(step: str) -> str:
    """The first action on this line. A later verb belongs on its own line."""
    text = step or ""
    if re.search(r"end tool", text, re.I):
        return "end"
    match = re.search(r"\b(double-?click|type|press|click)\b", text, re.I)
    if not match:
        if re.search(r"\b(ctrl\+|alt\+|win\+)\b", text, re.I):
            return "key"
        return ""
    word = match.group(1).casefold()
    if word.startswith("double"):
        return "double_click"
    if word == "type":
        return "type"
    if word == "press":
        return "key"
    if word == "click":
        return "click"
    return ""


def split_plan_line(body: str) -> list[str]:
    """One verb per piece. 'click inside, then type hello' becomes two steps."""
    matches = list(re.finditer(r"\b(?:double-?click|type|press|click)\b", body or "", re.I))
    if len(matches) <= 1:
        text = (body or "").strip()
        return [text] if text else []
    parts: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        piece = body[match.start() : end]
        piece = re.sub(r"(?:,|\s)+(?:then|and then|and)\s*$", "", piece, flags=re.I)
        piece = piece.strip(" ,.;")
        if piece:
            parts.append(piece)
    return parts or [body.strip()]


def expand_plan(plan: str) -> str:
    """Numbered lines with two actions become two numbered lines."""
    numbered: list[str] = []
    for line in (plan or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^\d+[\.\)]\s+(.*)$", stripped)
        if not match:
            continue
        for part in split_plan_line(match.group(1)):
            numbered.append(f"{len(numbered) + 1}. {part}")
    return "\n".join(numbered)


def non_end_steps(plan: str) -> int:
    """How many lines must be confirmed before the end button."""
    lines = plan_lines(plan)
    if not lines:
        return 1
    count = sum(1 for line in lines if step_kind(line) != "end")
    return max(1, count)


def action_matches_step(action: dict, step: str) -> bool:
    expected = step_kind(step)
    if not expected:
        return True
    kind = action_kind(action)
    if expected == "end":
        return _names_end_button(action) or kind in ("click", "double_click")
    if expected == "type":
        return kind in ("type", "paste")
    if expected == "key":
        return kind in ("key", "type", "paste")
    if expected == "double_click":
        return kind in ("double_click", "click")
    if expected == "click":
        return kind in ("click", "double_click")
    return True


def step_wants_content(step: str) -> bool:
    return bool(
        re.search(
            r"document|text area|text editor|editor|notepad|terminal|console|input|caret|inside",
            step or "",
            re.I,
        )
    )


_DOCUMENT_STEP = re.compile(
    r"document|text area|text editor|\beditor\b|terminal|console|\binput\b|caret|\binside\b",
    re.I,
)
_ADDRESS_STEP = re.compile(
    r"address\s*bar|omnibox|location\s*bar|\bhttps?://|\bwww\.|"
    r"\b[a-z0-9][a-z0-9-]{0,62}\.(?:com|org|net|io|dev|ai|edu|gov)\b",
    re.I,
)


def _is_address_step(step: str) -> bool:
    """A browser address or URL. That text is not a file document."""
    return bool(_ADDRESS_STEP.search(step or ""))


def edits_document(state: HandsContext, step: str = "") -> bool:
    """True when typing belongs in a file or app document. Start search and address bars do not."""
    if state.expect_file:
        return True
    if _DOCUMENT_STEP.search(step or ""):
        return True
    if _is_address_step(step):
        return False
    return bool(state.locked and not state.shell)


def required_typing(request: str, plan: str) -> str:
    """Text the request said to type. Empty when it did not ask for typing."""
    blob = f"{request or ''}\n{plan or ''}"
    quoted = re.search(r"\btype\b[^\"'\n]{0,24}[\"']([^\"']+)[\"']", blob, re.I)
    if quoted:
        return quoted.group(1).strip()
    match = re.search(r"\btype(?:\s+the\s+text)?\s+(.+)", blob, re.I)
    if not match:
        return ""
    text = re.split(r"\b(?:on a|then|and then|press|ctrl\+|alt\+)\b", match.group(1), maxsplit=1, flags=re.I)[0]
    return text.strip(" .\"'")


def typing_landed(needed: str, landed: str) -> bool:
    if not needed:
        return True
    return needed.casefold() in (landed or "").casefold()


def prompt_has_action(text: str) -> bool:
    """True when a tool prompt is an instruction, not a vague paraphrase."""
    if _HANDS_VERB.search(text or ""):
        return True
    return bool(re.search(r"\b(win\+d|ctrl\+|alt\+|press|double-?click)\b", text or "", re.I))


def goal_ready(request: str, title: str) -> bool:
    """True when the foreground window title is the thing the user asked to open."""
    phrase = goal_phrase(request)
    if not phrase or not title:
        return False
    return bool(matched_title(phrase, [title]))


def controls_with_icons(controls: list[Control], icons: list[DesktopIcon]) -> list[Control]:
    """Desktop icon rects replace a UI Automation hit of the same name."""
    names = {_norm_name(icon.name) for icon in icons if icon.name}
    rest = [control for control in controls if _norm_name(control.name) not in names]
    head: list[Control] = []
    for icon in icons:
        if icon.icon_width <= 0 or icon.icon_height <= 0 or not icon.name:
            continue
        head.append(
            Control(
                0,
                "Icon",
                icon.name,
                "",
                int(icon.icon_left),
                int(icon.icon_top),
                int(icon.icon_width),
                int(icon.icon_height),
            )
        )
    merged = head + rest
    return [
        Control(i, item.role, item.name, item.auto_id, item.left, item.top, item.width, item.height)
        for i, item in enumerate(merged[:MAX_CONTROLS], start=1)
    ]


class _LVITEMW(ctypes.Structure):
    _fields_ = [
        ("mask", wintypes.UINT),
        ("iItem", ctypes.c_int),
        ("iSubItem", ctypes.c_int),
        ("state", wintypes.UINT),
        ("stateMask", wintypes.UINT),
        ("pszText", ctypes.c_void_p),
        ("cchTextMax", ctypes.c_int),
        ("iImage", ctypes.c_int),
        ("lParam", ctypes.c_ssize_t),
        ("iIndent", ctypes.c_int),
        ("iGroupId", ctypes.c_int),
        ("cColumns", wintypes.UINT),
        ("puColumns", ctypes.c_void_p),
        ("piColFmt", ctypes.c_void_p),
        ("iGroup", ctypes.c_int),
    ]


def desktop_listview() -> int:
    """SysListView32 that holds the desktop icons. Does not focus or click."""
    if sys.platform != "win32":
        return 0
    user32 = _user32()
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowExW.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
    ]
    user32.FindWindowExW.restype = wintypes.HWND

    def under(parent: int) -> int:
        if not parent:
            return 0
        view = int(user32.FindWindowExW(parent, None, "SHELLDLL_DefView", None) or 0)
        if not view:
            return 0
        return int(user32.FindWindowExW(view, None, "SysListView32", None) or 0)

    progman = int(user32.FindWindowW("Progman", None) or 0)
    found = under(progman)
    if found:
        return found
    previous = 0
    for _ in range(64):
        worker = int(user32.FindWindowExW(None, previous or None, "WorkerW", None) or 0)
        if not worker or worker == previous:
            break
        found = under(worker)
        if found:
            return found
        previous = worker
    return 0


def _screen_rect(hwnd: int, rect: wintypes.RECT) -> tuple[int, int, int, int] | None:
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        return None
    user32 = _user32()
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    origin = wintypes.POINT(int(rect.left), int(rect.top))
    far = wintypes.POINT(int(rect.right), int(rect.bottom))
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(far)):
        return None
    return int(origin.x), int(origin.y), int(far.x - origin.x), int(far.y - origin.y)


def read_desktop_icons(limit: int = 36) -> list[DesktopIcon]:
    """Icon and label rectangles from the desktop list, in screen pixels."""
    if sys.platform != "win32":
        return []
    hwnd = desktop_listview()
    if not hwnd:
        return []
    user32 = _user32()
    kernel32 = ctypes.windll.kernel32
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.VirtualAllocEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_size_t,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.VirtualAllocEx.restype = ctypes.c_void_p
    kernel32.VirtualFreeEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    kernel32.VirtualFreeEx.restype = wintypes.BOOL
    kernel32.WriteProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.WriteProcessMemory.restype = wintypes.BOOL
    kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.ReadProcessMemory.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    process = kernel32.OpenProcess(0x0008 | 0x0010 | 0x0020 | 0x0400, False, pid.value)
    if not process:
        return []
    remote = kernel32.VirtualAllocEx(process, None, 8192, 0x1000 | 0x2000, 0x04)
    if not remote:
        kernel32.CloseHandle(process)
        return []
    base = int(remote)
    item_size = ctypes.sizeof(_LVITEMW)
    text_at = base + item_size
    rect_at = text_at + 520
    icons: list[DesktopIcon] = []
    try:
        count = int(user32.SendMessageW(hwnd, 0x1004, 0, 0))
        count = max(0, min(count, int(limit)))
        for index in range(count):
            item = _LVITEMW()
            item.pszText = text_at
            item.cchTextMax = 260
            written = ctypes.c_size_t()
            if not kernel32.WriteProcessMemory(
                process, ctypes.c_void_p(base), ctypes.byref(item), item_size, ctypes.byref(written)
            ):
                continue
            user32.SendMessageW(hwnd, 0x1073, index, base)
            raw = ctypes.create_unicode_buffer(260)
            if not kernel32.ReadProcessMemory(
                process, ctypes.c_void_p(text_at), raw, 520, ctypes.byref(written)
            ):
                continue
            name = raw.value.strip()
            if not name:
                continue

            def read_part(code: int) -> tuple[int, int, int, int] | None:
                part = wintypes.RECT(int(code), 0, 0, 0)
                if not kernel32.WriteProcessMemory(
                    process, ctypes.c_void_p(rect_at), ctypes.byref(part), 16, ctypes.byref(written)
                ):
                    return None
                if not user32.SendMessageW(hwnd, 0x100E, index, rect_at):
                    return None
                got = wintypes.RECT()
                if not kernel32.ReadProcessMemory(
                    process, ctypes.c_void_p(rect_at), ctypes.byref(got), 16, ctypes.byref(written)
                ):
                    return None
                return _screen_rect(hwnd, got)

            icon_rect = read_part(1) or (0, 0, 0, 0)
            label_rect = read_part(2) or icon_rect
            if icon_rect[2] <= 0 or icon_rect[3] <= 0:
                continue
            icons.append(
                DesktopIcon(
                    name[:80],
                    icon_rect[0],
                    icon_rect[1],
                    icon_rect[2],
                    icon_rect[3],
                    label_rect[0],
                    label_rect[1],
                    label_rect[2],
                    label_rect[3],
                )
            )
        return icons
    finally:
        kernel32.VirtualFreeEx(process, ctypes.c_void_p(base), 0, 0x8000)
        kernel32.CloseHandle(process)


def resolve_control(action: dict, controls: list[Control]) -> Control | None:
    """Match n / target. None means the caller should use x,y."""
    raw_n = action.get("n", action.get("control"))
    if raw_n is not None and str(raw_n).strip():
        token = str(raw_n).strip()
        if token.lstrip("-").isdigit():
            number = int(token)
            for control in controls:
                if control.index == number:
                    return control
            raise ValueError(f"no control n={number}")
    target = str(action.get("target") or action.get("name") or "").strip()
    if not target:
        return None
    folded = target.casefold()
    exact = [
        c
        for c in controls
        if c.name.casefold() == folded or c.auto_id.casefold() == folded
    ]
    pool = exact or [
        c
        for c in controls
        if folded in c.name.casefold() or (c.auto_id and folded in c.auto_id.casefold())
    ]
    if len(pool) == 1:
        return pool[0]
    if not pool:
        raise ValueError(f"no control matching {target!r}")
    listed = ", ".join(f"{c.index} {c.name or c.role}" for c in pool[:6])
    raise ValueError(f"several controls match {target!r}: {listed}")


def format_controls(controls: list[Control], shot: Shot) -> str:
    lines: list[str] = []
    for control in controls:
        cx = control.left + control.width / 2
        cy = control.top + control.height / 2
        try:
            ix, iy = screen_to_image(cx, cy, shot)
        except ValueError:
            continue
        if ix < 0 or iy < 0 or ix >= shot.iw or iy >= shot.ih:
            continue
        label = (control.name or control.auto_id or "(unnamed)").replace("\n", " ")
        if len(label) > 60:
            label = label[:60].rstrip() + "…"
        lines.append(f"{control.index}. {control.role} \"{label}\" at ({ix},{iy})")
    return "\n".join(lines)


def _com_call(ip: ctypes.c_void_p, index: int, restype, argtypes, args):
    addr = ip.value if isinstance(ip, ctypes.c_void_p) else int(ip)
    vtable = ctypes.cast(addr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
    func = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[0][index])
    return func(addr, *args)


def _release(ip: ctypes.c_void_p | None) -> None:
    if ip is not None and getattr(ip, "value", None):
        _com_call(ip, 2, ctypes.c_ulong, [], [])


def _ensure_com() -> None:
    if sys.platform != "win32":
        return
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = ctypes.c_long
    # 0 = initialized, 1 = already initialized. A negative code means the
    # thread already has a different COM mode; UI Automation still works.
    ole32.CoInitializeEx(None, 0)


def _uia() -> ctypes.c_void_p:
    _ensure_com()
    ole32 = ctypes.windll.ole32
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = ctypes.c_long
    created = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(
        ctypes.byref(_guid("ff48dba4-60ef-4201-aa87-54103eef594e")),
        None,
        1,
        ctypes.byref(_guid("30cbe57d-d9d0-452a-ab13-7ac5ac4825ee")),
        ctypes.byref(created),
    )
    if hr < 0 or not created.value:
        raise RuntimeError("UI Automation is unavailable")
    return created


def _bstr(elem: ctypes.c_void_p, index: int) -> str:
    holder = ctypes.c_void_p()
    hr = _com_call(
        elem, index, ctypes.c_long, [ctypes.POINTER(ctypes.c_void_p)], [ctypes.byref(holder)]
    )
    if hr < 0 or not holder.value:
        return ""
    try:
        return ctypes.wstring_at(holder.value)
    finally:
        ctypes.windll.oleaut32.SysFreeString(holder)


def _int_prop(elem: ctypes.c_void_p, index: int) -> int:
    holder = ctypes.c_int()
    hr = _com_call(
        elem, index, ctypes.c_long, [ctypes.POINTER(ctypes.c_int)], [ctypes.byref(holder)]
    )
    if hr < 0:
        return 0
    return int(holder.value)


def _bool_prop(elem: ctypes.c_void_p, index: int) -> bool:
    holder = wintypes.BOOL()
    hr = _com_call(
        elem, index, ctypes.c_long, [ctypes.POINTER(wintypes.BOOL)], [ctypes.byref(holder)]
    )
    if hr < 0:
        return False
    return bool(holder.value)


def _rect_prop(elem: ctypes.c_void_p) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    hr = _com_call(
        elem, 43, ctypes.c_long, [ctypes.POINTER(wintypes.RECT)], [ctypes.byref(rect)]
    )
    if hr < 0:
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        return None
    return int(rect.left), int(rect.top), width, height


def _element_from_handle(uia: ctypes.c_void_p, hwnd: int) -> ctypes.c_void_p | None:
    elem = ctypes.c_void_p()
    hr = _com_call(
        uia,
        6,
        ctypes.c_long,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        [ctypes.c_void_p(hwnd), ctypes.byref(elem)],
    )
    if hr < 0 or not elem.value:
        return None
    return elem


def _walker(uia: ctypes.c_void_p) -> ctypes.c_void_p:
    walker = ctypes.c_void_p()
    hr = _com_call(
        uia, 14, ctypes.c_long, [ctypes.POINTER(ctypes.c_void_p)], [ctypes.byref(walker)]
    )
    if hr < 0 or not walker.value:
        raise RuntimeError("UI Automation tree walker is unavailable")
    return walker


def _child(walker: ctypes.c_void_p, elem: ctypes.c_void_p, index: int) -> ctypes.c_void_p | None:
    nxt = ctypes.c_void_p()
    hr = _com_call(
        walker,
        index,
        ctypes.c_long,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        [elem, ctypes.byref(nxt)],
    )
    if hr < 0 or not nxt.value:
        return None
    return nxt


def read_controls(hwnd: int) -> list[Control]:
    """Named controls inside one window, in tree order. Capped for a small model."""
    if sys.platform != "win32" or not hwnd:
        return []
    uia = _uia()
    root = None
    walker = None
    try:
        root = _element_from_handle(uia, hwnd)
        if root is None:
            return []
        walker = _walker(uia)
        found: list[Control] = []
        _walk_controls(walker, root, 0, found, [0])
        return found
    finally:
        _release(walker)
        _release(root)
        _release(uia)


def _wm_gettext(hwnd: int, limit: int = 8000) -> str:
    """Edit control text. Does not type or change the clipboard."""
    if not hwnd or sys.platform != "win32":
        return ""
    user32 = _user32()
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    length = int(user32.SendMessageW(hwnd, 0x000E, 0, 0) or 0)
    if length <= 0:
        return ""
    length = min(length, limit)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.SendMessageW(hwnd, 0x000D, length + 1, ctypes.addressof(buf))
    return buf.value


def _child_edit_text(hwnd: int) -> str:
    if not hwnd or sys.platform != "win32":
        return ""
    user32 = _user32()
    user32.EnumChildWindows.argtypes = [
        wintypes.HWND,
        ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
        wintypes.LPARAM,
    ]
    user32.EnumChildWindows.restype = wintypes.BOOL
    found: list[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(child, _lparam):
        cls = window_class(int(child)).casefold()
        if cls == "edit" or "richedit" in cls:
            text = _wm_gettext(int(child))
            if text.strip():
                found.append(text)
        return True

    user32.EnumChildWindows(hwnd, _cb, 0)
    if not found:
        return ""
    return max(found, key=len)


def _value_text(elem: ctypes.c_void_p) -> str:
    """UI Automation Value pattern. Empty when the control has none."""
    pattern = ctypes.c_void_p()
    hr = _com_call(
        elem,
        16,
        ctypes.c_long,
        [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
        [ctypes.c_int(10002), ctypes.byref(pattern)],
    )
    if hr < 0 or not pattern.value:
        return ""
    try:
        return _bstr(pattern, 4)
    finally:
        _release(pattern)


def _collect_values(walker: ctypes.c_void_p, elem: ctypes.c_void_p, depth: int, found: list[str]) -> None:
    if depth > 8 or len(found) >= 8:
        return
    child = _child(walker, elem, 4)
    while child is not None and len(found) < 8:
        nxt = None
        try:
            role = _ROLE.get(_int_prop(child, 21), "")
            if role in ("Edit", "Document"):
                text = _value_text(child).strip()
                if text:
                    found.append(text[:8000])
            _collect_values(walker, child, depth + 1, found)
            nxt = _child(walker, child, 6)
        finally:
            _release(child)
        child = nxt


def read_document_text(hwnd: int) -> str:
    """Text in the editor document. Empty when it cannot be read. Does not type."""
    if sys.platform != "win32" or not hwnd:
        return ""
    child = _child_edit_text(hwnd).strip()
    if child:
        return child[:8000]
    uia = None
    root = None
    walker = None
    try:
        uia = _uia()
        root = _element_from_handle(uia, hwnd)
        if root is None:
            return ""
        walker = _walker(uia)
        found: list[str] = []
        own = _value_text(root).strip()
        if own:
            found.append(own[:8000])
        _collect_values(walker, root, 0, found)
        if not found:
            return ""
        return max(found, key=len)
    except Exception:
        return ""
    finally:
        _release(walker)
        _release(root)
        _release(uia)


def focused_is_document(hwnd: int) -> bool:
    """True when keyboard focus is an edit or document inside the window, below the title bar."""
    if sys.platform != "win32" or not hwnd:
        return False
    uia = None
    elem = None
    try:
        uia = _uia()
        holder = ctypes.c_void_p()
        hr = _com_call(
            uia, 8, ctypes.c_long, [ctypes.POINTER(ctypes.c_void_p)], [ctypes.byref(holder)]
        )
        if hr < 0 or not holder.value:
            return False
        elem = holder
        role = _ROLE.get(_int_prop(elem, 21), "")
        if role not in ("Edit", "Document"):
            return False
        rect = _rect_prop(elem)
        if not rect:
            return False
        sx = int(rect[0] + rect[2] / 2)
        sy = int(rect[1] + rect[3] / 2)
        if not _inside_client(hwnd, sx, sy):
            return False
        try:
            _left, top, _width, _height = window_rect(hwnd)
        except Exception:
            return False
        return sy >= top + 48
    except Exception:
        return False
    finally:
        _release(elem)
        _release(uia)


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


def caret_in_document(hwnd: int) -> bool:
    """True when the caret sits in the window's client area, below the title bar."""
    if sys.platform != "win32" or not hwnd:
        return False
    user32 = _user32()
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(_GUITHREADINFO)]
    user32.GetGUIThreadInfo.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    thread = int(user32.GetWindowThreadProcessId(hwnd, None) or 0)
    info = _GUITHREADINFO()
    info.cbSize = ctypes.sizeof(_GUITHREADINFO)
    if not thread or not user32.GetGUIThreadInfo(thread, ctypes.byref(info)):
        return False
    caret = int(info.hwndCaret or 0)
    if not caret:
        return False
    origin = wintypes.POINT(int(info.rcCaret.left), int(info.rcCaret.top))
    if not user32.ClientToScreen(caret, ctypes.byref(origin)):
        return False
    if not _inside_client(hwnd, int(origin.x), int(origin.y)):
        return False
    try:
        _left, top, _width, _height = window_rect(hwnd)
    except Exception:
        return False
    return int(origin.y) >= top + 48


def document_ready(hwnd: int) -> bool:
    """The click landed in the document when focus or the caret is there."""
    return focused_is_document(hwnd) or caret_in_document(hwnd)


def content_click_landed(state: HandsContext) -> bool:
    """True only when the click point is inside the document and focus followed it."""
    if state.shell or not state.locked or len(state.content_at) != 2:
        return False
    x, y = int(state.content_at[0]), int(state.content_at[1])
    if not _inside_client(state.lock_hwnd, x, y):
        return False
    try:
        _left, top, _width, _height = window_rect(state.lock_hwnd)
    except Exception:
        return False
    if y < top + 48:
        return False
    deadline = time.time() + 1.0
    while True:
        if document_ready(state.lock_hwnd):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.15)


def document_contains(state: HandsContext, needed: str) -> bool:
    """True when the editor document itself contains the typed text."""
    needle = (needed or "").strip()
    if not needle or state.shell or not state.locked:
        return False
    deadline = time.time() + 1.2
    while True:
        live = read_document_text(state.lock_hwnd)
        if needle.casefold() in (live or "").casefold():
            state.landed_text = live
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.2)


def _walk_controls(
    walker: ctypes.c_void_p,
    elem: ctypes.c_void_p,
    depth: int,
    found: list[Control],
    visits: list[int],
) -> None:
    if depth > 8 or len(found) >= MAX_CONTROLS or visits[0] > 250:
        return
    visits[0] += 1
    child = _child(walker, elem, 4)
    while child is not None and len(found) < MAX_CONTROLS:
        nxt = None
        try:
            offscreen = _bool_prop(child, 38)
            if not offscreen:
                role_id = _int_prop(child, 21)
                role = _ROLE.get(role_id, "")
                name = _bstr(child, 23).strip()
                auto_id = _bstr(child, 29).strip()
                rect = _rect_prop(child)
                if role and rect and (name or auto_id or role in _UNNAMED_OK):
                    if not name and not auto_id:
                        unnamed = sum(1 for c in found if not c.name and not c.auto_id)
                        keep = unnamed < 8
                    else:
                        keep = True
                    if keep:
                        left, top, width, height = rect
                        found.append(
                            Control(
                                len(found) + 1,
                                role,
                                name[:80],
                                auto_id[:80],
                                left,
                                top,
                                width,
                                height,
                            )
                        )
                _walk_controls(walker, child, depth + 1, found, visits)
            nxt = _child(walker, child, 6)
        finally:
            _release(child)
        child = nxt


def _harness_window(title: str) -> bool:
    name = (title or "").casefold()
    return name.startswith("sm0l") or name.startswith("sm0k")


def list_top_windows() -> list[tuple[int, str]]:
    """Visible top-level windows as (hwnd, title). Skips sm0l itself."""
    if sys.platform != "win32":
        return []
    user32 = _user32()
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.EnumWindows.argtypes = [
        ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
        wintypes.LPARAM,
    ]
    user32.EnumWindows.restype = wintypes.BOOL
    dwmapi = ctypes.windll.dwmapi
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        cloaked = wintypes.DWORD()
        if dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0:
            if cloaked.value:
                return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title or _harness_window(title):
            return True
        found.append((int(hwnd), title))
        return True

    user32.EnumWindows(_cb, 0)
    return found


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """(left, top, width, height) in this process's pixel space."""
    user32 = _user32()
    rect = wintypes.RECT()
    dwmapi = ctypes.windll.dwmapi
    hr = dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))
    if hr != 0 or rect.right <= rect.left or rect.bottom <= rect.top:
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError("could not read the window rectangle")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 20 or height < 20:
        raise RuntimeError("window is too small to capture")
    return int(rect.left), int(rect.top), width, height


def client_center(hwnd: int) -> tuple[int, int] | None:
    """Center of the client area, below the title bar. Does not click."""
    if not hwnd or sys.platform != "win32":
        return None
    user32 = _user32()
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 20 or height < 20:
        return None
    point = wintypes.POINT(width // 2, max(8, height // 2))
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return None
    return int(point.x), int(point.y)


def _inside_client(hwnd: int, x: int, y: int) -> bool:
    if not hwnd or sys.platform != "win32":
        return False
    user32 = _user32()
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return False
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return False
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    return origin.x <= x < origin.x + width and origin.y <= y < origin.y + height


def content_point(state: HandsContext, controls: list[Control]) -> tuple[int, int] | None:
    """A point inside the document. Title-bar and off-window centers are skipped."""
    edit = input_control(controls)
    if edit is not None and not is_chrome(edit):
        sx = int(edit.left + edit.width / 2)
        sy = int(edit.top + edit.height / 2)
        if _inside_client(state.lock_hwnd, sx, sy):
            try:
                _left, top, _width, _height = window_rect(state.lock_hwnd)
            except Exception:
                top = sy
            if sy >= top + 48:
                return sx, sy
    return client_center(state.lock_hwnd)


def focus_window(hwnd: int) -> None:
    user32 = _user32()
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.AttachThreadInput.restype = wintypes.BOOL
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)
        time.sleep(0.15)
    user32.ShowWindow(hwnd, 5)
    this_thread = ctypes.windll.kernel32.GetCurrentThreadId()
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)
    attached = False
    if target_thread and target_thread != this_thread:
        attached = bool(user32.AttachThreadInput(this_thread, target_thread, True))
    try:
        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(this_thread, target_thread, False)


_SHELL_CLASSES = {"progman", "workerw", "syslistview32", "shelldll_defview"}
_KEEP_UP = _SHELL_CLASSES | {"shell_traywnd", "shell_secondarytraywnd"}


def window_class(hwnd: int) -> str:
    if sys.platform != "win32" or not hwnd:
        return ""
    user32 = _user32()
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    buf = ctypes.create_unicode_buffer(256)
    if not user32.GetClassNameW(hwnd, buf, 256):
        return ""
    return buf.value


def foreground_hwnd() -> int:
    if sys.platform != "win32":
        return 0
    user32 = _user32()
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    return int(user32.GetForegroundWindow() or 0)


def _hwnd_title(hwnd: int) -> str:
    if sys.platform != "win32" or not hwnd:
        return ""
    user32 = _user32()
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    length = int(user32.GetWindowTextLengthW(hwnd) or 0)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return " ".join(buf.value.split())


def foreground_title() -> str:
    return _hwnd_title(foreground_hwnd())


def _same_lock(hwnd: int, state: HandsContext) -> bool:
    if not hwnd or not state.lock_hwnd:
        return False
    if int(hwnd) == int(state.lock_hwnd):
        return True
    if sys.platform != "win32":
        return False
    user32 = _user32()
    user32.IsChild.argtypes = [wintypes.HWND, wintypes.HWND]
    user32.IsChild.restype = wintypes.BOOL
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    if user32.IsChild(state.lock_hwnd, hwnd):
        return True
    root = int(user32.GetAncestor(hwnd, 2) or 0)
    if root and root == int(state.lock_hwnd):
        return True
    if state.shell and window_class(hwnd).casefold() in _SHELL_CLASSES:
        return True
    user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetWindow.restype = wintypes.HWND
    owner = int(user32.GetWindow(hwnd, 4) or 0)
    if owner and owner in (int(state.lock_hwnd), int(state.editor_hwnd)):
        return True
    root_owner = int(user32.GetAncestor(hwnd, 3) or 0)
    if root_owner and root_owner == int(state.lock_hwnd):
        return True
    return False


def _focus_attempts(state: HandsContext) -> bool:
    for _ in range(2):
        try:
            focus_window(state.lock_hwnd)
        except Exception:
            continue
        time.sleep(0.08)
        if _same_lock(foreground_hwnd(), state):
            state.focus_title = foreground_title()
            if state.shell:
                state.shell_focused = True
            return True
    return False


def _find_titled(title: str) -> tuple[int, str] | None:
    """Visible window for this title. A 'name - App' title also tries the name."""
    text = (title or "").strip()
    if not text or is_desktop_query(text):
        return None
    chosen = choose_window(list_top_windows(), text)
    if chosen:
        return chosen
    if " - " in text:
        stem = text.split(" - ", 1)[0].strip()
        if stem and not is_desktop_query(stem):
            return choose_window(list_top_windows(), stem)
    return None


def _shell_ready(state: HandsContext) -> bool:
    """Progman and WorkerW never become the foreground window. A live shell is enough."""
    if not state.shell or not _window_alive(state.lock_hwnd):
        return False
    state.shell_focused = True
    state.focus_title = foreground_title() or "Desktop"
    return True


def confirm_focus(state: HandsContext) -> None:
    """Focus the locked window. A closed window is found again, then the desktop."""
    if not state.locked:
        raise ValueError("lock a window before using the pointer")
    if state.shell and _shell_ready(state):
        return
    if _focus_attempts(state):
        return
    if state.shell and lock_shell(state) and _focus_attempts(state):
        return
    if not state.shell:
        chosen = _find_titled(state.lock_title or state.query)
        if chosen:
            state.lock_hwnd, state.lock_title = chosen
            state.shell = False
            state.pinned = True
            state.editor_hwnd = chosen[0]
            if _focus_attempts(state):
                return
    if not state.shell and not _window_alive(state.lock_hwnd):
        if return_to_desktop(state) and state.shell and _window_alive(state.lock_hwnd):
            _focus_attempts(state)
            return
    if _shell_ready(state):
        return
    title = state.lock_title or "the target"
    raise RuntimeError(f"could not lock focus on {title}")


def find_editor_window(name: str) -> tuple[int, str] | None:
    """Visible window whose title is this file or icon. Does not focus or click."""
    if not name:
        return None
    hits = [
        (hwnd, title)
        for hwnd, title in list_top_windows()
        if editor_match(name, [title]) or matched_title(name, [title])
    ]
    if not hits:
        return None
    front = foreground_hwnd()
    for hwnd, title in hits:
        if int(hwnd) == int(front):
            return hwnd, title
    return hits[0]


def _wait_for_editor(
    name: str,
    seconds: float = 2.0,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[int, str] | None:
    """Poll until the editor window exists. Does not click."""
    deadline = time.time() + max(0.2, float(seconds))
    while True:
        if should_cancel and should_cancel():
            return None
        found = find_editor_window(name)
        if found:
            return found
        if time.time() >= deadline:
            return None
        time.sleep(0.2)


def _lock_miss(message: str) -> bool:
    return "could not lock focus" in (message or "").casefold()


def _recover_lock(state: HandsContext) -> bool:
    """Find the same window again, or the editor, or the desktop. Does not count as a failed action."""
    try:
        if not state.shell:
            chosen = _find_titled(state.lock_title or state.query)
            if chosen:
                state.lock_hwnd, state.lock_title = chosen
                state.shell = False
                state.pinned = True
                state.editor_hwnd = chosen[0]
                try:
                    confirm_focus(state)
                except Exception:
                    return _window_alive(state.lock_hwnd)
                return True
        if state.expect_file and not state.pinned:
            found = find_editor_window(state.expect_file)
            if found:
                adopt_window(state, found[0], found[1], state.expect_file)
                try:
                    confirm_focus(state)
                except Exception:
                    return _window_alive(state.lock_hwnd)
                return True
        if state.shell or not _window_alive(state.lock_hwnd):
            if return_to_desktop(state) and _window_alive(state.lock_hwnd):
                try:
                    confirm_focus(state)
                except Exception:
                    return True
                return True
            if lock_shell(state):
                state.pinned = False
                return _window_alive(state.lock_hwnd)
    except Exception:
        return False
    return _window_alive(state.lock_hwnd)


def adopt_window(state: HandsContext, hwnd: int, title: str, filename: str = "") -> None:
    """Lock the editor (or other opened window) and remember the desktop handle."""
    if state.shell and state.lock_hwnd:
        state.desktop_hwnd = state.lock_hwnd
    state.shell = False
    state.lock_hwnd = int(hwnd)
    state.lock_title = title
    state.editor_hwnd = int(hwnd)
    state.focus_title = title
    if filename:
        state.expect_file = filename
    stem = file_stem(filename) or title
    if stem:
        state.query = stem
    state.pinned = True
    state.input_ready = False
    state.content_at = ()


def return_to_desktop(state: HandsContext) -> bool:
    """Put the lock back on the same desktop handle, or find the shell again."""
    if state.editor_hwnd and not _window_alive(state.editor_hwnd):
        state.editor_hwnd = 0
    if _window_alive(state.desktop_hwnd):
        state.lock_hwnd = state.desktop_hwnd
        state.lock_title = "Desktop"
        state.shell = True
        state.shell_focused = False
        state.pinned = False
        state.input_ready = False
        return True
    if lock_shell(state):
        state.pinned = False
        state.input_ready = False
        return True
    return False


def desktop_folder() -> str:
    if sys.platform != "win32":
        return ""
    shell32 = ctypes.windll.shell32
    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(_GUID),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    shell32.SHGetKnownFolderPath.restype = ctypes.HRESULT
    folder_id = _guid("B4BFCC3A-DB2C-424C-B029-7FE99A87C641")
    pointer = ctypes.c_void_p()
    hr = shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(pointer))
    if hr == 0 and pointer.value:
        try:
            return ctypes.wstring_at(pointer.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(pointer)
    return os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")


def desktop_file_path(name: str) -> str:
    """Desktop file matching this icon name. Empty when it is not there. Does not open it."""
    folder = desktop_folder()
    if not folder or not name:
        return ""
    direct = os.path.join(folder, name)
    if os.path.isfile(direct):
        return direct
    needle = _norm_name(name)
    stem = _norm_name(file_stem(name))
    try:
        entries = os.listdir(folder)
    except OSError:
        return ""
    for entry in entries:
        if _norm_name(entry) == needle or _norm_name(file_stem(entry)) == stem:
            path = os.path.join(folder, entry)
            if os.path.isfile(path):
                return path
    return ""


def file_contains(path: str, needed: str) -> bool:
    """True when this file's text contains needed. Does not open the file in an editor."""
    if not path or not (needed or "").strip():
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            data = handle.read(16000)
    except OSError:
        return False
    return needed.casefold() in data.casefold()


def _file_stamp(name: str) -> tuple[int, float]:
    path = desktop_file_path(name)
    if not path:
        return (0, 0.0)
    try:
        stat = os.stat(path)
    except OSError:
        return (0, 0.0)
    return (int(stat.st_size), float(stat.st_mtime))


def saved_contains(name: str, needed: str, before: tuple[int, float]) -> bool:
    """True when the desktop file holds the text, or was rewritten when no text was required."""
    path = desktop_file_path(name)
    if not path:
        return False
    deadline = time.time() + 2.0
    while True:
        try:
            stat = os.stat(path)
            stamp = (int(stat.st_size), float(stat.st_mtime))
        except OSError:
            stamp = (0, 0.0)
        if needed and file_contains(path, needed):
            return True
        if not needed and stamp != before and stamp != (0, 0.0):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.25)


def launch_desktop_file(name: str) -> str:
    """Open a desktop file in its registered editor. Returns the new window title, or ''."""
    path = desktop_file_path(name)
    if not path or sys.platform != "win32":
        return ""
    shell32 = ctypes.windll.shell32
    shell32.ShellExecuteW.argtypes = [
        wintypes.HWND,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_int,
    ]
    shell32.ShellExecuteW.restype = wintypes.HINSTANCE
    result = int(shell32.ShellExecuteW(None, "open", path, None, None, 1) or 0)
    if result <= 32:
        return ""
    return confirm_opened(file_stem(name) or name, visible_window_titles, tries=6, pause=0.25)


def ensure_file_editor(state: HandsContext, name: str) -> str:
    """Lock the file's editor, opening the desktop file once if that window is missing."""
    if not name:
        return ""
    if state.locked and not state.shell and _window_alive(state.lock_hwnd):
        title = state.lock_title or ""
        if editor_match(name, [title]) or matched_title(name, [title]):
            state.editor_hwnd = state.lock_hwnd
            state.expect_file = name
            return title
        if state.pinned:
            return ""
    found = find_editor_window(name)
    if found:
        adopt_window(state, found[0], found[1], name)
        return found[1]
    if not looks_like_file(name):
        return ""
    if state.launched_file == _norm_name(name):
        return ""
    state.launched_file = _norm_name(name)
    title = launch_desktop_file(name)
    if not title:
        return ""
    found = find_editor_window(name)
    if found:
        adopt_window(state, found[0], found[1], name)
        return found[1]
    return title


def minimize_interfering(state: HandsContext) -> int:
    """Minimize other titled windows so the desktop is visible. Does not close them."""
    state.desktop_cleared = True
    if sys.platform != "win32" or not state.shell:
        return 0
    user32 = _user32()
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.EnumWindows.argtypes = [
        ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
        wintypes.LPARAM,
    ]
    user32.EnumWindows.restype = wintypes.BOOL
    dwmapi = ctypes.windll.dwmapi
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        if state.lock_hwnd and int(hwnd) == int(state.lock_hwnd):
            return True
        cloaked = wintypes.DWORD()
        if dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0:
            if cloaked.value:
                return True
        if window_class(hwnd).casefold() in _KEEP_UP:
            return True
        length = int(user32.GetWindowTextLengthW(hwnd) or 0)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title or _harness_window(title):
            return True
        found.append(int(hwnd))
        return True

    user32.EnumWindows(_cb, 0)
    for hwnd in found:
        user32.ShowWindow(hwnd, _SW_MINIMIZE)
    if found:
        time.sleep(0.25)
    return len(found)


def _clipboard_open():
    user32 = _user32()
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.EnumClipboardFormats.argtypes = [wintypes.UINT]
    user32.EnumClipboardFormats.restype = wintypes.UINT
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p
    return user32, kernel32


def clipboard_snapshot() -> list[tuple[int, bytes]]:
    """Copy every memory-backed clipboard format. Refuses if one cannot be copied."""
    user32, kernel32 = _clipboard_open()
    if not user32.OpenClipboard(None):
        raise RuntimeError("clipboard is busy")
    try:
        items: list[tuple[int, bytes]] = []
        fmt = 0
        while True:
            fmt = int(user32.EnumClipboardFormats(fmt))
            if not fmt:
                break
            handle = user32.GetClipboardData(fmt)
            if not handle:
                raise RuntimeError(f"clipboard format {fmt} could not be read")
            size = int(kernel32.GlobalSize(handle))
            if size <= 0:
                raise RuntimeError(f"clipboard format {fmt} is not plain memory")
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                raise RuntimeError(f"clipboard format {fmt} could not be locked")
            try:
                items.append((fmt, ctypes.string_at(ptr, size)))
            finally:
                kernel32.GlobalUnlock(handle)
        return items
    finally:
        user32.CloseClipboard()


def clipboard_restore(items: list[tuple[int, bytes]]) -> None:
    user32, kernel32 = _clipboard_open()
    if not user32.OpenClipboard(None):
        raise RuntimeError("clipboard is busy")
    allocated: list[ctypes.c_void_p] = []
    try:
        if not user32.EmptyClipboard():
            raise RuntimeError("could not empty the clipboard")
        for fmt, data in items:
            handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
            if not handle:
                raise RuntimeError("could not allocate clipboard memory")
            allocated.append(handle)
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                raise RuntimeError("could not lock clipboard memory")
            try:
                ctypes.memmove(ptr, data, len(data))
            finally:
                kernel32.GlobalUnlock(handle)
            if not user32.SetClipboardData(fmt, handle):
                raise RuntimeError(f"could not restore clipboard format {fmt}")
            allocated.pop()
    except Exception:
        for handle in allocated:
            kernel32.GlobalFree(handle)
        raise
    finally:
        user32.CloseClipboard()


def clipboard_set_text(text: str) -> None:
    data = text.encode("utf-16-le") + b"\x00\x00"
    clipboard_restore([(_CF_UNICODETEXT, data)])


def paste_text(text: str) -> str:
    """Put text on the clipboard, send ctrl+v, then restore the previous clipboard."""
    if len(text) > TYPE_CAP:
        text = text[:TYPE_CAP]
    saved = clipboard_snapshot()
    try:
        clipboard_set_text(text)
        time.sleep(0.05)
        _press_chord("ctrl+v")
        time.sleep(0.3)
    finally:
        clipboard_restore(saved)
    note = f"pasted {len(text)} characters and restored the clipboard"
    if len(text) >= TYPE_CAP:
        note += f" (capped at {TYPE_CAP})"
    return note


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    import struct
    import zlib

    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_rgb_png(width: int, height: int, rgb: bytes) -> bytes:
    import struct
    import zlib

    if width <= 0 or height <= 0 or len(rgb) != width * height * 3:
        raise ValueError("rgb size")
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )


def _num(value: Any, default: float | None = None) -> float:
    if value is None or value is False:
        if default is None:
            raise ValueError("missing number")
        return float(default)
    return float(value)


def _user32():
    if sys.platform != "win32":
        raise RuntimeError("hands_on controls the Windows desktop only")
    return ctypes.windll.user32


def virtual_screen() -> tuple[int, int, int, int]:
    user32 = _user32()
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    return (
        int(user32.GetSystemMetrics(76)),
        int(user32.GetSystemMetrics(77)),
        int(user32.GetSystemMetrics(78)),
        int(user32.GetSystemMetrics(79)),
    )


def window_physical_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """GetWindowRect in physical pixels, the same space as screen capture."""
    user32 = _user32()
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if not hwnd or not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def move_topmost(hwnd: int, x: int, y: int) -> None:
    """Move a window to physical pixels without resizing or activating it."""
    user32 = _user32()
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), int(x), int(y), 0, 0, 0x0001 | 0x0010 | 0x0040)


def dpi_at(x: int, y: int) -> int:
    user32 = _user32()
    user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    user32.MonitorFromPoint.restype = wintypes.HMONITOR
    monitor = user32.MonitorFromPoint(wintypes.POINT(int(x), int(y)), 2)
    if not monitor:
        return 96
    try:
        shcore = ctypes.windll.shcore
        shcore.GetDpiForMonitor.argtypes = [
            wintypes.HMONITOR,
            ctypes.c_int,
            ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(wintypes.UINT),
        ]
        shcore.GetDpiForMonitor.restype = ctypes.c_long
        dx = wintypes.UINT()
        dy = wintypes.UINT()
        if shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dx), ctypes.byref(dy)) == 0 and dx.value:
            return int(dx.value)
    except Exception:
        pass
    return 96


def _abort_pressed() -> bool:
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_F12) & 0x8000)


def _bind_gdi():
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, ctypes.c_int]
    gdi32.SetStretchBltMode.restype = ctypes.c_int
    gdi32.StretchBlt.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.StretchBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    return user32, gdi32


def _bgra_to_rgb(bgra: bytes, width: int, height: int, flip: bool) -> bytes:
    stride = width * 4
    rgb = bytearray(width * height * 3)
    for y in range(height):
        src_y = (height - 1 - y) if flip else y
        src = src_y * stride
        dst = y * width * 3
        for x in range(width):
            i = src + x * 4
            o = dst + x * 3
            rgb[o] = bgra[i + 2]
            rgb[o + 1] = bgra[i + 1]
            rgb[o + 2] = bgra[i]
    return bytes(rgb)


def _mark_cursor(bgra: bytearray, width: int, height: int, ix: int, iy: int) -> None:
    if ix < 0 or iy < 0 or ix >= width or iy >= height:
        return

    def put(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            o = (y * width + x) * 4
            bgra[o : o + 4] = b"\xff\x00\xff\xff"

    for d in range(-8, 9):
        put(ix + d, iy)
        put(ix, iy + d)


def capture_screen(
    region: tuple[int, int, int, int] | None = None,
    mark_cursor: bool = True,
) -> Shot:
    """Grab the virtual desktop, or one window rect. Long edge is at most MAX_EDGE."""
    user32, gdi32 = _bind_gdi()
    if region is None:
        vx = int(user32.GetSystemMetrics(76))
        vy = int(user32.GetSystemMetrics(77))
        vw = int(user32.GetSystemMetrics(78))
        vh = int(user32.GetSystemMetrics(79))
    else:
        vx, vy, vw, vh = (int(v) for v in region)
    if vw <= 0 or vh <= 0:
        raise RuntimeError("no desktop to capture")
    scale = min(1.0, MAX_EDGE / float(max(vw, vh)))
    iw = max(1, int(round(vw * scale)))
    ih = max(1, int(round(vh * scale)))

    screen = user32.GetDC(0)
    if not screen:
        raise RuntimeError("GetDC failed")
    mem = gdi32.CreateCompatibleDC(screen)
    bmp = gdi32.CreateCompatibleBitmap(screen, iw, ih)
    if not mem or not bmp:
        if bmp:
            gdi32.DeleteObject(bmp)
        if mem:
            gdi32.DeleteDC(mem)
        user32.ReleaseDC(0, screen)
        raise RuntimeError("could not create a screenshot bitmap")
    old = gdi32.SelectObject(mem, bmp)
    try:
        gdi32.SetStretchBltMode(mem, _HALFTONE)
        ok = gdi32.StretchBlt(mem, 0, 0, iw, ih, screen, vx, vy, vw, vh, _SRCCOPY)
        if not ok:
            raise RuntimeError("StretchBlt failed")
        gdi32.SelectObject(mem, old)
        old = None
        header = _BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        header.biWidth = iw
        header.biHeight = -ih
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0
        buf = ctypes.create_string_buffer(iw * ih * 4)
        lines = gdi32.GetDIBits(mem, bmp, 0, ih, buf, ctypes.byref(header), 0)
        flip = False
        if lines == 0:
            header.biHeight = ih
            lines = gdi32.GetDIBits(mem, bmp, 0, ih, buf, ctypes.byref(header), 0)
            flip = True
        if lines == 0:
            raise RuntimeError("GetDIBits failed")
        raw = bytearray(buf.raw)
        pt = wintypes.POINT()
        if mark_cursor and user32.GetCursorPos(ctypes.byref(pt)):
            ix = int(round((pt.x - vx) * iw / float(vw)))
            iy = int(round((pt.y - vy) * ih / float(vh)))
            if flip:
                iy = ih - 1 - iy
            _mark_cursor(raw, iw, ih, ix, iy)
        png = encode_rgb_png(iw, ih, _bgra_to_rgb(bytes(raw), iw, ih, flip))
    finally:
        if old:
            gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(0, screen)

    attachment = save_encoded(png, "image/png", "hands.png")
    return Shot(attachment, vx, vy, vw, vh, iw, ih)


def stamp_end_button(
    path: str,
    shot: Shot,
    rect: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Paint the end button into the screenshot so the vision model can read it."""
    box = image_box(shot, rect)
    if not box or not path:
        return box
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
        from PyQt6.QtCore import QRectF
    except Exception:
        return box
    img = QImage(path)
    if img.isNull():
        return box
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    img.setDevicePixelRatio(1.0)
    x, y, w, h = box
    painter = QPainter(img)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = max(4, min(16, h // 3))
        painter.setPen(QPen(QColor(167, 139, 250, 230), max(1, h // 16)))
        painter.setBrush(QColor(18, 14, 28, 180))
        painter.drawRoundedRect(QRectF(x, y, w, h), radius, radius)
        painter.setPen(QColor(255, 255, 255, 240))
        font = QFont("Segoe UI")
        font.setBold(True)
        font.setPixelSize(max(8, min(22, int(h * 0.42))))
        painter.setFont(font)
        painter.drawText(
            QRectF(x, y, w, h),
            int(Qt.AlignmentFlag.AlignCenter),
            END_LABEL,
        )
    finally:
        painter.end()
    img.save(path, "PNG")
    return box


def _set_cursor(x: int, y: int) -> None:
    user32 = _user32()
    user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    user32.SetCursorPos.restype = wintypes.BOOL
    if not user32.SetCursorPos(int(x), int(y)):
        raise RuntimeError(f"SetCursorPos failed at {x},{y}")


def _mouse_event(flags: int, data: int = 0) -> None:
    user32 = _user32()
    user32.mouse_event.argtypes = [
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t,
    ]
    user32.mouse_event(flags, 0, 0, ctypes.c_uint32(data & 0xFFFFFFFF).value, 0)


def _click_button(button: str, times: int) -> None:
    button = (button or "left").lower()
    if button == "right":
        down, up = _RIGHTDOWN, _RIGHTUP
    elif button == "middle":
        down, up = _MIDDLEDOWN, _MIDDLEUP
    else:
        down, up = _LEFTDOWN, _LEFTUP
    for i in range(max(1, times)):
        _mouse_event(down)
        time.sleep(0.02)
        _mouse_event(up)
        if i + 1 < times:
            time.sleep(0.05)


def _point(action: dict, shot: Shot, key_x: str = "x", key_y: str = "y") -> tuple[int, int, int, int]:
    ix = _num(action.get(key_x))
    iy = _num(action.get(key_y))
    sx, sy = image_to_screen(
        ix, iy, vx=shot.vx, vy=shot.vy, vw=shot.vw, vh=shot.vh, iw=shot.iw, ih=shot.ih,
    )
    return int(round(ix)), int(round(iy)), sx, sy


def _tap_vk(vk: int) -> None:
    user32 = _user32()
    user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t]
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.01)
    user32.keybd_event(vk, 0, _KEYUP, 0)


def _vk_of(token: str) -> tuple[int, bool]:
    if token in _VK:
        return _VK[token], False
    if len(token) != 1:
        raise ValueError(f"unknown key {token}")
    user32 = _user32()
    user32.VkKeyScanW.argtypes = [wintypes.WCHAR]
    user32.VkKeyScanW.restype = ctypes.c_short
    scan = int(user32.VkKeyScanW(token))
    if scan < 0:
        raise ValueError(f"cannot type {token!r} as a key")
    return scan & 0xFF, bool(scan & 0x100)


def _send_unicode(code: int) -> None:
    user32 = _user32()
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    down = INPUT()
    down.type = 1
    down.union.ki = _KEYBDINPUT(0, code, _UNICODE, 0, 0)
    up = INPUT()
    up.type = 1
    up.union.ki = _KEYBDINPUT(0, code, _UNICODE | _KEYUP, 0, 0)
    sent = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
    sent += user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))
    if sent < 2:
        raise RuntimeError("SendInput failed while typing")


def _type_text(text: str, should_cancel: Callable[[], bool] | None) -> str:
    if len(text) > TYPE_CAP:
        text = text[:TYPE_CAP]
    count = 0
    for ch in text:
        if should_cancel and should_cancel():
            return f"type cancelled after {count} characters"
        if _abort_pressed():
            return f"type aborted (F12) after {count} characters"
        if ch == "\r":
            continue
        if ch == "\n":
            _tap_vk(0x0D)
        elif ch == "\t":
            _tap_vk(0x09)
        else:
            data = ch.encode("utf-16-le")
            for i in range(0, len(data), 2):
                _send_unicode(int.from_bytes(data[i : i + 2], "little"))
        count += 1
        time.sleep(0.008)
    note = f"typed {count} characters"
    if len(text) >= TYPE_CAP:
        note += f" (capped at {TYPE_CAP})"
    return note


def _press_chord(spec: str) -> str:
    tokens = parse_keys(spec)
    downs: list[int] = []
    try:
        for token in tokens:
            vk, shift = _vk_of(token)
            if shift and 0x10 not in downs:
                _tap_hold(0x10, downs)
            if vk not in downs:
                _tap_hold(vk, downs)
        time.sleep(0.03)
    finally:
        user32 = _user32()
        for vk in reversed(downs):
            user32.keybd_event(vk, 0, _KEYUP, 0)
    return "key " + "+".join(tokens)


def _tap_hold(vk: int, downs: list[int]) -> None:
    user32 = _user32()
    user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t]
    user32.keybd_event(vk, 0, 0, 0)
    downs.append(vk)


def _in_shot(sx: float, sy: float, shot: Shot, margin: int = 12) -> bool:
    return (shot.vx - margin) <= sx < (shot.vx + shot.vw + margin) and (
        shot.vy - margin
    ) <= sy < (shot.vy + shot.vh + margin)


def _target_key(control: Control | None, sx: int, sy: int) -> str:
    if control is not None:
        named = _norm_name(control.name or control.auto_id or "")
        return named or f"n{control.index}"
    return f"pt{int(sx)},{int(sy)}"


def note_miss(state: HandsContext) -> None:
    """Next try for this target uses a different point."""
    key = state.last_target or ""
    if not key:
        state.retry_index += 1
    elif state.retry_name != key:
        state.retry_name = key
        state.retry_index = 1
    else:
        state.retry_index += 1
    if state.retry_index > 6:
        state.missed_icon = state.retry_name or key


def _locate(
    action: dict,
    shot: Shot,
    state: HandsContext,
    controls: list[Control],
    key_x: str = "x",
    key_y: str = "y",
    allow_control: bool = True,
) -> tuple[int, int, int, int, Control | None]:
    if not state.locked:
        raise ValueError("lock a window before using the pointer")
    control = resolve_control(action, controls) if allow_control else None
    if control is not None:
        sx = int(control.left + control.width / 2)
        sy = int(control.top + control.height / 2)
    else:
        if action.get(key_x) is None or action.get(key_y) is None:
            raise ValueError("click needs n, target, or x and y")
        ix_raw = _num(action.get(key_x))
        iy_raw = _num(action.get(key_y))
        if ix_raw < -1 or iy_raw < -1 or ix_raw > shot.iw + 1 or iy_raw > shot.ih + 1:
            raise ValueError("point is outside the locked window")
        _ix, _iy, sx, sy = _point(action, shot, key_x, key_y)
        if allow_control:
            near = nearest_labeled(controls, sx, sy)
            if near is not None:
                control = near
                sx = int(control.left + control.width / 2)
                sy = int(control.top + control.height / 2)
    if not _in_shot(sx, sy, shot):
        if control is not None:
            raise ValueError(f"{control.role} {control.name or control.index} is outside the locked window")
        raise ValueError("point is outside the locked window")
    key = _target_key(control, sx, sy)
    state.last_target = key
    origin_x, origin_y = sx, sy
    if state.retry_name == key and state.retry_index:
        dpi = 96
        if sys.platform == "win32":
            try:
                dpi = dpi_at(origin_x, origin_y)
            except Exception:
                dpi = 96
        sx, sy = shift_point(origin_x, origin_y, state.retry_index, dpi)
        if not _in_shot(sx, sy, shot, margin=48):
            sx, sy = origin_x, origin_y
    ix, iy = screen_to_image(sx, sy, shot)
    return ix, iy, int(sx), int(sy), control


def _aim(state: HandsContext) -> None:
    if state.locked:
        confirm_focus(state)


def _names_end_button(action: dict) -> bool:
    blob = " ".join(
        str(action.get(key) or "")
        for key in ("target", "title", "text", "name", "label")
    ).casefold()
    return "end tool" in blob


def prepare_action(action: dict, controls: list[Control], step: str) -> dict:
    """A title-bar click on a terminal or editor step hits the input instead."""
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    if kind == "doubleclick":
        kind = "double_click"
    if kind not in ("click", "double_click", "right_click"):
        return action
    if _names_end_button(action):
        return action
    try:
        control = resolve_control(action, controls)
    except ValueError:
        return action
    edit = input_control(controls)
    wants = step_wants_content(step) or prefers_input(step, action)
    if wants and edit is not None:
        nxt = dict(action)
        nxt["n"] = edit.index
        nxt.pop("target", None)
        nxt.pop("name", None)
        nxt.pop("x", None)
        nxt.pop("y", None)
        return nxt
    if control is None or not is_chrome(control):
        return action
    if edit is None:
        return action
    if control.role not in ("TitleBar", "Window") and not wants:
        return action
    nxt = dict(action)
    nxt["n"] = edit.index
    nxt.pop("target", None)
    nxt.pop("name", None)
    nxt.pop("x", None)
    nxt.pop("y", None)
    return nxt


def _is_alt_f4(spec: str) -> bool:
    try:
        return parse_keys(spec) == ["alt", "f4"]
    except ValueError:
        return False


def _is_save_key(spec: str) -> bool:
    try:
        tokens = [token.casefold() for token in parse_keys(spec)]
    except ValueError:
        return False
    return "s" in tokens and ("ctrl" in tokens or "control" in tokens)


def _is_new_tab(spec: str) -> bool:
    """Ctrl+T opens a browser tab. It is not a document edit and not a click."""
    try:
        tokens = [token.casefold() for token in parse_keys(spec)]
    except ValueError:
        return False
    return tokens in (["ctrl", "t"], ["control", "t"])


_ADDRESS_FIELD = re.compile(r"address|omnibox|\burl\b|location", re.I)
_BLANK_ADDRESS = re.compile(r"^(?:|about:blank|about:newtab|about:home)$", re.I)
_SEARCH_HINT = re.compile(r"search|find a|type a url|enter a url", re.I)


def _focused_field() -> tuple[str, str, str] | None:
    """Name, automation id, and value of the focused control. None when UIA cannot see it."""
    if sys.platform != "win32":
        return None
    uia = None
    elem = None
    try:
        uia = _uia()
        holder = ctypes.c_void_p()
        hr = _com_call(
            uia, 8, ctypes.c_long, [ctypes.POINTER(ctypes.c_void_p)], [ctypes.byref(holder)]
        )
        if hr < 0 or not holder.value:
            return None
        elem = holder
        return _bstr(elem, 23).strip(), _bstr(elem, 29).strip(), _value_text(elem).strip()
    except Exception:
        return None
    finally:
        _release(elem)
        _release(uia)


def _address_value_ready(value: str) -> bool:
    folded = (value or "").strip()
    if _BLANK_ADDRESS.match(folded):
        return True
    return bool(
        folded
        and _SEARCH_HINT.search(folded)
        and "://" not in folded
        and "." not in folded
    )


def address_bar_ready() -> bool:
    """The focused control is an address bar that is empty or still showing a search hint."""
    field = _focused_field()
    if field is None:
        return False
    name, auto_id, value = field
    if not _ADDRESS_FIELD.search(f"{name} {auto_id}"):
        return False
    return _address_value_ready(value)


def _collect_address(
    walker: ctypes.c_void_p,
    elem: ctypes.c_void_p,
    depth: int,
    found: list[str],
) -> None:
    if depth > 6 or found:
        return
    child = _child(walker, elem, 4)
    while child is not None and not found:
        nxt = None
        try:
            name = _bstr(child, 23)
            auto_id = _bstr(child, 29)
            if _ADDRESS_FIELD.search(f"{name} {auto_id}"):
                value = _value_text(child).strip()
                if value:
                    found.append(value)
                    return
            _collect_address(walker, child, depth + 1, found)
            nxt = _child(walker, child, 6)
        finally:
            _release(child)
        child = nxt


def _address_in_window(hwnd: int) -> str:
    """Value of the first address field in this window. Empty when none is readable."""
    if sys.platform != "win32" or not hwnd:
        return ""
    uia = None
    root = None
    walker = None
    try:
        uia = _uia()
        root = _element_from_handle(uia, hwnd)
        if root is None:
            return ""
        walker = _walker(uia)
        found: list[str] = []
        _collect_address(walker, root, 0, found)
        return found[0] if found else ""
    except Exception:
        return ""
    finally:
        _release(walker)
        _release(root)
        _release(uia)


def new_tab_landed(state: HandsContext, title_before: str) -> bool:
    """Ctrl+T landed when the window title changes, or the address bar is a fresh tab.

    A title that already says New tab is not enough. The title has to change,
    or the focused address bar has to be empty.
    """
    before = " ".join((title_before or "").split()).casefold()
    deadline = time.time() + 1.5
    while True:
        title = _hwnd_title(state.lock_hwnd) if state.lock_hwnd else ""
        if not title:
            title = foreground_title()
        current = " ".join(title.split()).casefold()
        if current and before and current != before:
            return True
        try:
            if address_bar_ready():
                return True
        except Exception:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(0.2)


def _url_host(text: str) -> str:
    match = re.search(
        r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9.-]*\.[a-z]{2,})",
        text or "",
        re.I,
    )
    if not match:
        return ""
    return match.group(1).casefold().rstrip(".")


def address_typed(state: HandsContext, text: str, title_before: str) -> bool:
    """A typed URL counts when the title or the address bar contains that host."""
    host = _url_host(text)
    if not host:
        return False
    before = " ".join((title_before or "").split()).casefold()
    deadline = time.time() + 1.5
    while True:
        title = _hwnd_title(state.lock_hwnd) if state.lock_hwnd else ""
        if not title:
            title = foreground_title()
        current = " ".join(title.split()).casefold()
        if host in current and current != before:
            return True
        value = ""
        field = _focused_field()
        if field is not None and _ADDRESS_FIELD.search(f"{field[0]} {field[1]}"):
            value = field[2]
        if host not in value.casefold() and state.lock_hwnd:
            value = _address_in_window(state.lock_hwnd)
        if host in (value or "").casefold():
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.2)


def _is_win_d(spec: str) -> bool:
    try:
        tokens = parse_keys(spec)
    except ValueError:
        return False
    return tokens in (["win", "d"], ["meta", "d"], ["super", "d"])


def _prepare_editor(state: HandsContext, require_document: bool) -> None:
    """Focus the file's editor and click its document before typing or an edit chord."""
    if state.expect_file and (state.shell or not _window_alive(state.editor_hwnd)):
        ensure_file_editor(state, state.expect_file)
    if state.shell or not state.locked:
        raise ValueError("type needs the editor focused, not the desktop")
    confirm_focus(state)
    try:
        live = read_controls(state.lock_hwnd)
    except Exception:
        live = []
    point = content_point(state, live)
    if point is not None:
        _set_cursor(point[0], point[1])
        time.sleep(0.04)
        _click_button("left", 1)
        state.input_ready = True
    elif require_document:
        raise ValueError("type needs a click in the editor document")


def _click_input(state: HandsContext, controls: list[Control]) -> None:
    """Click the edit or document center so typing does not land on the title bar."""
    edit = input_control(controls)
    if edit is None:
        return
    sx = int(edit.left + edit.width / 2)
    sy = int(edit.top + edit.height / 2)
    key = _target_key(edit, sx, sy)
    state.last_target = key
    if state.retry_name == key and state.retry_index:
        dpi = 96
        if sys.platform == "win32":
            try:
                dpi = dpi_at(sx, sy)
            except Exception:
                dpi = 96
        sx, sy = shift_point(sx, sy, state.retry_index, dpi)
    _set_cursor(sx, sy)
    time.sleep(0.04)
    _click_button("left", 1)


def end_click_point(
    action: dict,
    shot: Shot,
    button_rect: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Image and screen point when this pointer action is the end button. No cursor move."""
    if not button_rect:
        return None
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    if kind in ("rightclick",):
        kind = "right_click"
    if kind in ("doubleclick",):
        kind = "double_click"
    if kind not in ("click", "double_click", "right_click"):
        return None
    if action.get("x") is not None and action.get("y") is not None:
        try:
            ix, iy, sx, sy = _point(action, shot)
        except Exception:
            pass
        else:
            if hits_end_button(sx, sy, button_rect):
                return ix, iy, sx, sy
    if _names_end_button(action):
        left, top, right, bottom = (int(v) for v in button_rect)
        sx = (left + right) // 2
        sy = (top + bottom) // 2
        ix, iy = screen_to_image(sx, sy, shot)
        return ix, iy, sx, sy
    return None


def _press_end_button(button_rect: tuple[int, int, int, int]) -> None:
    """Left-click the center of the on-screen end button."""
    left, top, right, bottom = (int(v) for v in button_rect)
    _set_cursor((left + right) // 2, (top + bottom) // 2)
    time.sleep(0.04)
    _click_button("left", 1)


def open_desktop_target(
    action: dict,
    shot: Shot,
    state: HandsContext,
    controls: list[Control],
    request: str,
    should_cancel: Callable[[], bool] | None = None,
    button_rect: tuple[int, int, int, int] | None = None,
) -> tuple[str, bool] | None:
    """Open a named desktop icon by its real position and stop when the window appears.

    Returns None when the action is not that open attempt, so the caller uses perform.
    """
    if not state.shell:
        return None
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    if kind in ("doubleclick",):
        kind = "double_click"
    if kind not in ("click", "double_click"):
        return None
    if end_click_point(action, shot, button_rect) is not None:
        return None
    try:
        _ix, _iy, sx, sy, control = _locate(action, shot, state, controls)
    except ValueError:
        return None
    phrase = goal_phrase(request)
    clicked = find_icon(state.desktop_icons, control.name) if control is not None else None
    wanted = find_icon(state.desktop_icons, phrase) if phrase else None
    icon: DesktopIcon | None = None
    if clicked is not None and (
        not phrase or _norm_name(phrase) in _norm_name(clicked.name) or _norm_name(clicked.name) in _norm_name(phrase)
    ):
        icon = clicked
    elif clicked is None and wanted is not None:
        icon = wanted
    elif control is not None and phrase and (
        _norm_name(phrase) in _norm_name(control.name) or _norm_name(control.name) in _norm_name(phrase)
    ):
        icon = None
    else:
        return None
    target = icon.name if icon is not None else (control.name if control is not None else phrase)
    if not target:
        return None
    state.expect_file = target
    if kind != "double_click":
        return None
    if state.missed_icon and state.missed_icon == _norm_name(target):
        return f"{target} did not open on the last try; do not click it again", False
    found = find_editor_window(target)
    if found:
        adopt_window(state, found[0], found[1], target)
        state.missed_icon = ""
        state.retry_name = ""
        state.retry_index = 0
        state.last_target = _norm_name(target)
        return f"editor focused: {found[1]}", False
    confirm_focus(state)
    dpi = dpi_at(sx, sy) if sys.platform == "win32" else 96
    points = attempt_points(icon, sx, sy, dpi)
    if not points:
        return None
    key = _norm_name(target)
    idx = state.retry_index if state.retry_name == key else 0
    if idx >= len(points) + 4:
        state.missed_icon = key
        return f"{target} did not open on the last try; do not click it again", False
    if idx < len(points):
        px, py = points[idx]
    else:
        ox, oy = points[0]
        px, py = shift_point(ox, oy, idx - len(points) + 1, dpi)
    state.last_target = key
    state.retry_name = key
    state.retry_index = idx + 1
    if should_cancel and should_cancel():
        return "cancelled", True
    _set_cursor(px, py)
    time.sleep(0.04)
    _click_button("left", 2)
    found = _wait_for_editor(target, 2.0, should_cancel)
    if found:
        adopt_window(state, found[0], found[1], target)
        state.missed_icon = ""
        state.retry_name = ""
        state.retry_index = 0
        return f"editor focused: {found[1]}", False
    return f"double_click {target} screen ({px},{py})", False


def perform(
    action: dict,
    shot: Shot,
    *,
    state: HandsContext | None = None,
    controls: list[Control] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    button_rect: tuple[int, int, int, int] | None = None,
    click_button: bool = False,
    step: str = "",
) -> tuple[str, bool]:
    """Run one action. Returns (log line, finished)."""
    state = state or HandsContext()
    controls = controls or []
    kind = str(action.get("action") or action.get("name") or "").strip().lower().replace("-", "_")
    if kind in ("rightclick",):
        kind = "right_click"
    if kind in ("doubleclick",):
        kind = "double_click"
    if kind == "done":
        text = str(action.get("text") or action.get("reason") or "done").strip()
        return f"done: {text[:1500]}", True
    if kind == "lock":
        title = str(action.get("title") or action.get("window") or "").strip()
        if not title:
            raise ValueError("lock needs a title")
        if is_desktop_query(title):
            if state.shell and state.locked:
                return f"already locked to {state.lock_title}", False
            if not lock_shell(state):
                raise ValueError("desktop shell was not found")
            state.pinned = False
            state.input_ready = False
            return f"locked to {state.lock_title}", False
        if (
            state.locked
            and not state.shell
            and title.casefold() in (state.lock_title or "").casefold()
        ):
            state.pinned = True
            return f"already locked to {state.lock_title}", False
        chosen = _find_titled(title)
        if not chosen:
            raise ValueError(f"no visible window matching {title!r}")
        state.query = title
        state.shell = False
        state.pinned = True
        state.input_ready = False
        state.lock_hwnd, state.lock_title = chosen
        state.editor_hwnd = chosen[0]
        return f"locked to {state.lock_title}", False
    if kind == "wait":
        ms = int(_num(action.get("ms"), 400))
        ms = min(max(ms, 0), 3000)
        waited = 0
        while waited < ms:
            if should_cancel and should_cancel():
                return "wait cancelled", True
            if _abort_pressed():
                return "aborted (F12)", True
            step = min(50, ms - waited)
            time.sleep(step / 1000)
            waited += step
        return f"wait {ms}ms", False
    if kind in ("click", "double_click", "right_click"):
        if end_click_point(action, shot, button_rect) is not None:
            if not state.locked:
                raise ValueError("lock a window before using the pointer")
            if click_button and button_rect is not None:
                try:
                    _press_end_button(button_rect)
                except Exception:
                    pass
            return "clicked end tool rounds", True
        confirm_focus(state)
        ix, iy, sx, sy, control = _locate(action, shot, state, controls)
        aimed_content = False
        state.content_at = ()
        if state.want_content and not state.shell:
            point = content_point(state, controls)
            if point is not None:
                sx, sy = int(point[0]), int(point[1])
                try:
                    ix, iy = screen_to_image(sx, sy, shot)
                except Exception:
                    pass
                edit = input_control(controls)
                if edit is not None:
                    cx = int(edit.left + edit.width / 2)
                    cy = int(edit.top + edit.height / 2)
                    control = edit if (sx, sy) == (cx, cy) else None
                else:
                    control = None
                aimed_content = True
                state.content_at = (sx, sy)
        state.input_ready = aimed_content
        _set_cursor(sx, sy)
        time.sleep(0.04)
        times = 2 if kind == "double_click" else 1
        button = "right" if kind == "right_click" else str(action.get("button") or "left")
        _click_button(button, times)
        if control is not None:
            label = control.name or control.auto_id or control.role
            return f"{kind} n={control.index} {label} screen ({sx},{sy})", False
        return f"{kind} image ({ix},{iy}) screen ({sx},{sy})", False
    if kind == "drag":
        confirm_focus(state)
        ix, iy, sx, sy, _c1 = _locate(action, shot, state, controls)
        ix2, iy2, sx2, sy2, _c2 = _locate(
            action, shot, state, controls, "x2", "y2", allow_control=False
        )
        _set_cursor(sx, sy)
        time.sleep(0.04)
        _mouse_event(_LEFTDOWN)
        steps = max(8, int(((sx2 - sx) ** 2 + (sy2 - sy) ** 2) ** 0.5 / 12))
        try:
            for i in range(1, steps + 1):
                if should_cancel and should_cancel():
                    return "drag cancelled", True
                t = i / steps
                _set_cursor(int(sx + (sx2 - sx) * t), int(sy + (sy2 - sy) * t))
                time.sleep(0.01)
        finally:
            _mouse_event(_LEFTUP)
        return (
            f"drag image ({ix},{iy})->({ix2},{iy2}) screen ({sx},{sy})->({sx2},{sy2})",
            False,
        )
    if kind in ("type", "paste"):
        text = str(action.get("text") or "")
        use_paste = kind == "paste" or len(text) >= PASTE_AT
        editing = edits_document(state, step)
        if editing and state.locked and not state.shell and text.strip():
            try:
                already = read_document_text(state.lock_hwnd)
            except Exception:
                already = ""
            if text.casefold() in already.casefold():
                state.landed_text = already
                preview = text.replace("\n", "\\n")[:40]
                return f"typed already present: {preview}", False
        if state.locked and editing and not state.shell:
            _prepare_editor(state, require_document=True)
        elif state.shell and editing:
            raise ValueError("type needs the editor focused, not the desktop")
        elif use_paste and not state.locked:
            raise ValueError("lock a window before pasting")
        if use_paste:
            try:
                note = paste_text(text)
            except Exception as e:
                if kind == "paste":
                    raise ValueError(f"paste refused, clipboard unchanged: {e}") from e
                note = _type_text(text, should_cancel) + " (paste unavailable)"
        else:
            note = _type_text(text, should_cancel)
        preview = text.replace("\n", "\\n")[:40]
        stop = "cancelled" in note or "aborted" in note
        return f"{note}: {preview}", stop
    if kind == "key":
        spec = str(action.get("keys") or action.get("key") or "")
        if _is_new_tab(spec):
            if state.locked:
                try:
                    confirm_focus(state)
                except Exception:
                    if state.shell and lock_shell(state):
                        confirm_focus(state)
                    else:
                        raise
            state.title_before = _hwnd_title(state.lock_hwnd) or foreground_title()
            return _press_chord(spec), False
        if _is_alt_f4(spec):
            if state.expect_file and (state.shell or not state.pinned):
                found = find_editor_window(state.expect_file)
                if found:
                    adopt_window(state, found[0], found[1], state.expect_file)
            if state.pinned and _window_alive(state.lock_hwnd) and not state.shell:
                state.editor_hwnd = state.lock_hwnd
            if state.shell or not _window_alive(state.editor_hwnd):
                raise ValueError("alt+f4 needs the editor in front")
            confirm_focus(state)
            note = _press_chord(spec)
            time.sleep(0.25)
            if not _window_alive(state.editor_hwnd):
                state.editor_hwnd = 0
                return_to_desktop(state)
            return note, False
        if is_edit_chord(spec):
            _prepare_editor(state, require_document=False)
            return _press_chord(spec), False
        if is_global_chord(spec):
            if state.shell and _is_win_d(spec):
                try:
                    confirm_focus(state)
                except Exception:
                    if lock_shell(state):
                        try:
                            confirm_focus(state)
                        except Exception:
                            pass
                if state.locked and _same_lock(foreground_hwnd(), state):
                    return "key win+d (desktop already showing)", False
            return _press_chord(spec), False
        if state.locked:
            try:
                confirm_focus(state)
            except Exception:
                if state.shell and lock_shell(state):
                    confirm_focus(state)
                else:
                    raise
        return _press_chord(spec), False
    if kind == "scroll":
        confirm_focus(state)
        ix, iy, sx, sy, control = _locate(action, shot, state, controls)
        dy = int(_num(action.get("dy"), -3))
        dy = min(max(dy, -10), 10)
        _set_cursor(sx, sy)
        time.sleep(0.03)
        _mouse_event(_WHEEL, dy * 120)
        if control is not None:
            return f"scroll {dy} on n={control.index} {control.name or control.role}", False
        return f"scroll {dy} at image ({ix},{iy})", False
    raise ValueError(f"unknown action {kind or '(empty)'}")


def _hide_sm0l() -> list[int]:
    user32 = _user32()
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if _harness_window(buf.value):
            found.append(int(hwnd))
        return True

    user32.EnumWindows(_cb, 0)
    for hwnd in found:
        user32.ShowWindow(hwnd, _SW_MINIMIZE)
    if found:
        time.sleep(0.45)
    return found


def _restore(hwnds: list[int]) -> None:
    user32 = _user32()
    for hwnd in hwnds:
        user32.ShowWindow(hwnd, _SW_RESTORE)


def _delete_shot(shot: Shot | None) -> None:
    if not shot:
        return
    path = (shot.attachment or {}).get("path")
    if not path:
        return
    try:
        from pathlib import Path

        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def screen_signature(region: tuple[int, int, int, int] | None = None) -> tuple:
    """Coarse color sample. The pointer is not drawn, so the cursor alone is not a change."""
    shot = capture_screen(region, mark_cursor=False)
    path = str((shot.attachment or {}).get("path") or "")
    try:
        from PyQt6.QtGui import QImage

        img = QImage(path)
        if img.isNull():
            return tuple()
        img = img.convertToFormat(QImage.Format.Format_RGB32)
        width = max(1, img.width())
        height = max(1, img.height())
        cells: list[tuple[int, int, int]] = []
        for gy in range(8):
            for gx in range(12):
                x = min(width - 1, int((gx + 0.5) * width / 12))
                y = min(height - 1, int((gy + 0.5) * height / 8))
                color = int(img.pixel(x, y)) & 0xFFFFFFFF
                cells.append(
                    (
                        ((color >> 16) & 255) // 24,
                        ((color >> 8) & 255) // 24,
                        (color & 255) // 24,
                    )
                )
        return tuple(cells)
    finally:
        _delete_shot(shot)


def _ready_to_end(
    request: str,
    plan: str,
    state: HandsContext,
    done_actions: int,
    steps_needed: int,
) -> bool:
    """The end button is for a finished plan whose requested text actually landed."""
    if done_actions < steps_needed:
        return False
    return typing_landed(required_typing(request, plan), state.landed_text)


def _auto_end(
    log: list[str],
    turn: int,
    line: str,
    aim: tuple[int, int, int, int] | None,
    click_button: bool,
) -> None:
    """Click the end button once the goal is confirmed."""
    if click_button and aim is not None:
        try:
            _press_end_button(aim)
        except Exception:
            pass
    if line:
        log.append(f"{turn}. {line}")
    log.append(f"{turn}. clicked end tool rounds")


def _report(status: str, goal: str, log: list[str]) -> str:
    task = (goal or "").strip()
    if len(task) > 900:
        task = task[:900].rstrip() + "…"
    steps = "\n".join(log) if log else "(no actions)"
    return f"hands on {status}\nplan:\n{task}\nsteps:\n{steps}"


def _refresh_lock(state: HandsContext) -> str | None:
    """Keep the locked window. If it closed, find that title again, then the desktop."""
    if sys.platform != "win32":
        return None
    user32 = _user32()
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL

    def take(chosen: tuple[int, str]) -> None:
        state.lock_hwnd, state.lock_title = chosen
        state.shell = False
        state.pinned = True
        state.editor_hwnd = chosen[0]

    if state.editor_hwnd and not _window_alive(state.editor_hwnd):
        state.editor_hwnd = 0
        if not state.shell:
            chosen = _find_titled(state.lock_title or state.query)
            if chosen:
                take(chosen)
                return None
            if return_to_desktop(state):
                return None
    if state.shell:
        if state.lock_hwnd and user32.IsWindow(state.lock_hwnd):
            return None
        if lock_shell(state):
            state.pinned = False
            return None
        return "stopped: desktop shell is gone"
    if state.lock_hwnd and not user32.IsWindow(state.lock_hwnd):
        chosen = _find_titled(state.lock_title or state.query)
        if chosen:
            take(chosen)
            return None
        if state.expect_file and not state.pinned:
            found = find_editor_window(state.expect_file)
            if found:
                adopt_window(state, found[0], found[1], state.expect_file)
                return None
        if return_to_desktop(state):
            return None
        return f"stopped: locked window is gone ({state.lock_title})"
    if state.query and not state.locked:
        if is_desktop_query(state.query):
            lock_shell(state)
        else:
            chosen = _find_titled(state.query)
            if chosen:
                take(chosen)
    return None


def tool_hands_on(
    workspace,
    prompt: str = "",
    max_turns: int = DEFAULT_TURNS,
    host: str = "",
    model: str = "",
    num_ctx: int = 0,
    should_cancel: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_window: Callable[[bool], None] | None = None,
    window: str = "",
    on_badge: Callable[[int, int, int, int], None] | None = None,
    button_rect: Callable[[], tuple[int, int, int, int] | None] | None = None,
    end_requested: Callable[[], bool] | None = None,
    brief: str = "",
    provider: str = "xai",
    api_key: str = "",
    reasoning: str = "off",
    reason_track=None,
    **_k: Any,
) -> str:
    del workspace  # desktop tool; the workspace is not the target
    request = str(prompt or "").strip()
    if not request:
        return "ERROR: hands_on needs prompt — copy the user's request unchanged."
    if sys.platform != "win32":
        return "ERROR: hands_on controls the Windows desktop only."
    if not (model or "").strip():
        return "ERROR: hands_on has no model to look at the screen."
    cloud = (provider or "xai") in ("xai", "meta")
    if cloud and not str(api_key or "").strip():
        who = "Meta" if (provider or "") == "meta" else "xAI"
        return f"ERROR: hands_on needs the {who} API key so the cloud model can see the screen."
    if not cloud and not (host or "").strip():
        return "ERROR: hands_on has no local model host."
    try:
        turns = int(max_turns or DEFAULT_TURNS)
    except (TypeError, ValueError):
        turns = DEFAULT_TURNS
    turns = min(max(turns, 1), MAX_TURNS_CAP)

    def stop_reason() -> str | None:
        if end_requested and end_requested():
            return "ended from the end tool rounds button"
        if _abort_pressed():
            return "aborted (F12)"
        if should_cancel and should_cancel():
            return "cancelled"
        return None

    def watching() -> bool:
        return stop_reason() is not None

    look_only = is_look_only(request)
    log: list[str] = []
    bad = 0
    done_actions = 0
    steps_needed = 1
    hidden: list[int] = []
    goal = request
    plan = ""
    state = HandsContext()
    try:
        if on_window:
            on_window(True)
            time.sleep(0.4)
        else:
            hidden = _hide_sm0l()
        if not look_only and on_status:
            on_status("hands on · writing a plan from the chat")
        plan = make_hands_plan(
            host,
            model,
            request,
            str(brief or ""),
            num_ctx,
            watching,
            provider,
            api_key,
            reasoning=reasoning,
            reason_track=reason_track,
        )
        why = stop_reason()
        if why:
            return _report(why, request, [])
        if not plan:
            plan = fallback_plan(request)
        plan = executable_plan(plan, request)
        goal = focus_text(plan, request)
        if on_status:
            on_status("hands on · plan\n" + plan)
        requested = str(window or "").strip()
        if is_desktop_query(requested):
            requested = "desktop"
        elif not requested:
            requested = infer_window_query(request, list_top_windows())
        state = HandsContext(query=requested)
        if is_desktop_query(requested) or not requested:
            if not lock_shell(state):
                return _report("could not lock desktop", goal, [])
        else:
            chosen = _find_titled(requested)
            if chosen:
                state.lock_hwnd, state.lock_title = chosen
                state.shell = False
                state.editor_hwnd = chosen[0]
                state.pinned = True
                state.query = requested
            else:
                note = closed_window_note(requested)
                if not lock_shell(state):
                    return _report(f"could not lock {requested}", goal, [note])
                log.append(note)
                if on_status:
                    on_status("hands on · " + note)
        steps_needed = non_end_steps(plan)
        if state.shell and not look_only and not state.desktop_cleared:
            try:
                minimize_interfering(state)
            except Exception:
                state.desktop_cleared = True
        for turn in range(1, turns + 1):
            why = stop_reason()
            if why:
                return _report(why, goal, log)
            gone = _refresh_lock(state)
            if gone:
                return _report(gone, goal, log)
            state.focus_title = foreground_title()
            if state.shell and state.expect_file and not state.pinned:
                found = find_editor_window(state.expect_file)
                if found:
                    adopt_window(state, found[0], found[1], state.expect_file)
                    if step_kind(current_step(plan, done_actions)) == "double_click":
                        state.held_kind = ""
                        state.retry_name = ""
                        state.retry_index = 0
                        done_actions += 1
                        log.append(f"{turn}. editor focused: {found[1]}")
            if on_status:
                where = state.lock_title or "unlocked"
                on_status(
                    f"hands on · {where} · screenshot {turn}/{turns} · "
                    "end tool rounds is on screen · F12 or Stop aborts"
                )
            shot: Shot | None = None
            controls: list[Control] = []
            titles = visible_window_titles() if look_only or not state.locked else []
            windows = "\n".join(f"- {title}" for title in titles)
            aim: tuple[int, int, int, int] | None = None
            live_button = False
            region = None
            try:
                step_now = current_step(plan, done_actions)
                bare_win = bool(log) and bool(re.search(r"\bkey win$", log[-1].strip(), re.I))
                start_type = (
                    state.shell
                    and bare_win
                    and step_kind(step_now) == "type"
                    and not edits_document(state, step_now)
                )
                if state.locked and not look_only and not start_type:
                    try:
                        confirm_focus(state)
                    except Exception:
                        if _recover_lock(state):
                            log.append(
                                f"{turn}. focus lost; locked {state.lock_title or 'the target'} again"
                            )
                        else:
                            log.append(
                                f"{turn}. could not lock focus on {state.lock_title or 'the target'}"
                            )
                            if on_status:
                                on_status("hands on · retry the same step")
                            time.sleep(0.3)
                            continue
                    if not state.shell:
                        try:
                            region = window_rect(state.lock_hwnd)
                        except Exception:
                            region = None
                    try:
                        controls = read_controls(state.lock_hwnd)
                    except Exception:
                        controls = []
                    if state.shell:
                        try:
                            state.desktop_icons = read_desktop_icons()
                        except Exception:
                            state.desktop_icons = []
                        controls = controls_with_icons(controls, state.desktop_icons)
                bounds = region if region is not None else virtual_screen()
                place = end_button_place(*bounds)
                if on_badge:
                    try:
                        on_badge(*place)
                    except Exception:
                        pass
                live = None
                if button_rect:
                    try:
                        live = button_rect()
                    except Exception:
                        live = None
                live_button = bool(live)
                if live:
                    aim = tuple(int(v) for v in live)  # type: ignore[assignment]
                else:
                    x, y, w, h = place
                    aim = (x, y, x + w, y + h)
                why = stop_reason()
                if why:
                    return _report(why, goal, log)
                shot = capture_screen(region)
                path = str((shot.attachment or {}).get("path") or "")
                box = stamp_end_button(path, shot, aim)
                messages = hands_messages(
                    goal,
                    _shot_text(
                        shot.iw,
                        shot.ih,
                        log,
                        lock_title="" if look_only else state.lock_title,
                        controls_text="" if look_only else format_controls(controls, shot),
                        look_only=look_only,
                        windows=windows,
                        shell=bool(state.shell) and not look_only,
                        focus_title="" if look_only else state.focus_title,
                        editor=bool(state.editor_hwnd) and not state.shell and not look_only,
                        badge=badge_sentence(
                            box,
                            when=(
                                "After the description, you may click it."
                                if look_only
                                else "Click it only after the plan is finished."
                            ),
                        ),
                    ),
                    shot.attachment,
                )
                from .reasoning import hands_predict

                options: dict[str, Any] = {
                    "temperature": 0.1,
                    "num_predict": hands_predict(provider, reasoning, look=look_only),
                    "reasoning": reasoning,
                }
                if num_ctx and int(num_ctx) > 0:
                    options["num_ctx"] = int(num_ctx)
                result = _look(
                    provider,
                    host,
                    api_key,
                    model,
                    messages,
                    options,
                    180,
                    watching,
                    reason_track=reason_track,
                    reason_label="REASONING · hands on",
                )
            finally:
                _delete_shot(shot)
            why = stop_reason()
            if why:
                return _report(why, goal, log)
            content = ((result.get("message") or {}).get("content") or "").strip()
            if shot is None:
                return _report("failed: no screenshot", goal, log)
            if look_only:
                described = describe_look(content, titles)
                action = None
                try:
                    action = parse_action(content)
                except Exception:
                    action = None
                if action is not None and end_click_point(action, shot, aim) is not None:
                    try:
                        perform(
                            action,
                            shot,
                            state=state,
                            controls=controls,
                            should_cancel=watching,
                            button_rect=aim,
                            click_button=live_button,
                        )
                    except Exception:
                        pass
                    lines = list(described) if described else list(_look_fallback(titles))
                    lines.append("ended from the end tool rounds button")
                    return _report("done", goal, lines)
                if described:
                    return _report("done", goal, described)
                bad += 1
                if action is not None and is_finish_action(action):
                    log.append(f"{turn}. that was a placeholder; describe what is actually on the screen")
                else:
                    log.append(f"{turn}. ignored a click; describe the screenshot with done")
                if bad >= 2:
                    return _report("done", goal, _look_fallback(titles))
                continue
            try:
                action = parse_action(content)
            except Exception as e:
                bad += 1
                log.append(f"{turn}. rejected: {e}")
                if bad >= 3:
                    return _report("stopped (3 unreadable actions)", goal, log)
                continue
            step = current_step(plan, done_actions)
            if is_finish_action(action) and not step_is_describe(step):
                if _ready_to_end(request, plan, state, done_actions, steps_needed):
                    _auto_end(log, turn, "", aim, live_button)
                    return _report("ended from the end tool rounds button", goal, log)
                bad += 1
                shown = step or "the next plan step"
                log.append(f"{turn}. do this step literally: {shown}")
                if on_status:
                    on_status("hands on · do the plan step")
                if bad >= 3:
                    return _report("stopped before the plan was started", goal, log)
                continue
            if state.held_kind and not repeats_held(state.held_kind, action):
                if end_click_point(action, shot, aim) is None:
                    bad += 1
                    log.append(f"{turn}. previous step is not confirmed; repeat it")
                    if on_status:
                        on_status("hands on · previous step is not confirmed")
                    if bad >= 3:
                        return _report("stopped before the plan was verified", goal, log)
                    continue
            if (
                step
                and not step_is_describe(step)
                and not action_matches_step(action, step)
                and end_click_point(action, shot, aim) is None
                and not is_finish_action(action)
            ):
                needed = required_typing(request, plan)
                typing_now = (
                    action_kind(action) in ("type", "paste")
                    and bool(needed)
                    and not typing_landed(needed, state.landed_text)
                    and not state.shell
                )
                if not typing_now:
                    bad += 1
                    log.append(f"{turn}. do this step literally: {step}")
                    if on_status:
                        on_status("hands on · do the plan step")
                    if bad >= 3:
                        return _report("stopped before the plan was started", goal, log)
                    continue
            if (
                not step_is_describe(step)
                and not _ready_to_end(request, plan, state, done_actions, steps_needed)
                and (
                    is_finish_action(action)
                    or end_click_point(action, shot, aim) is not None
                )
            ):
                bad += 1
                if done_actions >= steps_needed and not typing_landed(
                    required_typing(request, plan), state.landed_text
                ):
                    log.append(f"{turn}. typed text is not confirmed; click the document and type it")
                else:
                    log.append(f"{turn}. plan is not finished; do the next step before ending")
                if on_status:
                    on_status("hands on · plan is not finished")
                if bad >= 3:
                    return _report("stopped before the plan was started", goal, log)
                continue
            if step_is_describe(step) and is_finish_action(action):
                try:
                    line, _finished = perform(
                        action,
                        shot,
                        state=state,
                        controls=controls,
                        should_cancel=watching,
                        button_rect=aim,
                        click_button=False,
                    )
                except Exception as e:
                    bad += 1
                    log.append(f"{turn}. rejected: {e}")
                    if bad >= 3:
                        return _report("stopped (3 unreadable actions)", goal, log)
                    continue
                bad = 0
                done_actions += 1
                state.retry_name = ""
                state.retry_index = 0
                state.held_kind = ""
                log.append(f"{turn}. {line}")
                if on_status:
                    on_status(f"hands on · {line}")
                if _ready_to_end(request, plan, state, done_actions, steps_needed):
                    _auto_end(log, turn, "", aim, live_button)
                    return _report("ended from the end tool rounds button", goal, log)
                continue
            state.want_content = bool(step_wants_content(step) or prefers_input(step, action))
            editor_before = int(state.editor_hwnd or 0)
            state.title_before = _hwnd_title(state.lock_hwnd) or foreground_title()
            file_before = _file_stamp(state.expect_file)
            action = prepare_action(action, controls, step)
            before_titles = visible_window_titles()
            try:
                before_sig = screen_signature(region)
            except Exception:
                before_sig = None
            try:
                opened = open_desktop_target(
                    action,
                    shot,
                    state,
                    controls,
                    request,
                    watching,
                    aim,
                )
                if opened is None:
                    line, finished = perform(
                        action,
                        shot,
                        state=state,
                        controls=controls,
                        should_cancel=watching,
                        button_rect=aim,
                        click_button=live_button,
                        step=step,
                    )
                else:
                    line, finished = opened
            except Exception as e:
                message = str(e)
                if _lock_miss(message):
                    if _recover_lock(state):
                        log.append(f"{turn}. focus lost; locked {state.lock_title or 'the target'} again")
                    else:
                        log.append(f"{turn}. focus lost; retry the same step")
                    if on_status:
                        on_status("hands on · " + log[-1].split(". ", 1)[-1])
                    continue
                bad += 1
                if "editor" in message.casefold() or needs_editor(action):
                    state.held_kind = action_kind(action) or "type"
                log.append(f"{turn}. rejected: {message}")
                if bad >= 3:
                    return _report("stopped (3 unreadable actions)", goal, log)
                continue
            bad = 0
            why = stop_reason()
            if line.startswith("editor focused"):
                state.held_kind = ""
                state.retry_name = ""
                state.retry_index = 0
                done_actions += 1
                log.append(f"{turn}. {line}")
                if on_status:
                    on_status(f"hands on · {line}")
                if _ready_to_end(request, plan, state, done_actions, steps_needed):
                    _auto_end(log, turn, "", aim, live_button)
                    return _report("ended from the end tool rounds button", goal, log)
                continue
            if finished or why:
                log.append(f"{turn}. {line}")
                if on_status:
                    on_status(f"hands on · {line}")
                if why and why.startswith("ended"):
                    status = why
                elif "end tool rounds" in line:
                    status = "ended from the end tool rounds button"
                elif line.startswith("done"):
                    status = "done"
                elif why:
                    status = why
                else:
                    status = "done"
                return _report(status, goal, log)
            if "did not open" in line:
                state.held_kind = "double_click"
                log.append(f"{turn}. {line}")
                if on_status:
                    on_status(f"hands on · {line}")
                continue
            if line.startswith("wait"):
                log.append(f"{turn}. {line}")
                if on_status:
                    on_status(f"hands on · {line}")
                continue
            time.sleep(0.3)
            after_titles = visible_window_titles()
            try:
                after_sig = screen_signature(region)
            except Exception:
                after_sig = None
            name = state.expect_file
            kind_now = action_kind(action)
            spec = str(action.get("keys") or action.get("key") or "")
            opened = ""
            if line.startswith("double_click") and state.shell and name:
                found = find_editor_window(name)
                if found:
                    adopt_window(state, found[0], found[1], name)
                    opened = found[1]
                else:
                    opened = ensure_file_editor(state, name)
            pixels_changed = set(before_titles) != set(after_titles) or (
                before_sig is not None
                and after_sig is not None
                and before_sig != tuple()
                and after_sig != tuple()
                and before_sig != after_sig
            )
            if line.startswith("double_click") and name:
                confirmed = bool(opened) and not state.shell and bool(state.editor_hwnd)
            elif kind_now in ("type", "paste"):
                text_now = str(action.get("text") or "")
                if _is_address_step(f"{step} {text_now}") and not state.expect_file:
                    confirmed = address_typed(state, text_now, state.title_before)
                    if confirmed and text_now.strip() and not typing_landed(text_now, state.landed_text):
                        state.landed_text = (
                            f"{state.landed_text}\n{text_now}".strip() if state.landed_text else text_now
                        )
                elif edits_document(state, step):
                    if state.locked and not state.shell:
                        confirmed = document_contains(state, text_now)
                    else:
                        confirmed = False
                    if not confirmed and text_now.strip() and state.expect_file:
                        path = desktop_file_path(state.expect_file)
                        stamp = _file_stamp(state.expect_file)
                        if (
                            path
                            and stamp != file_before
                            and stamp != (0, 0.0)
                            and file_contains(path, text_now)
                        ):
                            try:
                                with open(path, encoding="utf-8", errors="replace") as handle:
                                    state.landed_text = handle.read(16000)
                            except OSError:
                                state.landed_text = text_now
                            confirmed = True
                else:
                    appeared = set(after_titles) - set(before_titles)
                    confirmed = pixels_changed or bool(appeared)
                    if confirmed and text_now.strip() and not typing_landed(text_now, state.landed_text):
                        state.landed_text = (
                            f"{state.landed_text}\n{text_now}".strip() if state.landed_text else text_now
                        )
            elif kind_now == "key" and _is_alt_f4(spec):
                confirmed = bool(editor_before) and not _window_alive(editor_before)
            elif kind_now == "key" and _is_new_tab(spec):
                confirmed = new_tab_landed(state, state.title_before)
            elif kind_now == "key" and _is_save_key(spec):
                confirmed = saved_contains(
                    state.expect_file,
                    required_typing(request, plan),
                    file_before,
                )
            elif (
                state.want_content
                and not state.shell
                and kind_now in ("click", "double_click")
            ):
                confirmed = content_click_landed(state)
            else:
                confirmed = (
                    "desktop already showing" in line
                    or line.startswith("locked")
                    or line.startswith("already locked")
                    or (action_matches_step(action, step) and pixels_changed)
                )
            if confirmed:
                state.retry_name = ""
                state.retry_index = 0
                state.missed_icon = ""
                state.held_kind = ""
                done_actions += 1
                shown = f"{line}; editor {opened}" if opened else line
                log.append(f"{turn}. {shown}")
                if on_status:
                    on_status(f"hands on · {shown}")
                if _ready_to_end(request, plan, state, done_actions, steps_needed):
                    _auto_end(log, turn, "", aim, live_button)
                    return _report("ended from the end tool rounds button", goal, log)
                continue
            if kind_now == "key" and _is_new_tab(spec):
                state.held_kind = "key"
                log.append(f"{turn}. {line}; new tab was not confirmed")
                if on_status:
                    on_status("hands on · new tab was not confirmed")
                continue
            typed_now = str(action.get("text") or "")
            if (
                kind_now in ("type", "paste")
                and _is_address_step(f"{step} {typed_now}")
                and not state.expect_file
            ):
                state.held_kind = "type"
                log.append(f"{turn}. {line}; address was not confirmed")
                if on_status:
                    on_status("hands on · address was not confirmed")
                continue
            note_miss(state)
            state.held_kind = held_from_line(line) or state.held_kind
            if state.retry_index > 6:
                log.append(f"{turn}. {line}; not confirmed after shifted tries")
                state.retry_name = ""
                state.retry_index = 0
            else:
                log.append(f"{turn}. {line}; not confirmed; next click will shift")
            if on_status:
                on_status("hands on · " + log[-1].split(". ", 1)[-1])
        return _report(f"stopped after {turns} turns", goal, log)
    except Exception as e:
        return _report(f"failed: {e}", goal, log)
    finally:
        if on_window:
            try:
                on_window(False)
            except Exception:
                pass
        elif hidden:
            try:
                _restore(hidden)
            except Exception:
                pass
