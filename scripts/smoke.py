"""Headless checks — no GUI, no Ollama required.

hands_on checks stay off the pointer, the keyboard, and the clipboard.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import parse_text_tool_calls
from src.compact import compact_threshold, estimate_tokens, split_for_compact
from src.media import attachments_to_b64, prepare_ollama_messages, save_encoded
from src.tools import clip, html_to_text, run_tool, tool_edit_file, tool_write_file


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print("ok", msg)


def main() -> None:
    calls = parse_text_tool_calls(
        'hello <tool_call>{"name":"search","arguments":{"query":"qwen"}}</tool_call>'
    )
    check(len(calls) == 1 and calls[0]["function"]["name"] == "search", "xml tool parse")

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a" * 50},
        {"role": "assistant", "content": "b" * 50},
        {"role": "user", "content": "c" * 50},
        {"role": "assistant", "content": "d" * 50, "tool_calls": [{"function": {"name": "search"}}]},
        {"role": "tool", "content": "hits"},
        {"role": "user", "content": "now"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    prefix, suffix = split_for_compact(msgs, keep_user_turns=2)
    check(suffix[0]["role"] == "user", "compact split on user turn")
    check(estimate_tokens(msgs) > 0, "token estimate")
    check(compact_threshold(32768) > 8000, "32k compact threshold")
    check(compact_threshold(8192) < compact_threshold(32768), "threshold scales with window")
    check("truncated" in clip("x" * 100, 20), "clip")
    check("Hello" in html_to_text("<html><script>x</script><p>Hello</p></html>"), "html text")

    tmp = ROOT / ".smoke_ws"
    tmp.mkdir(exist_ok=True)
    r = tool_write_file(tmp, "n.py", "print(1)\n")
    check("Wrote" in r, "write_file")
    r = tool_edit_file(tmp, "n.py", "print(1)", "print(2)")
    check("Edited" in r, "edit_file")
    r = run_tool("read_file", {"path": "n.py"}, tmp)
    check("print(2)" in r, "read_file")
    r = run_tool("list_dir", {"path": "."}, tmp)
    check("n.py" in r, "list_dir")

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )
    att = save_encoded(png, "image/png", "dot.png")
    check(Path(att["path"]).is_file(), "media save")
    b64 = attachments_to_b64([att])
    check(len(b64) == 1 and len(b64[0]) > 8, "media b64")
    msgs = [{"role": "user", "content": "see", "attachments": [att]}]
    prep = prepare_ollama_messages(msgs)
    check("images" in prep[0] and "attachments" not in prep[0], "ollama images field")
    check(
        estimate_tokens(msgs) > estimate_tokens([{"role": "user", "content": "see"}]),
        "image token estimate",
    )
    from src.dpapi import protect, unprotect
    from src.personality import SUBAGENT_RUNTIME, build_system_prompt
    from src.tools import schemas_for, subagent_schemas
    from src.xai_client import to_openai_messages
    from src.config import Settings

    secret = "sk-test-not-a-real-key"
    blob = protect(secret)
    check(unprotect(blob) == secret and secret not in blob, "dpapi roundtrip")
    s = Settings()
    s.set_xai_key(secret)
    check(s.xai_key() == secret and secret not in s.xai_key_dpapi, "settings key stays encrypted")
    grok = build_system_prompt(tmp / "ops_xai", provider="xai")
    muse = build_system_prompt(tmp / "ops_meta", provider="meta")
    local = build_system_prompt(tmp / "ops_local", provider="ollama")
    check("You are Grok" in grok and "ask_local" in grok and "sm0l" in grok, "grok system prompt")
    check("You are Muse" in muse and "ask_local" in muse and "sm0l" in muse, "muse system prompt")
    check("sm0l harness" in local and "You are not Grok" in local and "not Muse" in local, "local harness prompt")
    check("sm0l harness" in SUBAGENT_RUNTIME and "cloud agent" in SUBAGENT_RUNTIME, "subagent prompt")
    check(
        "read_file" in SUBAGENT_RUNTIME and "shell" in SUBAGENT_RUNTIME and "hands_on" in SUBAGENT_RUNTIME,
        "subagent prompt names its tools",
    )
    xai_tools = {t["function"]["name"] for t in schemas_for("xai")}
    meta_tools = {t["function"]["name"] for t in schemas_for("meta")}
    local_tools = {t["function"]["name"] for t in schemas_for("ollama")}
    check(
        "ask_local" in xai_tools and "ask_local" in meta_tools and "ask_local" not in local_tools,
        "ask_local for cloud providers",
    )
    check("hands_on" in xai_tools and "hands_on" in meta_tools and "hands_on" in local_tools, "hands_on on every provider")
    sub_tools = {t["function"]["name"] for t in subagent_schemas()}
    check(sub_tools == local_tools and "ask_local" not in sub_tools, "subagent gets the local tools")
    check("hands_on" in grok and "hands_on" in muse and "hands_on" in local, "prompts mention hands_on")
    converted = to_openai_messages(
        [{"role": "user", "content": "look", "attachments": [att]}]
    )
    check(isinstance(converted[0]["content"], list), "xai image parts")
    image_url = converted[0]["content"][1]["image_url"]["url"]
    check(image_url.startswith("data:image/png;base64,"), "png screenshot is labeled png")
    plain = to_openai_messages([{"role": "user", "content": "hi"}])
    check(plain[0]["content"] == "hi", "text stays a string")

    from src.hands import (
        HANDS_SYS,
        Control,
        Shot,
        badge_sentence,
        choose_window,
        clean_plan,
        describe_look,
        encode_rgb_png,
        end_button_place,
        end_click_point,
        attempt_points,
        confirm_opened,
        controls_with_icons,
        current_step,
        desktop_file_path,
        editor_match,
        input_control,
        is_edit_chord,
        is_global_chord,
        looks_like_file,
        nearest_labeled,
        needs_editor,
        prepare_action,
        repeats_held,
        shift_point,
        step_is_describe,
        step_kind,
        non_end_steps,
        edits_document,
        _is_new_tab,
        file_contains,
        step_wants_content,
        action_matches_step,
        executable_plan,
        prompt_has_action,
        required_typing,
        typing_landed,
        fallback_plan,
        find_icon,
        find_shell_desktop,
        goal_phrase,
        matched_title,
        focus_text,
        format_controls,
        hands_brief,
        hands_messages,
        hits_end_button,
        is_desktop_query,
        image_box,
        image_to_screen,
        infer_window_query,
        is_look_only,
        parse_action,
        parse_keys,
        perform,
        plan_step_count,
        reply_aside,
        resolve_control,
        stamp_end_button,
        window_argument,
        closed_window_note,
        _locate,
        _shot_text,
    )
    from src.hands import DesktopIcon, HandsContext, lock_shell, read_desktop_icons

    goal = "Open Notepad and type hello — do not rephrase this."
    a = hands_messages(goal, "Screenshot 10x10 pixels.", {"path": "x.png"})
    b = hands_messages(goal, "Screenshot 20x20 pixels.\nRecent actions:\n1. click", None)
    check(a[0]["content"] == b[0]["content"] == HANDS_SYS, "hands system prompt stays fixed")
    check(a[1]["content"] == b[1]["content"] == goal, "hands task text stays verbatim")
    check(a[2]["content"].startswith(goal + "\n\n"), "screenshot turn repeats the task verbatim")
    check(b[2]["content"].startswith(goal + "\n\n"), "later turns keep the same task text")
    check("attachments" in a[2] and "attachments" not in b[2], "screenshot rides on the second user message")
    act = parse_action('sure\n```json\n{"action":"click","x":12,"y":4}\n```')
    check(act["action"] == "click" and act["x"] == 12, "parse click json")
    act = parse_action('[{"action":"done","text":"ok"},{"action":"click"}]')
    check(act["action"] == "done", "parse takes the first action only")
    check(parse_keys("ctrl+shift+t") == ["ctrl", "shift", "t"], "key chord split")
    sx, sy = image_to_screen(50, 25, vx=-1920, vy=0, vw=1920, vh=1080, iw=100, ih=50)
    check((sx, sy) == (-960, 540), "screenshot pixel maps onto the virtual desktop")
    sx, sy = image_to_screen(-5, -5, vx=0, vy=0, vw=100, vh=80, iw=10, ih=8)
    check((sx, sy) == (0, 0), "clicks clamp inside the desktop")
    tiny = encode_rgb_png(1, 1, bytes([255, 0, 0]))
    check(tiny.startswith(b"\x89PNG\r\n\x1a\n"), "png encoder")
    denied = run_tool("hands_on", {}, tmp)
    check("prompt" in denied.lower(), "hands_on requires the task text")
    asked = "hey hows it going? could you use your hands on tool and tell me what you see?"
    check(is_look_only(asked), "what you see is a look")
    check(not is_look_only("open Notepad and type hello"), "opening an app is not a look")
    prose = 'The desktop shows a browser and a terminal.\n{"action":"click","x":4,"y":5}'
    check("browser" in reply_aside(prose), "words around a click are kept")
    seen = describe_look('{"action":"done","text":"a browser and a terminal"}', ["Notepad"])
    check(seen is not None and "browser" in seen[0] and "Notepad" in seen[1], "done text is the description")
    check(describe_look('{"action":"click","x":1,"y":2}', ["Notepad"]) is None, "a bare click is not a description")
    check(describe_look('{"action":"done","text":"what is visible"}', ["Notepad"]) is None, "a placeholder is not a description")
    check(describe_look('{"action":"done","text":"what you see"}', ["Notepad"]) is None, "what you see is not a description")
    brief = hands_brief(
        [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "make a new .txt titled hello world"},
            {"role": "tool", "content": "noise"},
            {"role": "assistant", "content": "I will use the screen."},
            {"role": "user", "content": "use hands on mode only to reach the desktop and create the .txt file"},
        ]
    )
    check("hello world" in brief and "desktop" in brief and "noise" not in brief, "plan reads the chat, not tool output")
    planned = clean_plan(
        "<think>secret</think>\n1. Press Win+D.\n2. Right-click the desktop.\n"
        + "\n".join(f"{i}. step {i}" for i in range(3, 12))
    )
    check("secret" not in planned and planned.startswith("1. Press Win+D."), "plan drops thinking")
    check(len(planned.splitlines()) == 8, "plan stays at eight steps")
    seen_plan = fallback_plan("try again now, tell me what you see please")
    check("Describe" in seen_plan and "end tool rounds" in seen_plan, "a look gets a describe plan")
    file_plan = fallback_plan("get to my desktop, then make a new .txt titled hello world")
    check("Win+D" in file_plan and "only after" in file_plan, "a desktop change does not end first")
    check(plan_step_count(file_plan) == 3, "fallback plan has three steps")
    check(plan_step_count("just a sentence") == 0, "an unnumbered plan has no steps")
    frozen = focus_text(file_plan, "get to my desktop, then make a new .txt titled hello world")
    again = hands_messages(frozen, "Screenshot 10x10.", None)
    later = hands_messages(frozen, "Screenshot 20x20.", None)
    check(again[1]["content"] == later[1]["content"] == frozen, "the plan stays fixed on later turns")
    check(later[2]["content"].startswith(frozen + "\n\n"), "every screenshot repeats the plan")

    windows = [
        (1, "Untitled - Notepad"),
        (2, "Document - Word"),
        (3, "Videos - File Explorer"),
    ]
    check(choose_window(windows, "Notepad")[0] == 1, "choose notepad")
    check(choose_window(windows, "word", foreground=2)[0] == 2, "prefer the foreground match")
    check(
        infer_window_query("open Notepad and type hi", windows) == "Notepad",
        "infer the window the user named",
    )
    check(infer_window_query("edit the file", windows) == "", "generic words do not pick a window")
    check(
        is_desktop_query("desktop") and is_desktop_query("Shell") and is_desktop_query("program manager"),
        "desktop and shell are window targets",
    )
    check(
        not is_desktop_query("Notepad") and not is_desktop_query("Desktop - File Explorer"),
        "an app title is not the desktop shell",
    )
    check(window_argument("shell", "make a file") == "desktop", "shell is kept when the user did not say shell")
    check(window_argument("desktop", "get to my desktop") == "desktop", "desktop window param is kept")
    check(window_argument("Notepad", "get to my desktop") == "Notepad", "a passed window title is locked")
    check(window_argument("Notepad", "open Notepad") == "Notepad", "a named app window is kept")
    check(
        closed_window_note("Opera") == "Opera was not open, started from the desktop",
        "a closed app starts from the desktop",
    )
    dropped = executable_plan("1. Describe the screen.\n2. Click Notepad.", "click Notepad")
    check("Describe" not in dropped and "Notepad" in dropped, "describe steps are dropped")
    check(step_kind("Type function test.") == "type", "a type step stays a type step")
    mixed = executable_plan(
        "1. Click inside the text area, then type hello.\n2. Click the end tool rounds button.",
        "click inside the text area and type hello",
    )
    mixed_lines = [line for line in mixed.splitlines() if line[:1].isdigit()]
    check(len(mixed_lines) == 3, "a click and a type become two steps")
    check(step_kind(mixed_lines[0]) == "click", "the first verb is the click")
    check(step_kind(mixed_lines[1]) == "type", "the second verb is the type")
    check(step_kind(mixed_lines[2]) == "end", "the end step stays last")
    check(non_end_steps(mixed) == 2, "the end step is not a shortcut")
    check(
        step_kind("Click inside the text area, then type hello.") == "click",
        "the first verb wins",
    )
    check(step_kind("Double-click the file.") == "double_click", "a double-click is one verb")
    desktop_lock = HandsContext(shell=True, lock_hwnd=1, lock_title="Desktop")
    check(not edits_document(desktop_lock, "Type notepad"), "start search is not a document edit")
    check(not edits_document(desktop_lock, "Type the app name"), "an app name after Win is not a document edit")
    check(
        edits_document(HandsContext(shell=True, lock_hwnd=1, expect_file="notes.txt"), "Type hello"),
        "a file type still needs the editor",
    )
    check(
        edits_document(HandsContext(shell=False, lock_hwnd=1, lock_title="Notepad"), "Type hello"),
        "typing in an app window is a document edit",
    )
    check(
        edits_document(desktop_lock, "Type inside the document"),
        "a document step on the desktop still needs the editor",
    )
    browser_lock = HandsContext(shell=False, lock_hwnd=1, lock_title="Opera")
    check(not edits_document(browser_lock, "Type google.com"), "a url in a browser is not a document edit")
    check(edits_document(browser_lock, "Type hello"), "plain typing in an app window stays a document edit")
    check(
        edits_document(HandsContext(shell=False, lock_hwnd=1, expect_file="notes.txt"), "Type google.com"),
        "a file type stays a document edit",
    )
    check(_is_new_tab("ctrl+t") and _is_new_tab("control+t"), "ctrl+t is a new tab")
    check(not _is_new_tab("ctrl+s") and not _is_new_tab("ctrl+w") and not _is_new_tab("ctrl+shift+t"), "other chords are not a new tab")
    check(
        not action_matches_step({"action": "click", "n": 1}, "Type function test."),
        "a click does not satisfy a type step",
    )
    check(
        step_wants_content("Click inside the document text area."),
        "a document step wants the content area",
    )
    check(
        required_typing("type the text function test on a new line", "") == "function test",
        "required typing stops before the next clause",
    )
    check(
        required_typing('type "tool test line"', "") == "tool test line",
        "quoted typing is kept whole",
    )
    check(typing_landed("function test", "function test"), "typed text can match")
    check(not typing_landed("function test", ""), "missing text is not typed")
    check(
        prompt_has_action("click inside its text area and type function test"),
        "an action prompt is kept",
    )
    check(not prompt_has_action("tell me what it needs"), "a vague prompt is not an action")
    check(window_argument("", "make a file") == "", "an omitted window stays empty for desktop fallback")
    controls = [
        Control(1, "Button", "Save", "saveBtn", 10, 10, 40, 20),
        Control(2, "Button", "Save as", "saveAs", 60, 10, 40, 20),
        Control(3, "Edit", "Filename", "name", 10, 40, 80, 20),
    ]
    check(resolve_control({"n": 3}, controls).auto_id == "name", "control by number")
    check(resolve_control({"target": "Filename"}, controls).index == 3, "control by name")
    check(resolve_control({"target": "Save"}, controls).index == 1, "exact control name wins")
    check(resolve_control({"target": "saveBtn"}, controls).index == 1, "control by automation id")
    try:
        resolve_control({"target": "Sav"}, controls)
        raise SystemExit("FAIL: ambiguous save should fail")
    except ValueError as e:
        check("several" in str(e), "ambiguous control names")
    fake = Shot({}, 0, 0, 200, 100, 200, 100)
    check('1. Button "Save" at (30,20)' in format_controls(controls, fake), "control list uses screenshot pixels")
    try:
        perform({"action": "click", "x": 1, "y": 1}, fake)
        raise SystemExit("FAIL: click before lock")
    except ValueError as e:
        check("lock" in str(e), "click before lock is refused")
    screen = Shot({}, 0, 0, 1000, 800, 1000, 800)
    shell_state = HandsContext(lock_hwnd=1, lock_title="Desktop", shell=True, query="desktop")
    _ix, _iy, sx, sy, pointed = _locate({"x": 10, "y": 12}, screen, shell_state, [])
    check(pointed is None and (sx, sy) == (10, 12), "desktop shell allows a click without an app window")
    desktop_text = _shot_text(
        1000, 800, [], lock_title="Desktop", controls_text="", shell=True
    )
    check(
        "Locked to the desktop" in desktop_text and "whole screen" in desktop_text,
        "desktop lock tells the model it can click",
    )
    found = find_shell_desktop()
    check(found is not None and found[0] > 0 and found[1] == "Desktop", "desktop shell window exists")
    live_shell = HandsContext()
    check(
        lock_shell(live_shell) and live_shell.shell and live_shell.lock_title == "Desktop",
        "locking the shell records the desktop",
    )
    line, finished = perform({"action": "lock", "title": "shell"}, fake, state=HandsContext())
    check((not finished) and line == "locked to Desktop", "lock shell targets the desktop")
    asked = "navigate to my desktop. open the recycling bin. then end the tool"
    check(goal_phrase(asked) == "recycling bin", "open request names the recycling bin")
    check(matched_title("recycling bin", ["Recycle Bin"]) == "Recycle Bin", "recycle and recycling are the same window")
    check(matched_title("recycling bin", ["Notepad"]) == "", "a different window is not the goal")
    bin_icon = DesktopIcon("Recycle Bin", 80, 40, 40, 40, 70, 80, 60, 20)
    points = attempt_points(bin_icon, 57, 53, 96)
    check(points[0] == (100, 60), "the icon center is the first click")
    check(points[1] == (100, 90), "the label is the next click")
    check((57, 53) not in points, "a guessed corner point is not reused")
    scaled = attempt_points(bin_icon, 57, 53, 144)
    check(scaled[0] == (100, 60) and (67, 40) in scaled, "a miss also tries the other DPI scale")
    miss = HandsContext(shell=True, retry_name="recycle bin", retry_index=1)
    next_idx = miss.retry_index if miss.retry_name == "recycle bin" else 0
    check(next_idx == 1 and scaled[next_idx] != scaled[0], "a failed open uses the next point")
    miss.retry_index = next_idx + 1
    check(miss.retry_index == 2, "the failed open advances before the next try")
    check(find_icon([bin_icon], "recycling bin").name == "Recycle Bin", "icon text matches the request")
    replaced = controls_with_icons(
        [
            Control(1, "ListItem", "Recycle Bin", "", 10, 10, 20, 20),
            Control(2, "Button", "Save", "save", 1, 1, 4, 4),
        ],
        [bin_icon],
    )
    check(replaced[0].role == "Icon" and replaced[0].left == 80, "the real icon rect replaces the bad one")
    check(replaced[1].name == "Save" and replaced[1].index == 2, "other controls keep their place")
    seen = {"n": 0}

    def titles() -> list[str]:
        seen["n"] += 1
        return ["Recycle Bin"] if seen["n"] >= 2 else []

    check(confirm_opened("recycling bin", titles, tries=3, pause=0) == "Recycle Bin", "a new window confirms the open")
    live_icons = read_desktop_icons()
    check(isinstance(live_icons, list), "desktop icons can be read")
    check(all(item.icon_width > 0 and item.icon_height > 0 for item in live_icons), "each desktop icon has a size")
    labeled = Control(1, "Button", "Save", "", 10, 10, 40, 20)
    check(nearest_labeled([labeled], 36, 24) is labeled, "a nearby point snaps to the labeled center")
    check(nearest_labeled([labeled], 400, 400) is None, "a far point is not a labeled control")
    check(shift_point(100, 60, 0, 96) == (100, 60), "the first try is the center")
    check(shift_point(100, 60, 1, 96) != (100, 60), "the next try moves")
    snapped = _locate(
        {"x": 24, "y": 14},
        screen,
        HandsContext(lock_hwnd=1, lock_title="Desktop", shell=True),
        [labeled],
    )
    check(snapped[2:4] == (30, 20) and snapped[4] is labeled, "a raw point near a label clicks its center")
    numbered = "1. Press Win+D.\n2. Describe what you see.\n3. Click the end tool rounds button."
    check("Win+D" in current_step(numbered, 0), "the first unfinished step is next")
    check(step_is_describe(current_step(numbered, 1)), "a describe step may be narration")
    check(not step_is_describe(current_step(numbered, 0)), "an input step is not narration")
    gated = "1. Click the icon.\n2. Type hello.\n3. Click the end tool rounds button."
    check(step_kind(current_step(gated, 0)) == "click", "the open step comes before the end button")
    check(step_kind(current_step(gated, 1)) == "type", "typing stays ahead of the end button")
    check(step_kind(current_step(gated, 2)) == "end", "the end button is last")
    check(non_end_steps(gated) == 2, "both real steps are required")
    title_bar = Control(1, "TitleBar", "Notepad", "", 0, 0, 500, 30)
    editor = Control(2, "Edit", "", "", 8, 80, 400, 200)
    check(input_control([title_bar, editor]) is editor, "typing targets the edit control")
    rewritten = prepare_action(
        {"action": "click", "n": 1},
        [title_bar, editor],
        "click the terminal prompt",
    )
    check(rewritten.get("n") == 2 and "x" not in rewritten, "a title click on a prompt step hits the input")
    check(
        editor_match("hello_world.txt", ["hello_world.txt - Notepad"]) == "hello_world.txt - Notepad",
        "a text file matches its editor title",
    )
    check(editor_match("hello_world.txt", ["Recycle Bin"]) == "", "another window is not the file")
    check(looks_like_file("hello_world.txt") and not looks_like_file("Recycle Bin"), "text files are editor targets")
    check(is_global_chord("win") and is_global_chord("win+d") and is_global_chord("alt+f4"), "window hotkeys stay global")
    check(is_edit_chord("ctrl+a") and is_edit_chord("ctrl+s") and not is_edit_chord("alt+f4"), "save and select need the editor")
    check(not is_edit_chord("ctrl+t"), "a new tab is not a document chord")
    check(needs_editor({"action": "type", "text": "tool test line"}), "typing needs the editor")
    check(not needs_editor({"action": "key", "keys": "win"}), "the win key does not need the editor")
    check(not repeats_held("type", {"action": "key", "keys": "ctrl+s"}), "a later chord waits for the unconfirmed type")
    check(repeats_held("type", {"action": "type", "text": "again"}), "the unconfirmed type can be retried")
    check(desktop_file_path("no-such-hands-file-zz.txt") == "", "a missing desktop file is not opened")
    sample = tmp / "hands_file_check.txt"
    sample.write_text("function test\n", encoding="utf-8")
    check(file_contains(str(sample), "function test"), "file text is read from disk")
    check(not file_contains(str(sample), "missing line"), "a missing line is not in the file")
    sample.unlink()
    placed = end_button_place(0, 0, 1000, 800)
    check(placed == (376, 8, 248, 48), "end button sits at the top center")
    button = (placed[0], placed[1], placed[0] + placed[2], placed[1] + placed[3])
    box = image_box(screen, button)
    check(box == (376, 8, 248, 48), "end button maps onto screenshot pixels")
    check(image_box(screen, (2000, 2000, 2100, 2060)) is None, "an off-screen button is not stamped")
    center = end_click_point(
        {"action": "click", "x": box[0] + box[2] // 2, "y": box[1] + box[3] // 2},
        screen,
        button,
    )
    check(
        center is not None and hits_end_button(center[2], center[3], button),
        "a click on the end button counts",
    )
    check(
        end_click_point({"action": "click", "target": "end tool rounds"}, screen, button) is not None,
        "naming the end button counts",
    )
    check(end_click_point({"action": "click", "x": 1, "y": 1}, screen, button) is None, "a distant click is not the end button")
    try:
        perform({"action": "click", "x": 1, "y": 1}, screen, button_rect=button)
        raise SystemExit("FAIL: distant click should still need a lock")
    except ValueError as e:
        check("lock" in str(e), "only the end button is clickable before lock")
    check("end tool rounds" in HANDS_SYS, "system prompt names the end button")
    look = _shot_text(1000, 800, [], lock_title="", controls_text="", look_only=True, badge=badge_sentence(box))
    check("end tool rounds" in look and "Do not click anything else" in look, "a look may click only the end button")
    check("Center (500,32)" in look, "the look is told the end button center")

    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    wide = Shot({}, 0, 0, 320, 80, 320, 80)
    spot = end_button_place(0, 0, 320, 80)
    spot_rect = (spot[0], spot[1], spot[0] + spot[2], spot[1] + spot[3])
    raw = encode_rgb_png(320, 80, bytes([180, 180, 180]) * (320 * 80))
    fd, png_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd)
        Path(png_path).write_bytes(raw)
        painted = stamp_end_button(png_path, wide, spot_rect)
        check(painted == image_box(wide, spot_rect), "stamp uses the screenshot box")
        img = QImage(png_path)
        check(not img.isNull() and img.width() == 320, "stamped screenshot reloads")
        corner = img.pixelColor(0, 0)
        check(abs(corner.red() - 180) < 8, "stamp leaves the rest of the screenshot alone")
        whites = 0
        x, y, w, h = painted
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                color = img.pixelColor(xx, yy)
                if color.red() > 220 and color.green() > 220 and color.blue() > 220:
                    whites += 1
        check(whites > 20, "end button label is painted into the screenshot")
    finally:
        Path(png_path).unlink(missing_ok=True)

    from PyQt6.QtWidgets import QPlainTextEdit
    from src.agent import MAX_ROUNDS, NORMAL_CAP, ROUND_CAP, round_budget
    from src.ui import (
        ToolFeed,
        append_tool_block,
        format_tool_log,
        scroll_tool_log_to,
        tool_entry_label,
        tool_feed_html,
    )
    import src.ui as ui_mod

    check(not hasattr(ui_mod, "AUTO_CONTINUE"), "auto mode does not inject a prompt")
    check(round_budget(MAX_ROUNDS) == MAX_ROUNDS, "normal round budget")
    check(round_budget(100) == NORMAL_CAP, "normal round budget stays capped")
    check(round_budget(MAX_ROUNDS, autorun=True) == MAX_ROUNDS * 2, "auto mode doubles rounds")
    check(round_budget(NORMAL_CAP, autorun=True) == ROUND_CAP, "auto round budget doubles up to the cap")
    check(tool_entry_label("grep", result=False) == "tool grep", "tool label")
    check(tool_entry_label("hands_on", result=False) == "tool hands on", "hands on label")
    check(tool_entry_label("grep", result=True) == "grep result", "result label")
    feed = tool_feed_html([(0, "tool grep"), (1, "grep result"), (2, "tool hands on")])
    check(
        'href="mark:0"' in feed and 'href="mark:1"' in feed and 'href="mark:2"' in feed,
        "tool feed links",
    )
    check(feed.count(", ") == 2 and "tool grep" in feed and "hands on" in feed, "tool feed is one comma-separated line")
    edit = QPlainTextEdit()
    edit.setReadOnly(True)
    edit.resize(240, 80)
    first = append_tool_block(edit, format_tool_log("grep", '{"pattern": "a"}', result=False))
    second = append_tool_block(edit, format_tool_log("hands_on", "open opera", result=False))
    plain = edit.toPlainText()
    check("\n\n" in plain and plain.index("→ hands_on") > plain.index("→ grep"), "tool log separates entries")
    check(plain.split("→ grep", 1)[1].startswith('\n{"pattern": "a"}'), "tool log breaks the entry")
    scroll_tool_log_to(edit, first)
    check("grep" in edit.textCursor().selectedText(), "tool link selects the log entry")
    scroll_tool_log_to(edit, second)
    check("hands_on" in edit.textCursor().selectedText(), "tool link moves to the later entry")
    feed_widget = ToolFeed(lambda _href: None)
    feed_widget.resize(420, 20)
    feed_widget.set_links(tool_feed_html([(0, "tool grep"), (1, "grep result")]))
    check(
        feed_widget.height() >= feed_widget.fontMetrics().lineSpacing(),
        "tool bubble has room for the link line",
    )
    feed_widget.deleteLater()
    edit.deleteLater()

    from src.config import Settings
    from src.reasoning import (
        EFFORTS,
        ReasonTrack,
        ThinkSplit,
        chat_predict,
        cloud_attempts,
        cloud_effort,
        display_reasoning,
        hands_predict,
        model_options,
        ollama_think,
    )
    from src.ui import ReasoningCard

    check(Settings().reasoning_effort == "off", "reasoning setting defaults off")
    check(tuple(EFFORTS) == ("off", "low", "medium", "high"), "reasoning levels")
    check(cloud_effort("xai", "off") == "none", "grok off tries none")
    check(cloud_attempts("xai", "off") == ["none", "low", None], "grok off can fall back")
    check(cloud_effort("meta", "off") == "minimal", "muse off is minimal")
    check(cloud_attempts("meta", "high") == ["high", None], "muse high passes through")
    check(ollama_think("off") is False, "local off does not think")
    check(ollama_think("medium") == "medium", "local medium thinks")
    check("reasoning" not in model_options({"num_predict": 8, "reasoning": "high"}), "local options omit reasoning")
    check(hands_predict("meta", "off", look=False) == 680, "muse action keeps room for reasoning")
    check(hands_predict("xai", "off", look=False) == 280, "grok action budget stays")
    check(chat_predict("meta", "high", 131072) == 2624, "muse chat grows with high reasoning")
    check(display_reasoning("meta", "off", "") is None, "off shows nothing")
    check(display_reasoning("meta", "low", "") == "output restricted", "hidden muse reasoning is marked")
    check(display_reasoning("xai", "high", "") is None, "empty grok reasoning makes no line")
    check(display_reasoning("xai", "high", "", hidden=True) == "output restricted", "hidden grok reasoning is marked")
    check(display_reasoning("ollama", "low", "") is None, "empty local reasoning makes no line")
    check(display_reasoning("meta", "medium", "because") == "because", "visible reasoning is kept")
    events: list[tuple] = []
    track = ReasonTrack(lambda text, anchor, call, label: events.append((text, anchor, call, label)), "a")
    track.start("REASONING")
    track.finish("meta", "off", "")
    check(events == [], "off reasoning makes no line")
    track.start("REASONING")
    track.finish("meta", "low", "", hidden=True)
    check(len(events) == 1 and events[0][0] == "output restricted", "restricted line is one card")
    track.start("REASONING")
    track.finish("xai", "high", "")
    check(len(events) == 1, "empty grok finish adds no card")
    track.start("REASONING")
    track.add("hello", provider="xai", effort="high")
    track.finish("xai", "high", "hello")
    check(events[-1][0] == "hello" and events[-1][2] == events[-2][2], "reasoning text stays on one card")
    splitter = ThinkSplit()
    c1, r1 = splitter.feed("Say ")
    c2, r2 = splitter.feed("<thi")
    c3, r3 = splitter.feed("nk>secret</think> done")
    check(c1 == "Say " and c2 == "" and r2 == "", "partial think tag is held")
    check(r3 == "secret" and c3 == " done" and splitter.saw_think, "think split keeps the answer")
    jumped: list[int] = []
    card = ReasoningCard("REASONING", lambda: jumped.append(1))
    check(card._body_box.isHidden(), "reasoning starts collapsed")
    check(card._header.text().startswith("▸"), "reasoning header is collapsed")
    card._header.click()
    check(not card._body_box.isHidden(), "reasoning expands")
    check("message" in card._link.text(), "reasoning links to the related message")
    card._link.linkActivated.emit("message")
    check(jumped == [1], "reasoning link jumps to the message")
    card.set_text("output restricted")
    check(card._body.text() == "output restricted", "restricted reasoning line")
    check(card._title == "reasoning" and card._header.text().endswith("reasoning"), "reasoning header is lowercase")
    check("font-size: 11px" in card._body.styleSheet(), "reasoning text fits the pane")
    card.deleteLater()
    del app

    from src.session import choose_session_id
    from src.theme import APP_VERSION

    check(choose_session_id("b", ["a", "b"]) == "b", "launch keeps the last thread")
    check(choose_session_id("gone", ["a", "b"]) == "a", "launch falls back to the newest thread")
    check(choose_session_id(None, []) is None, "launch creates a thread only when none exist")
    check(APP_VERSION == (ROOT / "VERSION").read_text(encoding="utf-8").strip(), "window title matches VERSION")
    ui_src = (ROOT / "src" / "ui.py").read_text(encoding="utf-8")
    check("resume_session(settings.model)" in ui_src, "startup opens an existing thread")
    check("last.unlink" not in ui_src, "startup leaves the saved thread in place")

    print("all smoke checks passed")


if __name__ == "__main__":
    main()
