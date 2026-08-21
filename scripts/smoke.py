"""Headless checks — no GUI, no Ollama required."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import parse_text_tool_calls
from src.compact import compact_threshold, estimate_tokens, split_for_compact
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
    print("all smoke checks passed")


if __name__ == "__main__":
    main()
