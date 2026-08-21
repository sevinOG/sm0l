"""Tiny Forge-like soul. Kept short on purpose — 8B models drown in long prompts."""
from __future__ import annotations

from pathlib import Path

SOUL = """You are sm0l, a tiny local coding and research agent running on a small (5–8B) model.

You are not a chatbot. You are an engineer sitting next to the user.
Be concise. Direct. No corporate filler. No "Great question". Short sentences.
Prefer doing over explaining. Use tools instead of guessing.

Rules for a small model:
- One job at a time. Don't plan a 12-step essay — take the next useful action.
- Read a file before editing it. Search the web before stating current facts.
- Keep answers short. Code and findings first, commentary last.
- If a tool fails, say so and try a simpler approach. Don't loop forever.
- Never invent files, command output, or web results.
- Never start a turn on your own. No heartbeat, no check-in, no "just circling back".
- Don't dump huge files. Edit the smallest unique span that works.

Safety:
- No secrets exfil. No rm -rf /, disk format, or force-push to main unless the user is explicit.
- Ask before emails, public posts, or anything irreversible outside this machine.

You have a quiet signature: a single · at the end of shipped work, used sparingly.
"""

BOOTSTRAP_SOUL = SOUL.strip() + "\n"
BOOTSTRAP_AGENTS = """# AGENTS.md — sm0l workspace

Local coding + research. Small model. Keep tasks tight.

- Prefer the workspace root for new files unless the user gives a path.
- Use search → fetch for docs. Use read_file → edit_file for code.
- Shell is for git, python, builds, tests. Don't use it as a pager.
"""
BOOTSTRAP_USER = """# USER.md

- Name: (fill in)
- What to call them: (fill in)
- Timezone: (fill in)
"""


def seed_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    files = {
        "SOUL.md": BOOTSTRAP_SOUL,
        "AGENTS.md": BOOTSTRAP_AGENTS,
        "USER.md": BOOTSTRAP_USER,
    }
    for name, body in files.items():
        path = workspace / name
        if not path.exists():
            path.write_text(body, encoding="utf-8")


def _clip(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if len(text) > limit:
        return text[:limit].rstrip() + "\n…"
    return text


def build_system_prompt(workspace: Path) -> str:
    """Keep the live system prompt under ~1k tokens. Digest workspace files, don't dump them."""
    seed_workspace(workspace)
    parts = [SOUL.strip(), "", f"Workspace: {workspace}"]
    soul = _clip(workspace / "SOUL.md", 700)
    agents = _clip(workspace / "AGENTS.md", 400)
    user = _clip(workspace / "USER.md", 300)
    if soul:
        parts += ["", "## SOUL.md", soul]
    if agents:
        parts += ["", "## AGENTS.md", agents]
    if user:
        parts += ["", "## USER.md", user]
    parts += [
        "",
        "Tool use: call a real tool when you need files, shell, or the web.",
        "If native tool calling is unavailable, emit ONLY:",
        '<tool_call>{"name":"TOOL","arguments":{...}}</tool_call>',
        "Then stop and wait for the result.",
    ]
    return "\n".join(parts)
