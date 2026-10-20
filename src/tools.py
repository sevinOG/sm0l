"""Small tool set for 8B models: web + files + shell. Results are hard-truncated."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
import html
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

MAX = {
    "search": 3500,
    "fetch": 6000,
    "read_file": 8000,
    "list_dir": 3000,
    "grep": 4000,
    "shell": 4000,
    "ask_local": 6000,
    "hands_on": 4000,
    "generic": 4000,
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web with DuckDuckGo. Use for research, docs, errors, facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch a URL and return visible text. Use after search to read a page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file. Optional 1-based start line and line limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "1-based start line"},
                    "limit": {"type": "integer", "description": "max lines"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 file (creates parent folders). Overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact old_text with new_text once. old_text must be unique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders. Relative paths are from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex. Skips .git, node_modules, __pycache__.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "file or directory"},
                    "glob": {"type": "string", "description": "optional filename glob like *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_local",
            "description": (
                "Spawn an offline local GPU subagent (Ollama) from the active cloud agent via sm0l. "
                "The subagent has search, fetch, files, grep, shell, and hands_on, and runs them on this machine. "
                "It does not call the cloud and cannot spawn another subagent. "
                "Use for private or heavy local work. Pass a clear task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What the local subagent should do"},
                    "model": {"type": "string", "description": "Optional Ollama model name. Empty uses the harness default."},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the workspace. Use for git, python, builds, tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hands_on",
            "description": (
                "Drive the Windows desktop from screenshots. Grok looks at the screen and clicks, types, and presses keys. "
                "Use only when the user asked to operate the screen. Do not use the shell for that same request. "
                "Put their request in prompt. A plan is written from the recent chat and followed on every screenshot. "
                "Pass window as desktop or shell to drive the desktop. "
                "Pass an app title only when the user named that app. "
                "If window is omitted and no app was named, the desktop is used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user's request. A fixed plan is written from the recent chat before any click.",
                    },
                    "window": {
                        "type": "string",
                        "description": "desktop or shell for the desktop. Otherwise an app title copied from the user's words, such as Notepad.",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "How many screenshot turns. Default 12, max 24.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def clip(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n].rstrip() + f"\n… [truncated {len(text) - n} chars]"


def resolve_path(workspace: Path, path: str | None) -> Path:
    raw = (path or ".").strip() or "."
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = workspace / p
    return p


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


def _http_get(url: str, timeout: float = 18, user_agent: str | None = None) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent or USER_AGENT,
            "Accept": "text/html,application/json,text/plain,/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(800_000)
        charset = "utf-8"
        if "charset=" in ctype:
            charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        try:
            body = raw.decode(charset, errors="surrogateescape")
        except LookupError:
            body = raw.decode("utf-8", errors="surrogateescape")
        return resp.geturl(), body


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        t = data.strip()
        if t:
            self.parts.append(t + " ")


def html_to_text(doc: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(doc)
        p.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", doc)
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tool_search(query: str, **_k: Any) -> str:
    query = (query or "").strip()
    if not query:
        return "ERROR: empty query"
    chunks: list[str] = []
    q = urllib.parse.quote_plus(query)
    try:
        _, body = _http_get(
            f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1&no_redirect=1&t=sm0l"
        )
        data = json.loads(body)
        heading = data.get("Heading") or ""
        abstract = data.get("AbstractText") or data.get("Abstract") or ""
        answer = data.get("Answer") or ""
        source = data.get("AbstractURL") or ""
        if heading or abstract or answer:
            chunks.append("Instant answer:")
            if heading:
                chunks.append(heading)
            if answer:
                chunks.append(answer)
            if abstract:
                chunks.append(abstract)
            if source:
                chunks.append(f"Source: {source}")
        related = []
        for item in data.get("RelatedTopics") or []:
            if isinstance(item, dict) and item.get("Text"):
                related.append(f"- {item.get('Text')} {item.get('FirstURL') or ''}".strip())
            elif isinstance(item, dict):
                for sub in item.get("Topics") or []:
                    if sub.get("Text"):
                        related.append(f"- {sub.get('Text')}")
            if len(related) >= 6:
                break
        if related:
            chunks.append("Related:")
            chunks.extend(related[:6])
    except Exception as e:
        chunks.append(f"(instant answer skipped: {e})")

    try:
        _, body = _http_get(
            f"https://html.duckduckgo.com/html/?q={q}",
            user_agent=BROWSER_UA,
        )
        results = _parse_ddg_html(body)
        if not results:
            _, body = _http_get(
                f"https://lite.duckduckgo.com/lite/?q={q}",
                user_agent=BROWSER_UA,
            )
            results = _parse_ddg_lite(body)
        if results:
            chunks.append("Results:")
            for i, r in enumerate(results[:6], 1):
                chunks.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    except Exception as e:
        chunks.append(f"(html search skipped: {e})")

    text = "\n".join(chunks).strip() or f"No DuckDuckGo results for: {query}"
    return clip(text, MAX["search"])


def _parse_ddg_html(doc: str) -> list[dict]:
    results = []
    # result__a + result__snippet
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        doc,
        re.I | re.S,
    ):
        url = html.unescape(m.group(1))
        if "uddg=" in url:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            url = (parsed.get("uddg") or [url])[0]
        title = re.sub(r"<[^>]+>", "", html.unescape(m.group(2))).strip()
        snippet = re.sub(r"<[^>]+>", "", html.unescape(m.group(3))).strip()
        if title and url.startswith("http"):
            results.append({"title": title, "url": url, "snippet": snippet})
    if results:
        return results
    for m in re.finditer(r'<a[^>]+class="[^"]*result-link[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', doc, re.I | re.S):
        url = html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", "", html.unescape(m.group(2))).strip()
        if title:
            results.append({"title": title, "url": url, "snippet": ""})
    return results


def _parse_ddg_lite(doc: str) -> list[dict]:
    results = []
    for m in re.finditer(
        r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        doc,
        re.I | re.S,
    ):
        url = html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", "", html.unescape(m.group(2))).strip()
        if title and url.startswith("http") and "duckduckgo.com" not in url:
            results.append({"title": title, "url": url, "snippet": ""})
    return results[:8]


def tool_fetch(url: str, **_k: Any) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    try:
        final, body = _http_get(url, timeout=20)
    except Exception as e:
        return f"ERROR fetching {url}: {e}"
    if "<html" in body[:400].lower() or "<!doctype" in body[:400].lower():
        text = html_to_text(body)
    else:
        text = body
    return clip(f"URL: {final}\n\n{text}", MAX["fetch"])


def tool_read_file(workspace: Path, path: str, offset: int | None = None, limit: int | None = None, **_k: Any) -> str:
    p = resolve_path(workspace, path)
    if not p.exists():
        return f"ERROR: not found: {p}"
    if p.is_dir():
        return f"ERROR: {p} is a directory — use list_dir"
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR reading {p}: {e}"
    lines = raw.splitlines()
    start = 1
    if offset and int(offset) > 1:
        start = int(offset)
    end = len(lines)
    if limit and int(limit) > 0:
        end = min(len(lines), start + int(limit) - 1)
    slice_ = lines[start - 1 : end]
    numbered = "\n".join(f"{i}|{line}" for i, line in enumerate(slice_, start))
    header = f"{p}  lines {start}-{start + len(slice_) - 1} of {len(lines)}\n"
    return clip(header + numbered, MAX["read_file"])


def tool_write_file(workspace: Path, path: str, content: str, **_k: Any) -> str:
    p = resolve_path(workspace, path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if content is not None else "", encoding="utf-8")
    except Exception as e:
        return f"ERROR writing {p}: {e}"
    n = len((content or "").splitlines())
    return f"Wrote {p} ({n} lines, {len(content or '')} chars)"


def tool_edit_file(workspace: Path, path: str, old_text: str, new_text: str, **_k: Any) -> str:
    p = resolve_path(workspace, path)
    if not p.is_file():
        return f"ERROR: not found: {p}"
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR reading {p}: {e}"
    count = raw.count(old_text)
    if count == 0:
        return f"ERROR: old_text not found in {p}"
    if count > 1:
        return f"ERROR: old_text found {count} times in {p} — make it unique"
    p.write_text(raw.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {p}"


def tool_list_dir(workspace: Path, path: str | None = None, **_k: Any) -> str:
    p = resolve_path(workspace, path or ".")
    if not p.exists():
        return f"ERROR: not found: {p}"
    if p.is_file():
        return f"FILE {p} ({p.stat().st_size} bytes)"
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    lines = [f"{p}/"]
    try:
        kids = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except Exception as e:
        return f"ERROR listing {p}: {e}"
    for child in kids[:200]:
        if child.name in skip:
            continue
        mark = "/" if child.is_dir() else ""
        lines.append(f"  {child.name}{mark}")
    if len(kids) > 200:
        lines.append(f"  … {len(kids) - 200} more")
    return clip("\n".join(lines), MAX["list_dir"])


_SKIP_DIR = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def tool_grep(workspace: Path, pattern: str, path: str | None = None, glob: str | None = None, **_k: Any) -> str:
    root = resolve_path(workspace, path or ".")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    hits: list[str] = []
    files: list[Path] = []
    if root.is_file():
        files = [root]
    elif root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR]
            for name in filenames:
                if glob and not Path(name).match(glob):
                    continue
                files.append(Path(dirpath) / name)
                if len(files) > 400:
                    break
            if len(files) > 400:
                break
    else:
        return f"ERROR: not found: {root}"
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{fp}:{i}:{line[:240]}")
                if len(hits) >= 80:
                    return clip("\n".join(hits) + "\n… more hits", MAX["grep"])
    return clip("\n".join(hits) if hits else f"No matches for {pattern!r}", MAX["grep"])


_DENY = re.compile(
    r"(rm\s+-rf\s+[\\/]|format\s+[a-z]:|del\s+/s\s+/q\s+[a-z]:|"
    r"shutdown\s+/s|rd\s+/s\s+/q\s+[a-z]:|mkfs\.|:\(\)\s*\{)",
    re.I,
)


def tool_shell(workspace: Path, command: str, timeout: int = 60, **_k: Any) -> str:
    command = (command or "").strip()
    if not command:
        return "ERROR: empty command"
    if _DENY.search(command):
        return "ERROR: blocked destructive command"
    is_wsl = "wsl" in command.lower() or "bash" in command.lower()
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(workspace),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the process - works for native Windows
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
        if is_wsl:
            return f"ERROR: timed out after {timeout}s (WSL session)"
        return f"ERROR: timed out after {timeout}s (stop button pressed)"
    except Exception as e:
        return f"ERROR: {e}"
    out = (stdout or "") + (("\n" + stderr) if stderr else "")
    # WSL commands may have empty stdout but stderr has output
    if is_wsl and not stdout and stderr:
        out = stderr
    return clip(f"exit {proc.returncode}\n{out.strip()}".strip(), MAX["shell"])


_LOCAL = {"host": "http://127.0.0.1:11434", "model": ""}


def set_local_runtime(host: str, model: str) -> None:
    """Default Ollama target for ask_local. Called by the agent each turn."""
    _LOCAL["host"] = (host or "http://127.0.0.1:11434").rstrip("/")
    _LOCAL["model"] = (model or "").strip()


_SUBAGENT_ROUNDS = 8


def subagent_schemas() -> list[dict]:
    """Tools a local subagent may call. ask_local is omitted so it cannot spawn another one."""
    return schemas_for("ollama")


def _local_turn(
    host: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    reasoning: str = "off",
    reason_track=None,
) -> dict:
    from .ollama_client import chat_once
    from .reasoning import finish_call, ollama_think

    options = {"temperature": 0.2, "reasoning": reasoning}
    think = ollama_think(reasoning)
    if reason_track is not None:
        reason_track.start("REASONING · local")
    try:
        result = chat_once(
            host, model, messages, tools=tools, options=options, think=think, timeout=180
        )
    except Exception as e:
        low = str(e).lower()
        if tools and ("tool" in low or "does not support" in low or "400" in low):
            result = chat_once(
                host, model, messages, tools=None, options=options, think=think, timeout=180
            )
        else:
            raise
    msg = result.get("message") if isinstance(result.get("message"), dict) else None
    if isinstance(msg, dict):
        finish_call(
            reason_track,
            "ollama",
            reasoning,
            msg,
            hidden=bool(result.get("reasoning_hidden")),
        )
    elif reason_track is not None:
        reason_track.finish("ollama", reasoning, "", hidden=bool(result.get("reasoning_hidden")))
    return result


def _subagent_calls(message: dict) -> tuple[str, list[dict]]:
    from .agent import TOOL_RE, parse_text_tool_calls

    content = str(message.get("content") or "")
    calls = list(message.get("tool_calls") or [])
    if calls:
        return content, calls
    calls = parse_text_tool_calls(content)
    if calls:
        content = TOOL_RE.sub("", content).strip()
    return content, calls


def tool_ask_local(
    workspace: Path,
    task: str,
    model: str = "",
    provider: str = "",
    host: str = "",
    shell_timeout: int = 60,
    should_cancel: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_window: Callable[[bool], None] | None = None,
    on_badge: Callable[..., None] | None = None,
    button_rect: Callable[[], Any] | None = None,
    end_requested: Callable[[], bool] | None = None,
    reasoning: str = "off",
    reason_track=None,
    **_k: Any,
) -> str:
    """Offline subagent with the local tool set. The prompt names who spawned it."""
    from .agent import _args
    from .ollama_client import list_models
    from .personality import SUBAGENT_RUNTIME

    task = (task or "").strip()
    if not task:
        return "ERROR: empty task"
    caller = "Muse" if (provider or _k.get("provider") or "") == "meta" else "Grok"
    host = (host or _LOCAL["host"] or "http://127.0.0.1:11434").rstrip("/")
    chosen = (model or _LOCAL["model"] or "").strip()
    if not chosen:
        try:
            names = [m["name"] for m in list_models(host)]
        except Exception as e:
            return f"ERROR: Ollama model list failed: {e}"
        if not names:
            return "ERROR: no local Ollama model. Start Ollama and pull a model."
        chosen = names[0]
    tools = subagent_schemas()
    messages: list[dict] = [
        {"role": "system", "content": SUBAGENT_RUNTIME.strip()},
        {
            "role": "user",
            "content": (
                f"{caller} (via sm0l) is asking you, a local GPU subagent, to do this:\n\n" + task
            ),
        },
    ]
    used: list[str] = []
    text = ""
    try:
        for _round in range(_SUBAGENT_ROUNDS):
            if should_cancel and should_cancel():
                return "cancelled"
            result = _local_turn(
                host, chosen, messages, tools, reasoning=reasoning, reason_track=reason_track
            )
            msg = result.get("message") or {}
            text, calls = _subagent_calls(msg)
            if not calls:
                break
            assistant: dict[str, Any] = {"role": "assistant", "content": text, "tool_calls": calls}
            messages.append(assistant)
            for tc in calls:
                if should_cancel and should_cancel():
                    return "cancelled"
                fn = tc.get("function") or tc
                name = str(fn.get("name") or "")
                arguments = _args(fn.get("arguments"))
                if name == "ask_local":
                    output = "ERROR: the local subagent cannot spawn another subagent"
                else:
                    if name == "hands_on":
                        from .hands import prompt_has_action, window_argument

                        arguments = dict(arguments)
                        proposed = str(arguments.get("prompt") or "").strip()
                        if not proposed or not prompt_has_action(proposed):
                            arguments["prompt"] = task
                        arguments["window"] = window_argument(
                            str(arguments.get("window") or ""),
                            f"{task}\n{arguments.get('prompt') or ''}",
                        )
                    if on_status:
                        on_status(f"local subagent · {name}")
                    output = run_tool(
                        name,
                        arguments,
                        workspace,
                        shell_timeout=shell_timeout,
                        host=host,
                        model=chosen,
                        provider="ollama",
                        api_key="",
                        num_ctx=0,
                        should_cancel=should_cancel,
                        on_status=on_status,
                        on_window=on_window,
                        on_badge=on_badge,
                        button_rect=button_rect,
                        end_requested=end_requested,
                        reasoning=reasoning,
                        reason_track=reason_track,
                    )
                used.append(name or "tool")
                messages.append(
                    {
                        "role": "tool",
                        "content": output,
                        "tool_name": name,
                        "name": name,
                    }
                )
        else:
            text = (text or "").strip()
            note = f"Stopped after {_SUBAGENT_ROUNDS} tool rounds."
            text = f"{note}\n{text}".strip()
    except Exception as e:
        return f"ERROR: local subagent failed: {e}"
    text = (text or "").strip()
    header = f"[local subagent {chosen} — spawned by {caller} via sm0l]"
    if used:
        header += "\ntools: " + ", ".join(used)
    if not text:
        if used:
            text = "The subagent finished the tool calls and did not add a summary."
        else:
            return f"ERROR: local subagent {chosen} returned empty"
    return clip(f"{header}\n{text}", MAX["ask_local"])


def schemas_for(provider: str) -> list[dict]:
    if provider in ("xai", "meta"):
        return SCHEMAS
    return [s for s in SCHEMAS if (s.get("function") or {}).get("name") != "ask_local"]


_HANDLERS: dict[str, Callable[..., str]] = {
    "search": lambda workspace, **k: tool_search(**k),
    "fetch": lambda workspace, **k: tool_fetch(**k),
    "read_file": lambda workspace, **k: tool_read_file(workspace, **k),
    "write_file": lambda workspace, **k: tool_write_file(workspace, **k),
    "edit_file": lambda workspace, **k: tool_edit_file(workspace, **k),
    "list_dir": lambda workspace, **k: tool_list_dir(workspace, **k),
    "grep": lambda workspace, **k: tool_grep(workspace, **k),
    "shell": lambda workspace, **k: tool_shell(workspace, **k),
    "ask_local": lambda workspace, **k: tool_ask_local(workspace, **k),
    "hands_on": lambda workspace, **k: _hands_on(workspace, **k),
}


def _hands_on(workspace: Path, **k: Any) -> str:
    from .hands import tool_hands_on

    return tool_hands_on(workspace, **k)


def run_tool(
    name: str,
    arguments: dict,
    workspace: Path,
    shell_timeout: int = 60,
    *,
    host: str = "",
    model: str = "",
    provider: str = "",
    api_key: str = "",
    num_ctx: int = 0,
    should_cancel: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_window: Callable[[bool], None] | None = None,
    on_badge: Callable[..., None] | None = None,
    button_rect: Callable[[], Any] | None = None,
    end_requested: Callable[[], bool] | None = None,
    reasoning: str = "off",
    reason_track=None,
) -> str:
    fn = _HANDLERS.get(name)
    if not fn:
        return f"ERROR: unknown tool {name}"
    args = dict(arguments or {})
    if name == "shell":
        args["timeout"] = shell_timeout
    if name == "ask_local":
        args["provider"] = provider
        args["host"] = host
        args["shell_timeout"] = shell_timeout
        args["should_cancel"] = should_cancel
        args["on_status"] = on_status
        args["on_window"] = on_window
        args["on_badge"] = on_badge
        args["button_rect"] = button_rect
        args["end_requested"] = end_requested
        args["reasoning"] = reasoning
        args["reason_track"] = reason_track
    if name == "hands_on":
        args["host"] = host
        args["model"] = model
        args["provider"] = provider or "xai"
        args["api_key"] = api_key
        args["num_ctx"] = num_ctx
        args["should_cancel"] = should_cancel
        args["on_status"] = on_status
        args["on_window"] = on_window
        args["on_badge"] = on_badge
        args["button_rect"] = button_rect
        args["end_requested"] = end_requested
        args["reasoning"] = reasoning
        args["reason_track"] = reason_track
    try:
        out = fn(workspace=workspace, **args)
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR in {name}: {e}"
    if name == "hands_on":
        return clip(out, MAX["hands_on"])
    return out
