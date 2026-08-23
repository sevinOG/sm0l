# sm0l

Tiny local coding + research agent. Windows dashboard, no heartbeat, user prompts only.

Short soul, eight tools, autocompaction against the selected model's native context window.


## What it is

A PyQt6 dashboard (Eche-purple) that talks to a local Ollama daemon:

- Chat, sessions, workspace picker
- Model list / pull from the UI (no CLI)
- DuckDuckGo search + page fetch
- Files + shell for coding on this machine
- Autocompact using the **same** model when the window fills

It will not ping you, cron itself, or invent a turn. You send a message or nothing happens.

## Requirements

- Windows 10/11
- Python 3.11+ (to build or run from source)
- [Ollama](https://ollama.com) running locally
- A local chat model

## Run the EXE

After a build:
(run build.bat first)
```
dist\sm0l\sm0l.exe
```

Or double-click `RUN.bat`.

1. Start Ollama.
2. In the right panel, Pull `qwen2.5:7b` (or Refresh if you already have a model).
3. Set workspace if you want a project folder.
4. Type. Enter sends.

Config and sessions live in `%LOCALAPPDATA%\sm0l\`. Workspace defaults to `%USERPROFILE%\sm0l_workspace` (OPERATIONS.md is seeded there — identity prefs + workspace rules in one file).

## Build

```
BUILD.bat
```

That creates `.venv`, generates `assets/icon.ico`, and writes `dist\sm0l\sm0l.exe`.

From source without freezing:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python sm0l.py
```

## Design

| Constraint | How sm0l handles it |
|---|---|
| Small context | Detect native `context_length` via Ollama `/api/show`, compact at ~62% |
| Weak long-prompt following | RUNTIME + single OPERATIONS.md digest stay tiny |
| Tool-call drift | 8 tools, 8 round cap, XML fallback if native tools fail |
| Huge tool dumps | Hard clip on search/fetch/read/shell |
| No background noise | No heartbeat, no MEMORY.md spam |

## Tools

`search` `fetch` `read_file` `write_file` `edit_file` `list_dir` `grep` `shell`

Search uses the free DuckDuckGo instant-answer API plus HTML results. No API key.

## License

MIT.
