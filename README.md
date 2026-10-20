# sm0l

Windows desktop agent. Version 1.0.0. No heartbeat, user prompts only.

The composer switches among three providers:

- **xAI · Grok** — cloud chat, tools, and screen vision. The main agent when xAI is selected.
- **Meta · Muse** — the same cloud path on `https://api.meta.ai/v1`. Default model `muse-spark-1.3`.
- **Ollama · local GPU** — offline. With a cloud provider selected, Ollama is the `ask_local` subagent. With Ollama selected, the chat itself stays offline and does not need a cloud key.

Cloud keys are stored with Windows DPAPI. They are not written to config as plaintext.


## What it is

A PyQt6 dashboard that talks to the provider selected in the composer:

- Chat, sessions, workspace picker
- Grok, Muse, or a local model from the composer chip
- DuckDuckGo search + page fetch
- Files + shell for coding on this machine
- `hands_on` for the Windows desktop: screenshots, mouse, and keyboard, one confirmed step at a time
- Autocompact with the model selected in the composer

It will not ping you, cron itself, or invent a turn. You send a message or nothing happens.

## Requirements

- Windows 10/11
- Python 3.11+ (to build or run from source)
- An xAI key, a Meta key, or [Ollama](https://ollama.com) with a local chat model
- Keys are saved in Settings. Each one is encrypted with Windows DPAPI for this user.

## Run the EXE

After a build:
(run build.bat first)
```
dist\sm0l\sm0l.exe
```

Or double-click `RUN.bat`.

1. Pick Grok, Muse, or a local model from the chip under the composer.
2. For a cloud model, open Settings and save that provider's API key.
3. For a local model, start Ollama and pull one (or Refresh if you already have a model).
4. Set workspace if you want a project folder.
5. Type. Enter sends. Shift+Enter is a newline.
6. Paste, drop, or click **Image** to attach pictures to the message being composed (up to 4).

Config and sessions live in `%LOCALAPPDATA%\sm0l\`. A first launch copies an existing sm0k xAI key blob when sm0l does not have one yet. Workspace defaults to `%USERPROFILE%\sm0l_workspace` (OPERATIONS.md is seeded there — identity prefs + workspace rules in one file). Attached images are stored in `%LOCALAPPDATA%\sm0l\media\`.

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
| Cloud or offline | xAI and Meta use a 128k window unless Settings overrides it. Ollama uses the model's native window. |
| Compaction | The selected composer model writes the summary. Local mode does not call a cloud API. |
| Weak long-prompt following | RUNTIME + single OPERATIONS.md digest stay tiny |
| Tool-call drift | Tool round cap, XML fallback if native tools fail |
| Huge tool dumps | Hard clip on search/fetch/read/shell |
| No background noise | No heartbeat, no MEMORY.md spam |

## Tools

`search` `fetch` `read_file` `write_file` `edit_file` `list_dir` `grep` `shell` `ask_local` `hands_on`

`ask_local` is offered to Grok and Muse. It runs an Ollama model offline and names whichever cloud agent spawned it. Search uses the free DuckDuckGo instant-answer API plus HTML results. No API key.

## License

MIT.
