"""Persistent dashboard settings."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import config_path, default_workspace


@dataclass
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    model: str = ""
    num_ctx: int = 0  # 0 = auto from model native window
    temperature: float = 0.2
    workspace: str = ""
    compact_ratio: float = 0.62
    max_tool_rounds: int = 8
    shell_timeout: int = 60
    autorun: bool = False

    def resolved_workspace(self) -> Path:
        raw = (self.workspace or "").strip()
        p = Path(raw) if raw else default_workspace()
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_settings() -> Settings:
    path = config_path()
    if not path.is_file():
        s = Settings(workspace=str(default_workspace()))
        save_settings(s)
        return s
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in Settings.__dataclass_fields__.values()}
        return Settings(**{k: v for k, v in data.items() if k in known})
    except Exception:
        return Settings(workspace=str(default_workspace()))


def save_settings(s: Settings) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
