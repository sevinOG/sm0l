"""Persistent dashboard settings. Cloud keys are DPAPI ciphertext only."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .dpapi import protect, unprotect
from .paths import config_path, default_workspace
from .reasoning import normalize_effort

PROVIDERS = ("xai", "meta", "ollama")


@dataclass
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    model: str = ""  # local Ollama model
    xai_model: str = "grok-4"
    meta_model: str = "muse-spark-1.3"
    provider: str = "xai"  # "xai", "meta", or "ollama"
    xai_key_dpapi: str = ""  # Windows DPAPI blob, never plaintext
    meta_key_dpapi: str = ""  # Windows DPAPI blob, never plaintext
    num_ctx: int = 0  # 0 = auto from model native window
    temperature: float = 0.2
    workspace: str = ""
    compact_ratio: float = 0.62
    max_tool_rounds: int = 8
    shell_timeout: int = 60
    autorun: bool = False
    reasoning_effort: str = "off"  # off, low, medium, high

    def has_xai_key(self) -> bool:
        return bool((self.xai_key_dpapi or "").strip())

    def xai_key(self) -> str:
        return unprotect(self.xai_key_dpapi)

    def set_xai_key(self, plain: str) -> None:
        self.xai_key_dpapi = protect(plain.strip()) if (plain or "").strip() else ""

    def has_meta_key(self) -> bool:
        return bool((self.meta_key_dpapi or "").strip())

    def meta_key(self) -> str:
        return unprotect(self.meta_key_dpapi)

    def set_meta_key(self, plain: str) -> None:
        self.meta_key_dpapi = protect(plain.strip()) if (plain or "").strip() else ""

    def cloud_key(self) -> str:
        if (self.provider or "") == "meta":
            return self.meta_key()
        if (self.provider or "") == "xai":
            return self.xai_key()
        return ""

    def active_model(self) -> str:
        provider = self.provider if self.provider in PROVIDERS else "xai"
        if provider == "meta":
            return (self.meta_model or "muse-spark-1.3").strip()
        if provider == "xai":
            return (self.xai_model or "grok-4").strip()
        return (self.model or "").strip()

    def resolved_workspace(self) -> Path:
        raw = (self.workspace or "").strip()
        p = Path(raw) if raw else default_workspace()
        p.mkdir(parents=True, exist_ok=True)
        return p


def _legacy_sm0k() -> dict:
    """Settings saved by the sm0k fork. The DPAPI blob opens for this Windows user."""
    path = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "sm0k" / "config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_settings() -> Settings:
    path = config_path()
    raw: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}
    known = {f.name for f in Settings.__dataclass_fields__.values()}
    if raw:
        settings = Settings(**{k: v for k, v in raw.items() if k in known})
    else:
        settings = Settings(workspace=str(default_workspace()))
    if settings.provider not in PROVIDERS:
        settings.provider = "xai"
    legacy = _legacy_sm0k()
    changed = not path.is_file()
    cleaned_effort = normalize_effort(settings.reasoning_effort)
    if cleaned_effort != (settings.reasoning_effort or ""):
        settings.reasoning_effort = cleaned_effort
        changed = True
    else:
        settings.reasoning_effort = cleaned_effort
    if not settings.xai_key_dpapi and legacy.get("xai_key_dpapi"):
        settings.xai_key_dpapi = str(legacy.get("xai_key_dpapi") or "")
        changed = True
    if legacy.get("xai_model") and "xai_model" not in raw:
        settings.xai_model = str(legacy.get("xai_model") or settings.xai_model)
        changed = True
    if "provider" not in raw and legacy.get("provider") in PROVIDERS:
        settings.provider = str(legacy.get("provider"))
        changed = True
    if not (settings.model or "").strip() and legacy.get("model"):
        settings.model = str(legacy.get("model") or "")
        changed = True
    if not (settings.workspace or "").strip() and legacy.get("workspace"):
        settings.workspace = str(legacy.get("workspace") or "")
        changed = True
    if not (settings.workspace or "").strip():
        settings.workspace = str(default_workspace())
        changed = True
    if changed:
        save_settings(settings)
    return settings


def save_settings(s: Settings) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
