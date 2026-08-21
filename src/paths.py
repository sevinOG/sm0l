"""App / user-data / workspace paths. Works from source and frozen EXE."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return app_dir()


def user_data() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "sm0l"
    base.mkdir(parents=True, exist_ok=True)
    (base / "sessions").mkdir(exist_ok=True)
    return base


def sessions_dir() -> Path:
    p = user_data() / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_path() -> Path:
    return user_data() / "config.json"


def default_workspace() -> Path:
    p = Path.home() / "sm0l_workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def asset(*parts: str) -> Path | None:
    bases = [bundle_dir(), app_dir(), bundle_dir() / "_internal"]
    if is_frozen():
        bases.append(Path(sys.executable).resolve().parent / "_internal")
    for base in bases:
        cand = base.joinpath(*parts)
        if cand.is_file():
            return cand
        cand = base / "assets" / Path(*parts).name
        if cand.is_file():
            return cand
    return None
