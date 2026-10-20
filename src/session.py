"""Chat sessions on disk. No heartbeat files."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import sessions_dir, user_data


@dataclass
class Session:
    id: str
    title: str
    created: float
    updated: float
    model: str
    messages: list[dict] = field(default_factory=list)
    token_estimate: int = 0
    native_ctx: int = 0

    def path(self) -> Path:
        return sessions_dir() / f"{self.id}.json"

    def save(self) -> None:
        self.updated = time.time()
        self.path().write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def touch_title(self, user_text: str) -> None:
        if self.title and self.title != "new session":
            return
        line = " ".join(user_text.strip().split())
        self.title = (line[:48] + "…") if len(line) > 48 else (line or "new session")


def new_session(model: str = "") -> Session:
    now = time.time()
    s = Session(
        id=uuid.uuid4().hex[:12],
        title="new session",
        created=now,
        updated=now,
        model=model,
    )
    s.save()
    return s


def load_session(sid: str) -> Session | None:
    path = sessions_dir() / f"{sid}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session(
            id=data.get("id") or sid,
            title=data.get("title") or "session",
            created=float(data.get("created") or time.time()),
            updated=float(data.get("updated") or time.time()),
            model=data.get("model") or "",
            messages=list(data.get("messages") or []),
            token_estimate=int(data.get("token_estimate") or 0),
            native_ctx=int(data.get("native_ctx") or 0),
        )
    except Exception:
        return None


def list_sessions() -> list[dict[str, Any]]:
    items = []
    for path in sessions_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": data.get("id") or path.stem,
                    "title": data.get("title") or path.stem,
                    "updated": float(data.get("updated") or 0),
                    "model": data.get("model") or "",
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items


def delete_session(sid: str) -> None:
    path = sessions_dir() / f"{sid}.json"
    if path.is_file():
        path.unlink()


def choose_session_id(last_id: str | None, ordered_ids: list[str]) -> str | None:
    """Pick the last thread when it is still on disk, else the newest, else none."""
    ids = [str(item) for item in ordered_ids if item]
    if last_id and str(last_id) in ids:
        return str(last_id)
    return ids[0] if ids else None


def read_last_session_id() -> str | None:
    path = user_data() / "last_session.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = str((data or {}).get("id") or "")
    except Exception:
        return None
    return sid or None


def remember_session(sid: str) -> None:
    path = user_data() / "last_session.json"
    path.write_text(json.dumps({"id": sid}), encoding="utf-8")


def resume_session(model: str = "") -> Session:
    """Open the last thread if it still exists, else the newest, else a new one."""
    items = list_sessions()
    ids = [str(item.get("id") or "") for item in items]
    chosen = choose_session_id(read_last_session_id(), ids)
    if chosen:
        loaded = load_session(chosen)
        if loaded is not None:
            return loaded
    for sid in ids:
        loaded = load_session(sid)
        if loaded is not None:
            return loaded
    return new_session(model)
