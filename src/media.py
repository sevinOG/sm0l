"""Image attachments for Ollama vision. Files live under user_data()/media/."""
from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any

from .paths import user_data

MAX_ATTACH = 4
MAX_EDGE = 1568
IMAGE_TOKENS = 768  # per image; vision models spend a chunk of context on pixels
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def media_root() -> Path:
    p = user_data() / "media"
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_media_path(ext: str = ".jpg") -> Path:
    ext = ext if ext.startswith(".") else f".{ext}"
    if ext.lower() not in IMAGE_EXTS:
        ext = ".jpg"
    return media_root() / f"{uuid.uuid4().hex}{ext.lower()}"


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def attachment_count(msg: dict) -> int:
    atts = msg.get("attachments")
    if isinstance(atts, list) and atts:
        return len(atts)
    images = msg.get("images")
    if isinstance(images, list):
        return len(images)
    return 0


def attachments_of(msg: dict) -> list[dict]:
    atts = msg.get("attachments")
    if isinstance(atts, list):
        out = []
        for a in atts:
            if isinstance(a, dict) and a.get("path"):
                out.append(a)
            elif isinstance(a, str):
                out.append({"path": a, "name": Path(a).name, "mime": MIME_BY_EXT.get(Path(a).suffix.lower(), "image/png")})
        return out
    return []


def attachments_to_b64(attachments: list[dict] | None) -> list[str]:
    out: list[str] = []
    for a in attachments or []:
        raw = a.get("path") if isinstance(a, dict) else a
        p = Path(str(raw))
        if not p.is_file():
            continue
        out.append(base64.b64encode(p.read_bytes()).decode("ascii"))
    return out


def prepare_ollama_messages(messages: list[dict]) -> list[dict]:
    """Drop local attachment metadata; emit Ollama `images` as base64 strings."""
    prepared: list[dict] = []
    for m in messages:
        mm = {k: v for k, v in m.items() if k not in ("attachments", "images")}
        b64s: list[str] = []
        atts = attachments_of(m)
        if atts:
            b64s = attachments_to_b64(atts)
        else:
            raw = m.get("images")
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and item:
                        b64s.append(item)
        if b64s:
            mm["images"] = b64s
        prepared.append(mm)
    return prepared


def encode_qimage(image: Any) -> tuple[bytes, str, str]:
    """Resize and encode a QImage/QPixmap. Returns (bytes, mime, ext)."""
    from PyQt6.QtCore import QBuffer, QIODevice
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage

    if image is None:
        raise ValueError("empty image")
    if not isinstance(image, QImage):
        if hasattr(image, "toImage"):
            image = image.toImage()
        else:
            image = QImage(image)
    if image.isNull():
        raise ValueError("empty image")
    w, h = image.width(), image.height()
    if max(w, h) > MAX_EDGE:
        image = image.scaled(
            MAX_EDGE,
            MAX_EDGE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    fmt = "PNG" if image.hasAlphaChannel() else "JPEG"
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    quality = 85 if fmt == "JPEG" else 80
    if not image.save(buf, fmt, quality):
        raise ValueError("could not encode image")
    data = bytes(buf.data())
    if not data:
        raise ValueError("could not encode image")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    ext = ".png" if fmt == "PNG" else ".jpg"
    return data, mime, ext


def save_encoded(data: bytes, mime: str, name: str) -> dict:
    ext = ".png" if mime == "image/png" else ".jpg"
    path = new_media_path(ext)
    path.write_bytes(data)
    return {
        "path": str(path),
        "name": name or path.name,
        "mime": mime,
        "bytes": len(data),
    }
