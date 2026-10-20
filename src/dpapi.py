"""Windows DPAPI wrappers. The API key is encrypted for the current user."""
from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes

_CRYPTPROTECT_UI_FORBIDDEN = 0x01


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _blob(data: bytes) -> tuple[_DATA_BLOB, ctypes.Array]:
    buf = ctypes.create_string_buffer(data, len(data))
    blob = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    return blob, buf


def protect(plain: str) -> str:
    """Encrypt a secret for the current Windows user. Returns base64 ciphertext."""
    text = plain or ""
    if not text:
        return ""
    blob_in, _keep = _blob(text.encode("utf-8"))
    blob_out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        "sm0l",
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        err = ctypes.windll.kernel32.GetLastError()
        raise OSError(err, "CryptProtectData failed")
    try:
        raw = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        return base64.b64encode(raw).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def unprotect(cipher_b64: str) -> str:
    """Decrypt a DPAPI blob. Returns '' if the blob is empty or cannot be opened."""
    raw_b64 = (cipher_b64 or "").strip()
    if not raw_b64:
        return ""
    try:
        raw = base64.b64decode(raw_b64)
    except Exception:
        return ""
    blob_in, _keep = _blob(raw)
    blob_out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        return ""
    try:
        data = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        return data.decode("utf-8", errors="replace")
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)
