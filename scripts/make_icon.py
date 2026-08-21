"""Generate assets/icon.png + icon.ico (stdlib only)."""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
BG = (11, 14, 20, 255)
VIOLET = (139, 92, 246, 255)
CORE = (245, 245, 250, 255)


def _png(w: int, h: int, pixels: list[tuple[int, int, int, int]]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw.extend(pixels[y * w + x])
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(4))


def _draw(size: int) -> list[tuple[int, int, int, int]]:
    cx = cy = size / 2
    pixels = []
    for y in range(size):
        for x in range(size):
            dx = (x + 0.5 - cx) / size
            dy = (y + 0.5 - cy) / size
            # diamond (manhattan) + faint glow
            man = abs(dx) + abs(dy)
            rad = math.sqrt(dx * dx + dy * dy)
            glow = max(0.0, 1.0 - rad / 0.48)
            diamond = max(0.0, 1.0 - man / 0.34)
            edge = max(0.0, 1.0 - abs(man - 0.28) / 0.035)
            core = max(0.0, 1.0 - man / 0.10)
            t = min(1.0, glow * 0.35 + diamond * 0.85 + edge * 0.55)
            col = _lerp(BG, VIOLET, t)
            if core > 0:
                col = _lerp(col, CORE, core * 0.85)
            # slight vignette on corners so it reads as a rounded app icon
            corner = max(abs(dx), abs(dy))
            if corner > 0.46:
                fade = max(0.0, 1.0 - (corner - 0.46) / 0.04)
                col = _lerp((0, 0, 0, 0), col, fade)
            pixels.append(col)
    return pixels


def _ico(pngs: list[tuple[int, bytes]]) -> bytes:
    count = len(pngs)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries = b""
    payload = b""
    for size, data in pngs:
        w = 0 if size >= 256 else size
        h = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(data), offset)
        payload += data
        offset += len(data)
    return header + entries + payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png256 = _png(256, 256, _draw(256))
    png32 = _png(32, 32, _draw(32))
    png16 = _png(16, 16, _draw(16))
    (OUT / "icon.png").write_bytes(png256)
    (OUT / "icon.ico").write_bytes(_ico([(256, png256), (32, png32), (16, png16)]))
    print(f"wrote {OUT / 'icon.png'} and {OUT / 'icon.ico'}")


if __name__ == "__main__":
    main()
