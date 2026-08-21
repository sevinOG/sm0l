# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
try:
    root = Path(SPECPATH).resolve()
except NameError:
    root = Path(".").resolve()
icon_ico = str(root / "assets" / "icon.ico") if (root / "assets" / "icon.ico").is_file() else None
version = str(root / "version_info.txt") if (root / "version_info.txt").is_file() else None

a = Analysis(
    ["sm0l.py"],
    pathex=[str(root)],
    binaries=[],
    datas=[
        ("assets", "assets"),
        ("VERSION", "."),
    ],
    hiddenimports=[
        "PyQt6.sip",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "src",
        "src.main",
        "src.ui",
        "src.theme",
        "src.agent",
        "src.tools",
        "src.compact",
        "src.ollama_client",
        "src.session",
        "src.config",
        "src.paths",
        "src.personality",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy", "PIL"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sm0l",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=icon_ico,
    version=version,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="sm0l",
)
