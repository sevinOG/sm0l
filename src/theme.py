"""Eche-inspired dark purple visual system for the sm0l dashboard."""
from __future__ import annotations

import sys

APP_NAME = "sm0l"
APP_VERSION = "0.2.0"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"

BG = "#0B0E14"
BG_ALT = "#14151a"
BG_ELEVATED = "#1c1d24"
BG_INPUT = "#12131a"
BORDER = "#2a2b34"
BORDER_LIGHT = "#3a3c48"
BORDER_FOCUS = "#8B5CF6"
TEXT = "#e8e8ec"
TEXT_MUTED = "#8b8c9a"
TEXT_DIM = "#6e6f7c"
TEXT_BRIGHT = "#f5f5fa"
ACCENT = "#8B5CF6"
ACCENT_HOVER = "#7C3AED"
ACCENT_SOFT = "#a78bfa"
DANGER = "#3a2228"
DANGER_BORDER = "#6b3040"
DANGER_TEXT = "#f0c0c8"
SUCCESS = "#2a4a3a"
SUCCESS_BORDER = "#3d7a5a"
SUCCESS_TEXT = "#b8e0c8"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Segoe UI Variable", sans-serif;
    font-size: 13px;
}}
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
}}
QToolTip {{
    background-color: {BG_ELEVATED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px 8px;
}}
QLabel {{
    color: {TEXT};
    background: transparent;
}}
QLabel#Brand {{
    color: {ACCENT};
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 6px;
}}
QLabel#BrandSub {{
    color: {TEXT_MUTED};
    font-size: 10px;
    letter-spacing: 1.6px;
}}
QLabel#Section {{
    color: {ACCENT_SOFT};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#Muted {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel#Dim {{
    color: {TEXT_DIM};
    font-size: 11px;
}}
QLabel#StatusDot {{
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 10px;
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QLabel#StatusDot[state="online"] {{
    color: {SUCCESS_TEXT};
    border-color: {SUCCESS_BORDER};
    background-color: {SUCCESS};
}}
QLabel#StatusDot[state="offline"] {{
    color: {DANGER_TEXT};
    border-color: {DANGER_BORDER};
    background-color: {DANGER};
}}
QLabel#StatusDot[state="busy"] {{
    color: {ACCENT_SOFT};
    border-color: {ACCENT};
}}
QPushButton {{
    background-color: #2b2d38;
    color: {TEXT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: #353744;
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: #22242e;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    background-color: #1e1f26;
    border-color: {BORDER};
}}
QPushButton#primary {{
    background-color: {ACCENT};
    border-color: {ACCENT_HOVER};
    color: {TEXT_BRIGHT};
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#danger {{
    background-color: {DANGER};
    border-color: {DANGER_BORDER};
    color: {DANGER_TEXT};
    font-weight: 600;
}}
QPushButton#ghost {{
    background-color: transparent;
    border: 1px solid {BORDER_LIGHT};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid #2e2f3a;
    border-radius: 8px;
    padding: 8px 10px;
    selection-background-color: {ACCENT};
    selection-color: {TEXT_BRIGHT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
    border: 1px solid {BORDER_FOCUS};
}}
QPlainTextEdit, QTextEdit {{
    font-family: "Consolas", "Cascadia Mono", "Courier New", monospace;
    font-size: 13px;
}}
QListWidget {{
    padding: 4px;
    outline: none;
    border: none;
    background: transparent;
}}
QListWidget::item {{
    padding: 9px 10px;
    border-radius: 8px;
    margin: 2px 0;
    color: {TEXT_MUTED};
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: {TEXT_BRIGHT};
}}
QListWidget::item:hover {{
    background-color: #2b2d38;
    color: {TEXT};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {BG_ELEVATED};
    color: {TEXT};
    selection-background-color: {ACCENT};
    border: 1px solid {BORDER};
}}
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT_MUTED};
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {BG};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #2e2f3a;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_HOVER};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QFrame#Panel, QFrame#Card {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#AccentBar {{
    background-color: {ACCENT};
    max-height: 2px;
    min-height: 2px;
    border: none;
}}
QFrame#UserBubble {{
    background-color: #1a1630;
    border: 1px solid #3d2f66;
    border-radius: 12px;
}}
QFrame#AssistantBubble {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#ToolBubble {{
    background-color: #14151c;
    border: 1px solid #2a2b34;
    border-radius: 10px;
}}
QCheckBox {{
    color: #b0b0bc;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3a3c46;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT_HOVER};
}}
"""


def brand_icon():
    from PyQt6.QtGui import QIcon
    from .paths import asset

    icon = QIcon()
    for name in ("icon.png", "icon.ico"):
        p = asset("assets", name) or asset(name)
        if p:
            icon.addFile(str(p))
    return icon


def apply_theme(app) -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("sm0l.App.1")
        except Exception:
            pass
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    try:
        icon = brand_icon()
        if not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass
