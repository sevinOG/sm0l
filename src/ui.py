"""Windows dashboard — chat, sessions, context, tools. No CLI required."""
from __future__ import annotations

import html
import json
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal, QMimeData, QRectF
from PyQt6.QtGui import (
    QKeyEvent,
    QDesktopServices,
    QGuiApplication,
    QImage,
    QPixmap,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPalette,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QDialog,
    QComboBox,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextBrowser,
)

CASHAPP_TAG = "$reshi7"
CASHAPP_URL = "https://cash.app/$reshi7"
BTC_ADDRESS = "bc1qp989v95u54zpnmw9j75azwp9hrqnd0k6d7jp3lvv6z3yywpfdutszkkhg6"

from .agent import Agent, round_budget
from .compact import compact_threshold, estimate_tokens, reset_calibration
from .config import Settings, save_settings
from .media import MAX_ATTACH, attachments_of, encode_qimage, is_image_path, save_encoded
from .ollama_client import list_models, ping
from .meta_client import (
    DEFAULT_MODEL as META_MODEL,
    account_status as meta_status,
    list_models as meta_list_models,
)
from .xai_client import DEFAULT_CTX, DEFAULT_MODEL, account_status, list_models as xai_list_models
from .reasoning import EFFORTS, normalize_effort
from .personality import seed_workspace
from .session import (
    Session,
    delete_session,
    list_sessions,
    load_session,
    new_session,
    remember_session,
    resume_session,
)
from .theme import ACCENT, ACCENT_HOVER, ACCENT_SOFT, BG_INPUT, TEXT, TEXT_BRIGHT, TEXT_DIM, TEXT_MUTED

# How many times auto may resume the same transcript after the round budget.
# The budget itself is doubled in round_budget when auto is on.
AUTO_HOP_CAP = 24


class AgentWorker(QThread):
    event = pyqtSignal(str, dict)

    def __init__(self, settings: Settings, model: str, messages: list[dict]):
        super().__init__()
        self.settings = settings
        self.model = model
        self.messages = messages
        self.agent: Agent | None = None
        self.result_messages: list[dict] = []
        self.hands_gate = threading.Event()
        self.hands_end = threading.Event()
        self.hands_button: tuple[int, int, int, int] | None = None

    def run(self):
        def emit(kind: str, data: dict):
            if kind in ("hands_window", "hands_badge"):
                self.hands_gate.clear()
            if kind == "hands_window" and data.get("hide"):
                self.hands_end.clear()
                self.hands_button = None
            self.event.emit(kind, data)
            if kind in ("hands_window", "hands_badge"):
                # GUI thread moves the end button or the sm0l window, then opens the gate.
                self.hands_gate.wait(2.0)

        self.agent = Agent(
            self.settings.ollama_host,
            self.model,
            self.settings.resolved_workspace(),
            num_ctx_override=self.settings.num_ctx,
            temperature=self.settings.temperature,
            compact_ratio=self.settings.compact_ratio,
            max_tool_rounds=round_budget(
                self.settings.max_tool_rounds, autorun=self.settings.autorun
            ),
            shell_timeout=self.settings.shell_timeout,
            emit=emit,
            provider=self.settings.provider,
            api_key=self.settings.cloud_key(),
            local_model=self.settings.model,
            xai_model=self.settings.xai_model or DEFAULT_MODEL,
            reasoning_effort=self.settings.reasoning_effort,
        )
        self.agent.on_badge = lambda left, top, width, height: emit(
            "hands_badge",
            {"left": int(left), "top": int(top), "width": int(width), "height": int(height)},
        )
        self.agent.button_rect = lambda: self.hands_button
        self.agent.end_requested = self.hands_end.is_set
        try:
            self.result_messages = self.agent.run(self.messages)
        except Exception as e:
            self.event.emit("error", {"text": str(e)})
            self.result_messages = self.messages

    def cancel(self):
        if self.agent:
            self.agent.cancel()
        # Wait on thread end only — last_prompt_tokens == 0 is "no eval yet", not stopped.
        import time
        start = time.monotonic()
        while time.monotonic() - start < 5:
            if not self.isRunning():
                break
            time.sleep(0.1)


class EndToolRounds(QWidget):
    """Semi-transparent button Grok can see and click to stop hands_on."""

    def __init__(self, on_press):
        super().__init__(None)
        self._on_press = on_press
        self.setFixedSize(248, 48)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def paintEvent(self, _event):
        from .hands import END_LABEL

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -2, -2)
        radius = max(4, min(16, rect.height() / 3))
        painter.setPen(QPen(QColor(167, 139, 250, 230), 2))
        painter.setBrush(QColor(18, 14, 28, 170))
        painter.drawRoundedRect(rect, radius, radius)
        painter.setPen(QColor(255, 255, 255, 240))
        font = QFont("Segoe UI")
        font.setBold(True)
        font.setPixelSize(max(8, min(18, int(rect.height() * 0.38))))
        painter.setFont(font)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), END_LABEL)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_press:
            self._on_press()
        super().mousePressEvent(event)


def sync_end_button(widget: QWidget, left: int, top: int, width: int, height: int):
    """Show the end button inside a physical screen box. Returns its GetWindowRect."""
    from PyQt6.QtWidgets import QApplication

    from .hands import dpi_at, move_topmost, window_physical_rect

    dpi = dpi_at(int(left) + int(width) // 2, int(top) + int(height) // 2)
    dpr = max(1.0, dpi / 96.0)
    widget.setFixedSize(max(1, int(round(width / dpr))), max(1, int(round(height / dpr))))
    widget.move(int(round(left / dpr)), int(round(top / dpr)))
    widget.show()
    widget.raise_()
    app = QApplication.instance()
    if app:
        app.processEvents()
    hwnd = int(widget.winId())
    got = window_physical_rect(hwnd)
    for _ in range(3):
        if not got:
            break
        dx = int(left) - got[0]
        dy = int(top) - got[1]
        if abs(dx) <= 2 and abs(dy) <= 2:
            break
        pos = widget.pos()
        widget.move(pos.x() + int(round(dx / dpr)), pos.y() + int(round(dy / dpr)))
        if app:
            app.processEvents()
        got = window_physical_rect(hwnd) or got
    if got and (abs(got[0] - int(left)) > 2 or abs(got[1] - int(top)) > 2):
        move_topmost(hwnd, int(left), int(top))
        got = window_physical_rect(hwnd) or got
    return got


class InputBox(QPlainTextEdit):
    send = pyqtSignal()
    attach = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.send.emit()
            return
        super().keyPressEvent(event)

    def canInsertFromMimeData(self, source: QMimeData) -> bool:
        if source and (source.hasImage() or source.hasUrls()):
            return True
        return super().canInsertFromMimeData(source)

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source is None:
            return
        if source.hasImage():
            img = source.imageData()
            if img is not None:
                self.attach.emit(img)
                return
        if source.hasUrls():
            paths = [
                u.toLocalFile()
                for u in source.urls()
                if u.isLocalFile() and is_image_path(u.toLocalFile())
            ]
            if paths:
                self.attach.emit(paths)
                return
        super().insertFromMimeData(source)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md and (md.hasImage() or md.hasUrls()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        self.insertFromMimeData(event.mimeData())
        event.acceptProposedAction()


def _thumb_row(attachments: list) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    for a in attachments:
        path = a.get("path") if isinstance(a, dict) else str(a)
        pix = QPixmap(str(path))
        if pix.isNull():
            continue
        pix = pix.scaled(
            140,
            140,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        lab = QLabel()
        lab.setPixmap(pix)
        lab.setToolTip(a.get("name") if isinstance(a, dict) else str(path))
        row.addWidget(lab)
    row.addStretch()
    return row


def tool_entry_label(name: str, *, result: bool) -> str:
    """Chat label for one tool call or its result. hands_on reads as hands on."""
    shown = " ".join((name or "tool").replace("_", " ").split()) or "tool"
    if result:
        return f"{shown} result"
    return f"tool {shown}"


def tool_feed_html(links: list[tuple[int, str]]) -> str:
    """Comma-separated links. The href is an index into the tool log."""
    parts: list[str] = []
    for mark, label in links:
        safe = html.escape(label, quote=False)
        parts.append(f'<a href="mark:{int(mark)}">{safe}</a>')
    return ", ".join(parts)


def format_tool_log(name: str, payload: str, *, result: bool) -> str:
    """One tool-log entry. The name and the body are on separate lines."""
    title = " ".join((name or "tool").split()) or "tool"
    arrow = "←" if result else "→"
    body = (payload or "").strip()
    if not body:
        return f"{arrow} {title}"
    return f"{arrow} {title}\n{body}"


def append_tool_block(edit: QPlainTextEdit, text: str) -> int:
    """Append one log entry. A blank line separates it from the previous entry."""
    doc = edit.document()
    cursor = QTextCursor(doc)
    cursor.movePosition(QTextCursor.MoveOperation.End)
    if edit.toPlainText().strip():
        cursor.insertText("\n\n")
    pos = cursor.position()
    cursor.insertText(text.strip("\n"))
    bar = edit.verticalScrollBar()
    bar.setValue(bar.maximum())
    return pos


def scroll_tool_log_to(edit: QPlainTextEdit, pos: int) -> None:
    """Put a log entry at the top of the tool panel and highlight its first line."""
    doc = edit.document()
    end = max(0, doc.characterCount() - 1)
    pos = min(max(0, int(pos)), end)
    cursor = QTextCursor(doc)
    cursor.setPosition(pos)
    cursor.movePosition(
        QTextCursor.MoveOperation.EndOfBlock,
        QTextCursor.MoveMode.KeepAnchor,
    )
    edit.setTextCursor(cursor)
    edit.ensureCursorVisible()
    rect = edit.cursorRect(cursor)
    bar = edit.verticalScrollBar()
    bar.setValue(max(0, bar.value() + rect.top() - 6))


class ToolFeed(QTextBrowser):
    """One growing line of tool links. The bubble height follows the text."""

    def __init__(self, on_link):
        super().__init__()
        self._fitting = False
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(0)
        font = QFont("Segoe UI")
        font.setPixelSize(13)
        self.setFont(font)
        self.document().setDocumentMargin(2)
        self.document().setDefaultStyleSheet(
            "body { color: " + TEXT + "; background: transparent; font-size: 13px; margin: 0; }"
            "a { color: " + ACCENT_SOFT + "; text-decoration: underline; }"
        )
        self.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; font-size: 13px; padding: 0; }"
        )
        self.viewport().setStyleSheet("background: transparent; border: none;")
        clear = QColor(0, 0, 0, 0)
        for widget in (self, self.viewport()):
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Base, clear)
            palette.setColor(QPalette.ColorRole.Window, clear)
            widget.setPalette(palette)
            widget.setAutoFillBackground(False)
        self.anchorClicked.connect(lambda url: on_link(url.toString()))

    def set_links(self, html_text: str) -> None:
        self.setHtml(html_text)
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._fitting:
            return
        self._fitting = True
        try:
            width = self.viewport().width()
            if width < 40:
                width = max(self.width(), 280)
            self.document().setTextWidth(width)
            # Extra room so the underline and the last wrapped line are not clipped.
            line = self.fontMetrics().lineSpacing() + 4
            height = max(line, int(self.document().size().height()) + 4)
            if self.height() != height:
                self.setFixedHeight(height)
        finally:
            self._fitting = False


_COLLAPSE_QSS = (
    f"QPushButton {{ color: {TEXT_DIM}; font-size: 11px; font-weight: 500;"
    " text-align: left; padding: 1px 0; border: none; background: transparent; }}"
    f"QPushButton:hover {{ color: {TEXT}; }}"
)


def _collapse_button(title: str) -> QPushButton:
    button = QPushButton("▸  " + title)
    button.setFlat(True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(_COLLAPSE_QSS)
    return button


def _quiet_title(title: str, fallback: str) -> str:
    return " ".join((title or fallback).split()).lower() or fallback


class ReasoningCard(QFrame):
    """One collapsed reasoning block. The body links back to its chat message."""

    def __init__(self, title: str, on_jump):
        super().__init__()
        self.setObjectName("ToolBubble")
        self._title = _quiet_title(title, "reasoning")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(0)
        self._header = _collapse_button(self._title)
        self._body_box = QWidget()
        self._body_box.setAutoFillBackground(False)
        self._body_box.setStyleSheet("background: transparent; border: none;")
        body_lay = QVBoxLayout(self._body_box)
        body_lay.setContentsMargins(0, 0, 0, 2)
        body_lay.setSpacing(2)
        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setStyleSheet(
            f"QLabel {{ color: {TEXT}; background: transparent; border: none; font-size: 11px; }}"
        )
        self._link = QLabel('<a href="message">message</a>')
        self._link.setTextFormat(Qt.TextFormat.RichText)
        self._link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._link.setOpenExternalLinks(False)
        self._link.setStyleSheet(
            "QLabel { background: transparent; border: none; font-size: 11px; }"
        )
        link_palette = self._link.palette()
        link_palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT_SOFT))
        self._link.setPalette(link_palette)
        self._link.linkActivated.connect(lambda _href: on_jump())
        body_lay.addWidget(self._body)
        body_lay.addWidget(self._link)
        self._body_box.setVisible(False)
        lay.addWidget(self._header)
        lay.addWidget(self._body_box)

        def _toggle() -> None:
            open_ = not self._body_box.isVisible()
            self._body_box.setVisible(open_)
            self._header.setText(("▾  " if open_ else "▸  ") + self._title)
            lay.setContentsMargins(6, 1, 6, 3 if open_ else 1)

        self._header.clicked.connect(_toggle)

    def set_text(self, value: str) -> None:
        self._body.setText(value or "")


def _bubble(kind: str, title: str, body: str, mono: bool = False, attachments: list | None = None) -> QFrame:
    frame = QFrame()
    frame.setObjectName(
        {"user": "UserBubble", "assistant": "AssistantBubble"}.get(kind, "ToolBubble")
    )
    shown = _quiet_title(title, "chat")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(8, 1, 8, 1)
    lay.setSpacing(0)
    header = _collapse_button(shown)
    lay.addWidget(header)
    body_box = QWidget()
    body_box.setAutoFillBackground(False)
    body_box.setStyleSheet("background: transparent; border: none;")
    body_lay = QVBoxLayout(body_box)
    body_lay.setContentsMargins(0, 0, 0, 4)
    body_lay.setSpacing(4)
    body_lbl = QLabel(body)
    body_lbl.setTextFormat(Qt.TextFormat.PlainText)
    body_lbl.setWordWrap(True)
    body_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if mono:
        body_lbl.setStyleSheet(
            f"QLabel {{ color: {TEXT_MUTED}; background: transparent; border: none;"
            " font-family: Consolas, Cascadia Mono, monospace; font-size: 12px; }}"
        )
    else:
        body_lbl.setStyleSheet(
            f"QLabel {{ color: {TEXT}; background: transparent; border: none; font-size: 13px; }}"
        )
    if body:
        body_lay.addWidget(body_lbl)
    if attachments:
        body_lay.addLayout(_thumb_row(attachments))
    if not body and not attachments:
        body_lay.addWidget(body_lbl)
    lay.addWidget(body_box)
    header.setText("▾  " + shown)
    lay.setContentsMargins(8, 1, 8, 4)

    def _toggle() -> None:
        open_ = not body_box.isVisible()
        body_box.setVisible(open_)
        header.setText(("▾  " if open_ else "▸  ") + shown)
        lay.setContentsMargins(8, 1, 8, 4 if open_ else 1)

    header.clicked.connect(_toggle)
    frame._body = body_lbl  # type: ignore[attr-defined]
    frame._raw = body  # type: ignore[attr-defined]
    frame._body_box = body_box  # type: ignore[attr-defined]
    return frame


class Dashboard(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.session = resume_session(settings.model)
        remember_session(self.session.id)
        self.worker: AgentWorker | None = None
        self._end_rounds: EndToolRounds | None = None
        self._stream_bubble: QFrame | None = None
        self._stream_text = ""
        self._tool_bubble: QFrame | None = None
        self._tool_marks: list[int] = []
        self._anchors: dict[str, QWidget] = {}
        self._reason_cards: dict[str, ReasoningCard] = {}
        self._latest_user: QWidget | None = None
        self.native_ctx = 0
        self.effective_ctx = 0
        self.last_prompt_tokens = 0  # real prompt_eval_count from last Ollama call
        self._live_used_tokens = 0  # compact/tokens estimate while a worker is in flight
        self._auto_resume = False
        self._user_stopped = False
        self._auto_hops = 0
        self._sending_auto = False
        self._pending: list[dict] = []
        self._local_models: list[str] = []
        self._xai_models: list[str] = []
        self._xai_ok = False
        self._xai_detail = "no API key"
        self._meta_models: list[str] = []
        self._meta_ok = False
        self._meta_detail = "no API key"
        self._ollama_ok = False
        self._build()
        seed_workspace(self.settings.resolved_workspace())
        self._refresh_sessions()
        self._reload_models()
        self._tick_ollama()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_ollama)
        self._timer.start(8000)
        self._update_ctx_bar()

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            "QFrame { background-color: #1c1d24; border: none; border-bottom: 1px solid #2a2b34; }"
        )
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 6, 16, 8)
        h.setSpacing(8)

        self.settings_btn = QPushButton("settings")
        self.settings_btn.setObjectName("headerBtn")
        self.settings_btn.setToolTip("xAI and Meta API keys (Windows DPAPI) and connection")
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.clicked.connect(self._open_settings)
        h.addWidget(self.settings_btn)

        self.donate_button = QPushButton("donate")
        self.donate_button.setObjectName("headerBtn")
        self.donate_button.setToolTip("Open the tip jar (optional, no pressure)")
        self.donate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.donate_button.clicked.connect(self._on_donate_clicked)
        h.addWidget(self.donate_button)

        self.provider_btn = QPushButton("provider")
        self.provider_btn.setObjectName("headerBtn")
        self.provider_btn.setToolTip("xAI, Meta, and Ollama status")
        self.provider_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.provider_btn.setProperty("state", "offline")
        self.provider_btn.clicked.connect(self._show_providers)
        h.addWidget(self.provider_btn)

        self.sessions_btn = QPushButton("sessions")
        self.sessions_btn.setObjectName("headerBtn")
        self.sessions_btn.setToolTip("Chats and the workspace folder")
        self.sessions_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sessions_btn.clicked.connect(self._open_sessions)
        h.addWidget(self.sessions_btn)
        h.addStretch()
        self._xai_ok = False
        self._ollama_ok = False
        outer.addWidget(header)

        accent = QFrame()
        accent.setObjectName("AccentBar")
        outer.addWidget(accent)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._sidebar())
        split.addWidget(self._chat_pane())
        split.addWidget(self._inspector())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([240, 720, 300])
        outer.addWidget(split, 1)
        self._build_sessions_dialog()

    def _sidebar(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Panel")
        wrap.setStyleSheet("QFrame#Panel { background-color: #0B0E14; border: none; border-radius: 0; }")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        lab = QLabel("reasoning")
        lab.setObjectName("Section")
        lay.addWidget(lab)
        self.reason_scroll = QScrollArea()
        self.reason_scroll.setObjectName("LogBubble")
        self.reason_scroll.setWidgetResizable(True)
        self.reason_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.reason_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.reason_scroll.setStyleSheet(
            "QScrollArea#LogBubble {"
            f" background-color: {BG_INPUT};"
            " border: 1px solid #2e2f3a;"
            " border-radius: 8px;"
            "}"
        )
        self.reason_host = QWidget()
        self.reason_host.setAutoFillBackground(False)
        self.reason_host.setStyleSheet("background: transparent; border: none;")
        self.reason_layout = QVBoxLayout(self.reason_host)
        self.reason_layout.setContentsMargins(10, 8, 8, 8)
        self.reason_layout.setSpacing(4)
        self.reason_layout.addStretch(1)
        self.reason_scroll.setWidget(self.reason_host)
        viewport = self.reason_scroll.viewport()
        viewport.setAutoFillBackground(False)
        viewport.setStyleSheet("background: transparent; border: none;")
        lay.addWidget(self.reason_scroll, 1)
        return wrap

    def _build_sessions_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("sessions")
        dlg.resize(420, 520)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)
        self.session_list = QListWidget()
        self.session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        self.session_list.customContextMenuRequested.connect(self._show_session_menu)
        lay.addWidget(self.session_list, 1)
        btn_row = QHBoxLayout()
        new_btn = QPushButton("new")
        new_btn.setObjectName("primary")
        new_btn.clicked.connect(self._new_session)
        del_btn = QPushButton("delete")
        del_btn.setObjectName("ghost")
        del_btn.clicked.connect(self._delete_session)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(del_btn)
        lay.addLayout(btn_row)
        lab = QLabel("workspace")
        lab.setObjectName("Section")
        lay.addWidget(lab)
        self.ws_label = QLabel(str(self.settings.resolved_workspace()))
        self.ws_label.setObjectName("Dim")
        self.ws_label.setWordWrap(True)
        lay.addWidget(self.ws_label)
        ws_btn = QPushButton("browse…")
        ws_btn.setObjectName("ghost")
        ws_btn.clicked.connect(self._browse_workspace)
        lay.addWidget(ws_btn)
        self._sessions_dialog = dlg

    def _open_sessions(self) -> None:
        self._fill_session_list()
        self._sessions_dialog.show()
        self._sessions_dialog.raise_()
        self._sessions_dialog.activateWindow()

    def _chat_pane(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_host = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_host)
        self.chat_layout.setContentsMargins(4, 2, 4, 2)
        self.chat_layout.setSpacing(4)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_host)
        lay.addWidget(self.chat_scroll, 1)

        self.round_label = QLabel("auto" if self.settings.autorun else "ready")
        self.round_label.setObjectName("Muted")
        self.round_label.setWordWrap(True)
        lay.addWidget(self.round_label)

        self.status_line = QLabel("ready")
        self.status_line.setObjectName("Muted")
        lay.addWidget(self.status_line)

        self.input = InputBox()
        self.input.setPlaceholderText("Message Grok  ·  Enter to send  ·  paste or image to attach")
        self.input.setFixedHeight(90)
        self.input.send.connect(self._send)
        self.input.attach.connect(self._on_input_attach)
        lay.addWidget(self.input)

        self.attach_strip = QWidget()
        self.attach_layout = QHBoxLayout(self.attach_strip)
        self.attach_layout.setContentsMargins(0, 0, 0, 0)
        self.attach_layout.setSpacing(8)
        self.attach_strip.hide()
        lay.addWidget(self.attach_strip)

        nav = QHBoxLayout()
        self.model_chip = QPushButton("model")
        self.model_chip.setObjectName("modelChip")
        self.model_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_chip.setToolTip("Choose Grok (xAI), Muse (Meta), or a local GPU model")
        self.model_chip.clicked.connect(self._open_model_menu)
        nav.addWidget(self.model_chip)
        self.attach_btn = QPushButton("image")
        self.attach_btn.setObjectName("ghost")
        self.attach_btn.setToolTip("Attach images to this message (or paste / drop)")
        self.attach_btn.clicked.connect(self._pick_images)
        nav.addWidget(self.attach_btn)
        nav.addStretch()
        self.autorun_toggle = QPushButton("auto")
        self.autorun_toggle.setObjectName("ghost")
        self.autorun_toggle.setCheckable(True)
        self.autorun_toggle.setChecked(self.settings.autorun)
        self.autorun_toggle.toggled.connect(self._on_autorun_toggled)
        self._style_autorun(self.settings.autorun)
        nav.addWidget(self.autorun_toggle)
        self.stop_btn = QPushButton("stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.send_btn = QPushButton("send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self._send)
        nav.addWidget(self.stop_btn)
        nav.addWidget(self.send_btn)
        lay.addLayout(nav)
        return wrap

    def _inspector(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Panel")
        wrap.setStyleSheet("QFrame#Panel { background-color: #0B0E14; border: none; border-radius: 0; }")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        lab = QLabel("context")
        lab.setObjectName("Section")
        lay.addWidget(lab)
        self.ctx_bar = QProgressBar()
        self.ctx_bar.setRange(0, 100)
        self.ctx_bar.setValue(0)
        lay.addWidget(self.ctx_bar)
        self.ctx_label = QLabel("tokens — / —  compact at —")
        self.ctx_label.setObjectName("Dim")
        self.ctx_label.setWordWrap(True)
        lay.addWidget(self.ctx_label)

        lab2 = QLabel("tools")
        lab2.setObjectName("Section")
        lay.addWidget(lab2)
        self.tool_log = QPlainTextEdit()
        self.tool_log.setReadOnly(True)
        self.tool_log.setPlaceholderText("Tool calls show up here. A chat link jumps to that entry.")
        lay.addWidget(self.tool_log, 1)

        effort_lab = QLabel("reasoning")
        effort_lab.setObjectName("Section")
        lay.addWidget(effort_lab)
        self.reason_combo = QComboBox()
        for name in EFFORTS:
            self.reason_combo.addItem(name)
        self.reason_combo.setCurrentIndex(EFFORTS.index(normalize_effort(self.settings.reasoning_effort)))
        self.reason_combo.currentIndexChanged.connect(self._on_reasoning_changed)
        lay.addWidget(self.reason_combo)

        self.agent_lab = QLabel("grok")
        self.agent_lab.setObjectName("Section")
        lay.addWidget(self.agent_lab)
        row = QHBoxLayout()
        row.setSpacing(8)
        col1 = QVBoxLayout()
        col1.setSpacing(4)
        col1.addWidget(self._lbl("temperature"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.2)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(self.settings.temperature)
        self.temp_spin.valueChanged.connect(lambda _: self._save_settings_from_ui())
        col1.addWidget(self.temp_spin)
        col2 = QVBoxLayout()
        col2.setSpacing(4)
        col2.addWidget(self._lbl("context"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setToolTip("0 uses a 128k window")
        self.ctx_spin.setRange(0, DEFAULT_CTX)
        self.ctx_spin.setSingleStep(8192)
        self.ctx_spin.setValue(self.settings.num_ctx)
        self.ctx_spin.valueChanged.connect(lambda _: self._save_settings_from_ui())
        col2.addWidget(self.ctx_spin)
        row.addLayout(col1)
        row.addLayout(col2)
        lay.addLayout(row)
        lay.addWidget(self._lbl("compact at"))
        self.compact_spin = QDoubleSpinBox()
        self.compact_spin.setRange(0.30, 0.90)
        self.compact_spin.setSingleStep(0.02)
        self.compact_spin.setDecimals(2)
        self.compact_spin.setValue(self.settings.compact_ratio)
        self.compact_spin.valueChanged.connect(lambda _: self._save_settings_from_ui())
        lay.addWidget(self.compact_spin)
        hint = QLabel("Compaction uses the model selected in the composer.")
        hint.setObjectName("Dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return wrap

    def _lbl(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("Dim")
        return l

    def _append_bubble(self, frame: QFrame) -> QFrame:
        stretch = self.chat_layout.count() - 1
        self.chat_layout.insertWidget(max(stretch, 0), frame)
        QTimer.singleShot(30, self._scroll_bottom)
        return frame

    def _scroll_bottom(self):
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _clear_chat(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._stream_bubble = None
        self._stream_text = ""
        self._tool_bubble = None
        self._clear_reasoning()

    def _clear_reasoning(self) -> None:
        self._anchors = {}
        self._reason_cards = {}
        self._latest_user = None
        layout = getattr(self, "reason_layout", None)
        if layout is None:
            return
        while layout.count() > 1:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _clear_tool_log(self) -> None:
        self.tool_log.clear()
        self._tool_marks = []
        self._tool_bubble = None

    def _append_tool_block(self, text: str) -> int:
        """Append one log entry. A blank line separates it from the previous entry."""
        return append_tool_block(self.tool_log, text)

    def _append_tool_note(self, line: str) -> None:
        """Extra line on the current log entry, such as a hands_on status."""
        text = (line or "").strip()
        if not text:
            return
        cursor = QTextCursor(self.tool_log.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n" + text)
        bar = self.tool_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _new_tool_feed_bubble(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ToolBubble")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 1, 8, 1)
        lay.setSpacing(0)
        header = _collapse_button("tools")
        body = QWidget()
        body.setAutoFillBackground(False)
        body.setStyleSheet("background: transparent; border: none;")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 4)
        body_lay.setSpacing(4)
        feed = ToolFeed(self._on_tool_link)
        body_lay.addWidget(feed)
        body.setVisible(False)
        lay.addWidget(header)
        lay.addWidget(body)

        def _toggle() -> None:
            open_ = not body.isVisible()
            body.setVisible(open_)
            header.setText("▾  tools" if open_ else "▸  tools")
            lay.setContentsMargins(8, 1, 8, 4 if open_ else 1)

        header.clicked.connect(_toggle)
        frame._feed = feed  # type: ignore[attr-defined]
        frame._links = []  # type: ignore[attr-defined]
        return frame

    def _grow_tool_feed(self, mark: int, label: str) -> None:
        """Add one comma-separated link to this turn's tool bubble."""
        if self._tool_bubble is None:
            self._tool_bubble = self._append_bubble(self._new_tool_feed_bubble())
        links = self._tool_bubble._links  # type: ignore[attr-defined]
        links.append((mark, label))
        self._tool_bubble._feed.set_links(tool_feed_html(links))  # type: ignore[attr-defined]
        self._scroll_bottom()

    def _record_tool(self, name: str, payload: str, *, result: bool) -> None:
        pos = self._append_tool_block(format_tool_log(name, payload, result=result))
        mark = len(self._tool_marks)
        self._tool_marks.append(pos)
        self._grow_tool_feed(mark, tool_entry_label(name, result=result))

    def _on_tool_link(self, href: str) -> None:
        text = str(href or "")
        if ":" in text:
            text = text.split(":")[-1]
        text = text.strip().strip("/")
        try:
            mark = int(text)
        except (TypeError, ValueError):
            return
        if mark < 0 or mark >= len(self._tool_marks):
            return
        scroll_tool_log_to(self.tool_log, self._tool_marks[mark])

    def _tool_payload(self, arguments, limit: int) -> str:
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False)
            except TypeError:
                arguments = str(arguments)
        return str(arguments or "")[:limit]

    def _render_history(self):
        self._clear_chat()
        self._clear_tool_log()
        for m in self.session.messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                continue
            if role == "user":
                self._append_tool_block("— turn —")
                frame = self._append_bubble(_bubble("user", "you", content, attachments=attachments_of(m)))
                self._latest_user = frame
            elif role == "assistant":
                if content.strip():
                    # Text breaks the tool run, so the next calls get a new bubble.
                    self._tool_bubble = None
                    self._append_bubble(_bubble("assistant", self._speaker(), content))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or tc
                    name = str(fn.get("name") or "tool")
                    self._record_tool(name, self._tool_payload(fn.get("arguments") or {}, 400), result=False)
            elif role == "tool":
                name = str(m.get("tool_name") or m.get("name") or "tool")
                self._record_tool(name, content[:800], result=True)
        self._update_ctx_bar()

    def _fill_session_list(self) -> None:
        self.session_list.blockSignals(True)
        try:
            self.session_list.clear()
            current = None
            for item in list_sessions():
                it = QListWidgetItem(item["title"])
                it.setData(Qt.ItemDataRole.UserRole, item["id"])
                self.session_list.addItem(it)
                if item["id"] == self.session.id:
                    current = it
            if current is not None:
                self.session_list.setCurrentItem(current)
        finally:
            self.session_list.blockSignals(False)

    def _refresh_sessions(self):
        self._fill_session_list()
        self._render_history()

    def _new_session(self):
        if self.worker and self.worker.isRunning():
            return
        self.session.save()
        self.session = new_session(self._current_model())
        remember_session(self.session.id)
        self._clear_tool_log()
        self._render_history()
        self._fill_session_list()
        self.status_line.setText("new session.")
        self.effective_ctx = 0
        self.last_prompt_tokens = 0
        self._live_used_tokens = 0
        reset_calibration()
        self._update_ctx_bar()

    def _delete_session(self, item=None):
        if self.worker and self.worker.isRunning():
            return
        if not isinstance(item, QListWidgetItem):
            item = self.session_list.currentItem()
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        was_open = self.session.id == sid
        delete_session(sid)
        if was_open:
            nxt = None
            for entry in list_sessions():
                loaded = load_session(entry["id"])
                if loaded:
                    nxt = loaded
                    break
            self.session = nxt or new_session(self._current_model())
            self._clear_tool_log()
            self._render_history()
        remember_session(self.session.id)
        self._fill_session_list()

    def _on_session_clicked(self, item: QListWidgetItem):
        if self.worker and self.worker.isRunning():
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            QMessageBox.warning(self, "Session Load", f"Failed to load session {sid}")
            return
        self.session.save()
        self.session = loaded
        self._render_history()

    def _on_session_selected(self, row: int):
        """Switch the open chat when the sessions list changes."""
        if self.worker and self.worker.isRunning():
            QTimer.singleShot(0, self._fill_session_list)
            return
        item = self.session_list.item(row)
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid == self.session.id:
            return
        loaded = load_session(sid)
        if not loaded:
            QMessageBox.warning(self, "session", f"Could not open {sid}")
            self._fill_session_list()
            return
        self.session.save()
        self.session = loaded
        remember_session(self.session.id)
        self._render_history()

    def _rename_session(self, item: QListWidgetItem):
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            return
        title, ok = QInputDialog.getText(self, "rename chat", "name", text=loaded.title or "")
        if not ok:
            return
        title = " ".join((title or "").split()) or "new session"
        loaded.title = title
        loaded.save()
        if self.session.id == loaded.id:
            self.session.title = title
        self._fill_session_list()

    def _show_session_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            return
        menu = QMenu(self)
        rename_btn = menu.addAction("rename")
        del_action = menu.addAction("delete")
        menu.addSeparator()
        refresh_action = menu.addAction("refresh list")

        def on_rename():
            self._rename_session(item)

        def on_delete():
            self._delete_session(item)

        def on_refresh():
            self._fill_session_list()
        
        rename_btn.triggered.connect(on_rename)
        del_action.triggered.connect(on_delete)
        refresh_action.triggered.connect(on_refresh)
        menu.exec(self.session_list.mapToGlobal(pos))

    def _current_model(self) -> str:
        return self.settings.active_model()

    def _speaker(self) -> str:
        if self.settings.provider == "xai":
            return "grok"
        if self.settings.provider == "meta":
            return "muse"
        return self._current_model() or "local"

    def _refresh_model_chip(self) -> None:
        provider = self.settings.provider
        if provider == "meta":
            label = f"{self.settings.meta_model or META_MODEL}  ·  Meta"
            section = "muse"
            placeholder = "Message Muse  ·  Enter to send  ·  paste or image to attach"
        elif provider == "xai":
            label = f"{self.settings.xai_model or DEFAULT_MODEL}  ·  xAI"
            section = "grok"
            placeholder = "Message Grok  ·  Enter to send  ·  paste or image to attach"
        else:
            label = f"{self.settings.model or 'local'}  ·  GPU"
            section = self.settings.model or "local"
            placeholder = "Message the local model  ·  Enter to send  ·  paste or image to attach"
        self.model_chip.setText(label)
        self.input.setPlaceholderText(placeholder)
        if getattr(self, "agent_lab", None) is not None:
            self.agent_lab.setText(section)
        self._update_ctx_bar()

    def _set_provider_model(self, provider: str, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        self.settings.provider = provider
        if provider == "xai":
            self.settings.xai_model = name
        elif provider == "meta":
            self.settings.meta_model = name
        else:
            self.settings.model = name
            if name not in self._local_models:
                self._local_models.append(name)
        save_settings(self.settings)
        self.session.model = name
        reset_calibration()
        self.effective_ctx = 0
        self._refresh_model_chip()
        self._refresh_ctx_for_model()

    def _open_model_menu(self) -> None:
        menu = QMenu(self)
        xai_menu = menu.addMenu("xAI  ·  Grok")
        names = list(self._xai_models or [DEFAULT_MODEL])
        for name in names:
            act = xai_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(self.settings.provider == "xai" and self.settings.xai_model == name)
            act.triggered.connect(lambda _=False, n=name: self._set_provider_model("xai", n))
        if not self.settings.has_xai_key():
            hint = xai_menu.addAction("add api key in settings")
            hint.triggered.connect(self._open_settings)
        meta_menu = menu.addMenu("Meta  ·  Muse")
        meta_names: list[str] = []
        for name in list(self._meta_models or [META_MODEL, "muse-wonder-1.0", "muse-spark-1.2"]):
            if name not in meta_names:
                meta_names.append(name)
        for name in meta_names:
            act = meta_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(self.settings.provider == "meta" and self.settings.meta_model == name)
            act.triggered.connect(lambda _=False, n=name: self._set_provider_model("meta", n))
        if not self.settings.has_meta_key():
            hint = meta_menu.addAction("add api key in settings")
            hint.triggered.connect(self._open_settings)
        local_menu = menu.addMenu("Ollama  ·  local GPU")
        local_names = list(self._local_models)
        if not local_names:
            empty = local_menu.addAction("No local models installed in Ollama")
            empty.setEnabled(False)
        for name in local_names:
            act = local_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(self.settings.provider == "ollama" and self.settings.model == name)
            act.triggered.connect(lambda _=False, n=name: self._set_provider_model("ollama", n))
        menu.exec(self.model_chip.mapToGlobal(self.model_chip.rect().bottomLeft()))

    def _refresh_ctx_for_model(self):
        model = self._current_model()
        if not model:
            return
        if getattr(self, "_ctx_model", None) != model or getattr(self, "_ctx_provider", None) != self.settings.provider:
            reset_calibration()
            self.effective_ctx = 0
            self._ctx_model = model
            self._ctx_provider = self.settings.provider
        self.native_ctx = self.settings.num_ctx or DEFAULT_CTX
        self.session.native_ctx = self.native_ctx
        self._update_ctx_bar()

    def _update_ctx_bar(self):
        if self.worker and self.worker.isRunning() and self._live_used_tokens:
            used = self._live_used_tokens
        else:
            used = estimate_tokens(self.session.messages)
        self.session.token_estimate = used
        if self.worker and self.worker.agent and self.worker.agent.effective_ctx:
            ctx = int(self.worker.agent.effective_ctx)
        elif self.effective_ctx:
            ctx = int(self.effective_ctx)
        else:
            # num_ctx == 0 means auto → native
            ctx = self.settings.num_ctx or self.native_ctx or 8192
        thresh = compact_threshold(ctx, self.settings.compact_ratio)
        pct = int(min(100, 100 * used / max(ctx, 1)))
        self.ctx_bar.setValue(pct)
        real = ""
        if self.last_prompt_tokens:
            real = f"   real {self.last_prompt_tokens:,}"
        who = {"xai": "grok", "meta": "muse"}.get(self.settings.provider, "local")
        self.ctx_label.setText(
            f"{used:,} / {ctx:,} tok   {who}   compact ≥ {thresh:,}{real}"
        )

    def _reload_models(self):
        try:
            models = list_models(self.settings.ollama_host)
        except Exception as e:
            self.status_line.setText(f"Ollama model list failed: {e}")
            models = []
        self._local_models = [m["name"] for m in models]
        if not self.settings.model and self._local_models:
            self.settings.model = self._local_models[0]
            save_settings(self.settings)
        if self._current_model():
            self._refresh_model_chip()
            self._refresh_ctx_for_model()

    def _paint_btn_state(self, btn: QPushButton, state: str) -> None:
        btn.setProperty("state", state)
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _tick_ollama(self):
        self._ollama_ok = ping(self.settings.ollama_host)
        key = self.settings.xai_key() if self.settings.has_xai_key() else ""
        if key:
            self._xai_ok, self._xai_detail = account_status(key)
        else:
            self._xai_ok, self._xai_detail = False, "no API key"
        meta_key = self.settings.meta_key() if self.settings.has_meta_key() else ""
        if meta_key:
            self._meta_ok, self._meta_detail = meta_status(meta_key)
        else:
            self._meta_ok, self._meta_detail = False, "no API key"
        provider = self.settings.provider
        if provider == "meta":
            state = "online" if self._meta_ok else ("partial" if meta_key else "offline")
        elif provider == "ollama":
            state = "online" if self._ollama_ok else "offline"
        else:
            state = "online" if self._xai_ok else ("partial" if key else "offline")
        if not (self.worker and self.worker.isRunning()):
            self.provider_btn.setText("provider")
            self._paint_btn_state(self.provider_btn, state)
        if key and not self._xai_models:
            try:
                self._xai_models = xai_list_models(key)
            except Exception:
                self._xai_models = []
        if meta_key and not self._meta_models:
            try:
                self._meta_models = meta_list_models(meta_key)
            except Exception:
                self._meta_models = []

    def _browse_workspace(self):
        path = QFileDialog.getExistingDirectory(
            self, "workspace folder", str(self.settings.resolved_workspace())
        )
        if not path:
            return
        self.settings.workspace = path
        save_settings(self.settings)
        seed_workspace(Path(path))
        self.ws_label.setText(path)

    def _on_reasoning_changed(self, index: int) -> None:
        if index < 0 or index >= len(EFFORTS):
            return
        self.settings.reasoning_effort = EFFORTS[index]
        save_settings(self.settings)

    def _jump_message(self, anchor: str) -> None:
        widget = self._anchors.get(anchor) or self._latest_user
        if widget is None:
            return
        self.chat_scroll.ensureWidgetVisible(widget, 20, 40)

    def _save_settings_from_ui(self):
        self.settings.temperature = float(self.temp_spin.value())
        self.settings.num_ctx = int(self.ctx_spin.value())
        self.settings.compact_ratio = float(self.compact_spin.value())
        save_settings(self.settings)
        self._refresh_ctx_for_model()

    def _set_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.attach_btn.setEnabled(not busy)
        self.input.setReadOnly(busy)
        self.session_list.setEnabled(not busy)
        if busy:
            self.provider_btn.setText("working")
            self._paint_btn_state(self.provider_btn, "partial")
        else:
            self.provider_btn.setText("provider")

    def _send(self):
        text = self.input.toPlainText().strip()
        atts = list(self._pending)
        if not text and not atts:
            return
        self.input.clear()
        self._pending.clear()
        self._refresh_attach_strip()
        self._auto_hops = 0
        self._dispatch(text, attachments=atts)

    def _on_input_attach(self, payload) -> None:
        if isinstance(payload, list):
            for p in payload:
                self._add_path(str(p))
            return
        self._add_qimage(payload, "paste.png")

    def _pick_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach images",
            str(self.settings.resolved_workspace()),
            "Images (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;All files (*)",
        )
        for f in files:
            self._add_path(f)

    def _add_path(self, path: str) -> None:
        if not is_image_path(path):
            self.status_line.setText(f"Not an image: {path}")
            return
        img = QImage(path)
        if img.isNull():
            self.status_line.setText(f"Could not read image: {path}")
            return
        self._add_qimage(img, Path(path).name)

    def _add_qimage(self, image, name: str) -> None:
        if len(self._pending) >= MAX_ATTACH:
            self.status_line.setText(f"Max {MAX_ATTACH} images per message")
            return
        try:
            data, mime, _ext = encode_qimage(image)
            att = save_encoded(data, mime, name)
        except Exception as e:
            self.status_line.setText(f"Could not attach image: {e}")
            return
        self._pending.append(att)
        self._refresh_attach_strip()
        self.status_line.setText(f"Attached {att.get('name') or 'image'} ({len(self._pending)}/{MAX_ATTACH})")

    def _remove_pending(self, idx: int) -> None:
        if 0 <= idx < len(self._pending):
            att = self._pending.pop(idx)
            try:
                Path(att["path"]).unlink(missing_ok=True)
            except Exception:
                pass
        self._refresh_attach_strip()

    def _refresh_attach_strip(self) -> None:
        while self.attach_layout.count():
            item = self.attach_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if not self._pending:
            self.attach_strip.hide()
            return
        for i, att in enumerate(self._pending):
            chip = QFrame()
            chip.setObjectName("ToolBubble")
            h = QHBoxLayout(chip)
            h.setContentsMargins(6, 4, 6, 4)
            h.setSpacing(6)
            pix = QPixmap(att["path"])
            thumb = QLabel()
            if not pix.isNull():
                thumb.setPixmap(
                    pix.scaled(
                        48,
                        48,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            h.addWidget(thumb)
            name = QLabel(att.get("name") or "image")
            name.setObjectName("Dim")
            h.addWidget(name)
            rm = QPushButton("×")
            rm.setObjectName("ghost")
            rm.setFixedWidth(28)
            rm.clicked.connect(lambda _checked=False, idx=i: self._remove_pending(idx))
            h.addWidget(rm)
            self.attach_layout.addWidget(chip)
        self.attach_layout.addStretch()
        self.attach_strip.show()

    def _start_worker(self, model: str) -> None:
        self._set_busy(True)
        self.worker = AgentWorker(self.settings, model, list(self.session.messages))
        self.worker.event.connect(self._on_agent_event)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _dispatch(self, text: str, attachments: list | None = None) -> bool:
        if self.worker and self.worker.isRunning():
            return False
        model = self._current_model()
        if not model:
            QMessageBox.information(
                self,
                "No model",
                "Pick a model under the composer.",
            )
            return False
        if self.settings.provider == "xai":
            if not self.settings.has_xai_key():
                QMessageBox.warning(
                    self,
                    "xAI key missing",
                    "Open settings and save an xAI API key. It is encrypted with Windows DPAPI.",
                )
                return False
            ok, detail = account_status(self.settings.xai_key())
            self._xai_ok, self._xai_detail = ok, detail
            if not ok:
                QMessageBox.warning(
                    self,
                    "xAI",
                    f"api.x.ai answered, but this key cannot run models.\n{detail}\n\n"
                    "If that says credits or spending limit, add credits or raise the cap in the xAI console.",
                )
                return False
        elif self.settings.provider == "meta":
            if not self.settings.has_meta_key():
                QMessageBox.warning(
                    self,
                    "Meta key missing",
                    "Open settings and save a Meta API key. It is encrypted with Windows DPAPI.",
                )
                return False
            ok, detail = meta_status(self.settings.meta_key())
            self._meta_ok, self._meta_detail = ok, detail
            if not ok:
                QMessageBox.warning(
                    self,
                    "Meta",
                    f"api.meta.ai answered, but this key cannot run models.\n{detail}",
                )
                return False
        elif not ping(self.settings.ollama_host):
            QMessageBox.warning(
                self,
                "Ollama offline",
                f"Nothing is listening at {self.settings.ollama_host}.\nStart Ollama and try again.",
            )
            return False
        self._user_stopped = False
        self._auto_resume = False
        self._sending_auto = False
        title_src = text or (
            (attachments[0].get("name") if attachments else "") or "image"
        )
        self.session.touch_title(title_src)
        msg: dict = {"role": "user", "content": text}
        if attachments:
            msg["attachments"] = list(attachments)
        self.session.messages.append(msg)
        self.session.model = model
        self.session.save()
        self._latest_user = self._append_bubble(_bubble("user", "you", text, attachments=attachments))
        self._stream_bubble = None
        self._stream_text = ""
        # Keep the open tool bubble. A new one starts only after assistant text.
        self._append_tool_block("— turn —")
        self.status_line.setText("Running…")
        if self.settings.autorun:
            self.round_label.setText("auto")
        self._start_worker(model)
        return True

    def _kick_auto(self):
        if not self.settings.autorun or self._user_stopped:
            return
        if self.worker and self.worker.isRunning():
            return
        model = self._current_model()
        if not model:
            return
        self._auto_hops += 1
        if self._auto_hops > AUTO_HOP_CAP:
            self.status_line.setText(
                f"Auto stopped after {AUTO_HOP_CAP} continues. Send a message to keep going."
            )
            self.round_label.setText("auto · capped")
            return
        self._auto_resume = False
        self._sending_auto = True
        self.status_line.setText("Auto continuing…")
        self.round_label.setText(f"auto · hop {self._auto_hops}")
        self._start_worker(model)

    def _stop(self):
        self._user_stopped = True
        self._auto_resume = False
        if self.worker:
            self.worker.cancel()
            self.status_line.setText("Stopping…")

    def _on_agent_event(self, kind: str, data: dict):
        if kind == "token":
            piece = data.get("text") or ""
            anchor = str(data.get("anchor") or "")
            self._stream_text += piece
            if self._stream_bubble is None:
                # Text breaks the tool run, so the next calls get a new bubble.
                self._tool_bubble = None
                self._stream_bubble = self._append_bubble(
                    _bubble("assistant", self._speaker(), self._stream_text)
                )
            else:
                self._stream_bubble._body.setText(self._stream_text)  # type: ignore[attr-defined]
            if anchor and self._stream_bubble is not None:
                self._anchors[anchor] = self._stream_bubble
            self._scroll_bottom()
        elif kind == "reasoning":
            call = str(data.get("call") or "")
            anchor = str(data.get("anchor") or "")
            label = str(data.get("label") or "REASONING")
            body = str(data.get("text") or "")
            if call and body:
                card = self._reason_cards.get(call)
                if card is None:
                    card = ReasoningCard(label, lambda a=anchor: self._jump_message(a))
                    self._reason_cards[call] = card
                    stretch = self.reason_layout.count() - 1
                    self.reason_layout.insertWidget(max(stretch, 0), card)
                card.set_text(body)
                bar = self.reason_scroll.verticalScrollBar()
                bar.setValue(bar.maximum())
        elif kind == "tool_start":
            # This round's text is already in its bubble. The next round starts empty.
            self._stream_bubble = None
            self._stream_text = ""
            name = data.get("name") or "tool"
            args = data.get("arguments") or {}
            preview = json.dumps(args, ensure_ascii=False)[:400]
            self._record_tool(name, preview, result=False)
            self.status_line.setText(f"tool: {name}")
        elif kind == "hands":
            line = data.get("text") or ""
            if line:
                self._append_tool_note(line)
        elif kind == "hands_window":
            if data.get("hide"):
                self.showMinimized()
                # A click from the previous run can set this after that run already hid.
                if self.worker:
                    self.worker.hands_end.clear()
            else:
                self._hide_end_rounds()
                self.showNormal()
                self.raise_()
                self.activateWindow()
            if self.worker:
                self.worker.hands_gate.set()
        elif kind == "hands_badge":
            try:
                self._place_end_rounds(data)
            finally:
                if self.worker:
                    self.worker.hands_gate.set()
        elif kind == "tool_end":
            name = data.get("name") or "tool"
            output = data.get("output") or ""
            self._record_tool(name, output[:800], result=True)
        elif kind == "compact":
            self.status_line.setText(data.get("text") or "compacting…")
            self._append_tool_block(f"compact: {data.get('text') or ''}")
            # session.messages is stale mid-turn; use the event estimate until the worker finishes.
            if data.get("estimate") is not None:
                self._live_used_tokens = int(data["estimate"])
            self._update_ctx_bar()
        elif kind == "status":
            self.status_line.setText(data.get("text") or "")
            rnd = data.get("round")
            mx = data.get("max_rounds")
            if rnd and mx:
                prefix = "auto · " if self.settings.autorun else ""
                self.round_label.setText(f"{prefix}{rnd}/{mx}")
        elif kind == "tokens":
            # P2-1: real prompt tokens from Ollama, after calibration.
            used = int(data.get("used") or 0)
            if used > 0:
                self.last_prompt_tokens = used
                self._live_used_tokens = used
                self._update_ctx_bar()
        elif kind == "error":
            self._auto_resume = False
            self.status_line.setText(data.get("text") or "error")
            self._append_bubble(_bubble("tool", "ERROR", data.get("text") or "", mono=True))
        elif kind == "done":
            complete = bool(data.get("complete", True))
            self._auto_resume = (
                (not complete) and self.settings.autorun and not self._user_stopped
            )
            if complete:
                self.status_line.setText("Done.")
                self.round_label.setText("auto · done" if self.settings.autorun else "done")
            else:
                self.status_line.setText("Round cap hit.")
            if data.get("num_ctx"):
                self.effective_ctx = int(data["num_ctx"])
            self._update_ctx_bar()

    def _on_worker_finished(self):
        if self.worker:
            self.session.messages = self.worker.result_messages
            self.session.token_estimate = estimate_tokens(self.session.messages)
            self.session.native_ctx = self.native_ctx
            agent = self.worker.agent
            if agent and agent.effective_ctx:
                self.effective_ctx = int(agent.effective_ctx)
            self.session.save()
        self._live_used_tokens = 0
        resume = self._auto_resume and self.settings.autorun and not self._user_stopped
        self._auto_resume = False
        self._set_busy(False)
        self._tick_ollama()
        self._update_ctx_bar()
        if resume:
            self.status_line.setText("Auto: continuing…")
            self.round_label.setText(f"auto · hop {self._auto_hops + 1}")
            QTimer.singleShot(250, self._kick_auto)

    def _style_autorun(self, on: bool):
        if on:
            self.autorun_toggle.setStyleSheet(
                f"QPushButton {{ color: {TEXT_BRIGHT}; background-color: {ACCENT};"
                f" border: 1px solid {ACCENT_HOVER}; border-radius: 6px;"
                " padding: 6px 12px; font-size: 12px; min-height: 16px; }}"
            )
            self.round_label.setText("auto")
        else:
            self.autorun_toggle.setStyleSheet("")
            self.round_label.setText("ready")

    def _on_autorun_toggled(self, checked: bool):
        self.settings.autorun = checked
        save_settings(self.settings)
        self._style_autorun(checked)
        running = bool(self.worker and self.worker.isRunning())
        if checked and not running and self.session.messages:
            last = self.session.messages[-1]
            last_text = str(last.get("content") or "")
            stalled = last.get("role") in ("assistant", "tool") or "max tool rounds" in last_text.lower()
            if stalled:
                self._user_stopped = False
                self._kick_auto()
        elif not checked:
            self._auto_resume = False

    def _open_settings(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("sm0l — settings")
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        title = QLabel("settings")
        title.setObjectName("Brand")
        lay.addWidget(title)
        note = QLabel(
            "xAI and Meta keys are encrypted with Windows DPAPI for this user. "
            "They are not written to config as plaintext."
        )
        note.setObjectName("Muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        xai_state = QLabel("A key is saved." if self.settings.has_xai_key() else "No key saved.")
        xai_state.setObjectName("Dim")
        lay.addWidget(xai_state)
        xai_edit = QLineEdit()
        xai_edit.setEchoMode(QLineEdit.EchoMode.Password)
        xai_edit.setPlaceholderText("Paste xAI API key")
        lay.addWidget(xai_edit)
        meta_state = QLabel("A key is saved." if self.settings.has_meta_key() else "No key saved.")
        meta_state.setObjectName("Dim")
        lay.addWidget(meta_state)
        meta_edit = QLineEdit()
        meta_edit.setEchoMode(QLineEdit.EchoMode.Password)
        meta_edit.setPlaceholderText("Paste Meta API key")
        lay.addWidget(meta_edit)
        host_lab = QLabel("ollama host")
        host_lab.setObjectName("Dim")
        lay.addWidget(host_lab)
        host_edit = QLineEdit(self.settings.ollama_host)
        lay.addWidget(host_edit)
        row = QHBoxLayout()
        save_btn = QPushButton("save")
        save_btn.setObjectName("primary")
        clear_xai_btn = QPushButton("clear xai")
        clear_xai_btn.setObjectName("headerBtn")
        clear_meta_btn = QPushButton("clear meta")
        clear_meta_btn.setObjectName("headerBtn")
        close_btn = QPushButton("close")
        close_btn.setObjectName("headerBtn")
        row.addWidget(save_btn)
        row.addWidget(clear_xai_btn)
        row.addWidget(clear_meta_btn)
        row.addStretch()
        row.addWidget(close_btn)
        lay.addLayout(row)

        def _refuse_plain(blob: str, plain: str) -> bool:
            return bool(plain) and plain in (blob or "")

        def save():
            xai_plain = xai_edit.text().strip()
            meta_plain = meta_edit.text().strip()
            self.settings.ollama_host = host_edit.text().strip() or "http://127.0.0.1:11434"
            if xai_plain:
                try:
                    self.settings.set_xai_key(xai_plain)
                except OSError as e:
                    QMessageBox.warning(dlg, "DPAPI failed", str(e))
                    return
                xai_edit.clear()
                xai_state.setText("A key is saved.")
            if meta_plain:
                try:
                    self.settings.set_meta_key(meta_plain)
                except OSError as e:
                    QMessageBox.warning(dlg, "DPAPI failed", str(e))
                    return
                meta_edit.clear()
                meta_state.setText("A key is saved.")
            save_settings(self.settings)
            if _refuse_plain(self.settings.xai_key_dpapi, xai_plain):
                QMessageBox.warning(dlg, "Key not protected", "Refusing to store a plaintext key.")
                self.settings.xai_key_dpapi = ""
                save_settings(self.settings)
                return
            if _refuse_plain(self.settings.meta_key_dpapi, meta_plain):
                QMessageBox.warning(dlg, "Key not protected", "Refusing to store a plaintext key.")
                self.settings.meta_key_dpapi = ""
                save_settings(self.settings)
                return
            self._xai_models = []
            self._meta_models = []
            self._tick_ollama()
            if xai_plain or meta_plain:
                self.status_line.setText("Settings saved. Keys stored with Windows DPAPI.")
            else:
                self.status_line.setText("Settings saved.")
            dlg.accept()

        def clear_xai():
            self.settings.set_xai_key("")
            save_settings(self.settings)
            xai_edit.clear()
            xai_state.setText("No key saved.")
            self._xai_ok = False
            self._xai_detail = "no API key"
            self._xai_models = []
            self._tick_ollama()

        def clear_meta():
            self.settings.set_meta_key("")
            save_settings(self.settings)
            meta_edit.clear()
            meta_state.setText("No key saved.")
            self._meta_ok = False
            self._meta_detail = "no API key"
            self._meta_models = []
            self._tick_ollama()

        save_btn.clicked.connect(save)
        clear_xai_btn.clicked.connect(clear_xai)
        clear_meta_btn.clicked.connect(clear_meta)
        close_btn.clicked.connect(dlg.reject)
        dlg.exec()

    def _show_providers(self) -> None:
        self._tick_ollama()
        dlg = QDialog(self)
        dlg.setWindowTitle("sm0l — providers")
        dlg.setMinimumWidth(420)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        title = QLabel("provider")
        title.setObjectName("Brand")
        lay.addWidget(title)

        def row(name: str, ok: bool, detail: str) -> QFrame:
            frame = QFrame()
            frame.setObjectName("Panel")
            h = QHBoxLayout(frame)
            h.setContentsMargins(12, 10, 12, 10)
            dot = QLabel("●")
            color = "#8fd4a8" if ok else "#e09098"
            dot.setStyleSheet(f"color: {color}; font-size: 16px;")
            lab = QLabel(name)
            lab.setStyleSheet("font-weight: 600;")
            sub = QLabel(detail)
            sub.setObjectName("Dim")
            h.addWidget(dot)
            h.addWidget(lab)
            h.addStretch()
            h.addWidget(sub)
            return frame

        xai_detail = self._xai_detail if self.settings.has_xai_key() else "no API key"
        meta_detail = self._meta_detail if self.settings.has_meta_key() else "no API key"
        oll_detail = "connected" if self._ollama_ok else "offline"
        lay.addWidget(row("xAI", self._xai_ok, xai_detail))
        lay.addWidget(row("Meta", self._meta_ok, meta_detail))
        lay.addWidget(row("Ollama", self._ollama_ok, oll_detail))
        selected = self.settings.provider if self.settings.provider in ("xai", "meta", "ollama") else "xai"
        selected_ok = {"xai": self._xai_ok, "meta": self._meta_ok, "ollama": self._ollama_ok}[selected]
        selected_name = {"xai": "xAI", "meta": "Meta", "ollama": "Ollama"}[selected]
        both = QLabel(
            f"{selected_name} connected" if selected_ok else "waiting on the selected provider"
        )
        both.setStyleSheet(
            "color: #8fd4a8; font-weight: 600;" if selected_ok else "color: #8b8c9a;"
        )
        lay.addWidget(both)
        close_btn = QPushButton("close")
        close_btn.setObjectName("headerBtn")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _on_donate_clicked(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("sm0l — tip jar")
        dlg.setMinimumWidth(480)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        title = QLabel("pls donate, im poor")
        title.setObjectName("Brand")
        lay.addWidget(title)
        sub = QLabel("single dad btw...")
        sub.setObjectName("Muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        cash = QFrame()
        cash.setObjectName("Panel")
        cl = QVBoxLayout(cash)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.addWidget(self._donate_section_title("cash app"))
        tag = QLabel(f"Cashtag: {CASHAPP_TAG}")
        tag.setObjectName("Dim")
        cl.addWidget(tag)
        crow = QHBoxLayout()
        open_cash = QPushButton(f"open {CASHAPP_TAG}")
        open_cash.setObjectName("primary")
        open_cash.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(CASHAPP_URL))
        )
        crow.addWidget(open_cash)
        copy_cash = QPushButton("copy cashtag")
        copy_cash.setObjectName("ghost")
        copy_cash.clicked.connect(
            lambda: (
                QGuiApplication.clipboard().setText(CASHAPP_TAG),
                self.status_line.setText(f"Copied Cash App {CASHAPP_TAG}"),
            )
        )
        crow.addWidget(copy_cash)
        crow.addStretch()
        cl.addLayout(crow)
        lay.addWidget(cash)

        btc = QFrame()
        btc.setObjectName("Panel")
        bl = QVBoxLayout(btc)
        bl.setContentsMargins(14, 12, 14, 12)
        bl.addWidget(self._donate_section_title("bitcoin"))
        bl.addWidget(
            QLabel(
                "Send BTC to this address from any wallet "
                "(Electrum, BlueWallet, Sparrow, mobile apps, exchange withdraw):"
            )
        )
        addr = QLineEdit(BTC_ADDRESS)
        addr.setReadOnly(True)
        addr.setMinimumHeight(34)
        bl.addWidget(addr)
        brow = QHBoxLayout()
        copy_btc = QPushButton("copy address")
        copy_btc.setObjectName("primary")
        copy_btc.clicked.connect(
            lambda: (
                QGuiApplication.clipboard().setText(BTC_ADDRESS),
                self.status_line.setText("Copied BTC address to clipboard"),
            )
        )
        brow.addWidget(copy_btc)
        brow.addStretch()
        bl.addLayout(brow)
        lay.addWidget(btc)

        close = QPushButton("close")
        close.setObjectName("ghost")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _donate_section_title(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("Section")
        return lab

    def _end_rounds_clicked(self):
        if self.worker:
            self.worker.hands_end.set()

    def _place_end_rounds(self, data: dict):
        if self._end_rounds is None:
            self._end_rounds = EndToolRounds(self._end_rounds_clicked)
        rect = sync_end_button(
            self._end_rounds,
            int(data.get("left") or 0),
            int(data.get("top") or 0),
            int(data.get("width") or 248),
            int(data.get("height") or 48),
        )
        if self.worker:
            self.worker.hands_button = rect

    def _hide_end_rounds(self):
        if self._end_rounds is not None:
            self._end_rounds.hide()

    def closeEvent(self, event):
        self._hide_end_rounds()
        try:
            self.session.save()
            self._save_settings_from_ui()
            remember_session(self.session.id)
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1500)
        super().closeEvent(event)
