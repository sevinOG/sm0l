"""Windows dashboard — chat, sessions, context, tools. No CLI required."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
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
    QVBoxLayout,
    QWidget,
    QSplitter,
    QScrollArea,
    QSpinBox,
)

CASHAPP_TAG = "$reshi7"
CASHAPP_URL = "https://cash.app/$reshi7"
BTC_ADDRESS = "bc1qp989v95u54zpnmw9j75azwp9hrqnd0k6d7jp3lvv6z3yywpfdutszkkhg6"

from .agent import Agent
from .compact import compact_threshold, estimate_tokens
from .config import Settings, save_settings
from .ollama_client import list_models, native_context_length, ping, pull_model
from .paths import user_data
from .personality import seed_workspace
from .session import Session, delete_session, list_sessions, load_session, new_session
from .theme import TEXT, TEXT_MUTED, TEXT_DIM

AUTO_CONTINUE = (
    "AUTOMODE ON PLEASE CONTINUE\n\n"
    "The tool-round cap was hit. Keep working the same task. "
    "Use tools. When the task is fully done, stop calling tools and give a brief final answer."
)


class AgentWorker(QThread):
    event = pyqtSignal(str, dict)

    def __init__(self, settings: Settings, model: str, messages: list[dict]):
        super().__init__()
        self.settings = settings
        self.model = model
        self.messages = messages
        self.agent: Agent | None = None
        self.result_messages: list[dict] = []

    def run(self):
        def emit(kind: str, data: dict):
            self.event.emit(kind, data)

        self.agent = Agent(
            self.settings.ollama_host,
            self.model,
            self.settings.resolved_workspace(),
            num_ctx_override=self.settings.num_ctx,
            temperature=self.settings.temperature,
            compact_ratio=self.settings.compact_ratio,
            max_tool_rounds=self.settings.max_tool_rounds,
            shell_timeout=self.settings.shell_timeout,
            emit=emit,
        )
        try:
            self.result_messages = self.agent.run(self.messages)
        except Exception as e:
            self.event.emit("error", {"text": str(e)})
            self.result_messages = self.messages

    def cancel(self):
        if self.agent:
            self.agent.cancel()
            # Wait for agent to actually stop, with hard timeout
            import time
            start = time.monotonic()
            while time.monotonic() - start < 5:
                if self.agent._cancel or self.agent.last_prompt_tokens == 0:
                    break
                time.sleep(0.1)


class PullWorker(QThread):
    status = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, host: str, name: str):
        super().__init__()
        self.host = host
        self.name = name

    def run(self):
        try:
            pull_model(self.host, self.name, on_status=self.status.emit)
            self.done.emit(self.name)
        except Exception as e:
            self.failed.emit(str(e))


class InputBox(QPlainTextEdit):
    send = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.send.emit()
            return
        super().keyPressEvent(event)


def _bubble(kind: str, title: str, body: str, mono: bool = False) -> QFrame:
    frame = QFrame()
    frame.setObjectName(
        {"user": "UserBubble", "assistant": "AssistantBubble"}.get(kind, "ToolBubble")
    )
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(4)
    cap = QLabel(title)
    cap.setObjectName("Dim")
    cap.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; letter-spacing: 1px;")
    body_lbl = QLabel(body)
    body_lbl.setTextFormat(Qt.TextFormat.PlainText)
    body_lbl.setWordWrap(True)
    body_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if mono:
        body_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-family: Consolas, Cascadia Mono, monospace; font-size: 12px;"
        )
    else:
        color = TEXT if kind != "user" else TEXT
        body_lbl.setStyleSheet(f"color: {color}; font-size: 13px;")
    lay.addWidget(cap)
    lay.addWidget(body_lbl)
    frame._body = body_lbl  # type: ignore[attr-defined]
    frame._raw = body  # type: ignore[attr-defined]
    return frame


class Dashboard(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        # Always start with a fresh session — no last-session restore
        self.session = new_session(settings.model)
        # Optionally delete old last_session.json so it doesn't come back
        last = user_data() / "last_session.json"
        if last.is_file():
            try:
                last.unlink()
            except Exception:
                pass
        self.worker: AgentWorker | None = None
        self.puller: PullWorker | None = None
        self._stream_bubble: QFrame | None = None
        self._stream_text = ""
        self.native_ctx = 0
        self._auto_resume = False
        self._user_stopped = False
        self._auto_hops = 0
        self._sending_auto = False
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
        header.setStyleSheet("background-color: #1c1d24; border-bottom: 1px solid #2a2b34;")
        h = QHBoxLayout(header)
        h.setContentsMargins(22, 12, 22, 12)
        h.setSpacing(16)
        brand_col = QVBoxLayout()
        brand_col.setSpacing(2)
        brand = QLabel("SM0L")
        brand.setObjectName("Brand")
        sub = QLabel("LOCAL AGENT  ·  OLLAMA  ·  5–8B")
        sub.setObjectName("BrandSub")
        brand_col.addWidget(brand)
        brand_col.addWidget(sub)
        h.addLayout(brand_col)
        h.addStretch()

        self.donate_button = QPushButton("pls donate, im poor")
        self.donate_button.setObjectName("donate")
        self.donate_button.setToolTip("Open the tip jar (optional, no pressure)")
        self.donate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.donate_button.clicked.connect(self._on_donate_clicked)
        h.addWidget(self.donate_button)

        self.status_dot = QLabel("ollama ?")
        self.status_dot.setObjectName("StatusDot")
        self.status_dot.setProperty("state", "offline")
        h.addWidget(self.status_dot)
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

    def _sidebar(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Panel")
        wrap.setStyleSheet("QFrame#Panel { border-radius: 0; border-top: none; border-bottom: none; }")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(10)

        lab = QLabel("MODEL")
        lab.setObjectName("Section")
        lay.addWidget(lab)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(36)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        lay.addWidget(self.model_combo)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.setObjectName("ghost")
        refresh.clicked.connect(self._reload_models)
        row.addWidget(refresh)
        row.addStretch()
        lay.addLayout(row)

        lab2 = QLabel("SESSIONS")
        lab2.setObjectName("Section")
        lay.addWidget(lab2)
        self.session_list = QListWidget()
        self.session_list.itemDoubleClicked.connect(self._rename_session)
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        self.session_list.customContextMenuRequested.connect(self._show_session_menu)
        lay.addWidget(self.session_list, 1)
        btn_row = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.setObjectName("primary")
        new_btn.clicked.connect(self._new_session)
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("ghost")
        del_btn.clicked.connect(self._delete_session)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(del_btn)
        lay.addLayout(btn_row)

        lab3 = QLabel("WORKSPACE")
        lab3.setObjectName("Section")
        lay.addWidget(lab3)
        self.ws_label = QLabel(str(self.settings.resolved_workspace()))
        self.ws_label.setObjectName("Dim")
        self.ws_label.setWordWrap(True)
        lay.addWidget(self.ws_label)
        ws_btn = QPushButton("Browse…")
        ws_btn.setObjectName("ghost")
        ws_btn.clicked.connect(self._browse_workspace)
        lay.addWidget(ws_btn)
        return wrap

    def _chat_pane(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_host = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_host)
        self.chat_layout.setContentsMargins(4, 4, 4, 4)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_host)
        lay.addWidget(self.chat_scroll, 1)

        self.round_label = QLabel("auto" if self.settings.autorun else "ready")
        self.round_label.setObjectName("Muted")
        self.round_label.setWordWrap(True)
        lay.addWidget(self.round_label)

        self.status_line = QLabel("Ready. User prompts only — no heartbeat.")
        self.status_line.setObjectName("Muted")
        lay.addWidget(self.status_line)

        self.input = InputBox()
        self.input.setPlaceholderText("Ask sm0l  ·  Enter to send  ·  Shift+Enter for newline")
        self.input.setFixedHeight(90)
        self.input.send.connect(self._send)
        lay.addWidget(self.input)

        nav = QHBoxLayout()
        nav.addStretch()
        self.autorun_toggle = QPushButton("Auto")
        self.autorun_toggle.setObjectName("ghost")
        self.autorun_toggle.setCheckable(True)
        self.autorun_toggle.setChecked(self.settings.autorun)
        self.autorun_toggle.toggled.connect(self._on_autorun_toggled)
        self._style_autorun(self.settings.autorun)
        nav.addWidget(self.autorun_toggle)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.setMinimumWidth(120)
        self.send_btn.clicked.connect(self._send)
        nav.addWidget(self.stop_btn)
        nav.addWidget(self.send_btn)
        lay.addLayout(nav)
        return wrap

    def _inspector(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Panel")
        wrap.setStyleSheet("QFrame#Panel { border-radius: 0; border-top: none; border-bottom: none; }")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(10)

        lab = QLabel("CONTEXT")
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

        lab2 = QLabel("TOOLS")
        lab2.setObjectName("Section")
        lay.addWidget(lab2)
        self.tool_log = QPlainTextEdit()
        self.tool_log.setReadOnly(True)
        self.tool_log.setPlaceholderText("Tool calls from this turn show up here.")
        lay.addWidget(self.tool_log, 1)

        lab3 = QLabel("SETTINGS")
        lab3.setObjectName("Section")
        lay.addWidget(lab3)
        lay.addWidget(self._lbl("Ollama host"))
        self.host_edit = QLineEdit(self.settings.ollama_host)
        self.host_edit.editingFinished.connect(self._save_settings_from_ui)
        lay.addWidget(self.host_edit)
        row = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(self._lbl("Temperature"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.2)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(self.settings.temperature)
        self.temp_spin.valueChanged.connect(lambda _: self._save_settings_from_ui())
        col1.addWidget(self.temp_spin)
        col2 = QVBoxLayout()
        col2.addWidget(self._lbl("num_ctx (0=auto)"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(0, 32768)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(self.settings.num_ctx)
        self.ctx_spin.valueChanged.connect(lambda _: self._save_settings_from_ui())
        col2.addWidget(self.ctx_spin)
        row.addLayout(col1)
        row.addLayout(col2)
        lay.addLayout(row)

        lay.addWidget(self._lbl("Pull model via Ollama"))
        pull_row = QHBoxLayout()
        self.pull_edit = QLineEdit()
        self.pull_edit.setPlaceholderText("qwen2.5:7b")
        self.pull_btn = QPushButton("Pull")
        self.pull_btn.setObjectName("ghost")
        self.pull_btn.clicked.connect(self._pull_model)
        pull_row.addWidget(self.pull_edit, 1)
        pull_row.addWidget(self.pull_btn)
        lay.addLayout(pull_row)
        hint = QLabel("No heartbeat. Compaction uses the selected model against its native window (capped 32k).")
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

    def _render_history(self):
        print(f"DEBUG _render_history: session={self.session.id}, messages={len(self.session.messages)}")
        self._clear_chat()
        print(f"DEBUG after clear, chat_layout count={self.chat_layout.count()}")
        for i, m in enumerate(self.session.messages):
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                continue
            print(f"DEBUG rendering message {i}: role={role}, content_len={len(content)}")
            if role == "user":
                self._append_bubble(_bubble("user", "YOU", content))
            elif role == "assistant":
                if content.strip():
                    self._append_bubble(_bubble("assistant", "SM0L", content))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or tc
                    name = fn.get("name") or "?"
                    args = fn.get("arguments") or {}
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    self._append_bubble(_bubble("tool", f"TOOL  {name}", str(args)[:500], mono=True))
            elif role == "tool":
                name = m.get("tool_name") or "tool"
                self._append_bubble(_bubble("tool", f"RESULT  {name}", content[:1200], mono=True))
        print(f"DEBUG after loop, chat_layout count={self.chat_layout.count()}")
        self._update_ctx_bar()

    def _refresh_sessions(self):
        self.session_list.clear()
        for item in list_sessions():
            it = QListWidgetItem(item["title"])
            it.setData(Qt.ItemDataRole.UserRole, item["id"])
            self.session_list.addItem(it)
            if item["id"] == self.session.id:
                self.session_list.setCurrentItem(it)
        
        # Render current session's history after refresh
        self._render_history()

    def _new_session(self):
        if self.worker and self.worker.isRunning():
            return
        self.session.save()
        self.session = new_session(self._current_model())
        self.tool_log.clear()
        self._clear_chat()
        self._refresh_sessions()
        self.status_line.setText("New session.")
        self._update_ctx_bar()

    def _delete_session(self):
        item = self.session_list.currentItem()
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        
        # Check if this is the last session remaining
        remaining = list_sessions()
        will_be_last = len(remaining) == 1 and remaining[0]["id"] == sid
        
        delete_session(sid)
        
        # If we deleted the last session, create a new one immediately
        if will_be_last:
            self.session = new_session(self._current_model())
            self._clear_chat()
        else:
            # Update self.session to point to an existing loaded session
            for s in remaining:
                loaded = load_session(s["id"])
                if loaded:
                    self.session = loaded
        
        # Force UI refresh by clearing and rebuilding the list
        self.session_list.clear()
        for item in list_sessions():
            it = QListWidgetItem(item["title"])
            it.setData(Qt.ItemDataRole.UserRole, item["id"])
            self.session_list.addItem(it)
            if item["id"] == self.session.id:
                self.session_list.setCurrentItem(it)

    def _on_session_clicked(self, item: QListWidgetItem):
        if self.worker and self.worker.isRunning():
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            QMessageBox.warning(self, "Session Load", f"Failed to load session {sid}")
            return
        print(f"DEBUG _on_session_clicked: switching to {sid}, messages={len(loaded.messages)}")
        self.session.save()
        self.session = loaded
        self.tool_log.clear()
        self._render_history()

    def _on_session_selected(self, row: int):
        """Handle session selection from sidebar."""
        if self.worker and self.worker.isRunning():
            return
        item = self.session_list.item(row)
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            QMessageBox.warning(self, "Session Load", f"Failed to load session {sid}")
            return
        print(f"DEBUG _on_session_selected: switching to {sid}, messages={len(loaded.messages)}")
        self.session.save()
        self.session = loaded
        self.tool_log.clear()
        self._render_history()

    def _rename_session(self, item: QListWidgetItem):
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            return
        # TODO: implement rename with custom dialog
        QMessageBox.information(self, "Rename", "Rename not yet implemented.")

    def _show_session_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        loaded = load_session(sid)
        if not loaded:
            return
        menu = QMenu(self)
        rename_btn = menu.addAction("Rename")
        del_action = menu.addAction("Delete")
        menu.addSeparator()
        refresh_action = menu.addAction("Refresh List")
        
        def on_rename():
            self._rename_session(item)
        
        def on_delete():
            self._delete_session()
        
        def on_refresh():
            self._refresh_sessions()
        
        rename_btn.triggered.connect(on_rename)
        del_action.triggered.connect(on_delete)
        refresh_action.triggered.connect(on_refresh)
        menu.exec(self.session_list.mapToGlobal(pos))

    def _current_model(self) -> str:
        return (self.model_combo.currentText() or self.settings.model or "").strip()

    def _on_model_changed(self, name: str):
        name = (name or "").strip()
        if not name:
            return
        self.settings.model = name
        save_settings(self.settings)
        self.session.model = name
        self._refresh_ctx_for_model()

    def _refresh_ctx_for_model(self):
        model = self._current_model()
        if not model:
            return
        try:
            self.native_ctx = native_context_length(self.settings.ollama_host, model)
        except Exception:
            self.native_ctx = 8192
        self.session.native_ctx = self.native_ctx
        self._update_ctx_bar()

    def _update_ctx_bar(self):
        used = estimate_tokens(self.session.messages)
        self.session.token_estimate = used
        ctx = self.settings.num_ctx or self.native_ctx or 8192
        thresh = compact_threshold(ctx, self.settings.compact_ratio)
        pct = int(min(100, 100 * used / max(ctx, 1)))
        self.ctx_bar.setValue(pct)
        self.ctx_label.setText(
            f"{used:,} / {ctx:,} tok   native {self.native_ctx or '—'}   compact ≥ {thresh:,}"
        )

    def _reload_models(self):
        current = self._current_model() or self.settings.model
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        try:
            models = list_models(self.settings.ollama_host)
        except Exception as e:
            self.status_line.setText(f"Ollama model list failed: {e}")
            models = []
        names = [m["name"] for m in models]
        self.model_combo.addItems(names)
        if current and current in names:
            self.model_combo.setCurrentText(current)
        elif self.settings.model and self.settings.model in names:
            self.model_combo.setCurrentText(self.settings.model)
        elif names:
            self.model_combo.setCurrentIndex(0)
            self.settings.model = names[0]
            save_settings(self.settings)
        self.model_combo.blockSignals(False)
        if self._current_model():
            self._refresh_ctx_for_model()

    def _tick_ollama(self):
        online = ping(self.settings.ollama_host)
        self.status_dot.setText("ollama on" if online else "ollama off")
        self.status_dot.setProperty("state", "online" if online else "offline")
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)

    def _browse_workspace(self):
        path = QFileDialog.getExistingDirectory(
            self, "Workspace folder", str(self.settings.resolved_workspace())
        )
        if not path:
            return
        self.settings.workspace = path
        save_settings(self.settings)
        seed_workspace(Path(path))
        self.ws_label.setText(path)

    def _save_settings_from_ui(self):
        self.settings.ollama_host = self.host_edit.text().strip() or "http://127.0.0.1:11434"
        self.settings.temperature = float(self.temp_spin.value())
        self.settings.num_ctx = int(self.ctx_spin.value())
        save_settings(self.settings)
        self._update_ctx_bar()

    def _pull_model(self):
        name = self.pull_edit.text().strip()
        if not name:
            return
        self.pull_btn.setEnabled(False)
        self.status_line.setText(f"Pulling {name}…")
        self.puller = PullWorker(self.settings.ollama_host, name)
        self.puller.status.connect(lambda s: self.status_line.setText(s))
        self.puller.done.connect(self._on_pulled)
        self.puller.failed.connect(self._on_pull_fail)
        self.puller.start()

    def _on_pulled(self, name: str):
        self.pull_btn.setEnabled(True)
        self.status_line.setText(f"Pulled {name}")
        self.settings.model = name
        save_settings(self.settings)
        self._reload_models()
        self.model_combo.setCurrentText(name)

    def _on_pull_fail(self, err: str):
        self.pull_btn.setEnabled(True)
        self.status_line.setText(f"Pull failed: {err}")
        QMessageBox.warning(self, "Pull failed", err)

    def _set_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.input.setReadOnly(busy)
        self.session_list.setEnabled(not busy)
        if busy:
            self.status_dot.setProperty("state", "busy")
            self.status_dot.setText("working")
            self.status_dot.style().unpolish(self.status_dot)
            self.status_dot.style().polish(self.status_dot)

    def _send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._auto_hops = 0
        self._dispatch(text, from_auto=False)

    def _dispatch(self, text: str, *, from_auto: bool) -> bool:
        if self.worker and self.worker.isRunning():
            return False
        model = self._current_model()
        if not model:
            QMessageBox.information(
                self,
                "No model",
                "Start Ollama, then Refresh, or Pull qwen2.5:7b from the right panel.",
            )
            return False
        if not ping(self.settings.ollama_host):
            QMessageBox.warning(
                self,
                "Ollama offline",
                f"Nothing is listening at {self.settings.ollama_host}.\nStart Ollama and try again.",
            )
            return False
        self._user_stopped = False
        self._auto_resume = False
        self._sending_auto = from_auto
        if not from_auto:
            self.session.touch_title(text)
        self.session.messages.append({"role": "user", "content": text})
        self.session.model = model
        self.session.save()
        title = "AUTO" if from_auto else "YOU"
        self._append_bubble(_bubble("user", title, text))
        self._stream_bubble = None
        self._stream_text = ""
        self.tool_log.appendPlainText("— auto —" if from_auto else "— turn —")
        self._set_busy(True)
        self.status_line.setText("Auto continuing…" if from_auto else "Running…")
        if self.settings.autorun:
            hop = f" · hop {self._auto_hops}" if from_auto and self._auto_hops else ""
            self.round_label.setText(f"auto{hop}")
        self.worker = AgentWorker(self.settings, model, list(self.session.messages))
        self.worker.event.connect(self._on_agent_event)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        return True

    def _kick_auto(self):
        if not self.settings.autorun or self._user_stopped:
            return
        if self.worker and self.worker.isRunning():
            return
        self._auto_hops += 1
        if self._auto_hops > 24:
            self.status_line.setText("Auto stopped after 24 continues. Send a message to keep going.")
            self.round_label.setText("auto · capped")
            return
        self._dispatch(AUTO_CONTINUE, from_auto=True)

    def _stop(self):
        self._user_stopped = True
        self._auto_resume = False
        if self.worker:
            self.worker.cancel()
            self.status_line.setText("Stopping…")

    def _on_agent_event(self, kind: str, data: dict):
        if kind == "token":
            piece = data.get("text") or ""
            self._stream_text += piece
            if self._stream_bubble is None:
                self._stream_bubble = self._append_bubble(
                    _bubble("assistant", "SM0L", self._stream_text)
                )
            else:
                self._stream_bubble._body.setText(self._stream_text)  # type: ignore[attr-defined]
            self._scroll_bottom()
        elif kind == "tool_start":
            self._stream_bubble = None
            name = data.get("name") or "tool"
            args = data.get("arguments") or {}
            preview = json.dumps(args, ensure_ascii=False)[:400]
            self.tool_log.appendPlainText(f"→ {name} {preview}")
            self._append_bubble(_bubble("tool", f"TOOL  {name}", preview, mono=True))
            self.status_line.setText(f"tool: {name}")
        elif kind == "tool_end":
            name = data.get("name") or "tool"
            output = data.get("output") or ""
            self.tool_log.appendPlainText(f"� {name}\n{output[:800]}\n")
            self._append_bubble(_bubble("tool", f"RESULT  {name}", output[:1200], mono=True))
        elif kind == "compact":
            self.status_line.setText(data.get("text") or "compacting…")
            self.tool_log.appendPlainText(f"compact: {data.get('text') or ''}")
        elif kind == "status":
            self.status_line.setText(data.get("text") or "")
            rnd = data.get("round")
            mx = data.get("max_rounds")
            if rnd and mx:
                prefix = "auto · " if self.settings.autorun else ""
                self.round_label.setText(f"{prefix}{rnd}/{mx}")
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
                self.native_ctx = int(data["num_ctx"])
            self._update_ctx_bar()

    def _on_worker_finished(self):
        if self.worker:
            self.session.messages = self.worker.result_messages
            self.session.token_estimate = estimate_tokens(self.session.messages)
            self.session.native_ctx = self.native_ctx
            self.session.save()
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
            self.autorun_toggle.setStyleSheet("color: white; background-color: #4caf50;")
            self.round_label.setText("auto")
        else:
            self.autorun_toggle.setStyleSheet("color: white; background-color: #333;")
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

    def _on_donate_clicked(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("sm0l — tip jar")
        dlg.setMinimumWidth(480)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(18, 18, 18, 18)
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
        cl.addWidget(self._donate_section_title("Cash App"))
        tag = QLabel(f"Cashtag: {CASHAPP_TAG}")
        tag.setObjectName("Dim")
        cl.addWidget(tag)
        crow = QHBoxLayout()
        open_cash = QPushButton(f"Open {CASHAPP_TAG}")
        open_cash.setObjectName("primary")
        open_cash.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(CASHAPP_URL))
        )
        crow.addWidget(open_cash)
        copy_cash = QPushButton("Copy cashtag")
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
        bl.addWidget(self._donate_section_title("Bitcoin (on-chain)"))
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
        copy_btc = QPushButton("Copy address")
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

        close = QPushButton("Close")
        close.setObjectName("ghost")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _donate_section_title(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("Section")
        return lab

    def closeEvent(self, event):
        try:
            self.session.save()
            self._save_settings_from_ui()
            # Save last session ID for next startup
            last_session_path = user_data() / "last_session.json"
            last_session_path.write_text(json.dumps({"id": self.session.id}), encoding="utf-8")
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1500)
        super().closeEvent(event)
