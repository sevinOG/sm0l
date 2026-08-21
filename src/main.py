"""sm0l entrypoint."""
from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication
    from src.theme import apply_theme, brand_icon, APP_TITLE
    from src.ui import Dashboard
    from src.config import load_settings

    app = QApplication(sys.argv)
    apply_theme(app)
    icon = brand_icon()
    win = Dashboard(load_settings())
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.setWindowTitle(APP_TITLE)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
