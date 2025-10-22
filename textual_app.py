"""
Launcher for the ICU Textual interface.
"""

from __future__ import annotations

from tui.app import IcuTextualApp


def main() -> None:
    app = IcuTextualApp()
    app.run()


if __name__ == "__main__":
    main()
