"""
Detection log view.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Label


class LogsView(Vertical):
    """Display detection history."""

    DEFAULT_CSS = """
    LogsView {
        layout: vertical;
        padding: 1;
        border: tall $surface 10%;
    }

    LogsView .actions {
        layout: horizontal;
    }

    LogsView .actions Button {
        margin-right: 1;
    }

    LogsView DataTable {
        height: 1fr;
    }
    """

    class Clear(Message):
        """User requested to clear detection history."""

        def __init__(self, sender: "LogsView") -> None:
            super().__init__()
            self.sender = sender

    def __init__(self, *, id: str = "logs") -> None:
        super().__init__(id=id)
        self.table = DataTable(zebra_stripes=True)
        self.clear_button: Button | None = None
        self._rows: List[str] = []
        self._row_limit = 100

    def compose(self):
        yield Label("Recent Detections", classes="title")
        self.clear_button = Button("Clear History", id="logs-clear", variant="default")
        yield self.clear_button
        self.table.add_columns("Time", "Camera", "Person", "Confidence", "Cooldown")
        yield self.table

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is self.clear_button:
            self.table.clear()
            self._rows.clear()
            self.post_message(self.Clear(self))

    def add_detection(
        self,
        *,
        timestamp: float,
        camera: str,
        person: str,
        confidence: float,
        cooldown_active: bool,
    ) -> None:
        if len(self._rows) >= self._row_limit:
            oldest = self._rows.pop(0)
            try:
                self.table.remove_row(oldest)
            except KeyError:
                pass
        time_label = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        row_key = self.table.add_row(
            time_label,
            camera,
            person,
            f"{confidence:.1f}%",
            "Yes" if cooldown_active else "No",
        )
        self._rows.append(row_key)
