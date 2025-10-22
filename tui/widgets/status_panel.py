"""
Status panel for contextual information.
"""

from __future__ import annotations

from textual.widgets import Static


class StatusPanel(Static):
    """Placeholder panel that displays contextual details."""

    def __init__(self) -> None:
        super().__init__("Select a camera to see details.")

    def update_detail(self, text: str) -> None:
        self.update(text)
