"""
Camera board table showing per-stream metrics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from textual.widgets import DataTable

from tui.state import CameraState


class CameraBoard(DataTable):
    """Tabular overview of camera streams."""

    def __init__(self) -> None:
        super().__init__()
        self._row_keys: Dict[str, str] = {}

    def on_mount(self) -> None:
        self.add_columns("Camera", "Status", "Processed", "Skipped", "Last Detection")
        self.cursor_type = "row"

    def update_camera(self, state: CameraState) -> None:
        row_key = self._row_keys.get(state.name)
        last_detection = self._format_detection(state.last_detection, state.last_detection_time)
        row = (
            state.name,
            state.status,
            f"{state.processed_frames}",
            f"{state.skipped_frames}",
            last_detection,
        )
        if row_key is not None:
            try:
                self.remove_row(row_key)
            except KeyError:
                pass
        new_key = self.add_row(*row, key=state.name)
        self._row_keys[state.name] = new_key

    def reset(self) -> None:
        self.clear()
        self._row_keys.clear()

    @staticmethod
    def _format_detection(label: str | None, timestamp: datetime | None) -> str:
        if not label:
            return "-"
        if not timestamp:
            return label
        return f"{label} @ {timestamp.strftime('%H:%M:%S')}"
