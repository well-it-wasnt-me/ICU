"""
Settings view to tweak runtime parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, Label


class SettingsView(Vertical):
    """Allow users to adjust runtime options."""

    DEFAULT_CSS = """
    SettingsView {
        layout: vertical;
        padding: 1;
        border: tall $surface 10%;
    }

    SettingsView Grid {
        grid-size: 2;
        grid-columns: 24 1fr;
        grid-gutter: 1 2;
    }

    SettingsView Input {
        width: 1fr;
    }

    SettingsView .buttons {
        layout: horizontal;
    }

    SettingsView .buttons Button {
        margin-right: 1;
    }
    """

    @dataclass
    class SettingsData:
        distance_threshold: float
        target_processing_fps: Optional[float]
        cpu_pressure_threshold: float
        pressure_backoff_factor: float
        use_gpu: bool

    class Apply(Message):
        """Emitted when settings should be applied."""

        def __init__(self, sender: "SettingsView", data: "SettingsView.SettingsData") -> None:
            super().__init__()
            self.sender = sender
            self.data = data

    class Reload(Message):
        """Emitted when configs should be reloaded."""

        def __init__(self, sender: "SettingsView") -> None:
            super().__init__()
            self.sender = sender

    def __init__(
        self,
        *,
        distance_threshold: float,
        target_processing_fps: Optional[float],
        cpu_pressure_threshold: float,
        pressure_backoff_factor: float,
        use_gpu: bool,
        id: str = "settings",
    ) -> None:
        super().__init__(id=id)
        self.distance_threshold_input: Input | None = None
        self.target_fps_input: Input | None = None
        self.cpu_threshold_input: Input | None = None
        self.backoff_input: Input | None = None
        self.use_gpu_checkbox: Checkbox | None = None

        self._initial = self.SettingsData(
            distance_threshold=distance_threshold,
            target_processing_fps=target_processing_fps,
            cpu_pressure_threshold=cpu_pressure_threshold,
            pressure_backoff_factor=pressure_backoff_factor,
            use_gpu=use_gpu,
        )

    def compose(self):
        yield Label("Runtime Settings", classes="title")
        with Grid():
            yield Label("Distance threshold")
            self.distance_threshold_input = Input(value=f"{self._initial.distance_threshold}")
            yield self.distance_threshold_input

            yield Label("Target FPS (blank = disabled)")
            fps_value = "" if self._initial.target_processing_fps in (None, 0.0) else f"{self._initial.target_processing_fps}"
            self.target_fps_input = Input(value=fps_value)
            yield self.target_fps_input

            yield Label("CPU pressure threshold")
            self.cpu_threshold_input = Input(value=f"{self._initial.cpu_pressure_threshold}")
            yield self.cpu_threshold_input

            yield Label("Backoff factor")
            self.backoff_input = Input(value=f"{self._initial.pressure_backoff_factor}")
            yield self.backoff_input

            yield Label("Use GPU")
            self.use_gpu_checkbox = Checkbox(value=self._initial.use_gpu)
            yield self.use_gpu_checkbox

        with Horizontal(classes="buttons"):
            yield Button("Apply", id="settings-apply", variant="success")
            yield Button("Reload Configs", id="settings-reload", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-apply":
            data = self._gather_settings()
            self.post_message(self.Apply(self, data))
        elif event.button.id == "settings-reload":
            self.post_message(self.Reload(self))

    def update_values(
        self,
        *,
        distance_threshold: float,
        target_processing_fps: Optional[float],
        cpu_pressure_threshold: float,
        pressure_backoff_factor: float,
        use_gpu: bool,
    ) -> None:
        self._initial = self.SettingsData(
            distance_threshold=distance_threshold,
            target_processing_fps=target_processing_fps,
            cpu_pressure_threshold=cpu_pressure_threshold,
            pressure_backoff_factor=pressure_backoff_factor,
            use_gpu=use_gpu,
        )
        if self.distance_threshold_input:
            self.distance_threshold_input.value = f"{distance_threshold}"
        if self.target_fps_input:
            self.target_fps_input.value = (
                "" if target_processing_fps in (None, 0.0) else f"{target_processing_fps}"
            )
        if self.cpu_threshold_input:
            self.cpu_threshold_input.value = f"{cpu_pressure_threshold}"
        if self.backoff_input:
            self.backoff_input.value = f"{pressure_backoff_factor}"
        if self.use_gpu_checkbox:
            self.use_gpu_checkbox.value = use_gpu

    def _gather_settings(self) -> "SettingsData":
        try:
            distance = float(self.distance_threshold_input.value)
        except (TypeError, ValueError):
            distance = self._initial.distance_threshold

        fps_input = self.target_fps_input.value.strip() if self.target_fps_input else ""
        if fps_input:
            try:
                target_fps = float(fps_input)
            except ValueError:
                target_fps = self._initial.target_processing_fps
        else:
            target_fps = None

        try:
            cpu_threshold = float(self.cpu_threshold_input.value)
        except (TypeError, ValueError):
            cpu_threshold = self._initial.cpu_pressure_threshold

        try:
            backoff = float(self.backoff_input.value)
        except (TypeError, ValueError):
            backoff = self._initial.pressure_backoff_factor

        use_gpu = self.use_gpu_checkbox.value if self.use_gpu_checkbox else self._initial.use_gpu

        return self.SettingsData(
            distance_threshold=distance,
            target_processing_fps=target_fps,
            cpu_pressure_threshold=cpu_threshold,
            pressure_backoff_factor=backoff,
            use_gpu=use_gpu,
        )
