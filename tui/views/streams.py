"""
Streams control view.
"""

from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label

from tui.widgets import CameraBoard


class StreamsView(Vertical):
    """Expose stream lifecycle controls and per-camera metrics."""

    DEFAULT_CSS = """
    StreamsView {
        layout: vertical;
        padding: 1;
        border: tall $surface 10%;
    }

    StreamsView .controls {
        layout: horizontal;
    }

    StreamsView .controls Button {
        margin-right: 1;
    }
    """

    running: reactive[bool] = reactive(False)

    class Start(Message):
        """User requested stream start."""

        def __init__(self, sender: "StreamsView") -> None:
            super().__init__()
            self.sender = sender

    class Stop(Message):
        """User requested stream stop."""

        def __init__(self, sender: "StreamsView") -> None:
            super().__init__()
            self.sender = sender

    def __init__(self, *, id: str = "streams") -> None:
        super().__init__(id=id)
        self.status_label: Label | None = None
        self.start_button: Button | None = None
        self.stop_button: Button | None = None
        self.camera_board = CameraBoard()

    def compose(self):
        yield Label("Stream Orchestration", classes="title")
        with Horizontal(classes="controls"):
            self.start_button = Button("Start Streams", id="streams-start", variant="success")
            yield self.start_button
            self.stop_button = Button("Stop Streams", id="streams-stop", variant="warning", disabled=True)
            yield self.stop_button
        self.status_label = Label("Streams idle.")
        yield self.status_label
        yield self.camera_board

    def watch_running(self, running: bool) -> None:
        if self.status_label:
            self.status_label.update("Streams running." if running else "Streams idle.")
        if self.start_button:
            self.start_button.disabled = running
        if self.stop_button:
            self.stop_button.disabled = not running

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is self.start_button:
            self.post_message(self.Start(self))
        elif event.button is self.stop_button:
            self.post_message(self.Stop(self))

    def update_camera(self, state) -> None:
        self.camera_board.update_camera(state)
