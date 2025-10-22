"""
Training view for the Textual UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textual.containers import Grid, Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, Label, Static


class TrainView(Static):
    """
    UI panel to configure and trigger training runs.
    """

    DEFAULT_CSS = """
    TrainView {
        layout: vertical;
        padding: 1;
        border: tall $surface 10%;
        height: 1fr;
        overflow: hidden auto;
    }

    TrainView Grid {
        grid-size: 2;
        grid-rows: auto;
        grid-columns: 20 1fr;
        grid-gutter: 1 2;
        width: 100%;
    }

    TrainView Input {
        width: 1fr;
    }

    TrainView .button-row {
        layout: horizontal;
        margin-top: 1;
    }

    TrainView .button-row Button {
        margin-right: 1;
    }

    TrainView .status {
        margin-top: 1;
    }
    """

    class Submit(Message):
        """Message emitted when the user requests training."""

        def __init__(self, sender: "TrainView", config: "TrainConfig") -> None:
            super().__init__()
            self.sender = sender
            self.config = config

    @dataclass
    class TrainConfig:
        train_dir: str
        model_path: str
        use_gpu: bool
        n_neighbors: Optional[int]

    def __init__(
        self,
        *,
        train_dir: str,
        model_path: str,
        use_gpu: bool,
        id: str = "train",
    ) -> None:
        super().__init__(id=id)
        self._train_dir = train_dir
        self._model_path = model_path
        self._use_gpu = use_gpu

        self.train_dir_input: Input | None = None
        self.model_path_input: Input | None = None
        self.n_neighbors_input: Input | None = None
        self.use_gpu_checkbox: Checkbox | None = None
        self.status_label: Label | None = None
        self.train_button: Button | None = None

    def compose(self):
        yield Label("Train KNN Model", classes="title")
        with Grid():
            yield Label("Training directory")
            self.train_dir_input = Input(value=self._train_dir, placeholder="poi")
            yield self.train_dir_input

            yield Label("Model output path")
            self.model_path_input = Input(value=self._model_path, placeholder="trained_knn_model.clf")
            yield self.model_path_input

            yield Label("Neighbors (optional)")
            self.n_neighbors_input = Input(placeholder="auto")
            yield self.n_neighbors_input

            yield Label("Use GPU")
            self.use_gpu_checkbox = Checkbox(value=self._use_gpu)
            yield self.use_gpu_checkbox

        with Horizontal(classes="button-row"):
            self.train_button = Button("Train Model", id="train-btn", variant="success")
            yield self.train_button
        self.status_label = Label("Ready.", classes="status")
        yield self.status_label

    def set_status(self, message: str | None, *, error: bool = False) -> None:
        if self.status_label:
            text = message or ""
            prefix = "[red]" if error else "[green]" if "completed" in text.lower() else ""
            suffix = "[/]" if prefix else ""
            self.status_label.update(f"{prefix}{text}{suffix}" if text else "Ready.")

    def set_running(self, running: bool) -> None:
        if self.train_button:
            self.train_button.disabled = running
            self.train_button.label = "Training..." if running else "Train Model"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.train_button or event.button is not self.train_button:
            return
        config = self._gather_config()
        self.post_message(self.Submit(self, config))

    def update_values(self, *, train_dir: str, model_path: str, use_gpu: bool) -> None:
        self._train_dir = train_dir
        self._model_path = model_path
        self._use_gpu = use_gpu
        if self.train_dir_input:
            self.train_dir_input.value = train_dir
        if self.model_path_input:
            self.model_path_input.value = model_path
        if self.use_gpu_checkbox:
            self.use_gpu_checkbox.value = use_gpu

    def _gather_config(self) -> "TrainConfig":
        train_dir = self.train_dir_input.value if self.train_dir_input else self._train_dir
        model_path = self.model_path_input.value if self.model_path_input else self._model_path
        use_gpu = self.use_gpu_checkbox.value if self.use_gpu_checkbox else self._use_gpu
        n_neighbors_raw = self.n_neighbors_input.value if self.n_neighbors_input else ""
        n_neighbors = None
        if n_neighbors_raw.strip():
            try:
                n_val = int(n_neighbors_raw.strip())
            except ValueError:
                n_val = None
            else:
                if n_val > 0:
                    n_neighbors = n_val
        return self.TrainConfig(
            train_dir=train_dir.strip() or self._train_dir,
            model_path=model_path.strip() or self._model_path,
            use_gpu=use_gpu,
            n_neighbors=n_neighbors,
        )
