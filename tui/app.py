"""
Textual application entry point for ICU.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import ContentSwitcher, DataTable, Footer, Header, Log

from face_recognizer import FaceRecognizer
from logger_setup import configure_logging, logger
from notifications import NotificationManager
from runtime_events import DetectionEvent, StreamLifecycleEvent, StreamMetricsEvent, TrainingEvent
from tui.event_bus import RuntimeEventBus
from tui.services import StreamSupervisor, TrainingManager, load_app_config, load_camera_config
from tui.state import CameraState
from tui.widgets import (
    CameraBoard,
    NavigationItem,
    NavigationRail,
    NavigationSelection,
    ResourceFooter,
    StatusPanel,
)
from tui.views import LogsView, SettingsView, StreamsView, TrainView


class IcuTextualApp(App[None]):
    """Main Textual application."""

    TITLE = "ICU Control Center"
    CSS_PATH = "app.css"

    BINDINGS = [
        Binding("t", "trigger_train", "Train"),
        Binding("s", "toggle_streams", "Start/Stop Streams"),
        Binding("f", "open_find_camera", "Find Cameras"),
        Binding("r", "reload_configs", "Reload Configs"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        camera_config_path: str = "configs/cameras.yaml",
        app_config_path: str = "configs/app.yaml",
        model_path: str = "trained_knn_model.clf",
        train_dir: str = "poi",
        distance_threshold: float = 0.5,
        use_gpu: bool = False,
    ) -> None:
        super().__init__()
        self.event_bus = RuntimeEventBus()
        self.training_manager = TrainingManager(event_bus=self.event_bus)
        self.stream_supervisor: Optional[StreamSupervisor] = None
        self.notification_manager: Optional[NotificationManager] = None

        self.camera_config_path = camera_config_path
        self.app_config_path = app_config_path
        self.model_path = model_path
        self.train_dir = train_dir
        self.distance_threshold = distance_threshold
        self.use_gpu = use_gpu

        self.camera_configs: list[dict] = []
        self.app_config: dict = {}
        self._camera_state: Dict[str, CameraState] = {}
        self._selected_camera: Optional[str] = None

        self.target_processing_fps: Optional[float] = None
        self.cpu_pressure_threshold: float = 85.0
        self.pressure_backoff_factor: float = 2.0

        self.camera_board: Optional[CameraBoard] = None
        self.status_panel: Optional[StatusPanel] = None
        self.log_panel: Optional[Log] = None
        self.resource_footer: Optional[ResourceFooter] = None
        self.content_switcher: Optional[ContentSwitcher] = None
        self.train_view: Optional[TrainView] = None
        self.streams_view: Optional[StreamsView] = None
        self.logs_view: Optional[LogsView] = None
        self.settings_view: Optional[SettingsView] = None
        self.log_file_path: Optional[str] = None
        self._log_tail_position: int = 0
        self._log_tail_buffer: str = ""
        self._log_poll_timer = None
        self._log_tail_window_bytes: int = 32768

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-body"):
            with Horizontal(id="content"):
                yield NavigationRail(
                    [
                        NavigationItem("dashboard", "Dashboard"),
                        NavigationItem("train", "Train"),
                        NavigationItem("streams", "Streams"),
                        NavigationItem("logs", "Logs"),
                        NavigationItem("settings", "Settings"),
                    ]
                )
                with Vertical(id="primary"):
                    with ContentSwitcher(initial="dashboard", id="main-switcher") as switcher:
                        self.content_switcher = switcher
                        with Vertical(id="dashboard"):
                            self.camera_board = CameraBoard()
                            yield self.camera_board
                            self.status_panel = StatusPanel()
                            yield self.status_panel
                        self.train_view = TrainView(
                            train_dir=self.train_dir,
                            model_path=self.model_path,
                            use_gpu=self.use_gpu,
                        )
                        yield self.train_view
                        self.streams_view = StreamsView()
                        yield self.streams_view
                        self.logs_view = LogsView()
                        yield self.logs_view
                        self.settings_view = SettingsView(
                            distance_threshold=self.distance_threshold,
                            target_processing_fps=self.target_processing_fps,
                            cpu_pressure_threshold=self.cpu_pressure_threshold,
                            pressure_backoff_factor=self.pressure_backoff_factor,
                            use_gpu=self.use_gpu,
                        )
                        yield self.settings_view
                with Vertical(id="secondary"):
                    self.log_panel = Log(max_lines=500)
                    yield self.log_panel
        self.resource_footer = ResourceFooter()
        yield self.resource_footer
        yield Footer()

    async def on_mount(self) -> None:
        configure_logging({})
        self.set_interval(0.5, self._drain_runtime_events)
        self.set_interval(2.0, self._refresh_resource_metrics)
        await self._load_configs()
        self._resolve_log_file_path(self.app_config)
        self._initialize_log_tail()
        self._log_poll_timer = self.set_interval(1.0, self._poll_log_file)

    async def on_unmount(self, event: events.Unmount) -> None:
        if self._log_poll_timer:
            self._log_poll_timer.stop()
            self._log_poll_timer = None
        if self.stream_supervisor and self.stream_supervisor.is_running():
            self.stream_supervisor.stop()

    async def on_navigation_selection(self, event: NavigationSelection) -> None:
        self._log(f"Selected navigation item {event.item_id!r}")
        if self.content_switcher:
            try:
                self.content_switcher.current = event.item_id
            except KeyError:
                pass

    def on_train_view_submit(self, event: TrainView.Submit) -> None:
        self._start_training(event.config)

    async def on_streams_view_start(self, event: StreamsView.Start) -> None:
        await self._start_streams()

    async def on_streams_view_stop(self, event: StreamsView.Stop) -> None:
        await self._stop_streams()

    def on_logs_view_clear(self, event: LogsView.Clear) -> None:
        self._log("Cleared detection history.")

    def on_settings_view_apply(self, event: SettingsView.Apply) -> None:
        data = event.data
        self.distance_threshold = max(0.0, data.distance_threshold)
        self.target_processing_fps = data.target_processing_fps if (data.target_processing_fps or 0) > 0 else None
        self.cpu_pressure_threshold = max(0.0, data.cpu_pressure_threshold)
        self.pressure_backoff_factor = max(1.0, data.pressure_backoff_factor)
        self.use_gpu = data.use_gpu
        self._log("Settings updated. Restart streams to apply changes.")
        if self.train_view:
            self.train_view.update_values(
                train_dir=self.train_dir,
                model_path=self.model_path,
                use_gpu=self.use_gpu,
            )
        if self.settings_view:
            self.settings_view.update_values(
                distance_threshold=self.distance_threshold,
                target_processing_fps=self.target_processing_fps,
                cpu_pressure_threshold=self.cpu_pressure_threshold,
                pressure_backoff_factor=self.pressure_backoff_factor,
                use_gpu=self.use_gpu,
            )

    async def on_settings_view_reload(self, event: SettingsView.Reload) -> None:
        await self.action_reload_configs()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_key = str(event.row_key.value)
        self._selected_camera = row_key
        state = self._camera_state.get(row_key)
        if state and self.status_panel:
            detail = (
                f"[b]Camera:[/b] {state.name}\n"
                f"[b]Status:[/b] {state.status}\n"
                f"[b]Processed:[/b] {state.processed_frames}\n"
                f"[b]Skipped:[/b] {state.skipped_frames}\n"
            )
            if state.last_detection and state.last_detection_time:
                detail += f"[b]Last Detection:[/b] {state.last_detection} at {state.last_detection_time.isoformat(timespec='seconds')}"
            self.status_panel.update_detail(detail)

    async def action_trigger_train(self) -> None:
        config = TrainView.TrainConfig(
            train_dir=self.train_dir,
            model_path=self.model_path,
            use_gpu=self.use_gpu,
            n_neighbors=None,
        )
        self._start_training(config)

    async def action_toggle_streams(self) -> None:
        if self.stream_supervisor and self.stream_supervisor.is_running():
            await self._stop_streams()
        else:
            await self._start_streams()

    async def action_open_find_camera(self) -> None:
        self._log("[cyan]Camera discovery UI not yet implemented.[/]")

    async def action_reload_configs(self) -> None:
        await self._load_configs()
        self._resolve_log_file_path(self.app_config)
        self._initialize_log_tail()

    async def action_quit(self) -> None:
        if self.stream_supervisor and self.stream_supervisor.is_running():
            await self._stop_streams()
        self.exit()

    def _start_training(self, config: TrainView.TrainConfig) -> None:
        if self.training_manager.is_running():
            message = "Training already in progress."
            self._log(message)
            if self.train_view:
                self.train_view.set_status(message, error=True)
            return

        self.train_dir = config.train_dir
        self.model_path = config.model_path
        self.use_gpu = config.use_gpu
        if self.train_view:
            self.train_view.update_values(
                train_dir=self.train_dir,
                model_path=self.model_path,
                use_gpu=self.use_gpu,
            )
        if self.settings_view:
            self.settings_view.update_values(
                distance_threshold=self.distance_threshold,
                target_processing_fps=self.target_processing_fps,
                cpu_pressure_threshold=self.cpu_pressure_threshold,
                pressure_backoff_factor=self.pressure_backoff_factor,
                use_gpu=self.use_gpu,
            )

        self._log("Starting training session...")
        if self.train_view:
            self.train_view.set_running(True)
            self.train_view.set_status("Preparing training data...")

        try:
            self.training_manager.start(
                train_dir=self.train_dir,
                model_save_path=self.model_path,
                use_gpu=self.use_gpu,
                n_neighbors=config.n_neighbors,
            )
        except Exception as exc:
            message = f"Failed to start training: {exc}"
            self._log(f"[red]{message}[/]")
            if self.train_view:
                self.train_view.set_status(message, error=True)
                self.train_view.set_running(False)

    async def _start_streams(self) -> None:
        if self.stream_supervisor and self.stream_supervisor.is_running():
            self._log("Streams already running.")
            return

        if not os.path.exists(self.model_path):
            self._log(f"[red]Model file not found: {self.model_path}[/]")
            return

        recognizer = FaceRecognizer(use_gpu=self.use_gpu)
        try:
            recognizer.load_model(self.model_path)
            if self.use_gpu:
                recognizer.initialize_facenet_pytorch_models()
        except Exception as exc:
            self._log(f"[red]Failed to load model: {exc}[/]")
            return

        notifier = self._create_notifier(self.app_config)
        self.stream_supervisor = StreamSupervisor(
            recognizer,
            notifier=notifier,
            event_bus=self.event_bus,
        )

        if not self.camera_configs:
            self._log("[yellow]No camera configurations loaded.[/]")
            if self.streams_view:
                self.streams_view.running = False
            self.stream_supervisor = None
            return

        self._log("Starting camera streams...")
        try:
            self.stream_supervisor.start(
                self.camera_configs,
                distance_threshold=self.distance_threshold,
                train_dir=self.train_dir,
                target_processing_fps=self.target_processing_fps,
                cpu_pressure_threshold=self.cpu_pressure_threshold,
                pressure_backoff_factor=self.pressure_backoff_factor,
            )
        except Exception as exc:
            self._log(f"[red]Failed to start streams: {exc}[/]")
            if self.streams_view:
                self.streams_view.running = False
            self.stream_supervisor = None
            return

        if self.streams_view:
            self.streams_view.running = True

    async def _stop_streams(self) -> None:
        if not self.stream_supervisor or not self.stream_supervisor.is_running():
            if self.streams_view:
                self.streams_view.running = False
            return

        self._log("Stopping streams...")
        self.stream_supervisor.stop()
        self.stream_supervisor = None
        if self.streams_view:
            self.streams_view.running = False

    async def _load_configs(self) -> None:
        await self._load_app_config()
        await self._load_camera_config()

    async def _load_app_config(self) -> None:
        try:
            config = load_app_config(self.app_config_path)
            self.app_config = config
        except FileNotFoundError:
            self._log(f"[yellow]App config not found: {self.app_config_path}[/]")
            self.app_config = {}
            return
        settings = config.get("settings", {})
        self.target_processing_fps = settings.get("target_processing_fps")
        self.cpu_pressure_threshold = settings.get("cpu_pressure_threshold", self.cpu_pressure_threshold)
        self.pressure_backoff_factor = settings.get("pressure_backoff_factor", self.pressure_backoff_factor)
        self._resolve_log_file_path(config)
        if self.settings_view:
            self.settings_view.update_values(
                distance_threshold=self.distance_threshold,
                target_processing_fps=self.target_processing_fps,
                cpu_pressure_threshold=self.cpu_pressure_threshold,
                pressure_backoff_factor=self.pressure_backoff_factor,
                use_gpu=self.use_gpu,
            )
        if self.train_view:
            self.train_view.update_values(
                train_dir=self.train_dir,
                model_path=self.model_path,
                use_gpu=self.use_gpu,
            )

    async def _load_camera_config(self) -> None:
        try:
            cameras, _ = load_camera_config(self.camera_config_path)
            self.camera_configs = list(cameras)
        except FileNotFoundError:
            self._log(f"[yellow]Camera config not found: {self.camera_config_path}[/]")
            self.camera_configs = []
            return
        self._log(f"Loaded {len(self.camera_configs)} cameras from {self.camera_config_path}")
        self._camera_state = {}
        if self.camera_board:
            self.camera_board.reset()
        if self.streams_view:
            self.streams_view.camera_board.reset()
        for camera in self.camera_configs:
            name = camera.get("name", "Unnamed")
            state = CameraState(name=name)
            self._camera_state[name] = state
            if self.camera_board:
                self.camera_board.update_camera(state)
            if self.streams_view:
                self.streams_view.update_camera(state)

    def _drain_runtime_events(self) -> None:
        for event in self.event_bus.drain():
            if isinstance(event, StreamLifecycleEvent):
                self._handle_stream_lifecycle(event)
            elif isinstance(event, StreamMetricsEvent):
                self._handle_stream_metrics(event)
            elif isinstance(event, DetectionEvent):
                self._handle_detection(event)
            elif isinstance(event, TrainingEvent):
                self._handle_training(event)

    def _handle_stream_lifecycle(self, event: StreamLifecycleEvent) -> None:
        state = self._ensure_camera_state(event.camera_name)
        state.status = event.status
        if event.message:
            self._log(f"[cyan]{event.camera_name}[/]: {event.message}")
        if self.camera_board:
            self.camera_board.update_camera(state)
        if self.streams_view:
            self.streams_view.update_camera(state)
            is_running = bool(self.stream_supervisor and self.stream_supervisor.is_running())
            self.streams_view.running = is_running

    def _handle_stream_metrics(self, event: StreamMetricsEvent) -> None:
        state = self._ensure_camera_state(event.camera_name)
        state.processed_frames = event.processed_frames
        state.skipped_frames = event.skipped_frames
        if self.camera_board:
            self.camera_board.update_camera(state)
        if self.streams_view:
            self.streams_view.update_camera(state)

    def _handle_detection(self, event: DetectionEvent) -> None:
        state = self._ensure_camera_state(event.camera_name)
        state.last_detection = event.person_name
        state.last_detection_time = datetime.fromtimestamp(event.timestamp)
        if self.camera_board:
            self.camera_board.update_camera(state)
        alert = (
            f"[green]{event.camera_name}[/] detected "
            f"[b]{event.person_name}[/] ({event.confidence:.1f}% confidence)"
        )
        if event.cooldown_active:
            alert += " [dim](cooldown active)[/]"
        self._log(alert)
        if self.logs_view:
            self.logs_view.add_detection(
                timestamp=event.timestamp,
                camera=event.camera_name,
                person=event.person_name,
                confidence=event.confidence,
                cooldown_active=event.cooldown_active,
            )

    def _handle_training(self, event: TrainingEvent) -> None:
        if self.train_view:
            if event.phase == "starting":
                self.train_view.set_running(True)
                self.train_view.set_status(event.message or "Training starting...")
            elif event.phase == "progress":
                self.train_view.set_status(event.message or "Training in progress...")
            elif event.phase == "completed":
                self.train_view.set_status(event.message or "Training completed.")
                self.train_view.set_running(False)
            elif event.phase == "failed":
                self.train_view.set_status(event.message or "Training failed.", error=True)
                self.train_view.set_running(False)

        if event.phase == "failed":
            self._log(f"[red]Training failed: {event.message}[/]")
        elif event.phase == "completed":
            self._log(f"[green]{event.message}[/]")
            model_path = event.details.get("model_path") if event.details else None
            if model_path:
                self.model_path = model_path
                if self.train_view:
                    self.train_view.update_values(
                        train_dir=self.train_dir,
                        model_path=self.model_path,
                        use_gpu=self.use_gpu,
                    )
        else:
            self._log(event.message or f"Training phase: {event.phase}")

    def _resolve_log_file_path(self, config: Optional[dict] = None) -> Optional[str]:
        candidate: Optional[str] = None
        if config:
            logging_cfg = config.get("logging", {})
            if isinstance(logging_cfg, dict):
                value = logging_cfg.get("file")
                if value:
                    candidate = os.path.abspath(os.fspath(value))

        if not candidate and self.log_file_path:
            candidate = self.log_file_path

        if not candidate:
            candidate = self._detect_log_file_from_handlers()

        if not candidate:
            candidate = os.path.abspath("face_recognition.log")

        self.log_file_path = candidate
        return candidate

    @staticmethod
    def _detect_log_file_from_handlers() -> Optional[str]:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    return os.path.abspath(handler.baseFilename)
                except (AttributeError, TypeError):
                    return handler.baseFilename
        return None

    def _initialize_log_tail(self) -> None:
        path = self.log_file_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                start = max(0, size - self._log_tail_window_bytes)
                fh.seek(start, os.SEEK_SET)
                data = fh.read()
        except OSError:
            return

        text = data.decode("utf-8", errors="replace") if data else ""
        lines = text.splitlines()
        if start > 0 and lines:
            lines = lines[1:]

        if self.log_panel:
            self.log_panel.clear()
            max_lines = getattr(self.log_panel, "max_lines", None)
            if max_lines:
                lines = lines[-max_lines:]
            for line in lines:
                self.log_panel.write_line(line)

        try:
            self._log_tail_position = os.path.getsize(path)
        except OSError:
            self._log_tail_position = 0
        self._log_tail_buffer = ""

    def _poll_log_file(self) -> None:
        path = self.log_file_path
        if not path:
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            return

        if size < self._log_tail_position:
            self._log_tail_position = 0

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._log_tail_position)
                data = fh.read()
                self._log_tail_position = fh.tell()
        except OSError:
            return

        if not data:
            return

        chunk = self._log_tail_buffer + data
        if not chunk:
            return

        if chunk.endswith("\n"):
            lines = chunk.splitlines()
            self._log_tail_buffer = ""
        else:
            lines = chunk.splitlines()
            if chunk and not chunk.endswith("\n"):
                if lines:
                    self._log_tail_buffer = lines.pop()
                else:
                    self._log_tail_buffer = chunk
                    lines = []
            else:
                self._log_tail_buffer = ""

        if not lines or not self.log_panel:
            return

        for line in lines:
            self.log_panel.write_line(line)

    def _refresh_resource_metrics(self) -> None:
        if not self.stream_supervisor:
            return
        snapshot = self.stream_supervisor.current_snapshot()
        if not snapshot or not self.resource_footer:
            return
        under_pressure = snapshot.cpu_percent >= self.cpu_pressure_threshold
        self.resource_footer.update_metrics(snapshot.cpu_percent, snapshot.memory_percent, under_pressure)

    def _create_notifier(self, config: dict) -> Optional[NotificationManager]:
        notifications = config.get("notifications", {}) if config else {}
        telegram_cfg = notifications.get("telegram", {})
        bot_token = telegram_cfg.get("bot_token")
        chat_id = telegram_cfg.get("chat_id")
        if bot_token and chat_id:
            return NotificationManager(
                telegram_bot_token=bot_token,
                telegram_chat_id=chat_id,
                timeout=telegram_cfg.get("timeout", 10),
                max_workers=telegram_cfg.get("max_workers", 2),
            )
        return None

    def _ensure_camera_state(self, camera_name: str) -> CameraState:
        state = self._camera_state.get(camera_name)
        if state is None:
            state = CameraState(name=camera_name)
            self._camera_state[camera_name] = state
        return state

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.log_panel and (not self.log_file_path or not os.path.exists(self.log_file_path)):
            self.log_panel.write_line(message)
