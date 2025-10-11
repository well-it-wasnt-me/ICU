"""
Terminal UI for live camera status.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Deque, Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from resource_monitor import ResourceMonitor, ResourceSnapshot


@dataclass
class CameraState:
    status: str = "initializing"
    detail: str = ""
    last_detection: str = "-"
    last_confidence: str = "-"
    updated_at: float = field(default_factory=time.time)


@dataclass
class DetectionEvent:
    camera: str
    person: str
    confidence: float
    timestamp: float


class TuiManager:
    """
    Render a Rich-based dashboard with camera and detection information.
    """

    def __init__(self, resource_monitor: Optional[ResourceMonitor] = None) -> None:
        self._console = Console()
        self._resource_monitor = resource_monitor
        self._events: "Queue[tuple]" = Queue()
        self._states: Dict[str, CameraState] = {}
        self._detections: Deque[DetectionEvent] = deque(maxlen=10)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._stop_event.set()
        self._events.put(("noop", "", "", ""))
        self._thread.join(timeout=1.5)

    def update_camera(self, camera: str, status: str, detail: str = "") -> None:
        self._events.put(("state", camera, status, detail))

    def notify_detection(self, camera: str, person: str, confidence: float) -> None:
        self._events.put(("detection", camera, person, confidence))

    def _run(self) -> None:
        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            screen=True,
        ) as live:
            last_refresh = 0.0
            while not self._stop_event.is_set():
                try:
                    event = self._events.get(timeout=0.2)
                except Empty:
                    event = None

                if event:
                    kind = event[0]
                    if kind == "state":
                        _, camera, status, detail = event
                        state = self._states.setdefault(camera, CameraState())
                        state.status = status
                        state.detail = detail
                        state.updated_at = time.time()
                    elif kind == "detection":
                        _, camera, person, confidence = event
                        state = self._states.setdefault(camera, CameraState())
                        state.last_detection = person
                        state.last_confidence = f"{confidence:.1f}%"
                        state.updated_at = time.time()
                        self._detections.appendleft(DetectionEvent(camera, person, confidence, time.time()))

                now = time.time()
                if event is not None or (now - last_refresh) >= 1.0:
                    live.update(self._render())
                    last_refresh = now

    def _render(self):
        table = Table(title="Camera Status", expand=True)
        table.add_column("Camera")
        table.add_column("State")
        table.add_column("Detail")
        table.add_column("Last Detection")
        table.add_column("Confidence")
        table.add_column("Updated")

        for camera, state in sorted(self._states.items()):
            delta = int(time.time() - state.updated_at)
            table.add_row(
                camera,
                state.status,
                state.detail or "-",
                state.last_detection,
                state.last_confidence,
                f"{delta}s ago",
            )

        recent_table = Table(show_header=True, header_style="bold", expand=True)
        recent_table.add_column("When")
        recent_table.add_column("Camera")
        recent_table.add_column("Person")
        recent_table.add_column("Confidence")
        for event in list(self._detections):
            delta = int(time.time() - event.timestamp)
            recent_table.add_row(f"{delta}s ago", event.camera, event.person, f"{event.confidence:.1f}%")

        panels = [Panel(table, title="Live Cameras", padding=(0, 1))]
        panels.append(Panel(recent_table, title="Recent Detections", padding=(0, 1)))

        snapshot = self._get_resource_snapshot()
        if snapshot:
            panels.append(self._render_resource_panel(snapshot))

        layout = Table.grid(expand=True)
        layout.add_row(*panels)
        return layout

    def _get_resource_snapshot(self) -> Optional[ResourceSnapshot]:
        if not self._resource_monitor:
            return None
        return self._resource_monitor.get_snapshot()

    @staticmethod
    def _render_resource_panel(snapshot: ResourceSnapshot) -> Panel:
        info = Table.grid(padding=(0, 1))
        info.add_row(f"CPU: {snapshot.cpu_percent:.1f}%")
        info.add_row(f"Memory: {snapshot.memory_percent:.1f}%")
        return Panel(info, title="Host Resources", padding=(0, 1))
