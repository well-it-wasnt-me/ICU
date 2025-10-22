"""
Background streaming supervisor that mirrors `main.py` orchestration logic.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable, Optional

from face_recognizer import FaceRecognizer
from logger_setup import logger
from notifications import NotificationManager
from resource_monitor import ResourceMonitor
from runtime_events import RuntimeEvent
from stream_processor import StreamProcessor
from tui.event_bus import RuntimeEventBus


class StreamSupervisor:
    """
    Manage lifecycle for multiple camera streams in a reusable fashion.

    The supervisor is responsible for spawning stream threads, wiring up resource
    monitoring, and relaying runtime events to the UI.
    """

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        notifier: Optional[NotificationManager] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        base_capture_dir: str = "captures",
    ) -> None:
        self.face_recognizer = face_recognizer
        self.notifier = notifier
        self.event_bus = event_bus
        self.base_capture_dir = base_capture_dir

        self._threads: list[threading.Thread] = []
        self._resource_monitor: Optional[ResourceMonitor] = None
        self._processor: Optional[StreamProcessor] = None
        self._lock = threading.Lock()

    def start(
        self,
        camera_configs: Iterable[dict],
        distance_threshold: float,
        train_dir: str,
        target_processing_fps: Optional[float] = None,
        cpu_pressure_threshold: float = 85.0,
        pressure_backoff_factor: float = 2.0,
    ) -> None:
        """
        Start processing the provided camera configurations.
        """
        with self._lock:
            if self._threads:
                raise RuntimeError("Streaming is already running.")

            target_interval = (
                target_processing_fps if target_processing_fps and target_processing_fps > 0 else None
            )

            resource_monitor: Optional[ResourceMonitor] = None
            if target_interval or cpu_pressure_threshold > 0:
                resource_monitor = ResourceMonitor()
                resource_monitor.start()

            reference_images = self.face_recognizer.load_reference_images(train_dir)

            processor = StreamProcessor(
                face_recognizer=self.face_recognizer,
                reference_images=reference_images,
                base_capture_dir=self.base_capture_dir,
                notifier=self.notifier,
                target_processing_fps=target_interval,
                resource_monitor=resource_monitor,
                cpu_pressure_threshold=cpu_pressure_threshold,
                pressure_backoff_factor=pressure_backoff_factor,
                event_publisher=self._publish_event,
            )

            threads: list[threading.Thread] = []
            for cam_config in camera_configs:
                thread = threading.Thread(
                    target=processor.process_stream,
                    args=(cam_config, distance_threshold),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
                time.sleep(0.3)

            self._threads = threads
            self._resource_monitor = resource_monitor
            self._processor = processor

    def stop(self) -> None:
        """
        Request all stream threads to stop and wait for them to finish.
        """
        with self._lock:
            processor = self._processor
            threads = list(self._threads)
            resource_monitor = self._resource_monitor

            self._processor = None
            self._threads = []
            self._resource_monitor = None

        if processor:
            processor.request_stop()

        for thread in threads:
            thread.join(timeout=5.0)

        if resource_monitor:
            resource_monitor.stop()

    def is_running(self) -> bool:
        with self._lock:
            return any(thread.is_alive() for thread in self._threads)

    def current_snapshot(self):
        with self._lock:
            monitor = self._resource_monitor
        if monitor is None:
            return None
        return monitor.get_snapshot()

    def _publish_event(self, event: RuntimeEvent) -> None:
        if self.event_bus is None:
            return
        self.event_bus.emit(event)
