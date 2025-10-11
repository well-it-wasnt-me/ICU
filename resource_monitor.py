"""
Background resource monitoring utilities.
"""

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float


class ResourceMonitor:
    """
    Periodically sample host resource usage.
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self._snapshot = ResourceSnapshot(time.time(), 0.0, 0.0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1.0)

    def get_snapshot(self) -> ResourceSnapshot:
        with self._lock:
            return self._snapshot

    def is_under_pressure(self, cpu_threshold: float = 85.0) -> bool:
        return self.get_snapshot().cpu_percent >= cpu_threshold

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)  # Prime measurement baseline
        while not self._stop_event.is_set():
            cpu = psutil.cpu_percent(interval=self.interval)
            memory = psutil.virtual_memory().percent
            snapshot = ResourceSnapshot(time.time(), cpu, memory)
            with self._lock:
                self._snapshot = snapshot
