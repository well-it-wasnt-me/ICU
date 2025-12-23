import time
from types import SimpleNamespace
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resource_monitor import ResourceMonitor


def test_resource_monitor_updates_snapshot(monkeypatch):
    cpu_samples = iter([5.0, 42.0, 51.0])
    cpu_calls = []

    def fake_cpu_percent(interval=None):
        cpu_calls.append(interval)
        try:
            return next(cpu_samples)
        except StopIteration:
            return 51.0

    monkeypatch.setattr("resource_monitor.psutil.cpu_percent", fake_cpu_percent)
    monkeypatch.setattr(
        "resource_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=33.3),
    )

    monitor = ResourceMonitor(interval=0.01)
    monitor.start()
    time.sleep(0.05)
    snapshot = monitor.get_snapshot()
    monitor.stop()

    assert cpu_calls, "cpu_percent should be called to generate snapshots."
    assert snapshot.cpu_percent == 51.0
    assert snapshot.memory_percent == 33.3
    assert monitor.is_under_pressure(cpu_threshold=50.0) is True
