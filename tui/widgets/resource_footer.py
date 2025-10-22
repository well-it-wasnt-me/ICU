"""
Footer widget displaying host resource usage.
"""

from __future__ import annotations

from textual.widgets import Static


class ResourceFooter(Static):
    """Simple status line for resource utilisation."""

    def __init__(self) -> None:
        super().__init__("CPU: --%  MEM: --%")
        self._cpu = None
        self._mem = None
        self._pressure = False

    def update_metrics(self, cpu_percent: float, mem_percent: float, under_pressure: bool = False) -> None:
        self._cpu = cpu_percent
        self._mem = mem_percent
        self._pressure = under_pressure
        pressure_flag = "!" if under_pressure else ""
        self.update(f"CPU: {cpu_percent:5.1f}%{pressure_flag}  MEM: {mem_percent:5.1f}%")
