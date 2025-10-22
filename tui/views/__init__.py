"""
High-level views for Textual content switcher panes.
"""

from .logs import LogsView
from .settings import SettingsView
from .streams import StreamsView
from .train import TrainView

__all__ = [
    "LogsView",
    "SettingsView",
    "StreamsView",
    "TrainView",
]
