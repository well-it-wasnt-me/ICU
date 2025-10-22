"""
Service layer for the Textual interface.

These utilities encapsulate long-running tasks (streaming, training, discovery)
so the UI can orchestrate them without duplicating business logic.
"""

from .config import load_app_config, load_camera_config
from .streaming import StreamSupervisor
from .training import TrainingManager

__all__ = [
    "load_app_config",
    "load_camera_config",
    "StreamSupervisor",
    "TrainingManager",
]
