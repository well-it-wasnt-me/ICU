"""
Lightweight state containers shared across Textual widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class CameraState:
    name: str
    status: str = "idle"
    processed_frames: int = 0
    skipped_frames: int = 0
    last_detection: Optional[str] = None
    last_detection_time: Optional[datetime] = None
