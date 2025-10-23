"""
Shared runtime event definitions for streaming and training telemetry.

These lightweight dataclasses allow modules to exchange structured updates
without creating a hard dependency on any specific UI implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

StreamStatus = Literal["connecting", "connected", "reconnecting", "stopped", "error"]


@dataclass(slots=True)
class RuntimeEvent:
    """Base event carrying a timestamp."""

    timestamp: float = field(default_factory=lambda: time.time())


@dataclass(slots=True)
class StreamLifecycleEvent(RuntimeEvent):
    """Lifecycle updates for an individual camera stream."""

    camera_name: str = ""
    status: StreamStatus = "connecting"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamMetricsEvent(RuntimeEvent):
    """Aggregated metrics emitted while processing frames."""

    camera_name: str = ""
    frame_count: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    last_latency_ms: Optional[float] = None


@dataclass(slots=True)
class DetectionEvent(RuntimeEvent):
    """Notification that a known person has been recognized."""

    camera_name: str = ""
    person_name: str = ""
    confidence: float = 0.0
    distance: float = 0.0
    capture_path: Optional[str] = None
    side_by_side_path: Optional[str] = None
    cooldown_active: bool = False


@dataclass(slots=True)
class PlateDetectionEvent(RuntimeEvent):
    """Notification that a license plate has been recognized."""

    camera_name: str = ""
    plate: str = ""
    confidence: float = 0.0
    occurrences: int = 0
    watchlist_hit: bool = False
    capture_path: Optional[str] = None
    crop_path: Optional[str] = None
    cooldown_active: bool = False


@dataclass(slots=True)
class TrainingEvent(RuntimeEvent):
    """Updates covering the lifecycle of a training session."""

    phase: Literal["starting", "progress", "completed", "failed"] = "starting"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
