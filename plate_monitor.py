"""
Utilities to persist license plate detections and trigger alerts.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from image_utils import ImageUtils
from logger_setup import logger
from notifications import NotificationManager
from plate_recognizer import PlateDetection
from runtime_events import PlateDetectionEvent


def _normalise_plate(plate: str) -> str:
    return "".join(ch for ch in plate.upper() if ch.isalnum())


@dataclass(slots=True)
class PlateHandlingResult:
    plate: str
    confidence: float
    occurrences: int
    watchlist_hit: bool
    capture_path: Optional[str]
    crop_path: Optional[str]
    notified: bool
    cooldown_active: bool
    timestamp: datetime

    def to_event(self, camera_name: str) -> PlateDetectionEvent:
        return PlateDetectionEvent(
            camera_name=camera_name,
            plate=self.plate,
            confidence=self.confidence,
            occurrences=self.occurrences,
            watchlist_hit=self.watchlist_hit,
            capture_path=self.capture_path,
            crop_path=self.crop_path,
            cooldown_active=self.cooldown_active,
            timestamp=self.timestamp.timestamp(),
        )


class PlateStore:
    """
    Thread-safe storage for detected plates and their occurrence counts.
    """

    def __init__(
        self,
        base_dir: str,
        summary_filename: str = "plates_summary.json",
        max_captures_per_plate: int = 20,
    ) -> None:
        self.base_dir = base_dir
        self.summary_path = os.path.join(base_dir, summary_filename)
        self.max_captures_per_plate = max(1, max_captures_per_plate)
        self._lock = threading.Lock()
        os.makedirs(self.base_dir, exist_ok=True)
        self._data = self._load()

    def record_detection(
        self,
        camera_name: str,
        plate: str,
        timestamp: datetime,
        capture_path: Optional[str],
        crop_path: Optional[str],
    ) -> int:
        """
        Register a detection for the given plate and return the updated occurrence count.
        """
        with self._lock:
            cameras = self._data.setdefault("cameras", {})
            camera_entry = cameras.setdefault(camera_name, {})
            plate_entry = camera_entry.setdefault(
                plate,
                {"count": 0, "last_seen": None, "captures": []},
            )

            plate_entry["count"] += 1
            plate_entry["last_seen"] = timestamp.isoformat()

            if capture_path or crop_path:
                plate_entry["captures"].append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "frame": capture_path,
                        "crop": crop_path,
                    }
                )
                if len(plate_entry["captures"]) > self.max_captures_per_plate:
                    plate_entry["captures"] = plate_entry["captures"][-self.max_captures_per_plate :]

            self._data["updated_at"] = timestamp.isoformat()
            self._persist()
            return plate_entry["count"]

    # Internal helpers -------------------------------------------------

    def _load(self) -> Dict:
        if not os.path.exists(self.summary_path):
            return {"cameras": {}, "updated_at": None}
        try:
            with open(self.summary_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.error("Failed to read plate summary %s: %s", self.summary_path, exc)
            return {"cameras": {}, "updated_at": None}

    def _persist(self) -> None:
        try:
            with open(self.summary_path, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2)
        except Exception as exc:
            logger.error("Failed to write plate summary %s: %s", self.summary_path, exc)


class PlateDetectionHandler:
    """
    Handle detected plates: persist captures, update counts, and notify watchers.
    """

    def __init__(
        self,
        store: PlateStore,
        notifier: Optional[NotificationManager],
        watchlist: Optional[Tuple[str, ...]] = None,
        alert_on_every_plate: bool = False,
        alert_on_watchlist: bool = True,
        capture_cooldown: int = 30,
        crop_padding: int = 8,
    ) -> None:
        self.store = store
        self.notifier = notifier
        self.alert_on_every_plate = alert_on_every_plate
        self.alert_on_watchlist = alert_on_watchlist
        self.capture_cooldown = max(0, capture_cooldown)
        self.crop_padding = max(0, crop_padding)
        self.watchlist = {_normalise_plate(entry) for entry in (watchlist or ())}
        self._last_capture: Dict[Tuple[str, str], float] = {}

    def handle(
        self,
        camera_name: str,
        frame: np.ndarray,
        detection: PlateDetection,
        timestamp: Optional[datetime] = None,
    ) -> PlateHandlingResult:
        """
        Persist the detection and emit alerts if configured.
        """
        timestamp = timestamp or datetime.utcnow()
        plate_key = _normalise_plate(detection.plate)

        capture_path, crop_path = None, None
        cooldown_active = self._is_in_cooldown(camera_name, plate_key, timestamp)
        if not cooldown_active:
            capture_path, crop_path = self._save_assets(
                camera_name=camera_name,
                frame=frame,
                detection=detection,
                timestamp=timestamp,
            )
            self._last_capture[(camera_name, plate_key)] = timestamp.timestamp()

        occurrences = self.store.record_detection(
            camera_name=camera_name,
            plate=plate_key,
            timestamp=timestamp,
            capture_path=capture_path,
            crop_path=crop_path,
        )

        watchlist_hit = plate_key in self.watchlist
        should_notify = (
            (watchlist_hit and self.alert_on_watchlist)
            or (self.alert_on_every_plate and not cooldown_active)
        )

        notified = False
        if should_notify and self.notifier and self.notifier.enabled:
            try:
                self.notifier.notify_plate_detection(
                    camera_name=camera_name,
                    plate_number=detection.plate,
                    confidence=detection.confidence * 100.0,
                    occurrences=occurrences,
                    capture_path=capture_path,
                    crop_path=crop_path,
                    timestamp=timestamp,
                    watchlist_hit=watchlist_hit,
                )
                notified = True
            except Exception as exc:
                logger.error("Failed to send plate notification: %s", exc)

        return PlateHandlingResult(
            plate=detection.plate,
            confidence=detection.confidence * 100.0,
            occurrences=occurrences,
            watchlist_hit=watchlist_hit,
            capture_path=capture_path,
            crop_path=crop_path,
            notified=notified,
            cooldown_active=cooldown_active,
            timestamp=timestamp,
        )

    # Internal helpers -------------------------------------------------

    def _is_in_cooldown(self, camera_name: str, plate: str, timestamp: datetime) -> bool:
        if self.capture_cooldown == 0:
            return False
        last = self._last_capture.get((camera_name, plate))
        if last is None:
            return False
        delta = timestamp.timestamp() - last
        return delta < self.capture_cooldown

    def _save_assets(
        self,
        camera_name: str,
        frame: np.ndarray,
        detection: PlateDetection,
        timestamp: datetime,
    ) -> Tuple[str, str]:
        base_dir = os.path.join(self.store.base_dir, camera_name, detection.plate)
        os.makedirs(base_dir, exist_ok=True)

        annotated = self._draw_annotation(frame.copy(), detection)
        annotated = ImageUtils.add_timestamp(annotated)
        crop = self._extract_crop(frame, detection)

        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        annotated_path = os.path.join(base_dir, f"{camera_name}_{detection.plate}_{ts}.jpg")
        crop_path = os.path.join(base_dir, f"{camera_name}_{detection.plate}_{ts}_crop.jpg")

        cv2.imwrite(annotated_path, annotated)
        cv2.imwrite(crop_path, crop)
        return annotated_path, crop_path

    def _draw_annotation(self, frame: np.ndarray, detection: PlateDetection) -> np.ndarray:
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = detection.plate
        cv2.putText(
            frame,
            label,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _extract_crop(self, frame: np.ndarray, detection: PlateDetection) -> np.ndarray:
        x, y, w, h = detection.expand_bbox(frame.shape, padding=self.crop_padding)
        return frame[y : y + h, x : x + w]
