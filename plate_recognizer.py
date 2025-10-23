"""
License Plate Recognition helpers.

Provides utilities to detect license plates in a frame and extract their text using OCR.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from logger_setup import logger


@dataclass(slots=True)
class PlateDetection:
    """Information about a detected license plate."""

    plate: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    raw_text: str
    ocr_confidence: float

    def expand_bbox(self, frame_shape: Tuple[int, int, int], padding: int = 6) -> Tuple[int, int, int, int]:
        """Return the bounding box expanded by `padding` pixels on each side."""
        height, width = frame_shape[:2]
        x, y, w, h = self.bbox
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        return x1, y1, x2 - x1, y2 - y1


class PlateRecognizer:
    """
    Detect license plates in a frame and extract their text using EasyOCR.

    The recognizer relies on OpenCV's Haar cascade to locate candidate plates and
    EasyOCR to read their contents. If EasyOCR is not installed, the recognizer
    is disabled automatically and emits a warning once.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        ocr_languages: Optional[Sequence[str]] = None,
        ocr_gpu: bool = False,
        min_confidence: float = 0.5,
        min_plate_length: int = 4,
        max_plate_length: int = 10,
        max_detections_per_frame: int = 5,
    ) -> None:
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.min_plate_length = min_plate_length
        self.max_plate_length = max_plate_length
        self.max_detections_per_frame = max_detections_per_frame
        self._ocr_languages = list(ocr_languages or ["en"])
        self._ocr_gpu = bool(ocr_gpu)
        self._reader = None
        self._reader_lock = threading.Lock()
        self._warned_about_easyocr = False

        cascade_default = os.path.join(
            cv2.data.haarcascades, "haarcascade_russian_plate_number.xml"
        )
        cascade_to_use = cascade_path or cascade_default
        self._cascade = cv2.CascadeClassifier(cascade_to_use)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load license plate cascade from {cascade_to_use}")

        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Return True when OCR is available and the recognizer is operational."""
        if not self._enabled:
            return False
        reader = self._ensure_reader()
        return reader is not None

    def detect(self, frame: np.ndarray) -> List[PlateDetection]:
        """
        Detect and read license plates present in the frame.

        :param frame: BGR image from OpenCV.
        :return: List of `PlateDetection` entries ordered by confidence.
        """
        if not self.enabled:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._detect_candidates(gray)
        if not detections:
            return []

        reader = self._ensure_reader()
        if not reader:
            return []

        results: List[PlateDetection] = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for x, y, w, h in detections[: self.max_detections_per_frame]:
            roi = frame_rgb[max(0, y - 6): y + h + 6, max(0, x - 6): x + w + 6]
            if roi.size == 0:
                continue
            try:
                ocr_results = reader.readtext(roi, detail=1, paragraph=False)
            except Exception as exc:
                logger.debug("EasyOCR failed to read plate: %s", exc)
                continue

            best_detection = self._select_best_result(ocr_results)
            if not best_detection:
                continue

            text, score = best_detection
            cleaned = self._clean_plate_text(text)
            if not cleaned:
                continue
            combined_confidence = float(score)
            if combined_confidence < self.min_confidence:
                continue

            results.append(
                PlateDetection(
                    plate=cleaned,
                    confidence=combined_confidence,
                    bbox=(x, y, w, h),
                    raw_text=text,
                    ocr_confidence=float(score),
                )
            )

        if not results:
            return []

        deduped = self._deduplicate(results)
        deduped.sort(key=lambda det: det.confidence, reverse=True)
        return deduped

    # Internal helpers -------------------------------------------------

    def _detect_candidates(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            results = self._cascade.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                outputRejectLevels=True,
            )
            boxes = results[0]
            weights = results[2]
            ordered = sorted(
                zip(boxes, weights),
                key=lambda entry: entry[1],
                reverse=True,
            )
            return [tuple(map(int, box)) for box, _ in ordered]
        except Exception:
            boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            return [tuple(map(int, box)) for box in boxes]

    def _ensure_reader(self):
        with self._reader_lock:
            if self._reader is not None:
                return self._reader
            try:
                import easyocr  # type: ignore
            except ImportError:
                if not self._warned_about_easyocr:
                    logger.warning(
                        "EasyOCR is not available. License plate recognition disabled. "
                        "Install 'easyocr' to enable this feature."
                    )
                    self._warned_about_easyocr = True
                self._enabled = False
                return None
            try:
                self._reader = easyocr.Reader(
                    self._ocr_languages,
                    gpu=self._ocr_gpu,
                )
            except Exception as exc:
                logger.error("Failed to initialise EasyOCR reader: %s", exc)
                self._enabled = False
                self._reader = None
            return self._reader

    def _select_best_result(self, ocr_results) -> Optional[Tuple[str, float]]:
        best: Tuple[str, float] | None = None
        for _, text, score in ocr_results:
            if not text:
                continue
            cleaned = self._clean_plate_text(text)
            if not cleaned:
                continue
            score = float(score)
            if best is None or score > best[1]:
                best = (text, score)
        return best

    def _clean_plate_text(self, text: str) -> Optional[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
        if not cleaned:
            return None
        if len(cleaned) < self.min_plate_length or len(cleaned) > self.max_plate_length:
            return None
        if not re.search(r"[A-Z]", cleaned) or not re.search(r"\d", cleaned):
            return None
        return cleaned

    def _deduplicate(self, detections: Sequence[PlateDetection]) -> List[PlateDetection]:
        unique: dict[str, PlateDetection] = {}
        for det in detections:
            key = det.plate
            stored = unique.get(key)
            if stored is None or det.confidence > stored.confidence:
                unique[key] = det
        return list(unique.values())
