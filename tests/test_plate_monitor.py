import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from plate_monitor import PlateDetectionHandler, PlateStore
from plate_recognizer import PlateDetection


class DummyNotifier:
    def __init__(self):
        self.enabled = True
        self.calls = []

    def notify_plate_detection(self, **payload):
        self.calls.append(payload)


def _make_detection():
    return PlateDetection(
        plate="ABC123",
        confidence=0.92,
        bbox=(10, 15, 80, 30),
        raw_text="ABC123",
        ocr_confidence=0.92,
    )


def test_plate_handler_persists_and_notifies(tmp_path):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    store = PlateStore(base_dir=str(tmp_path))
    notifier = DummyNotifier()
    handler = PlateDetectionHandler(
        store=store,
        notifier=notifier,
        watchlist=("ABC123",),
        alert_on_watchlist=True,
        capture_cooldown=0,
    )

    ts = datetime.utcnow()
    detection = _make_detection()
    result = handler.handle("TestCam", frame, detection, timestamp=ts)

    assert result.watchlist_hit is True
    assert result.notified is True
    assert result.capture_path is not None and os.path.exists(result.capture_path)
    assert result.crop_path is not None and os.path.exists(result.crop_path)
    assert notifier.calls, "Expected notifier to be called for watchlist hit."

    with open(store.summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    cam_entry = summary["cameras"]["TestCam"]["ABC123"]
    assert cam_entry["count"] == 1
    assert cam_entry["captures"], "Expected capture history to be stored."


def test_plate_handler_cooldown(tmp_path):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    store = PlateStore(base_dir=str(tmp_path))
    notifier = DummyNotifier()
    handler = PlateDetectionHandler(
        store=store,
        notifier=notifier,
        watchlist=("ABC123",),
        alert_on_watchlist=True,
        capture_cooldown=60,
    )

    ts = datetime.utcnow()
    detection = _make_detection()
    handler.handle("TestCam", frame, detection, timestamp=ts)
    result = handler.handle("TestCam", frame, detection, timestamp=ts + timedelta(seconds=10))

    assert result.cooldown_active is True
    assert result.capture_path is None
    assert result.crop_path is None
    with open(store.summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["cameras"]["TestCam"]["ABC123"]["count"] == 2


def test_add_plate_to_watchlist_persists(tmp_path):
    store = PlateStore(base_dir=str(tmp_path))
    watchlist_file = tmp_path / "watchlist.txt"
    handler = PlateDetectionHandler(
        store=store,
        notifier=None,
        watchlist=("ABC123",),
        watchlist_file=str(watchlist_file),
    )

    success, message = handler.add_plate_to_watchlist("XYZ-999")
    assert success is True
    assert "XYZ999" in handler.watchlist
    with open(watchlist_file, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    assert "XYZ999" in lines

    success, message = handler.add_plate_to_watchlist("xyz999")
    assert success is False, "Duplicate plates should not be added twice."
