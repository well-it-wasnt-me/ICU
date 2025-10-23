import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from notifications import NotificationManager


class ImmediateExecutor:
    def submit(self, func, *args, **kwargs):
        func(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


@pytest.fixture
def notification_manager(monkeypatch):
    # Prevent real HTTP sessions
    class DummySession:
        def __init__(self):
            self.calls = []

        def post(self, url, data=None, files=None, timeout=None):
            self.calls.append(
                {
                    "url": url,
                    "data": data,
                    "files": files,
                    "timeout": timeout,
                    "status_code": 200,
                    "text": "ok",
                }
            )
            class Response:
                status_code = 200
                text = "ok"

            return Response()

        def close(self):
            pass

    dummy_session = DummySession()
    monkeypatch.setattr("requests.Session", lambda: dummy_session)

    manager = NotificationManager(
        telegram_bot_token="token",
        telegram_chat_id="chat",
        enable_command_handler=False,
    )
    manager._executor = ImmediateExecutor()
    manager._session = dummy_session
    yield manager, dummy_session
    manager.shutdown()


def test_notify_detection_sends_photos(tmp_path, notification_manager, monkeypatch):
    manager, session = notification_manager
    capture = tmp_path / "capture.jpg"
    side = tmp_path / "side.jpg"
    capture.write_bytes(b"fake")
    side.write_bytes(b"fake")

    sent = []

    monkeypatch.setattr(
        manager,
        "_send_photo",
        lambda image_path, caption: sent.append(("photo", Path(image_path).name, caption)) or True,
    )
    monkeypatch.setattr(
        manager,
        "_send_text_message",
        lambda text: sent.append(("text", text)),
    )

    manager.notify_detection(
        camera_name="Cam",
        person_name="Person",
        confidence=88.8,
        capture_path=str(capture),
        side_by_side_path=str(side),
    )

    assert [s[0] for s in sent] == ["photo", "photo"], "Expected two photo messages to be sent."
    assert "capture.jpg" in sent[0][1], "First photo should use capture image."
    assert "side.jpg" in sent[1][1], "Second photo should use side-by-side image."


def test_notify_detection_falls_back_to_text(notification_manager, monkeypatch):
    manager, session = notification_manager
    sent = []

    monkeypatch.setattr(
        manager,
        "_send_photo",
        lambda image_path, caption: sent.append(("photo", image_path)) or False,
    )
    monkeypatch.setattr(
        manager,
        "_send_text_message",
        lambda text: sent.append(("text", text)),
    )

    manager.notify_detection(
        camera_name="Cam",
        person_name="Person",
        confidence=50.0,
        capture_path="/nonexistent/capture.jpg",
        side_by_side_path="/nonexistent/side.jpg",
    )

    assert sent, "Expected fallback text message to be sent."
    assert all(entry[0] != "photo" for entry in sent), "Photo should not be sent when files are missing."
    assert sent[0][0] == "text", "Should fall back to sending a text message."


def test_notify_plate_detection(notification_manager, tmp_path, monkeypatch):
    manager, session = notification_manager
    capture = tmp_path / "plate.jpg"
    crop = tmp_path / "crop.jpg"
    capture.write_bytes(b"fake")
    crop.write_bytes(b"fake")

    sent = []
    monkeypatch.setattr(
        manager,
        "_send_photo",
        lambda image_path, caption: sent.append(("photo", Path(image_path).name, caption)) or True,
    )
    monkeypatch.setattr(
        manager,
        "_send_text_message",
        lambda text: sent.append(("text", text)),
    )

    manager.notify_plate_detection(
        camera_name="Cam",
        plate_number="ABC123",
        confidence=87.5,
        occurrences=3,
        capture_path=str(capture),
        crop_path=str(crop),
        watchlist_hit=True,
    )

    assert [entry[0] for entry in sent] == ["photo", "photo"], "Expected two plate photos to be sent."
    assert "plate.jpg" in sent[0][1]
    assert "crop.jpg" in sent[1][1]
