import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import pytest
import numpy as np
from unittest.mock import patch
from stream_processor import StreamProcessor, get_frames_from_stream
import itertools

class DummyFaceRecognizer:
    def predict(self, img, distance_threshold):
        # Simulate face recognition by returning a dummy prediction
        return [("TestPerson", 0.4)]

@pytest.fixture
def dummy_reference_images():
    # Create a dummy reference image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    return {"TestPerson": dummy_image}

@pytest.fixture
def dummy_camera_config():
    return {
        "name": "TestCamera",
        "stream_url": "0",  # Use "0" to simulate a local webcam
        "process_frame_interval": 1,
        "capture_cooldown": 0
    }

@pytest.fixture
def dummy_stream_processor(dummy_reference_images):
    face_recognizer = DummyFaceRecognizer()
    return StreamProcessor(face_recognizer, dummy_reference_images)

def test_get_frames_from_stream_local_webcam(monkeypatch):
    class DummyVideoCapture:
        def __init__(self, source):
            self.frame_count = 0
            self.max_frames = 5

        def isOpened(self):
            return True

        def read(self):
            if self.frame_count < self.max_frames:
                self.frame_count += 1
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, frame
            else:
                return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", DummyVideoCapture)
    frames = list(itertools.islice(get_frames_from_stream("0"), 5))
    assert len(frames) == 5, "Expected 5 frames from the dummy webcam stream."

def test_process_stream(dummy_stream_processor, dummy_camera_config, tmp_path):
    # Change current working directory to the temporary path
    os.chdir(tmp_path)
    with patch("stream_processor.get_frames_from_stream") as mock_get_frames:
        mock_get_frames.return_value = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        dummy_stream_processor.process_stream(dummy_camera_config, distance_threshold=0.6)

        capture_dir = os.path.join(tmp_path, "captures", "TestCamera")
        assert os.path.exists(capture_dir), "Capture directory was not created."
        captured_files = os.listdir(capture_dir)
        # Expect 10 files: 5 annotated frames and 5 side-by-side images
        assert len(captured_files) == 10, (
            "Expected 10 captured files (5 annotated frames and 5 side-by-side images)."
        )
        # Verify that all filenames are unique (i.e. no overwriting occurred)
        assert len(set(captured_files)) == 10, "Duplicate filenames detected. Filenames should be unique."

def test_process_stream_logging(dummy_stream_processor, dummy_camera_config, caplog):
    with patch("stream_processor.get_frames_from_stream") as mock_get_frames:
        mock_get_frames.return_value = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        with caplog.at_level("INFO"):
            dummy_stream_processor.process_stream(dummy_camera_config, distance_threshold=0.6)
        assert any("Detected known face: TestPerson" in record.message for record in caplog.records), \
            "Expected log message about detected face not found."