import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("facenet_pytorch") is None, reason="facenet_pytorch not available")
def test_training_cli_creates_model(monkeypatch, tmp_path):
    """
    Exercise the `--train` CLI path without running heavy ML pipelines.

    The FaceRecognizer and image preprocessing are stubbed so the test only
    verifies argument wiring and that the model path is written.
    """
    import main  # Imported lazily so monkeypatch can hook symbols

    train_dir = tmp_path / "poi"
    train_dir.mkdir(parents=True)
    model_path = tmp_path / "trained_knn_model.clf"

    convert_calls = []

    monkeypatch.setattr(main.ImageUtils, "convert_images_to_rgb", lambda path: convert_calls.append(path))

    class DummyRecognizer:
        def __init__(self, use_gpu=False):
            self.device = "cpu"
            self.use_gpu = use_gpu
            self.initialised = False

        def initialize_facenet_pytorch_models(self):
            self.initialised = True

        def train(self, train_dir, model_save_path, n_neighbors=None, knn_algo="ball_tree", verbose=False):
            Path(model_save_path).write_text("dummy model")
            return object()

    monkeypatch.setattr(main, "FaceRecognizer", DummyRecognizer)
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")  # keep subprocess noise down

    argv = [
        "main.py",
        "--train",
        "--train_dir",
        str(train_dir),
        "--model_save_path",
        str(model_path),
    ]
    monkeypatch.setattr(main.sys, "argv", argv)

    main.main()

    assert convert_calls == [str(train_dir)], "Training should preprocess the provided train_dir."
    assert model_path.exists(), "Model file should be written by the stubbed trainer."
