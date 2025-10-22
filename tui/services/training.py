"""
Utilities for running training sessions in a background thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from face_recognizer import FaceRecognizer
from image_utils import ImageUtils
from logger_setup import logger
from runtime_events import TrainingEvent
from tui.event_bus import RuntimeEventBus


@dataclass
class TrainingResult:
    success: bool
    message: str = ""
    model_path: Optional[str] = None


class TrainingManager:
    """
    Execute model training asynchronously so the UI remains responsive.
    """

    def __init__(self, event_bus: Optional[RuntimeEventBus] = None) -> None:
        self.event_bus = event_bus
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._result: Optional[TrainingResult] = None

    def start(
        self,
        train_dir: str,
        model_save_path: str,
        use_gpu: bool = False,
        n_neighbors: Optional[int] = None,
        knn_algo: str = "ball_tree",
        verbose: bool = True,
    ) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Training is already running.")
            self._result = None
            thread = threading.Thread(
                target=self._run_training,
                args=(train_dir, model_save_path, use_gpu, n_neighbors, knn_algo, verbose),
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def last_result(self) -> Optional[TrainingResult]:
        return self._result

    def _run_training(
        self,
        train_dir: str,
        model_save_path: str,
        use_gpu: bool,
        n_neighbors: Optional[int],
        knn_algo: str,
        verbose: bool,
    ) -> None:
        self._emit(TrainingEvent(phase="starting", message="Preparing training data"))
        try:
            ImageUtils.convert_images_to_rgb(train_dir)
        except Exception as exc:
            message = f"Failed to preprocess training images: {exc}"
            logger.exception(message)
            self._record_result(False, message, None)
            self._emit(TrainingEvent(phase="failed", message=message))
            return

        recognizer = FaceRecognizer(use_gpu=use_gpu)
        if use_gpu:
            self._emit(TrainingEvent(phase="progress", message="Initializing GPU models"))
            recognizer.initialize_facenet_pytorch_models()

        self._emit(TrainingEvent(phase="progress", message="Training KNN classifier"))
        try:
            knn_clf = recognizer.train(
                train_dir,
                model_save_path=model_save_path,
                n_neighbors=n_neighbors,
                knn_algo=knn_algo,
                verbose=verbose,
            )
        except Exception as exc:
            message = f"Training failed: {exc}"
            logger.exception(message)
            self._record_result(False, message, None)
            self._emit(TrainingEvent(phase="failed", message=message))
            return

        if knn_clf is None:
            message = "No valid training samples were found."
            self._record_result(False, message, None)
            self._emit(TrainingEvent(phase="failed", message=message))
            return

        message = f"Training completed. Model saved to {model_save_path}"
        self._record_result(True, message, model_save_path)
        self._emit(TrainingEvent(phase="completed", message=message, details={"model_path": model_save_path}))

    def _record_result(self, success: bool, message: str, model_path: Optional[str]) -> None:
        self._result = TrainingResult(success=success, message=message, model_path=model_path)

    def _emit(self, event: TrainingEvent) -> None:
        if self.event_bus is None:
            return
        self.event_bus.emit(event)
