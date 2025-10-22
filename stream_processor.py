"""
Stream Processor Module.

Provides functionality to capture frames from video streams and process them for face recognition.
Includes a generator to yield frames from a stream and a class to handle stream processing.
"""

import os
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional
import av
import cv2
from image_utils import ImageUtils
from logger_setup import logger
from notifications import NotificationManager
from resource_monitor import ResourceMonitor
from runtime_events import (
    DetectionEvent,
    RuntimeEvent,
    StreamLifecycleEvent,
    StreamMetricsEvent,
)


def get_frames_from_stream(url, headers=None, reconnect_interval=300, stop_event: Optional[threading.Event] = None):
    """
    Generator function to yield video frames from a specified stream.

    Supports local webcam streams (if `url` is an integer in string format) and network streams.

    :param url: URL of the video stream or an index (as a string) for a local webcam.
    :param headers: Optional dictionary of HTTP headers for network streams.
    :param reconnect_interval: Time in seconds after which to reconnect to the stream otherwise you'll fill the entire hard disk
    :param stop_event: Optional threading.Event used to request termination of the stream.
    :yield: Video frame as a numpy array in BGR format.
    """
    start_time = time.time()
    try:
        # Attempt to treat the URL as a webcam index (integer)
        cam_index = int(url)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            logger.error(f"Cannot open local webcam {cam_index}")
            return

        # Continuously capture frames from the local webcam
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Failed to grab frame from local webcam {cam_index}. Reconnecting...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(cam_index)
                    start_time = time.time()  # Reset the reconnect timer
                    continue
                yield frame

                if stop_event and stop_event.is_set():
                    break

                # Reconnect after the specified interval
                if time.time() - start_time > reconnect_interval:
                    logger.info(f"Reconnect interval reached for webcam {cam_index}. Re-opening.")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(cam_index)
                    start_time = time.time()
        finally:
            cap.release()

    except ValueError:
        # URL is not an integer; treat as a network stream URL
        container = None
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                try:
                    if container is None:
                        if headers:
                            headers_str = '\r\n'.join([f"{key}: {value}" for key, value in headers.items()])
                            options = {'headers': headers_str}
                            container = av.open(url, options=options)
                        else:
                            container = av.open(url)
                        start_time = time.time()  # Reset timer upon connection
                        stream = container.streams.video[0]

                    # Decode each frame from the stream container
                    while True:
                        try:
                            frame = next(container.decode(stream))
                        except StopIteration:
                            break
                        except av.AVError as decode_err:
                            logger.warning(f"[Decode] {url}: {decode_err}. Skipping frame.")
                            continue

                        if stop_event and stop_event.is_set():
                            break

                        try:
                            img = frame.to_ndarray(format='bgr24')
                        except av.AVError as convert_err:
                            logger.warning(f"[Convert] {url}: {convert_err}. Dropping frame.")
                            continue

                        yield img

                        if stop_event and stop_event.is_set():
                            break

                        # Reconnect after the specified interval
                        if time.time() - start_time > reconnect_interval:
                            logger.info("Reconnect interval reached for stream. Restarting it.")
                            container.close()
                            container = None
                            break

                    if stop_event and stop_event.is_set():
                        break

                except Exception as e:
                    logger.error(f"Failed to open stream {url}: {e}")
                    if container is not None:
                        container.close()
                    container = None
                    time.sleep(5)
        finally:
            if container is not None:
                container.close()


class StreamProcessor:
    """
    A class to process video streams for face recognition.

    Captures frames from a video stream, applies face recognition,
    logs detections, and saves screenshots with annotations.
    """

    def __init__(
        self,
        face_recognizer,
        reference_images,
        base_capture_dir: str = 'captures',
        notifier: Optional[NotificationManager] = None,
        target_processing_fps: Optional[float] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        cpu_pressure_threshold: float = 85.0,
        pressure_backoff_factor: float = 2.0,
        event_publisher: Optional[Callable[[RuntimeEvent], None]] = None,
    ):
        """
        Initialize the StreamProcessor.

        :param face_recognizer: An instance of FaceRecognizer to perform face predictions.
        :param reference_images: Dictionary mapping person names to their reference images.
        :param base_capture_dir: Base directory to save capture images.
        """
        self.face_recognizer = face_recognizer
        self.reference_images = reference_images
        self.base_capture_dir = base_capture_dir
        self.notifier = notifier
        self.resource_monitor = resource_monitor
        self.cpu_pressure_threshold = cpu_pressure_threshold
        self.pressure_backoff_factor = pressure_backoff_factor
        self.min_processing_interval = (
            0 if not target_processing_fps else max(0.0, 1.0 / target_processing_fps)
        )
        self._stop_event = threading.Event()
        self._event_publisher = event_publisher

    def process_stream(self, camera_config, distance_threshold):
        """
        Process a video stream to detect faces and save capture images.

        Reads frames from the video stream specified in `camera_config`, processes every Nth frame,
        performs face recognition, and saves annotated frames and side-by-side screenshots for recognized faces.

        :param camera_config: Dictionary containing camera configuration parameters:
            - name: Identifier for the camera.
            - stream_url: URL or index for the video stream.
            - headers: (Optional) HTTP headers for network streams.
            - process_frame_interval: (Optional) Process every Nth frame.
            - capture_cooldown: (Optional) Minimum time (in seconds) between captures for the same person.
        :param distance_threshold: Float specifying the maximum allowed distance for a valid face match.
        """
        # Retrieve camera configuration parameters
        name = camera_config.get('name', 'Unnamed Camera')
        stream_url = camera_config['stream_url']
        headers = camera_config.get('headers', {})
        process_frame_interval = camera_config.get('process_frame_interval', 30)
        capture_cooldown = camera_config.get('capture_cooldown', 60)

        logger.info(f"[{name}] Connecting to the video stream...")
        self._emit_event(StreamLifecycleEvent(camera_name=name, status="connecting"))
        frames = get_frames_from_stream(stream_url, headers=headers, stop_event=self._stop_event)
        frame_count = 0
        processed_frames = 0
        skipped_frames = 0
        connected_emitted = False

        # Create directory to store captured images
        capture_dir = os.path.join(self.base_capture_dir, name)
        if not os.path.exists(capture_dir):
            os.makedirs(capture_dir)

        # Track last saved timestamp for each recognized person
        last_saved: Dict[str, float] = {}

        logger.info(f"[{name}] CONNECTED. Starting analysis process...")
        try:
            last_processed_at = 0.0
            for frame in frames:
                frame_count += 1
                if self._stop_event.is_set():
                    break
                if not connected_emitted:
                    self._emit_event(StreamLifecycleEvent(camera_name=name, status="connected"))
                    connected_emitted = True

                if not self._should_process_frame(frame_count, process_frame_interval):
                    skipped_frames += 1
                    continue

                now = time.time()
                min_interval = self._current_min_interval()
                if now - last_processed_at < min_interval:
                    skipped_frames += 1
                    continue
                last_processed_at = now
                start_time = time.perf_counter()
                img_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                predictions = self.face_recognizer.predict(
                    img_small,
                    distance_threshold=distance_threshold
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                processed_frames += 1
                self._emit_event(
                    StreamMetricsEvent(
                        camera_name=name,
                        frame_count=frame_count,
                        processed_frames=processed_frames,
                        skipped_frames=skipped_frames,
                        last_latency_ms=latency_ms,
                    )
                )

                for (pred_name, face_dist) in predictions:
                    if pred_name == "unknown":
                        continue  # Skip if face is not recognized

                    confidence = self._calculate_confidence(face_dist)

                    logger.info(f"[{name}] Detected known face: {pred_name} at {int(confidence)}%")

                    current_time = time.time()
                    last_time = last_saved.get(pred_name, 0)

                    # Skip saving if within the cooldown period
                    if (current_time - last_time) < capture_cooldown:
                        logger.info(f"[{name}] Skipping save for {pred_name}, still in cooldown.")
                        self._emit_event(
                            DetectionEvent(
                                camera_name=name,
                                person_name=pred_name,
                                confidence=confidence,
                                distance=face_dist,
                                cooldown_active=True,
                            )
                        )
                        continue

                    # Annotate the frame with a timestamp
                    annotated_frame = ImageUtils.add_timestamp(frame.copy())
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{name}_{pred_name}_{timestamp}.jpg"
                    filepath = os.path.join(capture_dir, filename)
                    try:
                        cv2.imwrite(filepath, annotated_frame)
                        logger.info(f"[{name}] Saved captured frame to {filepath}")
                        last_saved[pred_name] = current_time
                    except Exception as e:
                        logger.error(f"[{name}] Failed to save frame: {e}")

                    # Create a side-by-side screenshot with the reference image
                    ref_img = self.reference_images.get(pred_name, None)
                    side_by_side = ImageUtils.create_side_by_side_screenshot(
                        frame,
                        ref_img,
                        camera_name=name,
                        person_name=pred_name,
                        confidence=confidence
                    )

                    side_filename = f"{name}_{pred_name}_{timestamp}_sidebyside.jpg"
                    side_filepath = os.path.join(capture_dir, side_filename)
                    try:
                        cv2.imwrite(side_filepath, side_by_side)
                        logger.info(f"[{name}] Saved side-by-side screenshot to {side_filepath}")
                    except Exception as e:
                        logger.error(f"[{name}] Failed to save side-by-side screenshot: {e}")

                    if self.notifier:
                        self.notifier.notify_detection(
                            camera_name=name,
                            person_name=pred_name,
                            confidence=confidence,
                            capture_path=filepath,
                            side_by_side_path=side_filepath,
                            timestamp=datetime.utcnow(),
                        )
                    self._emit_event(
                        DetectionEvent(
                            camera_name=name,
                            person_name=pred_name,
                            confidence=confidence,
                            distance=face_dist,
                            capture_path=filepath,
                            side_by_side_path=side_filepath,
                        )
                    )

        except KeyboardInterrupt:
            logger.info(f"[{name}] Interrupted by user. Exiting...")
        except Exception as e:
            logger.error(f"[{name}] An error occurred: {e}")
            self._emit_event(
                StreamLifecycleEvent(
                    camera_name=name,
                    status="error",
                    message=str(e),
                )
            )
        finally:
            logger.info(f"[{name}] Stream processing terminated.")
            if hasattr(frames, "close"):
                try:
                    frames.close()
                except Exception:
                    pass
            self._emit_event(StreamLifecycleEvent(camera_name=name, status="stopped"))

    def request_stop(self) -> None:
        """
        Signal the processor to stop processing streams.
        """
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _should_process_frame(self, frame_count: int, base_interval: int) -> bool:
        if base_interval <= 1:
            return True
        if self.cpu_pressure_threshold > 0 and self.resource_monitor and self.resource_monitor.is_under_pressure(self.cpu_pressure_threshold):
            backoff = max(1, int(self.pressure_backoff_factor))
            return frame_count % (base_interval * backoff) == 0
        return frame_count % base_interval == 0

    def _current_min_interval(self) -> float:
        if self.min_processing_interval == 0:
            return 0
        if self.cpu_pressure_threshold > 0 and self.resource_monitor and self.resource_monitor.is_under_pressure(self.cpu_pressure_threshold):
            return self.min_processing_interval * self.pressure_backoff_factor
        return self.min_processing_interval

    @staticmethod
    def _calculate_confidence(face_distance: float) -> float:
        """
        Convert face distance (0 = identical, 1 = different) into a percentage confidence.
        """
        return max(0.0, min(100.0, (1.0 - min(face_distance, 1.0)) * 100.0))

    def _emit_event(self, event: RuntimeEvent) -> None:
        if not self._event_publisher:
            return
        try:
            self._event_publisher(event)
        except Exception:
            logger.debug("Failed to publish runtime event", exc_info=True)
