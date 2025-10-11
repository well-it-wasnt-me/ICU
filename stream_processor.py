"""
Stream Processor Module.

Provides functionality to capture frames from video streams and process them for face recognition.
Includes a generator to yield frames from a stream and a class to handle stream processing.
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional
import av
import cv2
from image_utils import ImageUtils
from logger_setup import logger
from notifications import NotificationManager
from resource_monitor import ResourceMonitor
from tui_manager import TuiManager


def get_frames_from_stream(url, headers=None, reconnect_interval=300):
    """
    Generator function to yield video frames from a specified stream.

    Supports local webcam streams (if `url` is an integer in string format) and network streams.

    :param url: URL of the video stream or an index (as a string) for a local webcam.
    :param headers: Optional dictionary of HTTP headers for network streams.
    :param reconnect_interval: Time in seconds after which to reconnect to the stream otherwise you'll fill the entire hard disk
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
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to grab frame from local webcam {cam_index}. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(cam_index)
                start_time = time.time()  # Reset the reconnect timer
                continue
            yield frame

            # Reconnect after the specified interval
            if time.time() - start_time > reconnect_interval:
                logger.info(f"Reconnect interval reached for webcam {cam_index}. Re-opening.")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(cam_index)
                start_time = time.time()

    except ValueError:
        # URL is not an integer; treat as a network stream URL
        container = None
        while True:
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
                for frame in container.decode(stream):
                    img = frame.to_ndarray(format='bgr24')
                    yield img

                    # Reconnect after the specified interval
                    if time.time() - start_time > reconnect_interval:
                        logger.info("Reconnect interval reached for stream. Restarting it.")
                        container.close()
                        container = None
                        break

            except Exception as e:
                logger.error(f"Failed to open stream {url}: {e}")
                if container is not None:
                    container.close()
                container = None
                time.sleep(5)


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
        tui: Optional[TuiManager] = None,
        show_preview: bool = False,
        preview_scale: float = 0.5,
        target_processing_fps: Optional[float] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        cpu_pressure_threshold: float = 85.0,
        pressure_backoff_factor: float = 2.0,
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
        self.tui = tui
        self.show_preview = show_preview
        self.preview_scale = preview_scale if preview_scale > 0 else 1.0
        self.resource_monitor = resource_monitor
        self.cpu_pressure_threshold = cpu_pressure_threshold
        self.pressure_backoff_factor = pressure_backoff_factor
        self.min_processing_interval = (
            0 if not target_processing_fps else max(0.0, 1.0 / target_processing_fps)
        )

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
        if self.tui:
            self.tui.update_camera(name, "connecting", stream_url)

        frames = get_frames_from_stream(stream_url, headers=headers)
        frame_count = 0

        # Create directory to store captured images
        capture_dir = os.path.join(self.base_capture_dir, name)
        if not os.path.exists(capture_dir):
            os.makedirs(capture_dir)

        # Track last saved timestamp for each recognized person
        last_saved: Dict[str, float] = {}

        logger.info(f"[{name}] CONNECTED. Starting analysis process...")
        if self.tui:
            self.tui.update_camera(name, "online", "analyzing")

        try:
            last_processed_at = 0.0
            for frame in frames:
                frame_count += 1

                if self.show_preview:
                    preview_frame = frame
                    if 0 < self.preview_scale != 1.0:
                        preview_frame = cv2.resize(frame, (0, 0), fx=self.preview_scale, fy=self.preview_scale)
                    cv2.imshow(name, preview_frame)

                if not self._should_process_frame(frame_count, process_frame_interval):
                    continue

                now = time.time()
                min_interval = self._current_min_interval()
                if now - last_processed_at < min_interval:
                    continue
                last_processed_at = now

                img_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                predictions = self.face_recognizer.predict(
                    img_small,
                    distance_threshold=distance_threshold
                )

                for (pred_name, face_dist) in predictions:
                    if pred_name == "unknown":
                        continue  # Skip if face is not recognized

                    confidence = self._calculate_confidence(face_dist)

                    logger.info(f"[{name}] Detected known face: {pred_name} at {int(confidence)}%")
                    if self.tui:
                        self.tui.notify_detection(name, pred_name, confidence)

                    current_time = time.time()
                    last_time = last_saved.get(pred_name, 0)

                    # Skip saving if within the cooldown period
                    if (current_time - last_time) < capture_cooldown:
                        logger.info(f"[{name}] Skipping save for {pred_name}, still in cooldown.")
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

                # Check if the 'q' key is pressed for a graceful exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info(f"[{name}] Interrupted by user. Exiting...")
            if self.tui:
                self.tui.update_camera(name, "interrupted")
        except Exception as e:
            logger.error(f"[{name}] An error occurred: {e}")
            if self.tui:
                self.tui.update_camera(name, "error", str(e))
        finally:
            logger.info(f"[{name}] Stream processing terminated.")
            if self.tui:
                self.tui.update_camera(name, "offline", "stopped")
            if self.show_preview:
                cv2.destroyWindow(name)

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
