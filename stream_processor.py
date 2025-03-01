"""
Stream Processor Module.

Provides functionality to capture frames from video streams and process them for face recognition.
Includes a generator to yield frames from a stream and a class to handle stream processing.
"""

import os
import time
from datetime import datetime
import av
import cv2
from image_utils import ImageUtils
from logger_setup import logger


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

    def __init__(self, face_recognizer, reference_images, base_capture_dir='captures'):
        """
        Initialize the StreamProcessor.

        :param face_recognizer: An instance of FaceRecognizer to perform face predictions.
        :param reference_images: Dictionary mapping person names to their reference images.
        :param base_capture_dir: Base directory to save capture images.
        """
        self.face_recognizer = face_recognizer
        self.reference_images = reference_images
        self.base_capture_dir = base_capture_dir

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

        frames = get_frames_from_stream(stream_url, headers=headers)
        frame_count = 0

        # Create directory to store captured images
        capture_dir = os.path.join(self.base_capture_dir, name)
        if not os.path.exists(capture_dir):
            os.makedirs(capture_dir)

        # Track last saved timestamp for each recognized person
        last_saved = {}

        logger.info(f"[{name}] CONNECTED. Starting analysis process...")

        try:
            for frame in frames:
                frame_count += 1
                # Process only every Nth frame to reduce processing load
                if frame_count % process_frame_interval == 0:
                    # Resize frame for faster processing
                    img_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                    predictions = self.face_recognizer.predict(
                        img_small,
                        distance_threshold=distance_threshold
                    )

                    for (pred_name, face_dist) in predictions:
                        if pred_name == "unknown":
                            continue  # Skip if face is not recognized

                        # Calculate confidence as a percentage based on the distance metric
                        confidence = (1.0 - (face_dist / distance_threshold)) * 100.0
                        confidence = max(0, min(100, confidence))  # Clamp confidence between 0 and 100

                        logger.info(f"[{name}] Detected known face: {pred_name} at {int(confidence)}%")

                        current_time = time.time()
                        last_time = last_saved.get(pred_name, 0)

                        # Skip saving if within the cooldown period
                        if (current_time - last_time) < capture_cooldown:
                            logger.info(f"[{name}] Skipping save for {pred_name}, still in cooldown.")
                            continue

                        # Annotate the frame with a timestamp
                        annotated_frame = ImageUtils.add_timestamp(frame.copy())
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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

                # Check if the 'q' key is pressed for a graceful exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info(f"[{name}] Interrupted by user. Exiting...")
        except Exception as e:
            logger.error(f"[{name}] An error occurred: {e}")
        finally:
            logger.info(f"[{name}] Stream processing terminated.")
