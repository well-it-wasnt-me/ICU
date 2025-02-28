import os
import time
from datetime import datetime
import av
import cv2
from image_utils import ImageUtils
from logger_setup import logger

def get_frames_from_stream(url, headers=None, reconnect_interval=300):
    """
    Generator function that yields video frames from a given stream URL.

    This function supports two modes:
      1. Local webcam: when the URL can be cast to an integer.
      2. Network/other streams: when the URL is a proper address.

    Parameters:
      url (str): URL of the video stream or an integer (as a string) for local webcam.
      headers (dict, optional): HTTP headers to use when opening a network stream.
      reconnect_interval (int, optional): Time interval (in seconds) after which to reconnect.

    Yields:
      frame (numpy.ndarray): The next video frame in BGR format.
    """
    start_time = time.time()
    try:
        # Attempt to treat the URL as a webcam index (integer)
        cam_index = int(url)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            logger.error(f"Cannot open local webcam {cam_index}")
            return

        # Continuous loop to capture frames from local webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                # Error reading frame; try to reconnect
                logger.error(f"Failed to grab frame from local webcam {cam_index}. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(cam_index)
                start_time = time.time()  # Reset the reconnect timer
                continue
            yield frame

            # Check if it's time to reconnect based on the reconnect interval
            if time.time() - start_time > reconnect_interval:
                logger.info(f"Reconnect interval reached for webcam {cam_index}. Re-opening.")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(cam_index)
                start_time = time.time()

    except ValueError:
        # The URL is not an integer, so treat it as a network stream URL
        container = None
        while True:
            try:
                if container is None:
                    # Open the network stream, optionally using custom headers if provided
                    if headers:
                        headers_str = '\r\n'.join([f"{key}: {value}" for key, value in headers.items()])
                        options = {'headers': headers_str}
                        container = av.open(url, options=options)
                    else:
                        container = av.open(url)
                    start_time = time.time()  # Reset timer upon connection
                    stream = container.streams.video[0]  # Get the first video stream

                # Decode each frame from the container
                for frame in container.decode(stream):
                    # Convert the frame to a numpy array in BGR format (compatible with OpenCV)
                    img = frame.to_ndarray(format='bgr24')
                    yield img

                    # Reconnect after the specified interval to refresh the connection
                    if time.time() - start_time > reconnect_interval:
                        logger.info("Reconnect interval reached for stream. Restarting it")
                        container.close()
                        container = None
                        break  # Exit the inner loop to restart connection


            except Exception as e:
                # Handle any errors in opening or decoding the stream
                logger.error(f"Well, i failed opening {url}: {e}")
                if container is not None:
                    container.close()
                container = None
                time.sleep(5)  # Wait before retrying the connection

class StreamProcessor:
    """
    Class to process video streams continuously for face recognition.

    It uses a provided face recognizer to detect known faces in the stream,
    logs detections, and saves screenshot captures of the detected faces.
    """

    def __init__(self, face_recognizer, reference_images, base_capture_dir='captures'):
        """
        Initialize the stream processor.

        Parameters:
          face_recognizer: An object with a predict() method to identify faces.
          reference_images (dict): Mapping of person names to their reference images.
          base_capture_dir (str): Base directory where capture images are stored.
        """
        self.face_recognizer = face_recognizer
        self.reference_images = reference_images
        self.base_capture_dir = base_capture_dir

    def process_stream(self, camera_config, distance_threshold):
        """
        Process a video stream to detect faces, log events, and save captures.

        Parameters:
          camera_config (dict): Configuration for the camera, including:
            - name: Name identifier for the camera.
            - stream_url: URL or index for the video stream.
            - headers: (Optional) HTTP headers for network streams.
            - process_frame_interval: (Optional) Process every Nth frame.
            - capture_cooldown: (Optional) Cooldown time (in seconds) between saves per person.
          distance_threshold (float): Threshold for face recognition distance metric.
        """
        # Retrieve camera configuration parameters
        name = camera_config.get('name', 'Unnamed Camera')
        stream_url = camera_config['stream_url']
        headers = camera_config.get('headers', {})
        process_frame_interval = camera_config.get('process_frame_interval', 30)
        capture_cooldown = camera_config.get('capture_cooldown', 60)

        logger.info(f"[{name}] Connecting to the video stream...")

        # Get frames from the specified video stream
        frames = get_frames_from_stream(stream_url, headers=headers)
        frame_count = 0

        # Create directory to save captured frames if it doesn't exist
        capture_dir = os.path.join(self.base_capture_dir, name)
        if not os.path.exists(capture_dir):
            os.makedirs(capture_dir)

        # Dictionary to track the last saved timestamp for each detected person
        last_saved = {}

        logger.info(f"[{name}] CONNECTED. Starting Analysis process...")

        try:
            for frame in frames:
                frame_count += 1
                # Process only every Nth frame to reduce processing load
                if frame_count % process_frame_interval == 0:
                    # Resize frame to half its original size for faster face recognition processing
                    img_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                    # Use the face recognizer to predict faces in the resized image
                    predictions = self.face_recognizer.predict(
                        img_small,
                        distance_threshold=distance_threshold
                    )

                    # Iterate over each face prediction
                    for (pred_name, face_dist) in predictions:
                        if pred_name == "unknown":
                            continue  # Skip processing if the face is not recognized

                        # Calculate a confidence percentage based on the face distance metric
                        confidence = (1.0 - (face_dist / distance_threshold)) * 100.0
                        confidence = max(0, min(100, confidence))  # Clamp confidence between 0 and 100

                        logger.info(f"[{name}] Detected known face: {pred_name} at {int(confidence)}%")

                        current_time = time.time()
                        last_time = last_saved.get(pred_name, 0)

                        # Check if the capture for this person is still in the cooldown period
                        if (current_time - last_time) < capture_cooldown:
                            logger.info(f"[{name}] Skipping save for {pred_name}, still in cooldown.")
                            continue

                        # Annotate the frame with a timestamp before saving
                        annotated_frame = ImageUtils.add_timestamp(frame.copy())
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{name}_{pred_name}_{timestamp}.jpg"
                        filepath = os.path.join(capture_dir, filename)
                        try:
                            cv2.imwrite(filepath, annotated_frame)
                            logger.info(f"[{name}] Saved captured frame to {filepath}")
                            # Update the last saved time for this person
                            last_saved[pred_name] = current_time
                        except Exception as e:
                            logger.error(f"[{name}] Failed to save frame: {e}")

                        # Create a side-by-side screenshot combining the current frame and the reference image
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

                # Check if the 'q' key is pressed to allow graceful exit from the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            # Allow the user to interrupt the process via keyboard (Ctrl+C)
            logger.info(f"[{name}] Interrupted by user. Exiting...")
        except Exception as e:
            # Log any unexpected errors that occur during stream processing
            logger.error(f"[{name}] An error occurred: {e}")
        finally:
            # Log termination of the stream processing
            logger.info(f"[{name}] Stream processing terminated.")
