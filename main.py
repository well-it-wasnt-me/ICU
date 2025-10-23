"""
Main Application Module.

This module serves as the entry point for the face recognition application.
It handles command-line argument parsing, training the model, loading configuration,
and initiating the stream processing.
"""

import argparse
import os
import yaml
import threading
import time

# Ensure facenet-pytorch uses the custom imresample function from ImageUtils
import facenet_pytorch.models.utils.detect_face as detect_face_module
from image_utils import ImageUtils

# Override the original imresample with custom_imresample
detect_face_module.imresample = ImageUtils.custom_imresample

from logger_setup import logger, configure_logging
from face_recognizer import FaceRecognizer
from stream_processor import StreamProcessor
from notifications import NotificationManager
from resource_monitor import ResourceMonitor
from stream_finder import CameraStreamFinder, DiscoveredStream
from plate_monitor import PlateDetectionHandler, PlateStore
from plate_recognizer import PlateRecognizer


def main():
    """
    Entry point of the face recognition application.

    Parses command-line arguments, trains or loads the KNN classifier,
    loads camera configurations, and starts stream processing threads for each camera.
    """
    parser = argparse.ArgumentParser(
        description='Face Recognition from Live Camera Stream'
    )
    parser.add_argument('--train_dir', type=str, default='poi', help='Directory with training images')
    parser.add_argument('--model_save_path', type=str, default='trained_knn_model.clf',
                        help='Path to save/load KNN model')
    parser.add_argument('--n_neighbors', type=int, default=None, help='Number of neighbors for KNN')
    parser.add_argument('--camera_config', '--config', dest='camera_config', type=str,
                        default='configs/cameras.yaml', help='Path to camera configuration file')
    parser.add_argument('--app_config', type=str, default='configs/app.yaml',
                        help='Path to application configuration file')
    parser.add_argument('--distance_threshold', type=float, default=0.5, help='Distance threshold for recognition')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU with facenet-pytorch')
    parser.add_argument('--target_processing_fps', type=float, default=0.0,
                        help='Target processing rate per camera (0 disables rate limiting)')
    parser.add_argument('--cpu_pressure_threshold', type=float, default=85.0,
                        help='CPU usage threshold to trigger adaptive throttling')
    parser.add_argument('--find-camera', action='store_true',
                        help='Interactively search for public camera streams by city')
    args = parser.parse_args()

    if args.find_camera:
        handle_camera_lookup()
        return
    # Convert images in training directory to RGB (if needed)
    ImageUtils.convert_images_to_rgb(args.train_dir)

    face_recognizer = FaceRecognizer(use_gpu=args.use_gpu)
    logger.info(f"Using device: {face_recognizer.device}")

    if args.train:
        logger.info("Training KNN classifier...")
        if args.use_gpu:
            # Initialize facenet-pytorch models for GPU-based training
            face_recognizer.initialize_facenet_pytorch_models()

        knn_clf = face_recognizer.train(
            args.train_dir,
            model_save_path=args.model_save_path,
            n_neighbors=args.n_neighbors,
            knn_algo='ball_tree',
            verbose=True
        )
        if knn_clf is not None:
            logger.info("Training complete!")
        else:
            logger.error("Training failed.")
        return

    # Load pre-trained model if not in training mode
    if not os.path.exists(args.model_save_path):
        logger.error(f"Model file {args.model_save_path} does not exist. Train first using --train.")
        return

    face_recognizer.load_model(args.model_save_path)

    # Load camera configuration
    camera_config_path = args.camera_config
    if not os.path.exists(camera_config_path):
        logger.error(f"Camera configuration file {camera_config_path} does not exist.")
        return

    try:
        with open(camera_config_path, 'r') as file:
            camera_config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing camera configuration: {exc}")
        return

    if isinstance(camera_config, list):
        cameras = camera_config
    elif isinstance(camera_config, dict):
        cameras = camera_config.get('cameras', [])
    else:
        cameras = []

    if not cameras:
        logger.error("No camera definitions found in the camera configuration file.")
        return

    # Load reference images for side-by-side screenshot captures
    reference_images = face_recognizer.load_reference_images(args.train_dir)

    # Initialize facenet-pytorch models for inference if GPU is used
    if args.use_gpu:
        face_recognizer.initialize_facenet_pytorch_models()

    app_config_path = args.app_config
    app_config = {}
    if app_config_path and os.path.exists(app_config_path):
        try:
            with open(app_config_path, 'r') as file:
                app_config = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing application configuration: {exc}")
            return
    else:
        logger.info(f"Application configuration file '{app_config_path}' not found. Using CLI/default settings.")

    settings = app_config.get('settings', {})
    target_processing_fps = settings.get('target_processing_fps', args.target_processing_fps)
    cpu_pressure_threshold = settings.get('cpu_pressure_threshold', args.cpu_pressure_threshold)

    configure_logging(app_config)

    target_fps = target_processing_fps if target_processing_fps and target_processing_fps > 0 else None
    resource_monitor = None
    if target_fps or cpu_pressure_threshold > 0:
        resource_monitor = ResourceMonitor()
        resource_monitor.start()

    notifications_cfg = app_config.get('notifications', {})
    telegram_cfg = notifications_cfg.get('telegram', {}) if notifications_cfg else {}

    notifier = None
    bot_token = telegram_cfg.get('bot_token')
    chat_id = telegram_cfg.get('chat_id')
    if bot_token and chat_id:
        notifier = NotificationManager(
            telegram_bot_token=bot_token,
            telegram_chat_id=chat_id,
            timeout=telegram_cfg.get('timeout', 10),
            max_workers=telegram_cfg.get('max_workers', 2),
            train_dir=args.train_dir,
            enable_command_handler=telegram_cfg.get('enable_commands', True),
            command_poll_timeout=telegram_cfg.get('command_poll_timeout', 20),
        )
    elif telegram_cfg and (bot_token or chat_id):
        logger.warning("Incomplete Telegram configuration detected. Notifications disabled.")

    plates_cfg = app_config.get('plates', {}) if app_config else {}
    plate_recognizer = None
    plate_handler = None
    if plates_cfg.get('enabled'):
        try:
            plate_recognizer = PlateRecognizer(
                cascade_path=plates_cfg.get('cascade_path'),
                ocr_languages=plates_cfg.get('ocr', {}).get('languages'),
                ocr_gpu=plates_cfg.get('ocr', {}).get('use_gpu', False),
                min_confidence=plates_cfg.get('min_confidence', 0.5),
                min_plate_length=plates_cfg.get('min_plate_length', 4),
                max_plate_length=plates_cfg.get('max_plate_length', 10),
                max_detections_per_frame=plates_cfg.get('max_detections_per_frame', 5),
            )
        except Exception as exc:
            logger.error("Failed to initialise plate recognizer: %s", exc)
            plate_recognizer = None

        if plate_recognizer and plate_recognizer.enabled:
            storage_cfg = plates_cfg.get('storage', {})
            base_dir = storage_cfg.get('base_dir', os.path.join('captures', 'plates'))
            summary_filename = storage_cfg.get('summary_file', 'plates_summary.json')
            max_captures = storage_cfg.get('max_captures_per_plate', 20)
            plate_store = PlateStore(
                base_dir=base_dir,
                summary_filename=summary_filename,
                max_captures_per_plate=max_captures,
            )
            notify_toggle = plates_cfg.get('use_notifications', True)
            handler_notifier = notifier if notify_toggle else None
            plate_handler = PlateDetectionHandler(
                store=plate_store,
                notifier=handler_notifier,
                watchlist=tuple(plates_cfg.get('watchlist', [])),
                alert_on_every_plate=plates_cfg.get('alert_on_every_plate', False),
                alert_on_watchlist=plates_cfg.get('alert_on_watchlist', True),
                capture_cooldown=plates_cfg.get('capture_cooldown', 30),
                crop_padding=plates_cfg.get('crop_padding', 8),
            )
        elif plate_recognizer and not plate_recognizer.enabled:
            logger.warning("Plate recognition disabled: OCR backend unavailable.")
            plate_recognizer = None

    stream_processor = StreamProcessor(
        face_recognizer=face_recognizer,
        reference_images=reference_images,
        base_capture_dir='captures',
        notifier=notifier,
        target_processing_fps=target_fps,
        resource_monitor=resource_monitor,
        cpu_pressure_threshold=cpu_pressure_threshold,
        plate_recognizer=plate_recognizer,
        plate_handler=plate_handler,
    )

    # Start processing each camera in a separate thread
    threads = []
    for cam_config in cameras:
        t = threading.Thread(
            target=stream_processor.process_stream,
            args=(cam_config, args.distance_threshold),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)

    try:
        while any(t.is_alive() for t in threads):
            if stream_processor.stop_requested():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping streams...")
        stream_processor.request_stop()
    finally:
        stream_processor.request_stop()
        for t in threads:
            t.join()

    logger.info("All camera threads finished.")
    if notifier:
        notifier.shutdown()
    if resource_monitor:
        resource_monitor.stop()


def handle_camera_lookup():
    city = input("Enter a city to find public camera streams: ").strip()
    if not city:
        logger.error("No city provided. Aborting search.")
        return

    finder = CameraStreamFinder()
    streams = finder.find_streams(city)
    if not streams:
        logger.info("No camera streams found for city '%s'.", city)
        return

    city_slug = city.strip().lower().replace(" ", "_")
    output_path = f"camera_streams_{city_slug}.yaml"
    try:
        serialized = [
            _serialize_stream(stream)
            for stream in streams
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(serialized, f, sort_keys=False)

        logger.info("Discovered %s stream(s):", len(streams))
        for stream in streams:
            logger.info("  [%s] %s", stream.protocol.upper(), stream.url)
        logger.info("Saved results to %s", output_path)
    except OSError as exc:
        logger.error("Failed to write camera search results: %s", exc)


def _serialize_stream(stream: DiscoveredStream) -> dict:
    data = {
        "url": stream.url,
        "protocol": stream.protocol,
    }
    if stream.headers:
        data["headers"] = stream.headers
    return data


if __name__ == "__main__":
    main()
