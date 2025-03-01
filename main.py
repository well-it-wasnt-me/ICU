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

from logger_setup import logger
from face_recognizer import FaceRecognizer
from stream_processor import StreamProcessor


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
    parser.add_argument('--config', type=str, default='cameras.yaml', help='Path to YAML config')
    parser.add_argument('--distance_threshold', type=float, default=0.5, help='Distance threshold for recognition')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU with facenet-pytorch')
    args = parser.parse_args()

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

    # Load camera configuration from YAML file
    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} does not exist.")
        return

    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            cameras = config.get('cameras', [])
            if not cameras:
                logger.error("No camera configurations found in the YAML file.")
                return
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        return

    # Load reference images for side-by-side screenshot captures
    reference_images = face_recognizer.load_reference_images(args.train_dir)

    # Initialize facenet-pytorch models for inference if GPU is used
    if args.use_gpu:
        face_recognizer.initialize_facenet_pytorch_models()

    stream_processor = StreamProcessor(
        face_recognizer=face_recognizer,
        reference_images=reference_images,
        base_capture_dir='captures'
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

    for t in threads:
        t.join()

    logger.info("All camera threads finished.")


if __name__ == "__main__":
    main()
