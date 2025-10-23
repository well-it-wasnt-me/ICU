"""
Face Recognizer Module.

This module provides the FaceRecognizer class which is responsible for:
- Determining the appropriate processing device (CUDA, MPS, or CPU)
- Training a KNN classifier using face embeddings from training images
- Making predictions on input frames with the trained classifier
- Loading reference images for known individuals
"""

import math
import os
import pickle
import sys
import threading
import cv2
from contextlib import nullcontext
import face_recognition
import torch
from PIL import Image
from face_recognition.face_recognition_cli import image_files_in_folder
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn import neighbors

from logger_setup import logger


class FaceRecognizer:
    """
    A class to handle face recognition operations including training, prediction, and model management.

    This class supports both GPU-based face recognition using facenet-pytorch and CPU-based recognition
    using the face_recognition library.
    """

    def __init__(self, use_gpu=False, device=None, knn_clf=None):
        """
        Initialize the FaceRecognizer.

        :param use_gpu: Boolean flag to indicate if GPU-based recognition should be used.
        :param device: Optional torch.device to override device selection.
        :param knn_clf: Pre-trained KNN classifier. If None, the classifier can be trained later.
        """
        self.use_gpu = use_gpu
        self.device = device if device else self.get_device(use_gpu)
        self.knn_clf = knn_clf
        self.models = None  # Tuple of (MTCNN, InceptionResnetV1) when GPU is used.
        self._dlib_lock = threading.Lock() if not use_gpu else None

    @staticmethod
    def get_device(use_gpu):
        """
        Determine the appropriate torch.device based on GPU availability.

        :param use_gpu: Boolean flag indicating if GPU should be used.
        :return: torch.device for GPU (cuda or mps) if available, otherwise CPU.
        """
        if use_gpu:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                logger.warning("GPU requested but not available. Using CPU.")
        return torch.device('cpu')

    @staticmethod
    def load_reference_images(train_dir):
        """
        Load reference images for each person from a training directory.

        The function expects each subdirectory in `train_dir` to be named after the person,
        and uses the first image found in each subdirectory as the reference image.

        :param train_dir: Directory containing subdirectories for each person with training images.
        :return: Dictionary mapping person names to their reference image (as a numpy array).
        """
        reference_images = {}
        for person_name in os.listdir(train_dir):
            person_folder = os.path.join(train_dir, person_name)
            if not os.path.isdir(person_folder):
                continue
            image_files = list(image_files_in_folder(person_folder))
            if len(image_files) == 0:
                continue
            ref_path = image_files[0]
            try:
                ref_img = cv2.imread(ref_path)
                reference_images[person_name] = ref_img
            except Exception as e:
                logger.error(f"Failed to load reference image {ref_path}: {e}")
        return reference_images

    def train(self, train_dir, model_save_path=None, n_neighbors=None,
              knn_algo='ball_tree', verbose=False):
        """
        Train the KNN classifier using face embeddings from training images.

        This method processes images for each person in the `train_dir`, extracts face embeddings,
        and trains a KNN classifier. Optionally, the trained model can be saved to disk.

        :param train_dir: Directory containing subdirectories for each person with training images.
        :param model_save_path: Optional file path to save the trained KNN classifier.
        :param n_neighbors: Number of neighbors for the KNN classifier. If None, computed automatically.
        :param knn_algo: KNN algorithm to use (default: 'ball_tree').
        :param verbose: Boolean flag to enable verbose logging.
        :return: Trained KNN classifier or None if training fails.
        """
        X = []
        y = []

        if self.use_gpu:
            logger.info(f"Using device for training: {self.device}")
            mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        else:
            mtcnn = None
            resnet = None

        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                if self.use_gpu:
                    img_pil = Image.fromarray(image).convert('RGB')
                    boxes, _ = mtcnn.detect(img_pil)
                    if boxes is None:
                        if verbose:
                            logger.warning(f"Image {img_path} not suitable: no face found")
                        continue

                    faces = mtcnn.extract(img_pil, boxes, save_path=None)
                    if faces is None or (faces.dim() == 3 and faces.shape[0] == 0):
                        if verbose:
                            logger.warning(f"Image {img_path} not suitable: no faces extracted")
                        continue
                    if faces.dim() == 3:
                        faces = faces.unsqueeze(0)

                    embeddings = resnet(faces.to(self.device)).detach().cpu().numpy()
                    if len(embeddings) != 1:
                        if verbose:
                            logger.warning(f"Image {img_path} has multiple faces. Skipping.")
                        continue

                    X.append(embeddings[0])
                    y.append(class_dir)
                else:
                    lock = self._dlib_lock or nullcontext()
                    with lock:
                        face_bounding_boxes = face_recognition.face_locations(image)
                        if len(face_bounding_boxes) != 1:
                            if verbose:
                                logger.warning(
                                    f"Image {img_path} not suitable: "
                                    f"{'No face found' if len(face_bounding_boxes) < 1 else 'More than one face found'}"
                                )
                            continue

                        encoding = face_recognition.face_encodings(
                            image, known_face_locations=face_bounding_boxes)[0]
                    X.append(encoding)
                    y.append(class_dir)

        if len(X) == 0:
            logger.error("No faces found in the training data. Aborting.")
            return None

        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                logger.info(f"Chose n_neighbors automatically: {n_neighbors}")

        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 algorithm=knn_algo,
                                                 weights='distance')
        knn_clf.fit(X, y)

        if model_save_path is not None:
            try:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"KNN classifier saved to {model_save_path}")
            except Exception as e:
                logger.error(f"Failed to save KNN classifier: {e}")

        self.knn_clf = knn_clf
        return knn_clf

    def load_model(self, model_path):
        """
        Load a pre-trained KNN classifier from disk.

        :param model_path: File path to the saved KNN classifier.
        :return: Loaded KNN classifier, or None if loading fails.
        """
        try:
            with open(model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)
            logger.info(f"Loaded KNN classifier from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load KNN classifier: {e}")
            self.knn_clf = None
        return self.knn_clf

    def initialize_facenet_pytorch_models(self):
        """
        Initialize the facenet-pytorch models (MTCNN and InceptionResnetV1) for GPU-based processing.
        """
        try:
            mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.models = (mtcnn, resnet)
        except Exception as e:
            logger.error(f"Failed to initialize facenet-pytorch models: {e}")
            sys.exit(1)

    def predict(self, X_frame, distance_threshold=0.5):
        """
        Predict the identities of faces present in the provided image frame.

        Depending on whether GPU-based processing is used, the method extracts face embeddings using
        either facenet-pytorch or the face_recognition library, and then uses the trained KNN classifier
        to determine the closest matches. A distance threshold is applied to filter out low-confidence matches.

        :param X_frame: A numpy array representing the image frame (BGR format).
        :param distance_threshold: A float specifying the threshold for valid face matching.
        :return: A list of dictionaries, each containing the predicted identity, face distance, and location
                 within the provided frame (as a tuple of (top, right, bottom, left)).
        """
        if not self.knn_clf:
            raise Exception("KNN classifier not loaded or trained.")

        if self.use_gpu:
            if not self.models:
                logger.error("Facenet-pytorch models must be provided when using GPU.")
                sys.exit(1)
            mtcnn, resnet = self.models
            img_pil = Image.fromarray(X_frame).convert('RGB')
            boxes, _ = mtcnn.detect(img_pil)
            if boxes is None:
                return []

            faces = mtcnn.extract(img_pil, boxes, save_path=None)
            if faces is None:
                return []

            if faces.dim() == 3:
                faces = faces.unsqueeze(0)

            embeddings = resnet(faces.to(self.device)).detach().cpu().numpy()
            closest_distances = self.knn_clf.kneighbors(embeddings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold
                           for i in range(len(embeddings))]

            predictions = []
            for i, (pred, rec, box) in enumerate(zip(self.knn_clf.predict(embeddings), are_matches, boxes)):
                face_distance = float(closest_distances[0][i][0])
                location = self._mtcnn_box_to_location(box, X_frame.shape)
                if rec:
                    predictions.append(
                        {
                            "name": pred,
                            "distance": face_distance,
                            "location": location,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "name": "unknown",
                            "distance": face_distance,
                            "location": location,
                        }
                    )
            return predictions

        else:
            lock = self._dlib_lock or nullcontext()
            with lock:
                X_face_locations = face_recognition.face_locations(X_frame)
                if len(X_face_locations) == 0:
                    return []
                faces_encodings = face_recognition.face_encodings(
                    X_frame,
                    known_face_locations=X_face_locations
                )
            closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold
                           for i in range(len(faces_encodings))]

            predictions = []
            for i, (pred, rec) in enumerate(zip(self.knn_clf.predict(faces_encodings), are_matches)):
                distance = float(closest_distances[0][i][0])
                location = self._clip_location(X_face_locations[i], X_frame.shape)
                if rec:
                    predictions.append(
                        {
                            "name": pred,
                            "distance": distance,
                            "location": location,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "name": "unknown",
                            "distance": distance,
                            "location": location,
                        }
                    )
            return predictions

    @staticmethod
    def _clip_location(location, frame_shape):
        """
        Ensure the face location stays within the frame boundaries.

        :param location: Tuple in (top, right, bottom, left) format.
        :param frame_shape: Shape tuple from the frame (height, width, channels).
        :return: Tuple of floats in (top, right, bottom, left) format within frame bounds.
        """
        top, right, bottom, left = location
        height, width = frame_shape[:2]
        max_y = max(0, height - 1)
        max_x = max(0, width - 1)
        top = float(max(0, min(max_y, top)))
        right = float(max(0, min(max_x, right)))
        bottom = float(max(0, min(max_y, bottom)))
        left = float(max(0, min(max_x, left)))
        return top, right, bottom, left

    @staticmethod
    def _mtcnn_box_to_location(box, frame_shape):
        """
        Convert an MTCNN bounding box to (top, right, bottom, left) format and clamp it.

        :param box: Bounding box from MTCNN in (left, top, right, bottom) order.
        :param frame_shape: Shape tuple from the frame (height, width, channels).
        :return: Tuple of floats in (top, right, bottom, left) format.
        """
        left, top, right, bottom = box
        return FaceRecognizer._clip_location((top, right, bottom, left), frame_shape)
