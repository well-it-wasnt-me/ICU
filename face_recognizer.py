import math
import os
import pickle
import sys

import cv2
import face_recognition
import torch
from PIL import Image
from face_recognition.face_recognition_cli import image_files_in_folder
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn import neighbors

from logger_setup import logger


class FaceRecognizer:
    """
    Handles all face recognition-related logic:
    - Determining CUDA/MPS/CPU device
    - Training the KNN classifier
    - Making predictions using the classifier
    - Loading reference images
    """

    def __init__(self, use_gpu=False, device=None, knn_clf=None):
        self.use_gpu = use_gpu
        self.device = device if device else self.get_device(use_gpu)
        self.knn_clf = knn_clf
        self.models = None  # (MTCNN, InceptionResnetV1) if GPU is used

    @staticmethod
    def get_device(use_gpu):
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
                pass
        return reference_images

    def train(self, train_dir, model_save_path=None, n_neighbors=None,
              knn_algo='ball_tree', verbose=False):
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
        Initialize MTCNN and Resnet if using GPU.
        """
        try:
            mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.models = (mtcnn, resnet)
        except Exception as e:
            logger.error(f"Failed to initialize facenet-pytorch models: {e}")
            sys.exit(1)

    def predict(self, X_frame, distance_threshold=0.5):
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
                face_distance = closest_distances[0][i][0]
                if rec:
                    predictions.append((pred, face_distance))
                else:
                    predictions.append(("unknown", face_distance))
            return predictions

        else:
            X_face_locations = face_recognition.face_locations(X_frame)
            if len(X_face_locations) == 0:
                return []

            faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
            closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold
                           for i in range(len(faces_encodings))]

            predictions = []
            for i, (pred, rec) in enumerate(zip(self.knn_clf.predict(faces_encodings), are_matches)):
                distance = closest_distances[0][i][0]
                if rec:
                    predictions.append((pred, distance))
                else:
                    predictions.append(("unknown", distance))
            return predictions
