import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
import threading

from tensorflow.keras.models import load_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ASLPredictor:

    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        ann_path="output/asl_ann_model.keras",
        scaler_path="output/scaler.pkl",
        encoder_path="output/label_encoder.pkl"
    ):

        # ======================================================
        # LOAD TRAINED ANN MODEL
        # ======================================================

        self.model = load_model(ann_path)

        # ======================================================
        # LOAD SCALER
        # ======================================================

        self.scaler = joblib.load(scaler_path)

        # ------------------------------------------------------
        # If the scaler was originally fit on a DataFrame, sklearn
        # stores the column names it saw as `feature_names_in_`.
        # We reuse those names when scaling new data so we don't
        # get the "X does not have valid feature names" warning.
        # If it wasn't fit with names, this will just be None and
        # we fall back to plain arrays (no behavior change).
        # ------------------------------------------------------

        self.feature_names = getattr(
            self.scaler,
            "feature_names_in_",
            None
        )

        # ======================================================
        # LOAD LABEL ENCODER
        # ======================================================

        self.label_encoder = joblib.load(encoder_path)

        # ======================================================
        # MEDIAPIPE HAND LANDMARKER
        # ======================================================

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,

            # IMPORTANT:
            # IMAGE mode matches how landmarks were extracted during
            # training (see landmark_dataset.ipynb). Each capture from
            # st.camera_input is a standalone photo, not a continuous
            # video stream, so IMAGE mode does a full fresh detection
            # every time instead of assuming temporal continuity and
            # reusing tracking state from the previous (unrelated)
            # photo — which was causing predictions to collapse onto
            # one letter regardless of the actual gesture shown.
            running_mode=vision.RunningMode.IMAGE,

            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        # ======================================================
        # THREAD LOCK
        # ======================================================

        self.lock = threading.Lock()

    # ==========================================================
    # EXTRACT HAND LANDMARKS
    # ==========================================================

    def extract_landmarks(self, frame):

        # ------------------------------------------------------
        # OpenCV BGR → RGB
        # ------------------------------------------------------

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------
        # Create MediaPipe image
        # ------------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ------------------------------------------------------
        # MediaPipe IMAGE inference
        # Matches training exactly: a full, independent detection
        # on this single photo, no cross-frame tracking assumptions.
        # ------------------------------------------------------

        result = self.landmarker.detect(mp_image)

        # ------------------------------------------------------
        # Check if hand exists
        # ------------------------------------------------------

        if len(result.hand_landmarks) == 0:
            return None, None

        # ------------------------------------------------------
        # Get first hand
        # ------------------------------------------------------

        landmarks = result.hand_landmarks[0]

        # ------------------------------------------------------
        # Extract x, y, z coordinates
        # 21 landmarks × 3 = 63 features
        # ------------------------------------------------------

        features = []

        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])

        features = np.array(features, dtype=np.float32)

        return features, landmarks

    # ==========================================================
    # PREDICTION
    # ==========================================================

    def predict(self, frame):

        # ------------------------------------------------------
        # Protect MediaPipe + ANN from simultaneous calls
        # ------------------------------------------------------

        with self.lock:

            features, landmarks = self.extract_landmarks(frame)

            # --------------------------------------------------
            # No hand detected
            # --------------------------------------------------

            if features is None:
                return None, 0.0, None

            # --------------------------------------------------
            # Reshape features
            # --------------------------------------------------

            features = features.reshape(1, -1)

            # --------------------------------------------------
            # Scale features
            #
            # If the scaler remembers the column names it was
            # trained with, wrap the array in a DataFrame using
            # those same names. This removes the sklearn
            # "feature names" warning and is a no-op otherwise.
            # --------------------------------------------------

            if self.feature_names is not None:
                features_df = pd.DataFrame(
                    features,
                    columns=self.feature_names
                )
                features_scaled = self.scaler.transform(features_df)
            else:
                features_scaled = self.scaler.transform(features)

            # --------------------------------------------------
            # ANN prediction
            # --------------------------------------------------

            probabilities = self.model.predict(
                features_scaled,
                verbose=0
            )[0]

            # --------------------------------------------------
            # Find predicted class
            # --------------------------------------------------

            predicted_index = np.argmax(probabilities)

            # --------------------------------------------------
            # Convert class index → ASL letter
            # --------------------------------------------------

            predicted_letter = self.label_encoder.inverse_transform(
                [predicted_index]
            )[0]

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            confidence = probabilities[predicted_index] * 100

            return predicted_letter, confidence, landmarks

    # ==========================================================
    # DRAW HAND LANDMARKS
    # ==========================================================

    def draw_landmarks(self, frame, landmarks):

        if landmarks is None:
            return frame

        h, w, _ = frame.shape

        # ------------------------------------------------------
        # MediaPipe hand connections
        # ------------------------------------------------------

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        # ------------------------------------------------------
        # Draw connections
        # ------------------------------------------------------

        for start, end in connections:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # ------------------------------------------------------
        # Draw landmark points
        # ------------------------------------------------------

        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

        return frame

    # ==========================================================
    # CLOSE MEDIAPIPE
    # ==========================================================

    def close(self):
        self.landmarker.close()