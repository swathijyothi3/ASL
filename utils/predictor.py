"""
ASL Vision — hand landmark extraction and ANN inference.

The pipeline is:

    frame → MediaPipe (full frame) → square crop around the hand
          → MediaPipe (crop) → 63 features → StandardScaler → ANN → letter

The crop step matters. The model was trained on 224x224 dataset images in
which the hand is centred and fills roughly 60% of a square frame. MediaPipe
returns landmark coordinates normalised to the *image* it was given, so a
16:9 webcam photo with a small hand in the corner produces feature values
that look nothing like the training data, and the ANN's accuracy collapses.
Re-detecting on a square crop puts the landmarks back into the coordinate
range the ANN was trained on.
"""

import threading

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import load_model


# ==========================================================
# HAND SKELETON
# ==========================================================

# MediaPipe's 21-landmark hand topology, grouped by finger so the
# overlay and the 3D view can colour each finger differently.

FINGERS = {
    "palm":   [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)],
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring":   [(13, 14), (14, 15), (15, 16)],
    "pinky":  [(17, 18), (18, 19), (19, 20)],
}

HAND_CONNECTIONS = [bone for bones in FINGERS.values() for bone in bones]

LANDMARK_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb tip",
    "Index MCP", "Index PIP", "Index DIP", "Index tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky tip",
]

# BGR, for the OpenCV overlay.
FINGER_COLORS_BGR = {
    "palm":   (200, 200, 200),
    "thumb":  (80, 180, 255),
    "index":  (120, 255, 120),
    "middle": (255, 200, 90),
    "ring":   (255, 130, 220),
    "pinky":  (120, 160, 255),
}

# Hex, for the Plotly 3D view.
FINGER_COLORS_HEX = {
    "palm":   "#cbd5e1",
    "thumb":  "#ffb454",
    "index":  "#4ade80",
    "middle": "#5ac8fa",
    "ring":   "#e879f9",
    "pinky":  "#a5b4fc",
}


# ==========================================================
# PREDICTION RESULT
# ==========================================================

class Prediction:
    """Everything the UI needs to describe one attempt at a frame."""

    def __init__(
        self,
        letter=None,
        confidence=0.0,
        probabilities=None,
        landmarks=None,
        display_landmarks=None,
        crop_box=None,
        used_crop=False,
    ):
        self.letter = letter
        self.confidence = confidence

        # {letter: probability in %} over every class the model knows.
        self.probabilities = probabilities or {}

        # (21, 3) array in the coordinate space the ANN actually saw.
        # This is what the 3D view plots.
        self.landmarks = landmarks

        # (21, 3) array mapped back onto the original frame, for the
        # 2D overlay drawn on top of the user's photo.
        self.display_landmarks = display_landmarks

        # (x, y, w, h) of the square crop in original-frame pixels.
        self.crop_box = crop_box

        self.used_crop = used_crop

    @property
    def hand_found(self):
        return self.landmarks is not None

    def top_k(self, k=3):
        """The k most likely letters, highest first."""
        ranked = sorted(
            self.probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:k]


# ==========================================================
# PREDICTOR
# ==========================================================

class ASLPredictor:

    # Both constants below are measured from the training set rather than
    # guessed; tools/tune_crop.py re-derives them and scores the result.

    # In the training images the hand's bounding box covers a mean 0.576
    # of the frame, so a crop 1/0.576 times the box reproduces that framing.
    # The measured optimum is flat from roughly 1.6 to 1.8, so this value
    # is not delicate.
    DEFAULT_ZOOM = 1.74

    # The hand is not centred in the training images — it sits low, with
    # the wrist near the bottom edge, averaging (0.48, 0.59) of the frame.
    # Matching that is principled, though the sweep shows it changes far
    # less than the zoom does.
    CROP_ANCHOR = (0.48, 0.59)

    # Training images are 224x224. Feeding the detector a crop of the
    # same size keeps its internal scaling consistent with training.
    CROP_SIZE = 224

    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        ann_path="output/asl_ann_model.keras",
        scaler_path="output/scaler.pkl",
        encoder_path="output/label_encoder.pkl",
        zoom=DEFAULT_ZOOM,
    ):

        self.zoom = zoom

        # ======================================================
        # TRAINED ARTEFACTS
        # ======================================================

        self.model = load_model(ann_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        self.classes = list(self.label_encoder.classes_)

        # The saved network has 26 output units but was only ever
        # trained on the 23 static letters present in the dataset.
        # The extra units are untrained, so we ignore them rather
        # than risk argmax landing on a class the encoder can't name.
        self.n_classes = len(self.classes)

        # If the scaler was fit on a DataFrame, sklearn remembers the
        # column names. Reusing them avoids the "X does not have valid
        # feature names" warning; it's a no-op when they're absent.
        self.feature_names = getattr(self.scaler, "feature_names_in_", None)

        # ======================================================
        # MEDIAPIPE HAND LANDMARKER
        # ======================================================

        # IMAGE mode matches how the training landmarks were extracted.
        # Each capture is an independent photo, so we want a fresh
        # detection every time rather than VIDEO mode's assumption that
        # consecutive frames are temporally related.

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # MediaPipe graphs and Keras models are not re-entrant, and
        # Streamlit caches one predictor across every browser session.
        self.lock = threading.Lock()

    # ==========================================================
    # DETECTION
    # ==========================================================

    def _detect(self, frame):
        """Run MediaPipe on a BGR frame. Returns an (21, 3) array or None."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        return np.array(
            [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
            dtype=np.float32,
        )

    def _square_crop(self, frame, landmarks, zoom, anchor):
        """
        Cut a square around the detected hand, matching the framing of
        the training images.

        Returns (crop, box) where box is (x, y, side) in original-frame
        pixels. The crop is padded by edge replication when the square
        runs past the border, so the hand always stays centred instead
        of being pushed off to one side.
        """

        h, w = frame.shape[:2]

        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h

        centre_x = (xs.min() + xs.max()) / 2.0
        centre_y = (ys.min() + ys.max()) / 2.0

        # Square side driven by the longer edge of the hand's bounding box.
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * zoom
        side = max(side, 16.0)

        # Offset the window so the hand lands where it sits in the
        # training images, rather than dead centre.
        anchor_x, anchor_y = anchor

        side_i = int(round(side))
        pad = side_i  # always enough to cover a fully off-frame square

        padded = cv2.copyMakeBorder(
            frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE
        )

        # Clamped so the slice is always a full square, whatever rounding
        # does at the edges. A short slice would silently change the
        # aspect ratio and with it every normalised coordinate.
        px = int(round(centre_x - side * anchor_x)) + pad
        py = int(round(centre_y - side * anchor_y)) + pad

        px = max(0, min(px, padded.shape[1] - side_i))
        py = max(0, min(py, padded.shape[0] - side_i))

        crop = padded[py:py + side_i, px:px + side_i]

        if crop.size == 0:
            return None, None

        crop = cv2.resize(
            crop,
            (self.CROP_SIZE, self.CROP_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # Reported back in original-frame pixels, after clamping, so the
        # overlay box and the landmark mapping agree with what was read.
        return crop, (px - pad, py - pad, side_i)

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================

    def _classify(self, landmarks):
        """(21, 3) landmarks → (letter, confidence, {letter: percent})."""

        features = landmarks.reshape(1, -1).astype(np.float32)

        if self.feature_names is not None:
            features = pd.DataFrame(features, columns=self.feature_names)

        features_scaled = np.asarray(
            self.scaler.transform(features), dtype=np.float32
        )

        # Calling the model directly is markedly faster than .predict()
        # for a single sample, which is all this app ever classifies.
        probabilities = np.asarray(self.model(features_scaled, training=False))[0]

        # Drop the untrained output units and renormalise so the
        # reported confidence is a true percentage over real classes.
        probabilities = probabilities[:self.n_classes]

        total = float(probabilities.sum())
        if total > 0:
            probabilities = probabilities / total

        index = int(np.argmax(probabilities))

        letter = self.classes[index]
        confidence = float(probabilities[index]) * 100.0

        distribution = {
            name: float(probabilities[i]) * 100.0
            for i, name in enumerate(self.classes)
        }

        return letter, confidence, distribution

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def predict(self, frame, use_crop=True, zoom=None, anchor=None):
        """
        Classify the hand gesture in a BGR frame.

        Always returns a Prediction; check `.hand_found` before using
        the letter.
        """

        zoom = self.zoom if zoom is None else zoom
        anchor = self.CROP_ANCHOR if anchor is None else anchor

        with self.lock:

            raw_landmarks = self._detect(frame)

            if raw_landmarks is None:
                return Prediction()

            landmarks = raw_landmarks
            display_landmarks = raw_landmarks
            crop_box = None
            used_crop = False

            if use_crop:
                crop, box = self._square_crop(frame, raw_landmarks, zoom, anchor)

                if crop is not None:
                    crop_landmarks = self._detect(crop)

                    # If the hand is no longer findable in the crop we
                    # keep the full-frame landmarks rather than fail.
                    if crop_landmarks is not None:
                        landmarks = crop_landmarks
                        crop_box = box
                        used_crop = True

                        display_landmarks = self._crop_to_frame(
                            crop_landmarks, box, frame.shape[:2]
                        )

            letter, confidence, distribution = self._classify(landmarks)

            return Prediction(
                letter=letter,
                confidence=confidence,
                probabilities=distribution,
                landmarks=landmarks,
                display_landmarks=display_landmarks,
                crop_box=crop_box,
                used_crop=used_crop,
            )

    @staticmethod
    def _crop_to_frame(landmarks, box, frame_shape):
        """Map crop-space landmarks back onto the original frame."""

        x0, y0, side = box
        h, w = frame_shape

        mapped = landmarks.copy()
        mapped[:, 0] = (x0 + landmarks[:, 0] * side) / w
        mapped[:, 1] = (y0 + landmarks[:, 1] * side) / h

        return mapped

    # ==========================================================
    # OVERLAY
    # ==========================================================

    def draw_landmarks(self, frame, landmarks, crop_box=None):
        """Draw the hand skeleton (and optionally the crop square)."""

        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        points = [
            (int(lm[0] * w), int(lm[1] * h))
            for lm in landmarks
        ]

        # Scale line and dot sizes with the image so the overlay looks
        # the same on a 224px thumbnail and a 1280px webcam photo.
        thickness = max(1, int(round(min(h, w) / 320)))
        radius = max(2, int(round(min(h, w) / 200)))

        if crop_box is not None:
            x0, y0, side = crop_box
            cv2.rectangle(
                frame,
                (int(x0), int(y0)),
                (int(x0 + side), int(y0 + side)),
                (90, 200, 255),
                max(1, thickness - 1),
            )

        for finger, bones in FINGERS.items():
            color = FINGER_COLORS_BGR[finger]

            for start, end in bones:
                cv2.line(frame, points[start], points[end], color, thickness)

        for index, point in enumerate(points):
            # The wrist is the anchor of the whole skeleton, so mark it.
            outer = radius + 2 if index == 0 else radius
            cv2.circle(frame, point, outer, (255, 255, 255), -1)
            cv2.circle(frame, point, max(1, outer - 2), (30, 41, 59), -1)

        return frame

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def close(self):
        self.landmarker.close()
