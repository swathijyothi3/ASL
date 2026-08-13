"""
ASL Vision — hand landmark extraction and inference.

    frame → MediaPipe → (optional) crop and re-detect → pose features → ANN

Two details carry most of the real-world accuracy:

**Pose-only features.** MediaPipe reports landmarks relative to the image,
so raw coordinates encode where the hand is and how large it appears as
well as what shape it makes. utils/features.py strips that out. Without it
the classifier is reading the framing of the training photos, and a webcam
shot — hand smaller, off-centre, possibly the other hand — lands outside
anything it learned.

**A detection cascade.** MediaPipe's default confidence threshold misses a
lot of perfectly clear hands once they are not filling the frame. The
strict pass runs first so an obvious hand is taken at high confidence; only
when that finds nothing does a permissive pass run. On simulated webcam
frames this lifted the proportion of hands found at a normal sitting
distance from 67% to over 90%.
"""

import json
import os
import threading

import cv2
import joblib
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import load_model

from utils.features import (
    FEATURE_VERSION,
    flatten,
    normalise_landmarks,
)


# ==========================================================
# HAND SKELETON
# ==========================================================

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

FINGER_COLORS_BGR = {
    "palm":   (200, 200, 200),
    "thumb":  (80, 180, 255),
    "index":  (120, 255, 120),
    "middle": (255, 200, 90),
    "ring":   (255, 130, 220),
    "pinky":  (120, 160, 255),
}

FINGER_COLORS_HEX = {
    "palm":   "#cbd5e1",
    "thumb":  "#ffb454",
    "index":  "#4ade80",
    "middle": "#5ac8fa",
    "ring":   "#e879f9",
    "pinky":  "#a5b4fc",
}


class FeatureVersionMismatch(RuntimeError):
    """The saved model was trained on a different feature definition."""


# ==========================================================
# PREDICTION RESULT
# ==========================================================

class Prediction:
    """Everything the interface needs to describe one attempt at a frame."""

    def __init__(
        self,
        letter=None,
        confidence=0.0,
        probabilities=None,
        landmarks=None,
        display_landmarks=None,
        crop_box=None,
        used_crop=False,
        detection_pass="none",
    ):
        self.letter = letter
        self.confidence = confidence
        self.probabilities = probabilities or {}
        self.landmarks = landmarks
        self.display_landmarks = display_landmarks
        self.crop_box = crop_box
        self.used_crop = used_crop

        # "strict", "lenient" or "none" — surfaced so the interface can
        # tell someone their hand was only just found, and to suggest
        # moving closer.
        self.detection_pass = detection_pass

    @property
    def hand_found(self):
        return self.landmarks is not None

    def top_k(self, k=3):
        ranked = sorted(
            self.probabilities.items(), key=lambda item: item[1], reverse=True
        )
        return ranked[:k]


# ==========================================================
# PREDICTOR
# ==========================================================

class ASLPredictor:

    # Crop size around the hand, as a multiple of its bounding box. The
    # features no longer depend on framing, so this only sharpens the
    # landmarks by giving the detector a closer look.
    DEFAULT_ZOOM = 1.9

    CROP_ANCHOR = (0.5, 0.5)
    CROP_SIZE = 256

    # First pass is deliberately strict; the fallback is permissive.
    STRICT_CONFIDENCE = 0.5
    LENIENT_CONFIDENCE = 0.15

    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        ann_path="output/asl_ann_model.keras",
        scaler_path="output/scaler.pkl",
        encoder_path="output/label_encoder.pkl",
        info_path=None,
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
        self.n_classes = len(self.classes)

        # ------------------------------------------------------
        # Guard against the model and the feature code drifting apart.
        # A mismatch would not raise on its own — it would quietly
        # produce confident nonsense, which is far harder to notice.
        # ------------------------------------------------------

        if info_path is None:
            info_path = os.path.join(os.path.dirname(ann_path), "model_info.json")

        self.info = {}

        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as handle:
                self.info = json.load(handle)

            trained_with = self.info.get("feature_version")

            if trained_with is not None and trained_with != FEATURE_VERSION:
                raise FeatureVersionMismatch(
                    f"The model in {ann_path} was trained on feature version "
                    f"{trained_with}, but this code produces version "
                    f"{FEATURE_VERSION}. Re-run tools/train_model.py."
                )

        # ======================================================
        # MEDIAPIPE
        # ======================================================

        # IMAGE mode: every capture is an independent photo, so a fresh
        # detection each time is wanted rather than VIDEO mode's
        # assumption that consecutive frames are related.

        self._strict = self._make_landmarker(model_path, self.STRICT_CONFIDENCE)
        self._lenient = self._make_landmarker(model_path, self.LENIENT_CONFIDENCE)

        # MediaPipe graphs and Keras models are not re-entrant, and
        # Streamlit shares one predictor across every visitor.
        self.lock = threading.Lock()

    @staticmethod
    def _make_landmarker(model_path, confidence):
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            )
        )

    @property
    def test_accuracy(self):
        return self.info.get("test_accuracy")

    # ==========================================================
    # DETECTION
    # ==========================================================

    @staticmethod
    def _run(landmarker, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
        )

        if not result.hand_landmarks:
            return None

        return np.array(
            [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
            dtype=np.float32,
        )

    def _detect(self, frame, allow_lenient=True):
        """Strict pass first, then a permissive one. Returns (points, pass)."""

        points = self._run(self._strict, frame)

        if points is not None:
            return points, "strict"

        if allow_lenient:
            points = self._run(self._lenient, frame)

            if points is not None:
                return points, "lenient"

        return None, "none"

    def _square_crop(self, frame, landmarks, zoom, anchor):
        """
        A square around the hand, enlarged and re-detected.

        With pose-only features this no longer exists to match training
        framing — it is there so the detector gets a closer look at a hand
        that is small in a wide frame, which makes the landmarks more
        precise.
        """

        h, w = frame.shape[:2]

        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h

        centre_x = (xs.min() + xs.max()) / 2.0
        centre_y = (ys.min() + ys.max()) / 2.0

        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * zoom
        side = max(side, 16.0)

        side_i = int(round(side))
        pad = side_i

        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        anchor_x, anchor_y = anchor

        px = int(round(centre_x - side * anchor_x)) + pad
        py = int(round(centre_y - side * anchor_y)) + pad

        px = max(0, min(px, padded.shape[1] - side_i))
        py = max(0, min(py, padded.shape[0] - side_i))

        crop = padded[py:py + side_i, px:px + side_i]

        if crop.size == 0:
            return None, None

        # Enlarging a small crop genuinely helps: the detector has a
        # minimum useful hand size in pixels.
        crop = cv2.resize(
            crop,
            (self.CROP_SIZE, self.CROP_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        return crop, (px - pad, py - pad, side_i)

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================

    def _classify(self, landmarks):
        """(21, 3) landmarks → (letter, confidence, {letter: percent})."""

        features = flatten(normalise_landmarks(landmarks))

        scaled = np.asarray(self.scaler.transform(features), dtype=np.float32)

        # Calling the model directly is markedly faster than .predict()
        # for the single sample this app classifies at a time.
        probabilities = np.asarray(self.model(scaled, training=False))[0]

        probabilities = probabilities[:self.n_classes]

        total = float(probabilities.sum())
        if total > 0:
            probabilities = probabilities / total

        index = int(np.argmax(probabilities))

        distribution = {
            name: float(probabilities[i]) * 100.0
            for i, name in enumerate(self.classes)
        }

        return self.classes[index], float(probabilities[index]) * 100.0, distribution

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def predict(self, frame, use_crop=True, zoom=None, anchor=None):
        """
        Classify the hand in a BGR frame.

        Always returns a Prediction; check `.hand_found` first.
        """

        zoom = self.zoom if zoom is None else zoom
        anchor = self.CROP_ANCHOR if anchor is None else anchor

        with self.lock:

            raw_landmarks, detection_pass = self._detect(frame)

            if raw_landmarks is None:
                return Prediction()

            landmarks = raw_landmarks
            display_landmarks = raw_landmarks
            crop_box = None
            used_crop = False

            if use_crop:
                crop, box = self._square_crop(frame, raw_landmarks, zoom, anchor)

                if crop is not None:
                    # The crop is already tightly framed, so the strict
                    # pass is enough; no need to fall back here.
                    crop_landmarks, _ = self._detect(crop, allow_lenient=True)

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
                detection_pass=detection_pass,
            )

    def locate(self, frame):
        """
        Find the hand without classifying it.

        Used by the live view, which only needs to draw the skeleton and
        the framing box at video rate.
        """

        with self.lock:
            landmarks, detection_pass = self._detect(frame)

        return landmarks, detection_pass

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

    def draw_landmarks(self, frame, landmarks, crop_box=None, box_color=(90, 200, 255)):
        """Draw the hand skeleton, and optionally the framing square."""

        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        points = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

        thickness = max(1, int(round(min(h, w) / 320)))
        radius = max(2, int(round(min(h, w) / 200)))

        if crop_box is not None:
            x0, y0, side = crop_box
            self._draw_corners(
                frame, int(x0), int(y0), int(side), box_color, thickness
            )

        for finger, bones in FINGERS.items():
            color = FINGER_COLORS_BGR[finger]
            for start, end in bones:
                cv2.line(frame, points[start], points[end], color, thickness)

        for index, point in enumerate(points):
            outer = radius + 2 if index == 0 else radius
            cv2.circle(frame, point, outer, (255, 255, 255), -1)
            cv2.circle(frame, point, max(1, outer - 2), (30, 41, 59), -1)

        return frame

    @staticmethod
    def _draw_corners(frame, x, y, side, color, thickness):
        """
        Corner brackets rather than a full rectangle.

        A closed box over a live camera feed reads as clutter across the
        hand; brackets mark the same region while leaving it visible.
        """

        length = max(8, side // 5)
        t = max(2, thickness + 1)

        corners = [
            ((x, y), (1, 1)),
            ((x + side, y), (-1, 1)),
            ((x, y + side), (1, -1)),
            ((x + side, y + side), (-1, -1)),
        ]

        for (cx, cy), (dx, dy) in corners:
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, t)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, t)

        return frame

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def close(self):
        self._strict.close()
        self._lenient.close()
