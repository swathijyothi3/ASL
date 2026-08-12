"""
Run this from your ASL_DL project root (same folder as app.py).
It tests several DIFFERENT known letters from your own dataset
and prints the full probability vector each time, so we can see
whether the model is actually distinguishing classes at all.
"""

import os
import cv2
import numpy as np
import joblib
import pandas as pd

from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# ==========================================================
# LOAD SAVED ARTIFACTS
# ==========================================================

model = load_model("output/asl_ann_model.keras")
scaler = joblib.load("output/scaler.pkl")
label_encoder = joblib.load("output/label_encoder.pkl")

print("Model input shape:", model.input_shape)
print("Model output classes:", model.output_shape)
print("Label encoder classes:", list(label_encoder.classes_))
print("Scaler expects features:", getattr(scaler, "feature_names_in_", "unknown"))
print()

# ==========================================================
# MEDIAPIPE (IMAGE mode, same as training)
# ==========================================================

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)


def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Could not read {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    if len(result.hand_landmarks) == 0:
        print(f"  No hand detected in {image_path}")
        return

    landmarks = result.hand_landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    features = np.array(features, dtype=np.float32).reshape(1, -1)

    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is not None:
        features_df = pd.DataFrame(features, columns=feature_names)
        features_scaled = scaler.transform(features_df)
    else:
        features_scaled = scaler.transform(features)

    probs = model.predict(features_scaled, verbose=0)[0]
    predicted_index = np.argmax(probs)
    predicted_letter = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probs[predicted_index] * 100

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(label_encoder.classes_[i], round(probs[i]*100, 2)) for i in top3_idx]

    print(f"  File: {os.path.basename(image_path)}")
    print(f"  Predicted: {predicted_letter}  ({confidence:.2f}%)")
    print(f"  Top 3: {top3}")
    print()


# ==========================================================
# TEST SEVERAL DIFFERENT LETTERS FROM YOUR OWN DATASET
# ==========================================================

DATASET_PATH = "dataset/ASL_Dataset/dataset"

test_letters = ["A", "B", "C", "D", "E", "F", "G"]

for letter in test_letters:
    folder = os.path.join(DATASET_PATH, f"{letter}-samples")
    if not os.path.isdir(folder):
        folder = os.path.join(DATASET_PATH, letter)
    if not os.path.isdir(folder):
        print(f"Skipping {letter}: folder not found")
        continue

    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        print(f"Skipping {letter}: no images")
        continue

    print(f"=== Testing true label: {letter} ===")
    predict_image(os.path.join(folder, images[0]))