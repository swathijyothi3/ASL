"""
Measure what the square-crop refinement is still worth.

Historical note: this crop originally existed to force webcam framing to
match the training photos, because the classifier was fed raw image
coordinates. That is no longer why it is here — utils/features.py reduces
each hand to pose only, so framing cannot affect the result.

What the crop does now is give MediaPipe a closer look at the hand, which
places the joints more precisely. Worth roughly 4 points of accuracy at
webcam distance; tools/evaluate.py --no-crop shows the comparison for the
whole pipeline.

It does two things:

  1. Reads the framing of the training set straight out of the landmark
     CSV — how much of the frame the hand covers, and where it sits —
     which is where DEFAULT_ZOOM and CROP_ANCHOR come from.

  2. Pastes dataset images into 1280x720 canvases to imitate a webcam,
     then scores three settings against the known labels:

       direct        the original 224x224 image     (accuracy ceiling)
       webcam-raw    simulated frame, no crop       (the bug)
       webcam-crop   simulated frame, crop applied  (the fix)

Run from the project root:

    python tools/tune_crop.py
    python tools/tune_crop.py --samples 3     # quicker
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

from utils.predictor import ASLPredictor  # noqa: E402


DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ASL_Dataset", "dataset")
LANDMARK_CSV = os.path.join(BASE_DIR, "output", "asl_landmarks.csv")

CANVAS_W, CANVAS_H = 1280, 720

# Fraction of canvas height the pasted sample covers — roughly, how far
# the signer is sitting from the camera.
HAND_FRACTIONS = [0.30, 0.45, 0.60]

ZOOMS = [1.4, 1.6, 1.74, 1.9, 2.1]

ANCHORS = [(0.5, 0.5), ASLPredictor.CROP_ANCHOR]


# ==========================================================
# WHAT THE TRAINING FRAMING ACTUALLY IS
# ==========================================================

def describe_training_framing():
    """Derive the crop constants from the landmarks used for training."""

    if not os.path.exists(LANDMARK_CSV):
        print("No landmark CSV, skipping the framing measurement.\n")
        return

    frame = pd.read_csv(LANDMARK_CSV)

    values = frame.drop(columns="label").to_numpy(dtype=np.float32)
    values = values.reshape(len(frame), 21, 3)

    x = values[:, :, 0]
    y = values[:, :, 1]

    side = np.maximum(x.max(1) - x.min(1), y.max(1) - y.min(1))

    centre_x = (x.max(1) + x.min(1)) / 2
    centre_y = (y.max(1) + y.min(1)) / 2

    print("Training-set framing, measured from output/asl_landmarks.csv")
    print(f"  samples                     {len(frame):,}")
    print(f"  hand box / frame            {side.mean():.3f}"
          f"  (median {np.median(side):.3f})")
    print(f"  implied zoom  1 / that      {1 / side.mean():.2f}"
          f"   → DEFAULT_ZOOM = {ASLPredictor.DEFAULT_ZOOM}")
    print(f"  hand centre                 "
          f"({centre_x.mean():.2f}, {centre_y.mean():.2f})"
          f"   → CROP_ANCHOR = {ASLPredictor.CROP_ANCHOR}")
    print("  the hand sits low in frame, so a centred crop would shift")
    print("  every y coordinate against what the network was fitted on.")
    print()


# ==========================================================
# WEBCAM SIMULATION
# ==========================================================

def simulate_webcam(image, fraction, slot):
    """Paste a 224x224 sample into a wide frame, the way a webcam sees it."""

    target = max(32, int(CANVAS_H * fraction))
    small = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)

    # Extend the sample's own background across the canvas so the paste
    # doesn't leave a rectangle the detector could lock onto.
    border = np.concatenate([
        image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]
    ])
    background = np.median(border, axis=0).astype(np.uint8)

    canvas = np.full((CANVAS_H, CANVAS_W, 3), background, dtype=np.uint8)

    offsets = [(0.5, 0.5), (0.32, 0.45), (0.68, 0.55), (0.5, 0.35), (0.4, 0.62)]
    fx, fy = offsets[slot % len(offsets)]

    x = max(0, min(int(CANVAS_W * fx - target / 2), CANVAS_W - target))
    y = max(0, min(int(CANVAS_H * fy - target / 2), CANVAS_H - target))

    canvas[y:y + target, x:x + target] = small

    return canvas


# ==========================================================
# SCORING
# ==========================================================

def load_samples(per_class):
    samples = []

    for folder in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(class_path):
            continue

        letter = folder.replace("-samples", "")

        names = sorted(
            name for name in os.listdir(class_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        for slot, name in enumerate(names[-per_class:]):
            image = cv2.imread(os.path.join(class_path, name))

            if image is not None:
                samples.append((image, letter, slot))

    return samples


def score(predictor, samples, transform=None, **kwargs):
    correct = 0
    detected = 0
    confidence_total = 0.0

    for image, letter, slot in samples:
        frame = transform(image, slot) if transform else image

        result = predictor.predict(frame, **kwargs)

        if not result.hand_found:
            continue

        detected += 1
        confidence_total += result.confidence

        if result.letter == letter:
            correct += 1

    total = len(samples) or 1

    return {
        "accuracy": 100.0 * correct / total,
        "detected": 100.0 * detected / total,
        "confidence": confidence_total / detected if detected else 0.0,
    }


def show(label, stats, best=False):
    marker = "  <-- best" if best else ""
    print(
        f"  {label:<34s} accuracy {stats['accuracy']:6.2f}%"
        f"   found {stats['detected']:6.2f}%"
        f"   conf {stats['confidence']:6.2f}%{marker}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5,
                        help="Images per letter (default 5).")
    args = parser.parse_args()

    describe_training_framing()

    if not os.path.isdir(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return 1

    predictor = ASLPredictor(
        model_path=os.path.join(BASE_DIR, "models", "hand_landmarker.task"),
        ann_path=os.path.join(BASE_DIR, "output", "asl_ann_model.keras"),
        scaler_path=os.path.join(BASE_DIR, "output", "scaler.pkl"),
        encoder_path=os.path.join(BASE_DIR, "output", "label_encoder.pkl"),
    )

    samples = load_samples(args.samples)

    print(f"Scoring {len(samples)} images across "
          f"{len(predictor.classes)} letters\n")

    print("Original 224x224 images — the crop should not make these worse")
    show("no crop", score(predictor, samples, use_crop=False))
    show("crop, tuned defaults", score(predictor, samples, use_crop=True))
    print()

    for fraction in HAND_FRACTIONS:
        print(f"Simulated webcam — hand covers {fraction:.0%} of frame height")

        def transform(image, slot, fraction=fraction):
            return simulate_webcam(image, fraction, slot)

        baseline = score(predictor, samples, transform, use_crop=False)
        show("no crop (the bug)", baseline)

        results = []

        for anchor in ANCHORS:
            for zoom in ZOOMS:
                stats = score(
                    predictor, samples, transform,
                    use_crop=True, zoom=zoom, anchor=anchor,
                )
                results.append((stats["accuracy"], anchor, zoom, stats))

        best = max(results, key=lambda item: item[0])[0]

        for accuracy, anchor, zoom, stats in results:
            label = f"crop zoom={zoom}  anchor={anchor[0]:.2f},{anchor[1]:.2f}"
            show(label, stats, best=accuracy == best)

        print()

    predictor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
