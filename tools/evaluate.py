"""
Measure the whole pipeline the way the app actually runs it.

This is not the training script's held-out score. That number is computed
on landmarks alone; this one starts from images, so it also captures
whether MediaPipe finds the hand at all — which is what limits the app in
practice.

Dataset photos are pasted into 1280x720 canvases to imitate a webcam at
various distances, and mirrored to imitate the other hand.

    python tools/evaluate.py
    python tools/evaluate.py --samples 3      # quicker
    python tools/evaluate.py --per-letter     # where the mistakes are
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.predictor import ASLPredictor  # noqa: E402

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ASL_Dataset", "dataset")

CANVAS_W, CANVAS_H = 1280, 720


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5,
                        help="Images per letter (default 5).")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip the crop-and-re-detect refinement.")
    parser.add_argument("--per-letter", action="store_true",
                        help="Break the webcam case down by letter.")
    return parser.parse_args()


def simulate_webcam(image, fraction, slot):
    """Paste a dataset photo into a wide frame, the way a webcam sees it."""

    target = max(32, int(CANVAS_H * fraction))
    small = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)

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


def rotate(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


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

        for slot, name in enumerate(names[:per_class]):
            image = cv2.imread(os.path.join(class_path, name))

            if image is not None:
                samples.append((image, letter, slot))

    return samples


CASES = [
    ("dataset photo, as-is", lambda i, s: i),
    ("mirrored (other hand)", lambda i, s: cv2.flip(i, 1)),
    ("tilted +20 deg", lambda i, s: rotate(i, 20)),
    ("tilted -20 deg", lambda i, s: rotate(i, -20)),
    ("webcam, hand 60% of frame", lambda i, s: simulate_webcam(i, 0.60, s)),
    ("webcam, hand 45% of frame", lambda i, s: simulate_webcam(i, 0.45, s)),
    ("webcam, hand 30% of frame", lambda i, s: simulate_webcam(i, 0.30, s)),
    ("webcam 45%, mirrored", lambda i, s: simulate_webcam(cv2.flip(i, 1), 0.45, s)),
    ("webcam 45%, tilted 20 deg", lambda i, s: simulate_webcam(rotate(i, 20), 0.45, s)),
]


def main():
    args = parse_args()

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

    print(f"{len(samples)} images per case, {len(predictor.classes)} letters")
    print(f"crop refinement: {'off' if args.no_crop else 'on'}\n")

    print(f"  {'case':<28s} {'correct':>8s} {'hand found':>11s} {'of those found':>15s}")
    print("  " + "-" * 66)

    per_letter = defaultdict(lambda: [0, 0])
    confusions = Counter()

    for name, transform in CASES:
        correct = found = 0

        for image, letter, slot in samples:
            frame = transform(image, slot)

            result = predictor.predict(frame, use_crop=not args.no_crop)

            if not result.hand_found:
                continue

            found += 1

            hit = result.letter == letter

            if hit:
                correct += 1
            elif name.startswith("webcam, hand 45%"):
                confusions[(letter, result.letter)] += 1

            if name.startswith("webcam, hand 45%"):
                per_letter[letter][0] += int(hit)
                per_letter[letter][1] += 1

        total = len(samples)
        share = 100.0 * correct / found if found else 0.0

        print(f"  {name:<28s} {100.0 * correct / total:7.1f}% "
              f"{100.0 * found / total:10.1f}% {share:14.1f}%")

    print("\n  'of those found' is accuracy counting only the frames where a")
    print("  hand was detected — it separates the classifier's job from the")
    print("  detector's.")

    if args.per_letter and per_letter:
        print("\nPer letter, webcam at 45% of frame")
        for letter in sorted(per_letter):
            hits, total = per_letter[letter]
            rate = 100.0 * hits / total if total else 0.0
            flag = "   <-- weak" if rate < 70 and total else ""
            print(f"  {letter}  {rate:6.1f}%  ({hits}/{total}){flag}")

        if confusions:
            print("\n  most common mistakes")
            for (true, predicted), count in confusions.most_common(8):
                print(f"    {true} read as {predicted}  x{count}")

    predictor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
