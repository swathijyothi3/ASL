"""
Run the full app pipeline over the dataset and report where it fails.

This is not the notebook's test-set score. The notebook evaluates the
network on landmarks that were extracted once, up front. This script
goes through the same path the app does — image → MediaPipe → crop →
network — so it also catches detection failures and framing problems.

    python tools/evaluate.py                # every image
    python tools/evaluate.py --limit 20     # 20 images per letter, faster
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

from utils.predictor import ASLPredictor  # noqa: E402


DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ASL_Dataset", "dataset")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum images per letter (default: all).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip the square-crop step, to compare against it.",
    )

    return parser.parse_args()


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

    per_class = defaultdict(lambda: {"total": 0, "correct": 0, "missed": 0})
    confusions = Counter()

    for folder in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(class_path):
            continue

        letter = folder.replace("-samples", "")

        names = sorted(
            name for name in os.listdir(class_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if args.limit:
            names = names[:args.limit]

        for name in names:
            image = cv2.imread(os.path.join(class_path, name))

            if image is None:
                continue

            result = predictor.predict(image, use_crop=not args.no_crop)

            stats = per_class[letter]
            stats["total"] += 1

            if not result.hand_found:
                stats["missed"] += 1
                continue

            if result.letter == letter:
                stats["correct"] += 1
            else:
                confusions[(letter, result.letter)] += 1

        stats = per_class[letter]
        rate = 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"  {letter}  {rate:6.2f}%   ({stats['correct']}/{stats['total']})")

    predictor.close()

    total = sum(s["total"] for s in per_class.values())
    correct = sum(s["correct"] for s in per_class.values())
    missed = sum(s["missed"] for s in per_class.values())

    print("\n" + "=" * 46)
    print(f"  Images            {total}")
    print(f"  Correct           {correct}  ({100.0 * correct / total:.2f}%)")
    print(f"  No hand detected  {missed}  ({100.0 * missed / total:.2f}%)")
    print(f"  Crop step         {'off' if args.no_crop else 'on'}")
    print("=" * 46)

    if confusions:
        print("\nMost common mistakes")
        for (true, predicted), count in confusions.most_common(10):
            print(f"  {true} read as {predicted}   ×{count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
