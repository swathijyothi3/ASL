"""
Train the ASL classifier and write the artefacts the app loads.

    python tools/train_model.py

Reads output/asl_landmarks.csv and writes, into output/:

    asl_ann_model.keras   the network
    scaler.pkl            fitted StandardScaler
    label_encoder.pkl     fitted LabelEncoder
    model_info.json       feature version, classes and measured accuracy

What changed from the original notebook, and why
------------------------------------------------
The notebook fed MediaPipe's landmarks to the network exactly as they came
out: coordinates relative to the image frame. That scores extremely well on
a held-out split of the same photo collection, because every photo in it is
framed the same way — but it means the model is reading the framing as much
as the sign. Point a webcam at yourself, where the hand is smaller, off to
one side and possibly the other hand, and the input no longer resembles
anything it was trained on.

Here the landmarks are reduced to pose only (see utils/features.py), and
the training set is augmented with mirrored, tilted and slightly noisy
copies. The held-out score barely moves; what changes is how the model
behaves on anything that is not a dataset photo.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.features import (  # noqa: E402
    FEATURE_VERSION,
    NUM_FEATURES,
    flatten,
    mirror_landmarks,
    normalise_landmarks,
    rotate_landmarks,
)

LANDMARK_CSV = os.path.join(BASE_DIR, "output", "asl_landmarks.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SEED = 42

# Augmentation strength. Rotation is kept moderate on purpose — see the
# note about K/P and G/Q in utils/features.py.
ROTATION_DEGREES = 18.0
SCALE_JITTER = 0.10
NOISE = 0.015
AUGMENTED_COPIES = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and report, but do not overwrite the artefacts.")
    return parser.parse_args()


def augment(points, rng):
    """One randomly tilted, resized and jittered copy of every hand."""

    angles = rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES, len(points))
    out = rotate_landmarks(points, angles)

    out = out * rng.uniform(1 - SCALE_JITTER, 1 + SCALE_JITTER, (len(out), 1, 1))
    out = out + rng.normal(0, NOISE, out.shape)

    return out.astype(np.float32)


def build_training_set(points, labels, rng):
    """Originals plus mirrors, then several augmented copies of both."""

    both = np.concatenate([points, mirror_landmarks(points)])
    both_labels = np.concatenate([labels, labels])

    chunks = [both]
    chunk_labels = [both_labels]

    for _ in range(AUGMENTED_COPIES):
        chunks.append(augment(both, rng))
        chunk_labels.append(both_labels)

    return np.concatenate(chunks), np.concatenate(chunk_labels)


def robustness_report(model, scaler, points, labels):
    """How the model holds up under things a real camera does."""

    def accuracy(transformed):
        features = scaler.transform(flatten(normalise_landmarks(transformed)))
        predicted = np.asarray(model(features, training=False)).argmax(axis=1)
        return 100.0 * (predicted == labels).mean()

    smaller = (points - points[:, :1, :]) * 0.4 + points[:, :1, :]

    shifted = points.copy()
    shifted[:, :, 0] += 0.25

    return [
        ("held-out test set", accuracy(points)),
        ("mirrored (other hand)", accuracy(mirror_landmarks(points))),
        ("tilted +20 deg", accuracy(rotate_landmarks(points, 20.0))),
        ("tilted -20 deg", accuracy(rotate_landmarks(points, -20.0))),
        ("tilted +35 deg", accuracy(rotate_landmarks(points, 35.0))),
        ("hand at 40% the size", accuracy(smaller)),
        ("shifted 25% across frame", accuracy(shifted)),
    ]


def main():
    args = parse_args()

    if not os.path.exists(LANDMARK_CSV):
        print(f"Missing {LANDMARK_CSV}. Run the extraction step first.")
        return 1

    rng = np.random.default_rng(SEED)

    frame = pd.read_csv(LANDMARK_CSV)

    labels = frame["label"].to_numpy()
    points = frame.drop(columns="label").to_numpy(dtype=np.float32).reshape(-1, 21, 3)

    print(f"{len(points):,} hands, {len(set(labels))} letters\n")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)

    train_points, test_points, train_y, test_y = train_test_split(
        points, encoded, test_size=0.20, random_state=SEED, stratify=encoded
    )

    augmented, augmented_y = build_training_set(train_points, train_y, rng)

    print(f"training hands  {len(train_points):,} "
          f"→ {len(augmented):,} after mirroring and augmentation")
    print(f"held-out hands  {len(test_points):,}\n")

    X_train = flatten(normalise_landmarks(augmented))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(NUM_FEATURES,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(len(encoder.classes_), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train, augmented_y,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=12,
                                 restore_best_weights=True)],
        verbose=2,
    )

    # ------------------------------------------------------
    # results
    # ------------------------------------------------------

    print("\nAccuracy")
    print("-" * 46)

    results = robustness_report(model, scaler, test_points, test_y)

    for name, value in results:
        print(f"  {name:<30s} {value:6.2f}%")

    test_accuracy = results[0][1]

    from sklearn.metrics import classification_report

    features = scaler.transform(flatten(normalise_landmarks(test_points)))
    predicted = np.asarray(model(features, training=False)).argmax(axis=1)

    print("\nPer-letter, on the held-out set")
    print(classification_report(test_y, predicted,
                                target_names=list(encoder.classes_),
                                zero_division=0))

    if args.dry_run:
        print("Dry run — nothing written.")
        return 0

    # ------------------------------------------------------
    # save
    # ------------------------------------------------------

    model.save(os.path.join(OUTPUT_DIR, "asl_ann_model.keras"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    info = {
        "feature_version": FEATURE_VERSION,
        "classes": list(encoder.classes_),
        "test_accuracy": round(float(test_accuracy), 2),
        "robustness": {name: round(float(value), 2) for name, value in results},
        "training_hands": int(len(train_points)),
        "augmented_rows": int(len(augmented)),
        "augmentation": {
            "mirror": True,
            "rotation_degrees": ROTATION_DEGREES,
            "scale_jitter": SCALE_JITTER,
            "noise": NOISE,
            "copies": AUGMENTED_COPIES,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "model_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")

    print(f"\nWritten to {OUTPUT_DIR}")
    print(f"  feature version {FEATURE_VERSION}, test accuracy {test_accuracy:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
