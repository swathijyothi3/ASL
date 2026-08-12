"""
Verify an ASL Vision install before running the app.

Checks the Python version, the required packages, the model files, and
then runs one real prediction end to end. Every failure prints what to
do about it.

    python tools/check_setup.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)


def path(*parts):
    return os.path.join(BASE_DIR, *parts)


PACKAGES = [
    ("streamlit", "streamlit"),
    ("cv2", "opencv-contrib-python-headless"),
    ("mediapipe", "mediapipe"),
    ("tensorflow", "tensorflow-cpu"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("plotly", "plotly"),
]

REQUIRED_FILES = [
    ("models/hand_landmarker.task", "MediaPipe hand landmark model"),
    ("output/asl_ann_model.keras", "trained neural network"),
    ("output/scaler.pkl", "fitted StandardScaler"),
    ("output/label_encoder.pkl", "fitted LabelEncoder"),
]

OPTIONAL_FILES = [
    ("output/asl_landmarks.csv", "landmark data — powers the 3D explorer"),
    ("dataset/ASL_Dataset/dataset", "sample photos — powers the alphabet guide"),
]

failures = []


def ok(message):
    print(f"  [ ok ] {message}")


def fail(message, remedy):
    print(f"  [FAIL] {message}")
    print(f"         → {remedy}")
    failures.append(message)


def warn(message, note):
    print(f"  [warn] {message}")
    print(f"         → {note}")


# ==========================================================
# PYTHON
# ==========================================================

print("\nPython")

version = sys.version_info

if version >= (3, 11):
    ok(f"Python {version.major}.{version.minor}.{version.micro}")
else:
    fail(
        f"Python {version.major}.{version.minor} is too old",
        "The pinned NumPy and pandas need Python 3.11 or newer.",
    )


# ==========================================================
# PACKAGES
# ==========================================================

print("\nPackages")

for module_name, package_name in PACKAGES:
    try:
        __import__(module_name)
        ok(package_name)
    except ImportError:
        fail(
            f"{package_name} is not installed",
            "Run: pip install -r requirements.txt",
        )


# ==========================================================
# FILES
# ==========================================================

print("\nModel files")

for relative, description in REQUIRED_FILES:
    if os.path.exists(path(relative)):
        size = os.path.getsize(path(relative)) / 1024
        ok(f"{relative}  ({size:,.0f} KB) — {description}")
    else:
        fail(
            f"{relative} is missing — {description}",
            "Re-clone the repository, or regenerate it with "
            "landmark_dataset.ipynb.",
        )

print("\nOptional files")

for relative, description in OPTIONAL_FILES:
    if os.path.exists(path(relative)):
        ok(f"{relative} — {description}")
    else:
        warn(
            f"{relative} is missing — {description}",
            "The app still runs; that section shows a notice instead.",
        )


# ==========================================================
# END-TO-END PREDICTION
# ==========================================================

if not failures:

    print("\nEnd-to-end prediction")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        import cv2

        from utils.predictor import ASLPredictor

        predictor = ASLPredictor(
            model_path=path("models", "hand_landmarker.task"),
            ann_path=path("output", "asl_ann_model.keras"),
            scaler_path=path("output", "scaler.pkl"),
            encoder_path=path("output", "label_encoder.pkl"),
        )

        ok(f"model loaded — {len(predictor.classes)} letters: "
           f"{' '.join(predictor.classes)}")

        # Checked across several letters rather than one, and with the
        # crop off: these are dataset images, already framed the way the
        # model expects, so re-framing them only adds noise to what is
        # meant to be a wiring check.

        checked = 0
        matched = 0

        for letter in ["A", "B", "C", "L", "V", "W"]:
            sample = path(
                "dataset", "ASL_Dataset", "dataset", f"{letter}-samples", "0.jpg"
            )

            if not os.path.exists(sample):
                continue

            frame = cv2.imread(sample)

            if frame is None:
                continue

            result = predictor.predict(frame, use_crop=False)
            checked += 1

            if not result.hand_found:
                print(f"         {letter}: no hand detected")
                continue

            if result.letter == letter:
                matched += 1
            else:
                print(
                    f"         {letter}: read as {result.letter} "
                    f"({result.confidence:.1f}%)"
                )

        if not checked:
            warn("no sample images to test with", "Skipped the prediction check.")

        elif matched == checked:
            ok(f"predicted all {checked} test images correctly")

        elif matched >= checked - 1:
            ok(f"predicted {matched} of {checked} test images correctly")
            warn(
                "one image was misread",
                "Normal — the model is accurate, not perfect.",
            )

        else:
            fail(
                f"only {matched} of {checked} test images were correct",
                "The scaler, label encoder and network may come from "
                "different training runs. Re-run landmark_dataset.ipynb "
                "so all three are written together.",
            )

        predictor.close()

    except Exception as error:  # noqa: BLE001 — reported to the user
        fail(f"prediction failed: {error}", "See the traceback above.")
        raise


# ==========================================================
# SUMMARY
# ==========================================================

print()

if failures:
    print(f"{len(failures)} problem(s) found. Fix them, then run this again.")
    sys.exit(1)

print("Everything checks out. Start the app with:  streamlit run app.py")
