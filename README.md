# 🤟 ASL Vision — American Sign Language Alphabet Recognition

A web app that reads a static American Sign Language letter from a photo or a
webcam capture. It uses **MediaPipe Hand Landmarker** to find 21 hand joints
and a small **neural network** to turn those joints into a letter.

Built with Python, TensorFlow/Keras, scikit-learn, MediaPipe, Plotly and
Streamlit.

---

## Features

- **Live camera view** — your hand is outlined as you move and the letter is
  read continuously, so you can adjust until it reads correctly. It runs
  entirely in your browser: the video never leaves your machine and the server
  does no per-frame work.
- **Three more ways in** — take a single photo, upload an image, or run a
  built-in example if you have no webcam.
- **Interactive 3D hand** — the 21 detected landmarks drawn as a rotatable
  skeleton. This is not decoration: depth is one of the three values per
  landmark that the classifier reads.
- **Compare against the reference** — overlay the average hand shape for the
  predicted letter to see how your sign differs.
- **Ranked predictions** — the five strongest candidates with their
  probabilities, so a close call is visible rather than hidden.
- **Word builder** — collect letters into a word or sentence and save it.
- **Alphabet guide** — a reference photo for every letter the model supports.
- **3D explorer** — compare any two letters' canonical hand shapes side by side.

---

## How it works

```
Frame (live video, photo, upload or sample)
        │
        ▼
MediaPipe Hand Landmarker  →  21 landmarks (x, y, depth)
   strict pass, then a permissive one if nothing was found
        │
        ▼
Square crop around the hand  →  detect again, for sharper joints
        │
        ▼
Centre on the wrist, scale by hand size   ← the important step
        │
        ▼
63 features  →  StandardScaler  →  neural network
        │
        ▼
Letter + probability for every class
```

### The mistake this project started with

MediaPipe reports each joint as a fraction of the image it was given. Fed in
raw, those numbers describe **where the hand is and how large it appears** just
as much as what shape it is making. Every training photo here is a 224×224
close-up of a centred hand, so a model trained on raw coordinates learns that
framing. Point a webcam at yourself — hand smaller, off to one side, possibly
the other hand — and the input lands outside anything it ever saw. It stays
confident and starts being wrong.

The fix is to reduce each hand to pose only: centre it on its own wrist and
divide by its own size, so position and distance drop out entirely
(`utils/features.py`). The training set is also mirrored, since a sign means
the same letter with either hand, and tilted copies are added so a crooked
wrist does not matter.

**Rotation is deliberately left in.** In ASL, P is K rotated downwards and Q is
G rotated downwards. A rotation-invariant model could never separate those
pairs, however well trained — so tilt is handled by augmentation rather than
by normalising it away.

### What that changed, measured

`tools/evaluate.py` pastes dataset photos into 1280×720 canvases to imitate a
webcam at various distances, then runs the whole pipeline against known labels.
Before and after the change:

| Condition | Raw coordinates | Pose-only |
|---|---|---|
| Dataset photo, as-is | 93.9% | **98.3%** |
| Mirrored (other hand) | 74.8% | **97.4%** |
| Tilted 20° | 73.9% | **95.7%** |
| Webcam, hand 60% of frame | 74.8% | **93.0%** |
| Webcam, hand 45% of frame | 59.1% | **88.7%** |
| Webcam 45%, mirrored | 52.2% | **81.7%** |

### Detection is now the limit, not classification

Counting only the frames where MediaPipe actually found a hand, the classifier
is right **95–99% of the time** in every condition above. What remains is the
detector: a hand at arm's length in a wide frame often is not found at all.

Two things help, both measured in `tools/`:

- **A detection cascade.** A strict pass runs first, so an obvious hand is
  taken at high confidence; only when that finds nothing does a permissive pass
  run. At a normal sitting distance this lifted hands-found from 67% to ~90%.
- **The crop.** Re-detecting on a close crop sharpens the joints — worth about
  4 points at webcam distance. Toggle it in the sidebar as "Smart framing".

This is why the interface keeps asking you to bring your hand closer: it is the
one thing you control that matters most.

---

## Project structure

```
ASL_VISION/
├── app.py                     # Streamlit interface
├── landmark_dataset.ipynb     # Dataset extraction + model training
├── requirements.txt           # Python dependencies (pinned)
├── packages.txt               # System libraries needed by OpenCV on Linux
│
├── .streamlit/
│   └── config.toml            # App theme
│
├── dataset/ASL_Dataset/dataset/   # Training photos, one folder per letter
│
├── models/
│   └── hand_landmarker.task   # MediaPipe landmark model
│
├── output/
│   ├── asl_ann_model.keras    # Trained network
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── label_encoder.pkl      # Fitted LabelEncoder
│   └── asl_landmarks.csv      # Extracted landmarks (features + labels)
│
├── utils/
│   ├── features.py            # Landmarks → pose-only features (shared)
│   ├── predictor.py           # Detection + inference
│   ├── webcam.py              # Live camera view (runs in the browser)
│   └── visuals.py             # Plotly figures, including the 3D hand
│
└── tools/
    ├── train_model.py         # Trains the shipped model
    ├── export_web_model.py    # Exports weights for the browser live view
    ├── evaluate.py            # Whole-pipeline accuracy, incl. webcam sims
    ├── tune_crop.py           # Measures the crop refinement
    └── check_setup.py         # Verifies an install before running the app
```

After retraining, regenerate the browser copy too, or the live view will keep
using the old weights:

```bash
python tools/export_web_model.py
```

`utils/features.py` is imported by both the trainer and the predictor on
purpose. If the two ever disagreed about how landmarks become numbers, the
result would be confident nonsense rather than an error — so the model records
a feature version and the app refuses to load on a mismatch.

---

## Run it locally

Python **3.11** is recommended. The pinned versions of NumPy and pandas
require 3.11 or newer.

```bash
git clone https://github.com/swathijyothi3/ASL_VISION.git
cd ASL_VISION
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux use `source venv/bin/activate` instead.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Check that everything loads:

```bash
python tools/check_setup.py
```

Start the app:

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). The camera
needs `localhost` or HTTPS — browsers block it on plain remote HTTP.

---

## Deploy to Streamlit Community Cloud

The repository is deployment-ready: the model, the landmark data and the
sample images are all committed, so nothing needs to be fetched at runtime.

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   the same GitHub account.
3. Choose **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `swathijyothi3/ASL_VISION`
   - **Branch:** `master`
   - **Main file path:** `app.py`
5. Open **Advanced settings** and set **Python version** to **3.11**.
6. Click **Deploy**. The first build takes several minutes, mostly
   installing TensorFlow.

### Two pins that the deployment depends on

Both were found by resolving `requirements.txt` against Linux rather than by
trial and error, and changing either one will break the deployed app.

**`tensorflow-cpu==2.20.0`** — version 2.21.0 was never published for Linux;
it exists only for Windows and macOS. Asking for it on Streamlit Cloud makes
the entire dependency install fail, which leaves the app with no third-party
packages at all and a confusing `ImportError` on the first `import`. `keras`
is pinned alongside it because the saved model was written by Keras 3.15.1
while TensorFlow only requires `keras>=3.10.0`.

**`mediapipe==0.10.31`** — from 0.10.32 onwards (1.0.0 included) mediapipe
depends on `opencv-contrib-python`, the desktop build. pip installs it over
the headless one, and that binary needs `libGL`, `libSM`, `libICE`, `libX11`,
`libxcb` and `libglib` present at import. `libglib2.0-0` cannot be added
through `packages.txt` without an apt conflict, so this is not fixable from
the system side. 0.10.31 declares no OpenCV dependency, so only the headless
build is installed and no system libraries are needed at all.

Before bumping mediapipe, check that the new version has not reintroduced the
dependency:

```bash
pip download --no-deps mediapipe==<version> -d /tmp/mp
```

then look for `opencv` in the wheel's `METADATA`.

`packages.txt` is kept as a safety net for OpenCV on Linux, but with the
headless build it is no longer doing any real work. Do not add
`libglib2.0-0` to it.

---

## Model and data

| | |
|---|---|
| Held-out accuracy | 100% on a 20% split the model never trained on |
| Letters | 23 |
| Training photos | 2,294 hands, 25,690 rows after mirroring and augmentation |
| Input | 63 features — 21 landmarks × (x, y, depth), wrist-centred and scaled |
| Architecture | `Dense(256) → Dense(128) → Dense(64) → softmax`, dropout 0.3 |
| Training | Adam, early stopping on validation loss |

Retrain with:

```bash
python tools/train_model.py
```

That rewrites the model, scaler, encoder and `output/model_info.json`, which
records the feature version and the measured accuracy the app displays.

### Supported letters

```
A B C D E F G I K L M N O P Q R S T U V W X Y
```

### Not supported

- **J and Z** are traced through the air rather than held. This model reads a
  single still frame, so there is no motion for it to see. Supporting them
  needs a sequence model such as an LSTM over several frames.
- **H** is a static sign and would work fine, but it is not present in this
  dataset. Adding photos of H and retraining is all it would take.

### Reading the accuracy figure honestly

**100% on the held-out split does not mean 100% on your camera.** That split
comes from the same photo collection as the training data: same lighting, same
backgrounds, same hands. It says the network separates these classes cleanly —
nothing more. The webcam simulations above (89% at a normal sitting distance)
are the more useful number, and even those are kinder than a real room.

What actually limits it, in order:

1. **Whether your hand is found at all.** At arm's length in a wide frame,
   MediaPipe often does not locate it. Hold your hand closer — roughly half the
   height of the picture.
2. **Genuinely similar signs.** M, N, S and T are all closed fists differing
   only in thumb placement. Most remaining mistakes are among these.

Other limitations:

- One hand at a time (`num_hands=1`).
- Static poses only — J and Z need motion.
- The live view fetches the MediaPipe runtime from a CDN the first time it
  runs. If that is blocked it says so, and the photo modes still work.

---

## Acknowledgements

- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — Google
- ASL Alphabet Dataset — Kaggle
