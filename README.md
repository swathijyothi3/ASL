# 🤟 ASL Vision — American Sign Language Alphabet Recognition

A web app that reads a static American Sign Language letter from a photo or a
webcam capture. It uses **MediaPipe Hand Landmarker** to find 21 hand joints
and a small **neural network** to turn those joints into a letter.

Built with Python, TensorFlow/Keras, scikit-learn, MediaPipe, Plotly and
Streamlit.

---

## Features

- **Three ways in** — take a photo with your camera, upload an image, or run a
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
Photo (camera, upload or sample)
        │
        ▼
MediaPipe Hand Landmarker  →  21 landmarks (x, y, depth)
        │
        ▼
Square crop around the hand  →  detect again
        │
        ▼
63 features  →  StandardScaler  →  neural network
        │
        ▼
Letter + probability for every class
```

### Why the crop step exists

MediaPipe reports landmark positions **relative to the image it is given**,
not in real-world units. Every training image here is a 224×224 close-up with
the hand filling most of the frame. A webcam photo is wide, and the hand
usually occupies a small part of it — so the same sign produces very different
numbers, far outside the range the network saw during training.

The app therefore crops a square around the detected hand and runs detection a
second time on that crop, which puts the coordinates back into the range the
model was trained on.

`tools/tune_crop.py` measures this. It pastes dataset images into 1280×720
canvases to imitate a webcam, then scores the pipeline against the known
labels:

| Hand covers … of the frame | Without the crop | With the crop | Hand found at all |
|---|---|---|---|
| 60% of frame height | 15.2% correct | **81.5%** correct | 82.6% |
| 45% of frame height | 4.3% correct | **63.0%** correct | 66.3% |
| 30% of frame height | 0.0% correct | 8.7% correct | 9.8% |

Read the last column together with the others. Once the crop is in place,
almost every hand that MediaPipe *finds* is classified correctly — 63.0 out of
66.3, and 81.5 out of 82.6. What limits the app after that is detection, not
classification: MediaPipe simply cannot locate a hand that occupies 30% of a
wide frame. That is why the interface keeps telling people to fill the frame.

The same script derives the two constants it uses, straight from the training
landmarks rather than by guesswork:

- the hand's bounding box covers a mean **0.576** of the training frame, so the
  crop is sized `1 / 0.576 ≈ 1.74` times the box;
- the hand sits **low** in those images, centred at about (0.48, 0.59), so the
  crop is anchored there rather than dead centre.

The sweep is flat between about 1.6 and 1.8, so the exact figure is not
delicate, and the low anchor turns out to matter less than the zoom does.

On the original 224×224 images the crop costs a little accuracy (100% → 98.9%)
— they are already framed the way the model expects. That trade is worth
making, since real users point a webcam at themselves rather than feeding in
dataset files. You can toggle it off in the sidebar ("Smart framing").

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
│   ├── predictor.py           # Landmark extraction + inference
│   └── visuals.py             # Plotly figures, including the 3D hand
│
└── tools/
    ├── tune_crop.py           # Measures the crop step and picks the zoom
    └── check_setup.py         # Verifies an install before running the app
```

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

`packages.txt` installs the system libraries OpenCV needs on Linux. Without it
the build fails with `libGL.so.1: cannot open shared object file`.

Leave that file as it is. Adding `libglib2.0-0` to it looks like the obvious
next step when debugging OpenCV imports, but it conflicts with a package
already present in the Streamlit Cloud image and fails the whole apt stage.
The four entries listed are the set that builds cleanly.

---

## Model and data

| | |
|---|---|
| Test accuracy | 99.13% on a held-out 20% split |
| Letters | 23 |
| Training photos | 2,326 |
| Input | 63 features — 21 landmarks × (x, y, depth) |
| Architecture | `Dense(128) → Dense(64) → Dropout(0.3) → Dense(32) → softmax` |
| Training | Adam, early stopping on validation loss |

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

The 99.13% is measured on a split of the same photo collection used for
training — similar lighting, backgrounds and hands. It says the network
separates these classes well; it does not promise the same accuracy on your
webcam. Expect more mistakes in real use, particularly between visually
similar signs such as M and N, or A and S.

Other limitations:

- One hand at a time (`num_hands=1`).
- Static poses only.
- Accuracy depends on framing, lighting and background.

---

## Acknowledgements

- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — Google
- ASL Alphabet Dataset — Kaggle
