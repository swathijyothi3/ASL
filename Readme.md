# 🤟 ASL Vision — Real-Time ASL Alphabet Recognition

ASL Vision recognizes American Sign Language (ASL) alphabet hand gestures from a static image or a live camera capture. It uses **MediaPipe Hand Landmarker** to extract hand landmarks and a custom **Artificial Neural Network (ANN)** to classify the gesture into a letter.

Built with Python, TensorFlow/Keras, scikit-learn, MediaPipe, and Streamlit.

---

## ✨ Features

- 🖼️ **Image Prediction** — upload a photo of a hand gesture and get the predicted letter
- 📷 **Live Recognition** — capture a photo from your webcam and get an instant prediction
- 🎯 21-point hand landmark detection with visual overlay
- 📊 Confidence score for every prediction
- 🧠 ANN trained on 63 numerical hand-landmark features (21 landmarks × x, y, z)

---

## 🧰 Tech Stack

| Component            | Tool                              |
|-----------------------|------------------------------------|
| Hand landmark detection | MediaPipe Hand Landmarker        |
| Classifier             | Artificial Neural Network (Keras/TensorFlow) |
| Feature scaling        | scikit-learn `StandardScaler`     |
| Label encoding          | scikit-learn `LabelEncoder`       |
| Web app / UI            | Streamlit                         |
| Image handling           | OpenCV, Pillow (PIL)             |

---

## ⚙️ How It Works

```
Input (image or webcam capture)
        │
        ▼
MediaPipe Hand Landmarker
   → detects 21 hand landmarks (x, y, z)
        │
        ▼
Crop tightly around the detected hand
   → re-run landmark detection on the crop
   → keeps landmark scale consistent regardless
     of how far the hand is from the camera
        │
        ▼
63 features → StandardScaler → ANN
        │
        ▼
Predicted ASL letter + confidence score
```

---

## 📁 Project Structure

```
ASL_DL/
├── app.py                     # Streamlit app (UI + prediction flow)
├── landmark_dataset.ipynb     # Dataset creation + model training notebook
├── requirements.txt
│
├── dataset/
│   └── ASL_Dataset/           # Raw training images, organized by letter
│
├── models/
│   └── hand_landmarker.task   # MediaPipe hand landmark detection model
│
├── output/
│   ├── asl_ann_model.keras    # Trained ANN model
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── label_encoder.pkl      # Fitted LabelEncoder
│   └── asl_landmarks.csv      # Extracted landmark dataset (features + labels)
│
└── utils/
    └── predictor.py           # ASLPredictor class — landmark extraction + inference
```

---

## 🚀 Setup & Run

1. **Clone/download the project**, then create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (e.g. `http://localhost:8501`) in your browser.

---

## 📊 Model Performance

- **Test Accuracy:** 99.13%
- **Architecture:** `Dense(128) → Dense(64) → Dropout(0.3) → Dense(32) → Dense(26, softmax)`
- **Input:** 63 features (21 landmarks × x, y, z)
- **Training samples:** ~96–102 images per class, balanced across all included letters

---

## 🔤 Supported Letters

The model currently supports **23 static ASL letters**:

```
A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
```

### Not currently supported:
- **J and Z** — these are *motion-based* signs (traced in the air), not static hand poses. Since this project classifies a single frame at a time, they're excluded by design. Supporting them would require a sequence-based approach (e.g. an LSTM over a series of frames) rather than a single-frame ANN.
- **H** — missing from the current training dataset. Unlike J/Z, H is a static sign, so it can be added by collecting/sourcing H images and retraining.

---

## 🛠️ Known Limitations & Future Work

- Static-image classification only — no temporal/motion sign support (J, Z)
- H letter missing from current dataset
- Single-hand detection only (`num_hands=1`)
- Accuracy depends on hand framing, lighting, and background consistency with training data

**Potential improvements:**
- Add H (and other missing) letters to the dataset and retrain
- Extend to word/phrase-level recognition using sequence models
- Add data augmentation for more robust real-world lighting/background variation
- Multi-hand support for two-handed signs

---

## 🙏 Acknowledgements

- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — Google
- ASL Alphabet Dataset — Kaggle