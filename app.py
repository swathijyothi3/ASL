"""
ASL Vision — American Sign Language alphabet recognition.

Streamlit front end for the MediaPipe + ANN pipeline in utils/predictor.py.
Run with:  streamlit run app.py
"""

import io
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from packaging.version import Version
from PIL import Image

from utils.predictor import ASLPredictor, FeatureVersionMismatch
from utils.visuals import (
    class_distribution,
    confidence_bars,
    hand_3d,
    hand_3d_comparison,
)


# ==========================================================
# PATHS
# ==========================================================

# Resolved against this file rather than the working directory, so the
# app behaves the same however it is launched.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


DATASET_PATH = project_path("dataset", "ASL_Dataset", "dataset")
LANDMARK_CSV = project_path("output", "asl_landmarks.csv")

# The live view needs streamlit-webrtc, which needs a WebRTC connection to
# survive the viewer's network. The snapshot modes work everywhere, so a
# missing or blocked live view degrades to a notice rather than an error.
try:
    from utils.live import live_view

    LIVE_AVAILABLE = True
except Exception:  # noqa: BLE001 — reported in the interface
    live_view = None
    LIVE_AVAILABLE = False


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="ASL Vision — Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# STREAMLIT VERSION COMPATIBILITY
# ==========================================================

# Streamlit 1.49 replaced use_container_width with width="stretch".
# Supporting both means the app runs on whatever version a marker,
# classmate or reviewer happens to have installed.

_NEW_WIDTH_API = Version(st.__version__) >= Version("1.49")

# Spread into any button or download_button that should fill its column.
STRETCH = {"width": "stretch"} if _NEW_WIDTH_API else {}

PLOTLY_CONFIG = {"displayModeBar": False}


def show_image(image, caption=None):
    if _NEW_WIDTH_API:
        st.image(image, caption=caption, width="stretch")
    else:
        st.image(image, caption=caption, use_container_width=True)


def show_chart(figure, key=None):
    if _NEW_WIDTH_API:
        st.plotly_chart(figure, width="stretch", key=key, config=PLOTLY_CONFIG)
    else:
        st.plotly_chart(
            figure, use_container_width=True, key=key, config=PLOTLY_CONFIG
        )


# ==========================================================
# STYLES
# ==========================================================

# The colour scheme lives in .streamlit/config.toml so Streamlit's own
# widgets match. Only the custom pieces are styled here.

st.markdown(
    """
    <style>

    .block-container { padding-top: 2.4rem; max-width: 1400px; }

    /* ---------- header ---------- */

    .hero {
        text-align: center;
        padding: 0.5rem 0 1.6rem 0;
    }
    .hero__title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
        background: linear-gradient(90deg, #38bdf8, #a78bfa 60%, #f472b6);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero__subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    /* ---------- predicted letter ---------- */

    .letter-card {
        border-radius: 20px;
        padding: 1.4rem 1rem 1.6rem 1rem;
        text-align: center;
        background: linear-gradient(160deg,
                    rgba(56,189,248,0.18), rgba(167,139,250,0.10));
        border: 1px solid rgba(56,189,248,0.35);
    }
    .letter-card__label {
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2.5px;
    }
    .letter-card__value {
        color: #f8fafc;
        font-size: 6.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.4rem 0 0.3rem 0;
        text-shadow: 0 0 42px rgba(56,189,248,0.55);
    }
    .letter-card__conf { color: #cbd5e1; font-size: 1rem; }

    .letter-card--empty {
        background: rgba(148,163,184,0.06);
        border: 1px dashed rgba(148,163,184,0.4);
    }
    .letter-card--empty .letter-card__value {
        color: #64748b;
        font-size: 3.6rem;
        text-shadow: none;
    }

    /* ---------- sentence builder ---------- */

    .sentence {
        min-height: 74px;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(148,163,184,0.25);
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 5px;
        font-family: ui-monospace, "Cascadia Code", monospace;
        color: #f8fafc;
        word-break: break-all;
    }
    .sentence--empty {
        color: #64748b;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: normal;
        font-family: inherit;
    }

    /* ---------- step cards ---------- */

    .step {
        height: 100%;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        background: rgba(148,163,184,0.07);
        border: 1px solid rgba(148,163,184,0.18);
    }
    .step__number {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .step__title {
        color: #f1f5f9;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0.35rem 0 0.4rem 0;
    }
    .step__body { color: #94a3b8; font-size: 0.9rem; line-height: 1.5; }

    /* ---------- alphabet grid ---------- */

    .letter-tag {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        color: #38bdf8;
        font-family: ui-monospace, monospace;
        margin-top: -0.4rem;
    }

    /* ---------- misc ---------- */

    div[data-testid="stCameraInput"] > label,
    div[data-testid="stFileUploader"] > label { font-weight: 600; }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 2.5rem 0 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# LOADING
# ==========================================================

@st.cache_resource(show_spinner="Loading the recognition model…")
def get_predictor():
    """The model is loaded once and shared by every visitor."""

    return ASLPredictor(
        model_path=project_path("models", "hand_landmarker.task"),
        ann_path=project_path("output", "asl_ann_model.keras"),
        scaler_path=project_path("output", "scaler.pkl"),
        encoder_path=project_path("output", "label_encoder.pkl"),
    )


@st.cache_data(show_spinner=False)
def load_reference_hands():
    """
    The average hand pose for each letter, taken from the landmark CSV
    produced during training.

    Averaging works because every training image is framed the same way,
    so the mean of the normalised coordinates is a clean canonical pose.
    Used for the 3D comparison and the alphabet explorer.
    """

    if not os.path.exists(LANDMARK_CSV):
        return {}, {}

    frame = pd.read_csv(LANDMARK_CSV)

    references = {}

    for letter, group in frame.groupby("label"):
        values = group.drop(columns="label").to_numpy(dtype=np.float32)
        references[str(letter)] = values.mean(axis=0).reshape(21, 3)

    counts = {
        str(letter): int(count)
        for letter, count in frame["label"].value_counts().sort_index().items()
    }

    return references, counts


@st.cache_data(show_spinner=False)
def load_sample_images():
    """One example photo per letter, for the alphabet guide."""

    samples = {}

    if not os.path.isdir(DATASET_PATH):
        return samples

    for folder in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(class_path):
            continue

        names = sorted(
            name for name in os.listdir(class_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if not names:
            continue

        image = cv2.imread(os.path.join(class_path, names[0]))

        if image is not None:
            letter = folder.replace("-samples", "")
            samples[letter] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return samples


@st.cache_data(show_spinner=False)
def sample_frame_for(letter):
    """A raw BGR training photo for a letter, used by the demo mode."""

    class_path = os.path.join(DATASET_PATH, f"{letter}-samples")

    if not os.path.isdir(class_path):
        return None

    names = sorted(
        name for name in os.listdir(class_path)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not names:
        return None

    return cv2.imread(os.path.join(class_path, names[0]))


@st.cache_data(show_spinner=False, max_entries=24)
def predict_bytes(image_bytes, smart_framing, zoom):
    """
    Decode an uploaded/captured image and classify it.

    Cached on the image itself so that clicking around the interface
    doesn't re-run MediaPipe and the network every time.
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = get_predictor().predict(frame, use_crop=smart_framing, zoom=zoom)

    return frame, result


# ==========================================================
# SESSION STATE
# ==========================================================

st.session_state.setdefault("sentence", "")
st.session_state.setdefault("last_added", None)


def append_letter(letter):
    st.session_state.sentence += letter
    st.session_state.last_added = letter


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero__title">🤟 ASL Vision</div>
        <div class="hero__subtitle">
            Show a hand sign — see the letter, the confidence, and the
            hand rebuilt in 3D.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

try:
    predictor = get_predictor()
    model_ready = True
    load_error = None
except Exception as error:  # noqa: BLE001 — surfaced to the user below
    predictor = None
    model_ready = False
    load_error = error

if not model_ready:
    st.error("The recognition model could not be loaded, so predictions are off.")

    if isinstance(load_error, FeatureVersionMismatch):
        st.markdown(
            """
            The saved model was trained on a different definition of the
            input features than this code produces. Predictions would be
            confident nonsense, so the app refuses to run.

            Rebuild the model from the current code:

            ```bash
            python tools/train_model.py
            ```
            """
        )
    else:
        st.markdown(
            """
            This usually means the model files are missing. The app expects:

            - `models/hand_landmarker.task`
            - `output/asl_ann_model.keras`
            - `output/scaler.pkl`
            - `output/label_encoder.pkl`
            """
        )

    with st.expander("Technical details"):
        st.exception(load_error)

    st.stop()


references, class_counts = load_reference_hands()

LETTERS = predictor.classes


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("### 🤟 ASL Vision")
    st.caption("Real-time ASL alphabet recognition")

    st.divider()

    st.markdown("#### Recognition settings")

    smart_framing = st.toggle(
        "Smart framing",
        value=True,
        help=(
            "After finding your hand, the app takes a closer look at just "
            "that square and re-reads the joints, which places them more "
            "precisely. Worth about 4 points of accuracy on webcam-sized "
            "hands. Turn it off to compare."
        ),
    )

    zoom = st.slider(
        "Framing tightness",
        min_value=1.2,
        max_value=2.6,
        value=float(ASLPredictor.DEFAULT_ZOOM),
        step=0.1,
        disabled=not smart_framing,
        help=(
            "How much space is kept around your hand. Lower crops tighter. "
            "The default matches the training photos."
        ),
    )

    threshold = st.slider(
        "Confidence needed",
        min_value=40,
        max_value=95,
        value=70,
        step=5,
        help=(
            "Below this the result is flagged as unsure instead of being "
            "shown as a confident answer."
        ),
    )

    st.divider()

    st.markdown("#### Model")

    accuracy = predictor.test_accuracy

    st.metric(
        "Held-out accuracy",
        f"{accuracy:.1f}%" if accuracy is not None else "—",
        help=(
            "Measured on signs the model never trained on. Real camera "
            "accuracy is lower — see the Model tab."
        ),
    )

    left, right = st.columns(2)
    left.metric("Letters", len(LETTERS))
    right.metric("Features", 63)

    if class_counts:
        st.caption(
            f"Trained on {sum(class_counts.values()):,} hand photos. "
            "J and Z need motion, so they are not included."
        )
    else:
        st.caption("J and Z need motion, so they are not included.")

    st.divider()

    st.caption(
        "Built with MediaPipe, TensorFlow/Keras, scikit-learn and Streamlit."
    )


# ==========================================================
# RESULT RENDERING
# ==========================================================

def confidence_verdict(confidence):
    """(streamlit callable, message) for a confidence score."""

    if confidence >= max(threshold, 85):
        return st.success, "Clear match — the hand shape is unambiguous."

    if confidence >= threshold:
        return st.info, "Good match, though a cleaner shot could confirm it."

    return st.warning, "Not sure about this one. Try the tips below."


def render_tips():
    with st.expander("Tips for a better reading"):
        st.markdown(
            """
            - **Get your hand closer — this matters more than anything else.**
              Once your hand is found it is read correctly almost every time,
              but a hand at arm's length in a wide frame often is not found
              at all. Aim for it covering about half the height of the
              picture.
            - **Keep the whole hand visible**, including the wrist.
            - **Light your hand from the front.** Avoid a bright window
              behind you, which turns your hand into a silhouette.
            - **A plainer background helps**, though it matters far less
              than size does.
            - **Either hand is fine**, and a tilted wrist is fine — the model
              is trained for both.
            """
        )


def render_result(frame, result, key_prefix):
    """The shared results panel used by every input mode."""

    if not result.hand_found:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            show_image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="No hand detected in this image",
            )

        with right:
            st.markdown(
                """
                <div class="letter-card letter-card--empty">
                    <div class="letter-card__label">NO HAND FOUND</div>
                    <div class="letter-card__value">—</div>
                    <div class="letter-card__conf">
                        Nothing to read in this frame yet.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.info("Move your hand into view and try again.")

        render_tips()
        return

    # ------------------------------------------------------
    # Annotated photo + predicted letter
    # ------------------------------------------------------

    annotated = predictor.draw_landmarks(
        frame.copy(),
        result.display_landmarks,
        crop_box=result.crop_box,
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        show_image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=(
                "21 landmarks detected — the blue square is the region "
                "the model actually read"
                if result.used_crop else
                "21 hand landmarks detected"
            ),
        )

        ok, encoded = cv2.imencode(".png", annotated)

        if ok:
            st.download_button(
                "Download this image",
                data=encoded.tobytes(),
                file_name=f"asl_vision_{result.letter}.png",
                mime="image/png",
                key=f"{key_prefix}_download",
            )

    with right:
        st.markdown(
            f"""
            <div class="letter-card">
                <div class="letter-card__label">PREDICTED LETTER</div>
                <div class="letter-card__value">{result.letter}</div>
                <div class="letter-card__conf">
                    {result.confidence:.1f}% confident
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.progress(min(result.confidence / 100.0, 1.0))

        notify, message = confidence_verdict(result.confidence)
        notify(message)

        st.button(
            f"➕  Add “{result.letter}” to my text",
            key=f"{key_prefix}_add",
            on_click=append_letter,
            args=(result.letter,),
            help="Builds up a word or sentence in the panel below.",
            **STRETCH,
        )

    # ------------------------------------------------------
    # How sure, and of what else
    # ------------------------------------------------------

    st.write("")

    bars, space = st.columns([1, 1], gap="large")

    with bars:
        st.markdown("##### How the model ranked the letters")
        show_chart(
            confidence_bars(result.top_k(5)),
            key=f"{key_prefix}_bars",
        )
        st.caption(
            "The five strongest candidates. A tall single bar means the "
            "model is not torn between similar hand shapes."
        )

    with space:
        st.markdown("##### Your hand in 3D")

        compare = st.toggle(
            "Overlay the reference sign",
            value=False,
            key=f"{key_prefix}_compare",
            help=(
                "Adds the average hand shape for this letter from the "
                "training data, so you can see how your sign differs."
            ),
        )

        reference = references.get(result.letter)

        if compare and reference is not None:
            figure = hand_3d_comparison(
                result.landmarks, reference, result.letter
            )
        else:
            figure = hand_3d(result.landmarks)

        show_chart(figure, key=f"{key_prefix}_3d")

        st.caption(
            "Drag to rotate, scroll to zoom, double-click to reset. "
            "Depth comes from MediaPipe — it is part of what the model reads."
        )

    render_tips()


# ==========================================================
# SENTENCE BUILDER
# ==========================================================

def render_sentence_builder():

    st.divider()
    st.markdown("### ✍️ Build a word")
    st.caption(
        "Add letters one at a time to spell something out, then copy or "
        "download the result."
    )

    text = st.session_state.sentence

    if text:
        st.markdown(
            f'<div class="sentence">{text.replace(" ", "&nbsp;")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sentence sentence--empty">'
            "Nothing yet — sign a letter above and press "
            "<b>Add to my text</b>."
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    space, back, clear, download = st.columns(4)

    with space:
        if st.button("␣  Space", key="sb_space", **STRETCH):
            st.session_state.sentence += " "
            st.rerun()

    with back:
        if st.button("⌫  Delete", key="sb_back", disabled=not text, **STRETCH):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()

    with clear:
        if st.button("🗑  Clear", key="sb_clear", disabled=not text, **STRETCH):
            st.session_state.sentence = ""
            st.rerun()

    with download:
        st.download_button(
            "⬇  Save text",
            data=text or "",
            file_name="asl_vision_text.txt",
            mime="text/plain",
            disabled=not text,
            key="sb_download",
            **STRETCH,
        )


# ==========================================================
# TABS
# ==========================================================

recognise_tab, alphabet_tab, explore_tab, how_tab, model_tab = st.tabs([
    "🎯  Recognise",
    "📖  Alphabet",
    "🧊  3D explorer",
    "⚙️  How it works",
    "📊  Model",
])


# ----------------------------------------------------------
# RECOGNISE
# ----------------------------------------------------------

with recognise_tab:

    with st.expander("👋  New here? Read this first", expanded=not st.session_state.sentence):
        st.markdown(
            """
            **Three steps:**

            1. Pick how you want to give the app a hand sign — a **live
               camera**, a **photo**, an **image file**, or a built-in
               **example**.
            2. Hold the sign up so your hand is a decent size in the frame.
               An outline appears around it once the app has found it.
            3. Read the predicted letter, and rotate the 3D hand to see
               what the model saw.

            You do not need to install or configure anything. If a result
            looks wrong, open **Tips for a better reading** underneath it.
            """
        )

    modes = ["🔴  Live camera", "📷  Take a photo",
             "🖼️  Upload an image", "✨  Use an example"]

    mode = st.radio(
        "How would you like to try it?",
        modes,
        horizontal=True,
        key="input_mode",
        help=(
            "Live camera outlines your hand and reads it as you move. "
            "Take a photo is a single snapshot, and works on any network. "
            "Upload accepts a JPG or PNG. Example uses a training photo, "
            "handy if you have no webcam."
        ),
    )

    st.write("")

    # ------------------------------------------------------
    # LIVE CAMERA
    # ------------------------------------------------------

    if mode.endswith("Live camera"):

        st.markdown("#### Hold a sign up to the camera")
        st.caption(
            "Your hand is outlined as soon as it is found, and the letter "
            "appears along the bottom of the picture. The view is mirrored, "
            "so it behaves like a mirror. Nothing is recorded or uploaded."
        )

        if not LIVE_AVAILABLE:
            st.warning(
                "The live view is unavailable in this deployment. "
                "**Take a photo** works just as well — it uses the same "
                "recognition, one frame at a time."
            )
        else:
            try:
                live_view(predictor, threshold=float(threshold), key="live_view")
            except Exception as error:  # noqa: BLE001 — shown to the user
                st.warning(
                    "The live view could not start here. **Take a photo** "
                    "runs the same recognition and works on any network."
                )
                with st.expander("Technical details"):
                    st.exception(error)

            st.info(
                "Press **START** and allow camera access. Hold each sign "
                "still for a moment — the letter is only shown once it "
                "settles, which stops it flickering between similar shapes."
            )

            with st.expander("The video will not start"):
                st.markdown(
                    """
                    The live view needs a direct video connection to your
                    browser, which some office, campus and mobile networks
                    block. Nothing is wrong with your camera or the app.

                    Switch to **Take a photo** — it runs exactly the same
                    recognition on a single frame and works on any network.
                    """
                )

        render_tips()

    # ------------------------------------------------------
    # SNAPSHOT CAMERA
    # ------------------------------------------------------

    elif mode.endswith("Take a photo"):

        st.markdown("#### Take a photo of your sign")
        st.caption(
            "Your browser will ask for camera permission. Nothing is "
            "uploaded anywhere — the photo is processed and then discarded."
        )

        photo = st.camera_input(
            "Position your hand, then press the capture button",
            key="camera",
            label_visibility="visible",
        )

        if photo is not None:
            frame, result = predict_bytes(
                photo.getvalue(), smart_framing, zoom
            )
            st.write("")
            render_result(frame, result, "cam")
        else:
            st.info(
                "Waiting for a photo. Allow camera access, make your sign, "
                "then press the capture button above."
            )

    # ------------------------------------------------------
    # UPLOAD
    # ------------------------------------------------------

    elif mode.endswith("image"):

        st.markdown("#### Upload a photo of a hand sign")

        uploaded = st.file_uploader(
            "JPG or PNG, one hand in the picture",
            type=["jpg", "jpeg", "png"],
            key="upload",
        )

        if uploaded is not None:
            frame, result = predict_bytes(
                uploaded.getvalue(), smart_framing, zoom
            )
            st.write("")
            render_result(frame, result, "up")
        else:
            st.info("Choose an image file to see a prediction.")

    # ------------------------------------------------------
    # EXAMPLE
    # ------------------------------------------------------

    else:

        st.markdown("#### Try a photo from the dataset")
        st.caption(
            "No camera needed. Pick a letter and the app runs the same "
            "pipeline on a real training photo."
        )

        chosen = st.selectbox(
            "Which letter would you like to test?",
            LETTERS,
            key="example_letter",
        )

        frame = sample_frame_for(chosen)

        if frame is None:
            st.warning(
                "The example images are not available in this deployment."
            )
        else:
            # Smart framing is deliberately skipped here. These photos are
            # already framed the way the model was trained on, so re-framing
            # them only nudges borderline cases around. It is the camera and
            # upload paths that need it.
            result = predictor.predict(frame, use_crop=False)

            st.write("")

            if result.hand_found and result.letter != chosen:
                st.warning(
                    f"The model read this **{chosen}** photo as "
                    f"**{result.letter}** — a useful reminder that "
                    "99% accuracy is not 100%."
                )

            render_result(frame, result, f"ex_{chosen}")

            st.caption(
                "These sample photos are already framed the way the model "
                "expects, so smart framing is skipped for them. It applies "
                "to the camera and upload modes, where your hand is usually "
                "a small part of a wide picture."
            )

    render_sentence_builder()


# ----------------------------------------------------------
# ALPHABET
# ----------------------------------------------------------

with alphabet_tab:

    st.markdown("### The signs this app can read")
    st.caption(
        "Copy these hand shapes to get the best results. Every photo below "
        "is taken from the data the model learned from."
    )

    samples = load_sample_images()

    if not samples:
        st.info("The reference photos are not available in this deployment.")
    else:
        letters = list(samples.keys())
        columns_per_row = 6

        for start in range(0, len(letters), columns_per_row):
            row = letters[start:start + columns_per_row]
            columns = st.columns(columns_per_row)

            for column, letter in zip(columns, row):
                with column:
                    show_image(samples[letter])
                    st.markdown(
                        f'<div class="letter-tag">{letter}</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    st.markdown("#### Letters that are missing, and why")

    missing_left, missing_right = st.columns(2)

    with missing_left:
        st.markdown(
            """
            **J and Z — by design**

            These two are not hand shapes at all: you trace them through
            the air. This model looks at a single still frame, so there is
            no movement for it to see. Reading them properly needs a model
            that works across a sequence of frames.
            """
        )

    with missing_right:
        st.markdown(
            """
            **H — missing from the data**

            H *is* a still hand shape, so nothing about the approach rules
            it out. It simply is not in this dataset. Adding photos of H
            and retraining would be enough to support it.
            """
        )


# ----------------------------------------------------------
# 3D EXPLORER
# ----------------------------------------------------------

with explore_tab:

    st.markdown("### Compare two signs in 3D")
    st.caption(
        "These are the average hand shapes the model learned, drawn from "
        "the landmark data. Rotating them shows why some letters are easy "
        "to tell apart and others are not."
    )

    if not references:
        st.info("The landmark data is not available in this deployment.")
    else:
        show_axes = st.toggle(
            "Show axes",
            value=False,
            help="Turn on the grid if you want a sense of scale.",
        )

        left, right = st.columns(2, gap="large")

        # Default to M and N — the pair the model most often confuses, so
        # the first thing anyone sees is a genuinely close comparison.
        def index_of(letter, fallback):
            return LETTERS.index(letter) if letter in LETTERS else fallback

        with left:
            first = st.selectbox(
                "Left hand shape", LETTERS,
                index=index_of("M", 0), key="explore_a",
            )
            show_chart(
                hand_3d(references[first], show_axes=show_axes),
                key="explore_fig_a",
            )
            st.caption(f"Average hand shape for **{first}**")

        with right:
            second = st.selectbox(
                "Right hand shape", LETTERS,
                index=index_of("N", min(1, len(LETTERS) - 1)),
                key="explore_b",
            )
            show_chart(
                hand_3d(references[second], show_axes=show_axes),
                key="explore_fig_b",
            )
            st.caption(f"Average hand shape for **{second}**")

        st.info(
            "Try **M** against **N**, or **A** against **S**. The shapes "
            "nearly overlap, which is exactly where the model's mistakes "
            "come from."
        )


# ----------------------------------------------------------
# HOW IT WORKS
# ----------------------------------------------------------

with how_tab:

    st.markdown("### From photo to letter")
    st.caption("Four steps, none of which involve the raw pixels reaching the classifier.")

    steps = [
        (
            "STEP 01",
            "Take a picture",
            "A camera capture, an uploaded file or a sample image. "
            "All three go down exactly the same path.",
        ),
        (
            "STEP 02",
            "Find the hand",
            "MediaPipe locates 21 joints — fingertips, knuckles, wrist — "
            "and returns each as an x, y and depth value.",
        ),
        (
            "STEP 03",
            "Reduce to pose",
            "The joints are centred on your wrist and scaled to a standard "
            "size, so where your hand is and how far away it sits drop out "
            "entirely. Only the shape remains.",
        ),
        (
            "STEP 04",
            "Classify",
            "Those 63 numbers are scaled and passed to the neural network, "
            "which returns a probability for each letter.",
        ),
    ]

    columns = st.columns(4, gap="medium")

    for column, (number, title, body) in zip(columns, steps):
        with column:
            st.markdown(
                f"""
                <div class="step">
                    <div class="step__number">{number}</div>
                    <div class="step__title">{title}</div>
                    <div class="step__body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.divider()

    st.markdown("#### Why step 3 is the important one")

    st.markdown(
        """
        MediaPipe reports each joint as a fraction of the picture it was
        given, not in centimetres. Raw, those numbers describe **where your
        hand is and how big it looks** as much as what shape it is making.

        Every training photo was a 224×224 close-up of a centred hand. A
        model fed the raw numbers therefore learns that framing, and a
        webcam — hand smaller, off to one side, maybe the other hand —
        falls outside anything it has seen. It stays confident and starts
        being wrong, which is worse than admitting uncertainty.

        Centring each hand on its own wrist and dividing by its own size
        removes position and distance completely. The training data is also
        mirrored, so either hand reads the same, and slightly tilted copies
        are added so a crooked wrist does not matter.

        One thing is deliberately *not* removed: rotation. In ASL, **P is K
        rotated downwards** and **Q is G rotated downwards**. A model blind
        to rotation could never separate those pairs, however well trained.
        """
    )

    st.divider()

    st.markdown("#### What the model actually sees")

    st.markdown(
        """
        Not your photo. The classifier only ever receives 63 numbers —
        21 landmarks × (x, y, depth). That is why the app runs quickly,
        why it works on plain backgrounds, and why the 3D view is a fair
        picture of its input rather than a decoration.
        """
    )


# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------

with model_tab:

    st.markdown("### Model and data")

    columns = st.columns(4)

    accuracy = predictor.test_accuracy

    columns[0].metric(
        "Held-out accuracy",
        f"{accuracy:.1f}%" if accuracy is not None else "—",
    )
    columns[1].metric("Letters covered", len(LETTERS))
    columns[2].metric("Input features", 63)
    columns[3].metric(
        "Training photos",
        f"{sum(class_counts.values()):,}" if class_counts else "—",
    )

    st.caption(
        "Measured on a held-out 20% of the dataset the network never saw "
        "while training. Read the next section before trusting it."
    )

    robustness = (predictor.info or {}).get("robustness") or {}

    if robustness:
        st.write("")
        st.markdown("#### Does it survive a real camera?")
        st.caption(
            "The same held-out signs, altered the way a webcam alters them. "
            "A model that only works on tidy dataset photos falls apart here."
        )

        rows = pd.DataFrame(
            {
                "Condition": list(robustness.keys()),
                "Accuracy": [f"{value:.1f}%" for value in robustness.values()],
            }
        )

        st.dataframe(rows, hide_index=True, **STRETCH)

    st.divider()

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.markdown("#### Photos per letter")

        if class_counts:
            show_chart(class_distribution(class_counts), key="dist")
            st.caption(
                "A roughly even spread, so no single letter dominates "
                "what the network learned."
            )
        else:
            st.info("The landmark data is not available in this deployment.")

    with right:
        st.markdown("#### Network")

        st.code(
            "Input               63 features\n"
            "Dense    128        ReLU\n"
            "Dense     64        ReLU\n"
            "Dropout  0.3\n"
            "Dense     64        ReLU\n"
            "Dense           softmax over letters",
            language="text",
        )

        st.markdown(
            """
            Trained with Adam and early stopping on validation loss, on
            pose-normalised landmarks. Every sign is also learned mirrored,
            tilted and slightly jittered, which is what makes it hold up
            away from the dataset.
            """
        )

    st.divider()

    st.markdown("#### Honest limitations")

    st.markdown(
        """
        - **Getting your hand *found* is now the hard part.** Once MediaPipe
          locates a hand, it is read correctly around 95–99% of the time.
          But at arm's length in a wide frame it often is not located at
          all. This is why the app keeps asking you to hold your hand
          closer — it is the single biggest thing you control.
        - **A high held-out score is not real-world accuracy.** That split
          comes from the same photo collection as the training data: same
          lighting, same backgrounds, same hands. Your camera is harder.
        - **One frame at a time.** J and Z are movements, so they are out
          of scope for this design.
        - **H is not in the dataset**, so the app cannot produce it.
        - **One hand only.** The detector is set to a single hand.
        - **M, N, S and T are genuinely close** — all closed fists differing
          by thumb placement. These are where mistakes concentrate.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    '<div class="footer">🤟 ASL Vision — MediaPipe + neural network, '
    "built with Streamlit</div>",
    unsafe_allow_html=True,
)
