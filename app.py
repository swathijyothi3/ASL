import sys
import asyncio

# ==========================================================
# WINDOWS FIX FOR streamlit-webrtc / aiortc
#
# On Windows, asyncio defaults to ProactorEventLoop, which is
# incompatible with aiortc's UDP-based STUN/ICE negotiation.
# This causes the webcam to connect but never deliver frames
# (silent "sendto on NoneType" crashes in aioice). Forcing the
# Selector event loop fixes it. Must run before any other
# asyncio-dependent imports (including streamlit_webrtc).
# ==========================================================

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import streamlit as st
import textwrap

from PIL import Image

from utils.predictor import ASLPredictor


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="ASL Vision",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    textwrap.dedent("""
    <style>

    /* ======================================================
       MAIN BACKGROUND
       ====================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #111827 50%,
                #172554 100%
            );
    }


    /* ======================================================
       GENERAL TEXT
       ====================================================== */

    .stApp p,
    .stApp li,
    .stApp span,
    .stApp label {
        color: #e5e7eb;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #ffffff;
    }


    /* ======================================================
       MAIN TITLE
       ====================================================== */

    .main-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }


    /* ======================================================
       CARD
       ====================================================== */

    .card {
        padding: 1.5rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1.2rem;
    }


    /* ======================================================
       PREDICTION BOX
       ====================================================== */

    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        background:
            linear-gradient(
                160deg,
                rgba(96, 165, 250, 0.15) 0%,
                rgba(255, 255, 255, 0.06) 100%
            );
        border: 1px solid rgba(96, 165, 250, 0.35);
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }

    .prediction-label {
        color: #93c5fd;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .prediction-value {
        color: #ffffff;
        font-size: 6.5rem;
        font-weight: 900;
        line-height: 1.1;
        margin: 0.5rem 0;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.6);
    }

    .confidence-text {
        color: #cbd5e1;
        font-size: 1.1rem;
    }

    .no-hand-box {
        padding: 2.5rem 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.25);
        text-align: center;
        color: #94a3b8;
        margin-top: 1rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {
        color: #111827;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #111827;
    }

    .accuracy-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2563eb;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] section {
        background: #f8fafc !important;
        border-radius: 15px;
    }

    [data-testid="stFileUploader"] label {
        color: #111827 !important;
    }


    /* ======================================================
       STEP CARDS
       ====================================================== */

    .step-card {
        padding: 1.2rem;
        min-height: 150px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .step-number {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
    }


    /* ======================================================
       WEBCAM FRAME
       ====================================================== */

    div[data-testid="stCustomComponentV1"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(96, 165, 250, 0.25);
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🤟 ASL Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    textwrap.dedent("""
    <div class="subtitle">
        Real-Time American Sign Language Recognition
        using MediaPipe + Artificial Neural Network
    </div>
    """),
    unsafe_allow_html=True
)


# ==========================================================
# LOAD PREDICTOR
# ==========================================================

@st.cache_resource
def load_predictor():
    return ASLPredictor(
        model_path="models/hand_landmarker.task",
        ann_path="output/asl_ann_model.keras",
        scaler_path="output/scaler.pkl",
        encoder_path="output/label_encoder.pkl"
    )


predictor = load_predictor()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤟 ASL Vision</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.header("About")

    st.write(
        "This application recognizes American Sign Language "
        "alphabet gestures using computer vision and "
        "deep learning."
    )

    st.markdown(
        """
        **Technologies used:**

        🤚 MediaPipe Hand Landmarker

        🧠 Artificial Neural Network

        📊 63 hand landmark features

        🎥 Real-time webcam processing

        🐍 Python

        🌐 Streamlit
        """
    )

    st.markdown("---")

    st.header("Model Performance")

    st.write("Test Accuracy")

    st.markdown(
        '<div class="accuracy-number">99.13%</div>',
        unsafe_allow_html=True
    )

    st.write("The ANN was evaluated on unseen test data.")


# ==========================================================
# TABS
# ==========================================================

tab1, tab2 = st.tabs(
    [
        "🖼️ Image Prediction",
        "📷 Live Recognition"
    ]
)


# ==========================================================
# IMAGE PREDICTION
# ==========================================================

with tab1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖼️ Upload an ASL Hand Image")
    st.write(
        "Upload an image containing one ASL hand gesture "
        "and let the ANN predict the sign."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "💡 Tip: For the best prediction, make sure "
        "the hand is clearly visible and well positioned."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)

            # RGB → BGR
            frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------

            letter, confidence, landmarks = predictor.predict(frame)

            col1, col2 = st.columns([1.4, 1])

            # --------------------------------------------------
            # IMAGE
            # --------------------------------------------------

            with col1:

                display_frame = frame.copy()

                if landmarks is not None:
                    display_frame = predictor.draw_landmarks(
                        display_frame, landmarks
                    )

                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                st.image(
                    display_frame,
                    caption="Detected Hand Landmarks",
                    use_column_width=True
                )

            # --------------------------------------------------
            # PREDICTION DISPLAY
            # --------------------------------------------------

            with col2:

                if letter is not None:

                    st.markdown("### 🎯 Predicted ASL Letter")

                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">ASL LETTER</div>
                            <div class="prediction-value">{letter}</div>
                            <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                        </div>
                        """).strip(),
                        unsafe_allow_html=True
                    )

                    st.write("Prediction Confidence")
                    st.progress(min(confidence / 100, 1.0))

                    if confidence >= 90:
                        st.success("Very high confidence prediction!")
                    elif confidence >= 70:
                        st.info("Good confidence prediction.")
                    else:
                        st.warning("Low confidence. Try a clearer image.")

                else:
                    st.markdown(
                        textwrap.dedent("""
                        <div class="no-hand-box">
                            ❌ No hand detected.<br>
                            Please upload a clearer hand image.
                        </div>
                        """).strip(),
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error("Unable to process this image.")
            print("IMAGE PREDICTION ERROR:", repr(e))


# ==========================================================
# LIVE WEBCAM RECOGNITION
# ==========================================================

with tab2:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Live ASL Recognition")
    st.write(
        "Position your hand gesture in front of the camera, "
        "then click the capture button to predict."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "💡 Tip: Allow camera permission when your browser asks. "
        "Then show your gesture and click the camera button below."
    )

    # ------------------------------------------------------
    # SNAPSHOT CAPTURE
    #
    # st.camera_input uses the browser's native camera API
    # directly (no aiortc / STUN / ICE negotiation), so it
    # avoids the WebRTC connection issues entirely. The user
    # shows their gesture and clicks the built-in capture
    # button to take a photo, which is then run through the
    # same predictor used in the Image Prediction tab.
    # ------------------------------------------------------

    camera_photo = st.camera_input(
        "Capture your hand gesture",
        key="live_camera_input"
    )

    if camera_photo is not None:

        try:
            image = Image.open(camera_photo).convert("RGB")
            image_array = np.array(image)

            frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            letter, confidence, landmarks = predictor.predict(frame)

            col1, col2 = st.columns([1.4, 1])

            with col1:

                display_frame = frame.copy()

                if landmarks is not None:
                    display_frame = predictor.draw_landmarks(
                        display_frame, landmarks
                    )

                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                st.image(
                    display_frame,
                    caption="Captured Frame",
                    use_column_width=True
                )

            with col2:

                if letter is not None:

                    st.markdown("### 🎯 Predicted ASL Letter")

                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">ASL LETTER</div>
                            <div class="prediction-value">{letter}</div>
                            <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                        </div>
                        """).strip(),
                        unsafe_allow_html=True
                    )

                    st.write("Prediction Confidence")
                    st.progress(min(confidence / 100, 1.0))

                    if confidence >= 90:
                        st.success("Very high confidence prediction!")
                    elif confidence >= 70:
                        st.info("Good confidence prediction.")
                    else:
                        st.warning("Low confidence. Try repositioning your hand.")

                else:
                    st.markdown(
                        textwrap.dedent("""
                        <div class="no-hand-box">
                            ❌ No hand detected.<br>
                            Try repositioning your hand and capture again.
                        </div>
                        """).strip(),
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error("Unable to process the captured frame.")
            print("LIVE CAPTURE ERROR:", repr(e))


# ==========================================================
# HOW THE SYSTEM WORKS
# ==========================================================

st.markdown("---")
st.subheader("⚙️ How the System Works")
st.write(
    "The application converts a hand gesture "
    "into a predicted ASL alphabet letter."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(
        """
        **01 — Input**

        Upload an image or use the live webcam.
        """
    )

with col2:
    st.info(
        """
        **02 — MediaPipe**

        Detects 21 hand landmarks from the image.
        """
    )

with col3:
    st.info(
        """
        **03 — ANN**

        Processes the 63 numerical landmark features.
        """
    )

with col4:
    st.info(
        """
        **04 — Prediction**

        Returns the ASL letter and confidence.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        🤟 ASL Vision • Deep Learning + Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)