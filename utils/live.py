"""
Live camera view — the hand outlined and read as you move.

Runs each video frame through the same predictor the snapshot modes use,
draws the skeleton and a framing bracket over the hand, and reports the
letter on the picture.

Two decisions worth knowing about:

**The picture is mirrored.** People expect a camera preview to behave like
a mirror; without it, raising your right hand makes the overlay appear on
what feels like the wrong side. This is safe here because the classifier
was trained with mirrored copies of every sign, so it reads either hand.

**The letter is only shown once it settles.** A per-frame reading flickers
between similar letters and looks broken even when it is mostly right.
A letter has to hold across most of a short window before it is announced,
which matches how someone actually forms a sign — move, hold, read.
"""

from collections import Counter, deque

import av
import cv2
import numpy as np

from streamlit_webrtc import WebRtcMode, webrtc_streamer


# Public STUN server, enough for most networks. Some corporate or mobile
# networks need a TURN relay, which Community Cloud does not provide —
# hence the snapshot modes remaining the reliable path.
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

MEDIA_CONSTRAINTS = {
    "video": {"width": {"ideal": 960}, "height": {"ideal": 720}},
    "audio": False,
}

# Frames are downscaled before detection. MediaPipe gains nothing from a
# larger picture here, and Community Cloud gives the app a single CPU.
WORKING_WIDTH = 720


class LiveSigner:
    """Video callback: detect, draw, classify, smooth."""

    def __init__(self, predictor, threshold=60.0, window=9, agreement=0.55):
        self.predictor = predictor
        self.threshold = threshold

        self.recent = deque(maxlen=window)
        self.agreement = agreement

        self.letter = None
        self.confidence = 0.0

    # ------------------------------------------------------

    def _settle(self, letter, confidence):
        """Only report a letter that holds across the window."""

        self.recent.append(letter)

        if not self.recent:
            return

        winner, count = Counter(self.recent).most_common(1)[0]

        if winner is not None and count >= self.agreement * self.recent.maxlen:
            self.letter = winner
            self.confidence = confidence if winner == letter else self.confidence
        elif winner is None:
            self.letter = None
            self.confidence = 0.0

    # ------------------------------------------------------

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Mirror first so everything drawn afterwards lines up with what
        # the person sees of themselves.
        image = cv2.flip(image, 1)

        height, width = image.shape[:2]

        scale = WORKING_WIDTH / width if width > WORKING_WIDTH else 1.0

        working = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image
        )

        try:
            result = self.predictor.predict(working)
        except Exception:  # noqa: BLE001 — never kill the video stream
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        if not result.hand_found:
            self._settle(None, 0.0)
            self._banner(image, None, 0.0)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        self._settle(result.letter, result.confidence)

        # Landmarks are normalised, so they map onto the full-size frame
        # regardless of the scale used for detection.
        self.predictor.draw_landmarks(
            image,
            result.display_landmarks,
            crop_box=self._scale_box(result.crop_box, scale),
            box_color=(120, 220, 120) if self.confidence >= self.threshold
            else (90, 200, 255),
        )

        self._banner(image, self.letter, self.confidence)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    @staticmethod
    def _scale_box(box, scale):
        if box is None or scale == 1.0:
            return box
        x, y, side = box
        return (x / scale, y / scale, side / scale)

    # ------------------------------------------------------

    def _banner(self, image, letter, confidence):
        """A readable strip along the bottom, rather than text on the picture."""

        height, width = image.shape[:2]

        strip = max(64, height // 8)
        top = height - strip

        panel = image[top:, :].copy()
        shaded = (panel * 0.35).astype(np.uint8)
        image[top:, :] = shaded

        if letter is None:
            cv2.putText(
                image, "Show your hand to the camera",
                (24, top + strip // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 210, 225), 2, cv2.LINE_AA,
            )
            return

        settled = confidence >= self.threshold
        colour = (120, 220, 120) if settled else (120, 200, 255)

        cv2.putText(
            image, letter,
            (28, top + strip - 14),
            cv2.FONT_HERSHEY_SIMPLEX, strip / 42.0, colour, 4, cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{confidence:.0f}% confident" if settled else "hold the sign steady",
            (28 + int(strip * 1.1), top + strip - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (225, 232, 240), 2, cv2.LINE_AA,
        )


def live_view(predictor, threshold=60.0, key="live"):
    """Render the streamer. Returns the context so callers can check state."""

    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=MEDIA_CONSTRAINTS,
        video_frame_callback=LiveSigner(predictor, threshold=threshold),
        async_processing=True,
    )
