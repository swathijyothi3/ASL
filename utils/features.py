"""
How a detected hand becomes numbers for the classifier.

Both training (tools/train_model.py) and inference (utils/predictor.py)
import from here, so the two can never drift apart — a mismatch between
them produces confident nonsense rather than an error, which is the worst
kind of bug to chase.

Why not feed MediaPipe's landmarks in raw
----------------------------------------
MediaPipe reports each landmark as a fraction of the image it was given.
Those numbers describe where the hand is and how big it appears as much as
what shape it is making. A model trained on them learns the framing of its
training photos: move the hand to one side, sit further from the camera,
or use the other hand, and the input drifts outside anything it saw.

Centring each hand on its own wrist and dividing by its own size removes
position and distance completely, leaving only the shape of the sign.

Why rotation is deliberately left in
------------------------------------
It is tempting to normalise rotation away too. Doing so would break the
alphabet: P is K rotated downwards, and Q is G rotated downwards. A
rotation-invariant model could not tell those pairs apart no matter how
well it was trained. Moderate tilt is handled by augmenting the training
data instead — see rotate_landmarks below.
"""

import numpy as np


# Bumped whenever the feature definition changes. The predictor checks it
# against what the model was trained with and refuses to load on mismatch.
FEATURE_VERSION = 2

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def normalise_landmarks(points):
    """
    Raw MediaPipe landmarks → pose-only features.

    Accepts a single (21, 3) hand or a batch (N, 21, 3), and returns the
    same shape. The wrist becomes the origin and the hand is scaled to a
    consistent size, so the same sign gives the same numbers whether it
    fills the frame or sits small in the corner of a webcam shot.
    """

    points = np.asarray(points, dtype=np.float32)

    single = points.ndim == 2

    if single:
        points = points[None, ...]

    # Wrist (landmark 0) to the origin.
    centred = points - points[:, :1, :]

    # Scale by the hand's own spread. The mean distance is used rather
    # than the maximum so one badly placed fingertip cannot resize
    # everything else.
    size = np.linalg.norm(centred, axis=2).mean(axis=1)
    size = np.maximum(size, 1e-6)

    scaled = centred / size[:, None, None]

    return scaled[0] if single else scaled


def flatten(points):
    """(…, 21, 3) → (…, 63), the shape the network expects."""

    points = np.asarray(points, dtype=np.float32)

    return points.reshape(-1, NUM_FEATURES) if points.ndim == 3 else points.reshape(1, -1)


# ==========================================================
# AUGMENTATION — training only
# ==========================================================

def mirror_landmarks(points):
    """
    Flip left/right.

    A sign means the same letter whichever hand makes it, so mirroring is
    a free doubling of the training data and is what makes the model work
    for left- and right-handed signers alike. It also removes any need to
    detect handedness at inference time.
    """

    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 0] *= -1.0

    return out


def rotate_landmarks(points, degrees):
    """
    Rotate about the wrist in the image plane.

    Used to teach tolerance of a tilted wrist. Keep the angles modest:
    large rotations would blur the K/P and G/Q distinctions, which are
    genuinely orientation-based.
    """

    points = np.asarray(points, dtype=np.float32)

    single = points.ndim == 2

    if single:
        points = points[None, ...]

    degrees = np.broadcast_to(np.asarray(degrees, dtype=np.float32), (len(points),))
    radians = np.radians(degrees)

    cos = np.cos(radians)[:, None]
    sin = np.sin(radians)[:, None]

    origin = points[:, :1, :]
    centred = points - origin

    x = centred[:, :, 0].copy()
    y = centred[:, :, 1].copy()

    rotated = centred.copy()
    rotated[:, :, 0] = x * cos - y * sin
    rotated[:, :, 1] = x * sin + y * cos

    out = rotated + origin

    return out[0] if single else out
