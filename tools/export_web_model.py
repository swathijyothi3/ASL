"""
Export the trained classifier so it can run in the browser.

    python tools/export_web_model.py

Writes output/web_model.json: the StandardScaler statistics and the dense
layers, as base64-encoded float32.

Why bother
----------
The live camera view runs MediaPipe in the browser and needs to turn the
landmarks it finds into a letter immediately. Sending every frame to the
server would need a video connection (which many networks block) and would
spend the app's single CPU on decoding video.

The network here is four dense layers — about 60k weights. Evaluating that
in JavaScript is a few lines of arithmetic, so the whole live view becomes
self-contained: no video leaves the browser, nothing to negotiate, and the
server does no per-frame work at all.
"""

import base64
import json
import os
import sys

import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.features import FEATURE_VERSION  # noqa: E402

OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def encode(array):
    """float32 array → base64, keeping the file compact and exact."""

    return base64.b64encode(
        np.ascontiguousarray(array, dtype=np.float32).tobytes()
    ).decode("ascii")


def main():
    from tensorflow.keras.models import load_model

    model = load_model(os.path.join(OUTPUT_DIR, "asl_ann_model.keras"))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    layers = []

    for layer in model.layers:
        weights = layer.get_weights()

        if not weights:
            continue  # dropout and friends have nothing to carry

        kernel, bias = weights

        activation = getattr(layer, "activation", None)
        name = getattr(activation, "__name__", "linear")

        layers.append({
            "shape": list(kernel.shape),
            "kernel": encode(kernel),
            "bias": encode(bias),
            "activation": name,
        })

    payload = {
        "feature_version": FEATURE_VERSION,
        "classes": list(encoder.classes_),
        "scaler_mean": encode(scaler.mean_),
        "scaler_scale": encode(scaler.scale_),
        "layers": layers,
    }

    path = os.path.join(OUTPUT_DIR, "web_model.json")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")

    size = os.path.getsize(path) / 1024

    print(f"wrote {path}  ({size:,.0f} KB)")
    print(f"  feature version {FEATURE_VERSION}")
    print(f"  {len(layers)} weighted layers, "
          f"{sum(np.prod(l['shape']) for l in layers):,} weights")
    print(f"  {len(payload['classes'])} classes")

    # ------------------------------------------------------
    # verify the exported copy agrees with Keras
    # ------------------------------------------------------

    def decode(text, shape=None):
        flat = np.frombuffer(base64.b64decode(text), dtype=np.float32)
        return flat.reshape(shape) if shape else flat

    mean = decode(payload["scaler_mean"])
    scale = decode(payload["scaler_scale"])

    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, (5, 63)).astype(np.float32)

    activations = (sample - mean) / scale

    for spec in layers:
        kernel = decode(spec["kernel"], spec["shape"])
        bias = decode(spec["bias"])

        activations = activations @ kernel + bias

        if spec["activation"] == "relu":
            activations = np.maximum(activations, 0.0)
        elif spec["activation"] == "softmax":
            shifted = activations - activations.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            activations = exp / exp.sum(axis=1, keepdims=True)

    reference = np.asarray(
        model(scaler.transform(sample).astype(np.float32), training=False)
    )

    difference = float(np.abs(activations - reference).max())

    print(f"  max disagreement with Keras: {difference:.2e}")

    if difference > 1e-4:
        print("  MISMATCH — the exported model does not reproduce Keras")
        return 1

    print("  exported model reproduces Keras")
    return 0


if __name__ == "__main__":
    sys.exit(main())
