"""
Live camera view that runs entirely in the browser.

MediaPipe has a JavaScript build, and the classifier is four dense layers —
about 60k weights — which is a few lines of arithmetic. So instead of
streaming video to the server, the whole thing runs on the viewer's
machine: camera, hand detection, and the letter.

Why not streamlit-webrtc
------------------------
The obvious way to do a live view in Streamlit is streamlit-webrtc, and it
was tried first. Two problems. It needs a WebRTC connection, which plenty
of office, campus and mobile networks block outright and which Community
Cloud cannot relay. More seriously, it pulls in PyAV, whose bundled ffmpeg
libraries clash with the ones inside OpenCV — the deployed app stopped
starting at all.

Doing it in the browser removes both problems, costs the server nothing
per frame, and runs at full frame rate rather than the handful of frames a
shared CPU could manage. The trade is that it needs the MediaPipe runtime
from a CDN; if that is blocked the view says so and the photo modes still
work.

The maths here must match utils/features.py exactly. Both centre the hand
on its wrist and divide by mean distance from it.
"""

import json
import os

import streamlit as st
import streamlit.components.v1 as components


# Pinned rather than floating, so a CDN-side release cannot change the
# behaviour of a deployed app without anyone touching it.
TASKS_VISION_VERSION = "0.10.22-rc.20250304"

WASM_BASE = (
    f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VISION_VERSION}/wasm"
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@st.cache_data(show_spinner=False)
def _web_model(path):
    """The exported weights, as a JSON string ready to inline."""

    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


_TEMPLATE = r"""
<div id="wrap">
  <div id="stage">
    <video id="cam" playsinline muted></video>
    <canvas id="view"></canvas>
    <div id="readout"><span id="letter">–</span><span id="note">starting the camera…</span></div>
  </div>
  <div id="bar">
    <button id="toggle">Stop camera</button>
    <span id="status">loading the hand detector…</span>
  </div>
</div>

<style>
  * { box-sizing: border-box; }
  #wrap { font-family: "Source Sans Pro", system-ui, sans-serif; color: #e2e8f0; }
  #stage {
    position: relative; width: 100%; border-radius: 16px; overflow: hidden;
    background: #0b1120; border: 1px solid rgba(148,163,184,0.25); line-height: 0;
  }
  #cam { display: none; }
  #view { width: 100%; height: auto; display: block; }
  #readout {
    position: absolute; left: 0; right: 0; bottom: 0;
    display: flex; align-items: baseline; gap: 18px;
    padding: 14px 20px; background: linear-gradient(transparent, rgba(2,6,23,0.92));
  }
  #letter { font-size: 3.2rem; font-weight: 800; line-height: 1; color: #64748b; }
  #letter.on { color: #4ade80; text-shadow: 0 0 26px rgba(74,222,128,0.45); }
  #note { font-size: 0.95rem; color: #cbd5e1; }
  #bar { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
  button {
    background: #1e293b; color: #e2e8f0; border: 1px solid rgba(148,163,184,0.35);
    border-radius: 8px; padding: 7px 16px; font-size: 0.9rem; cursor: pointer;
  }
  button:hover { border-color: #38bdf8; color: #38bdf8; }
  #status { font-size: 0.85rem; color: #94a3b8; }
  .bad { color: #fca5a5 !important; }
</style>

<script type="module">
const CONFIG = __CONFIG__;
const MODEL  = __MODEL__;

let backend = "?";

const video   = document.getElementById("cam");
const canvas  = document.getElementById("view");
const ctx     = canvas.getContext("2d");
const letterEl= document.getElementById("letter");
const noteEl  = document.getElementById("note");
const statusEl= document.getElementById("status");
const toggle  = document.getElementById("toggle");

function fail(message) {
  statusEl.textContent = message;
  statusEl.classList.add("bad");
}

/* ---------- the classifier, straight from output/web_model.json ---------- */

function bytes(b64) {
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return new Float32Array(out.buffer);
}

let net = null;

if (MODEL) {
  net = {
    classes: MODEL.classes,
    mean: bytes(MODEL.scaler_mean),
    scale: bytes(MODEL.scaler_scale),
    layers: MODEL.layers.map(l => ({
      shape: l.shape,
      kernel: bytes(l.kernel),
      bias: bytes(l.bias),
      activation: l.activation,
    })),
  };
}

/* Must mirror utils/features.py: centre on the wrist, divide by the mean
   distance from it. A change in one without the other is silent nonsense. */
function poseFeatures(points) {
  const wrist = points[0];
  const centred = points.map(p => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);

  let total = 0;
  for (const p of centred) total += Math.hypot(p[0], p[1], p[2]);

  const size = Math.max(total / centred.length, 1e-6);

  const out = new Float32Array(63);
  centred.forEach((p, i) => {
    out[i * 3]     = p[0] / size;
    out[i * 3 + 1] = p[1] / size;
    out[i * 3 + 2] = p[2] / size;
  });
  return out;
}

function classify(points) {
  if (!net) return null;

  let a = poseFeatures(points);
  for (let i = 0; i < a.length; i++) a[i] = (a[i] - net.mean[i]) / net.scale[i];

  for (const layer of net.layers) {
    const [inDim, outDim] = layer.shape;
    const next = new Float32Array(outDim);

    for (let j = 0; j < outDim; j++) next[j] = layer.bias[j];

    for (let i = 0; i < inDim; i++) {
      const v = a[i];
      if (v === 0) continue;
      const row = i * outDim;
      for (let j = 0; j < outDim; j++) next[j] += v * layer.kernel[row + j];
    }

    if (layer.activation === "relu") {
      for (let j = 0; j < outDim; j++) if (next[j] < 0) next[j] = 0;
    } else if (layer.activation === "softmax") {
      let max = -Infinity;
      for (const v of next) if (v > max) max = v;
      let sum = 0;
      for (let j = 0; j < outDim; j++) { next[j] = Math.exp(next[j] - max); sum += next[j]; }
      for (let j = 0; j < outDim; j++) next[j] /= sum;
    }
    a = next;
  }

  let best = 0;
  for (let j = 1; j < a.length; j++) if (a[j] > a[best]) best = j;

  return { letter: net.classes[best], confidence: a[best] * 100 };
}

/* Exposed deliberately. This is the one part of the pipeline that is a
   hand transcription rather than shared code, so it needs to be checkable
   against Python from a browser console:

     __aslClassify(landmarks)   // [{x, y, z} × 21] -> {letter, confidence}

   tools/export_web_model.py verifies the exported weights reproduce Keras;
   this hook is how the JavaScript arithmetic around them gets verified. */
window.__aslClassify = classify;

/* ---------- drawing ---------- */

const FINGERS = {
  palm:   [[0,5],[5,9],[9,13],[13,17],[0,17]],
  thumb:  [[0,1],[1,2],[2,3],[3,4]],
  index:  [[5,6],[6,7],[7,8]],
  middle: [[9,10],[10,11],[11,12]],
  ring:   [[13,14],[14,15],[15,16]],
  pinky:  [[17,18],[18,19],[19,20]],
};

const COLOURS = {
  palm: "#cbd5e1", thumb: "#ffb454", index: "#4ade80",
  middle: "#5ac8fa", ring: "#e879f9", pinky: "#a5b4fc",
};

/* Corner brackets rather than a closed box: it marks the region the model
   reads without drawing a line across the hand itself. */
function brackets(pts, w, h, settled) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of pts) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }

  const pad = 0.055;
  const x0 = (minX - pad) * w, y0 = (minY - pad) * h;
  const x1 = (maxX + pad) * w, y1 = (maxY + pad) * h;
  const len = Math.max(18, Math.min(x1 - x0, y1 - y0) * 0.22);

  ctx.strokeStyle = settled ? "#4ade80" : "#5ac8fa";
  ctx.lineWidth = Math.max(2, w / 320);
  ctx.lineCap = "round";

  const corners = [[x0,y0,1,1],[x1,y0,-1,1],[x0,y1,1,-1],[x1,y1,-1,-1]];
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(cx + dx * len, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * len);
    ctx.stroke();
  }
}

function skeleton(pts, w, h) {
  ctx.lineWidth = Math.max(2, w / 260);
  ctx.lineCap = "round";

  for (const [finger, bones] of Object.entries(FINGERS)) {
    ctx.strokeStyle = COLOURS[finger];
    ctx.beginPath();
    for (const [a, b] of bones) {
      ctx.moveTo(pts[a].x * w, pts[a].y * h);
      ctx.lineTo(pts[b].x * w, pts[b].y * h);
    }
    ctx.stroke();
  }

  const r = Math.max(3, w / 200);
  for (let i = 0; i < pts.length; i++) {
    ctx.beginPath();
    ctx.arc(pts[i].x * w, pts[i].y * h, i === 0 ? r + 2 : r, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff"; ctx.fill();
    ctx.beginPath();
    ctx.arc(pts[i].x * w, pts[i].y * h, (i === 0 ? r + 2 : r) - 2, 0, Math.PI * 2);
    ctx.fillStyle = "#1e293b"; ctx.fill();
  }
}

/* ---------- settling ---------- */

const recent = [];
let shown = null, shownConfidence = 0;

function settle(letter, confidence) {
  recent.push(letter);
  if (recent.length > CONFIG.window) recent.shift();

  const counts = {};
  for (const l of recent) counts[l] = (counts[l] || 0) + 1;

  let best = null, bestCount = 0;
  for (const [l, c] of Object.entries(counts)) if (c > bestCount) { best = l; bestCount = c; }

  if (best === "null" || best === null) { shown = null; shownConfidence = 0; return; }

  if (bestCount >= CONFIG.window * CONFIG.agreement) {
    shown = best;
    if (best === letter) shownConfidence = confidence;
  }
}

function readout() {
  if (!shown) {
    letterEl.textContent = "–";
    letterEl.classList.remove("on");
    noteEl.textContent = recent.some(Boolean)
      ? "hold the sign steady"
      : "show your hand to the camera";
    return;
  }
  const settled = shownConfidence >= CONFIG.threshold;
  letterEl.textContent = shown;
  letterEl.classList.toggle("on", settled);
  noteEl.textContent = settled
    ? `${shownConfidence.toFixed(0)}% confident`
    : "hold the sign steady";
}

/* ---------- main loop ---------- */

let landmarker = null, stream = null, running = false;

/* Detection runs on a small offscreen copy rather than the camera frame.
   Cost scales with pixels, and laptop webcams hand back 720p or more where
   a phone often gives far less — which is why this view could run smoothly
   on a phone and stutter on a laptop. A hand is perfectly findable at this
   size, and landmarks come back normalised so they map to any canvas. */
const detect = document.createElement("canvas");
const dctx = detect.getContext("2d", { willReadFrequently: true });

async function setup() {
  let vision;
  try {
    vision = await import(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${CONFIG.version}`
    );
  } catch (e) {
    fail("Could not load the hand detector. Use “Take a photo” instead.");
    return;
  }

  const files = await vision.FilesetResolver.forVisionTasks(CONFIG.wasm).catch(() => null);

  if (!files) {
    fail("Could not load the hand detector. Use “Take a photo” instead.");
    return;
  }

  const options = (delegate) => ({
    baseOptions: { modelAssetPath: CONFIG.handModel, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: CONFIG.detection,
    minHandPresenceConfidence: CONFIG.detection,
    minTrackingConfidence: CONFIG.detection,
  });

  // GPU is much faster where it works, but some laptop drivers and
  // browsers fail or fall into a slow path, so CPU is a real fallback
  // rather than a formality.
  try {
    landmarker = await vision.HandLandmarker.createFromOptions(files, options("GPU"));
    backend = "GPU";
  } catch (e) {
    try {
      landmarker = await vision.HandLandmarker.createFromOptions(files, options("CPU"));
      backend = "CPU";
    } catch (e2) {
      fail("The hand detector could not start: " + e2.message);
      return;
    }
  }

  statusEl.textContent = net
    ? "detector ready — allow camera access"
    : "detector ready, but the classifier is missing";

  await start();
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: CONFIG.captureWidth },
        height: { ideal: Math.round(CONFIG.captureWidth * 3 / 4) },
        frameRate: { ideal: 30, max: 30 },
        facingMode: "user",
      },
      audio: false,
    });
  } catch (e) {
    fail("Camera access was blocked. Allow it in your browser, then reload.");
    return;
  }

  video.srcObject = stream;
  await video.play();

  // Cap what is drawn. A 1080p webcam gains nothing here and costs fill
  // rate on every frame.
  const ratio = video.videoHeight / video.videoWidth;
  const width = Math.min(video.videoWidth, CONFIG.displayWidth);

  canvas.width = Math.round(width);
  canvas.height = Math.round(width * ratio);

  detect.width = CONFIG.detectWidth;
  detect.height = Math.round(CONFIG.detectWidth * ratio);

  running = true;
  toggle.textContent = "Stop camera";
  requestAnimationFrame(loop);
}

function stop() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  held = null;
  toggle.textContent = "Start camera";
  statusEl.textContent = "camera stopped";
}

toggle.addEventListener("click", () => (running ? stop() : start()));

/* The video is redrawn every frame so it stays smooth, but detection runs
   on a timer well below the display rate. Detecting on every frame is what
   made this stutter: it is by far the most expensive step, and running it
   more often than a hand actually moves buys nothing. */

let lastDetect = 0, held = null, lastTime = -1;
let frames = 0, fpsSince = 0, detections = 0;

function loop(now) {
  if (!running) return;

  const w = canvas.width, h = canvas.height;

  // Mirrored, so the preview behaves like a mirror.
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  const due = now - lastDetect >= 1000 / CONFIG.detectFps;

  if (landmarker && due && video.currentTime !== lastTime) {
    lastDetect = now;
    lastTime = video.currentTime;

    dctx.drawImage(video, 0, 0, detect.width, detect.height);

    let result = null;
    try {
      result = landmarker.detectForVideo(detect, now);
      detections++;
    } catch (e) { /* a dropped frame must not kill the loop */ }

    if (result && result.landmarks && result.landmarks.length) {
      const raw = result.landmarks[0];

      // Classify on the true landmarks; draw on mirrored ones so the
      // overlay sits on the hand as the viewer sees it.
      const guess = classify(raw);
      settle(guess ? guess.letter : null, guess ? guess.confidence : 0);

      held = raw.map(p => ({ x: 1 - p.x, y: p.y, z: p.z }));
    } else {
      settle(null, 0);
      held = null;
    }
    readout();
  }

  // Drawn from the last detection, so the outline stays on the hand
  // between detections instead of blinking.
  if (held) {
    brackets(held, w, h, shownConfidence >= CONFIG.threshold);
    skeleton(held, w, h);
  }

  frames++;
  if (now - fpsSince >= 1000) {
    const fps = Math.round(frames * 1000 / (now - fpsSince));
    const dps = Math.round(detections * 1000 / (now - fpsSince));
    statusEl.textContent =
      `running in your browser · ${fps} fps, ${dps} readings/s · ${backend}`;
    frames = 0; detections = 0; fpsSince = now;
  }

  requestAnimationFrame(loop);
}

setup();
</script>
"""


def browser_live_view(threshold=60.0, detection=0.3, height=620, model_path=None):
    """Render the in-browser live camera view."""

    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
            "web_model.json",
        )

    model_json = _web_model(model_path)

    config = {
        "version": TASKS_VISION_VERSION,
        "wasm": WASM_BASE,
        "handModel": HAND_MODEL_URL,
        "threshold": float(threshold),
        "detection": float(detection),
        "window": 8,
        "agreement": 0.5,

        # Performance. Detection cost scales with pixels, and a laptop
        # webcam hands back far more of them than a phone does — which is
        # why this had to stop running on the raw camera frame.
        #
        # 480 is not a guess: measured against the dataset, hands-found and
        # accuracy are flat from 1280 all the way down to 384 (MediaPipe
        # downscales internally regardless), and only start to slip at 320.
        # 480 keeps a margin for small hands at no measured cost.
        "captureWidth": 640,   # asked of the camera
        "displayWidth": 800,   # cap on what is drawn
        "detectWidth": 480,    # what the detector actually sees
        "detectFps": 15,       # detections per second, not per frame
    }

    html = (
        _TEMPLATE
        .replace("__CONFIG__", json.dumps(config))
        .replace("__MODEL__", model_json if model_json else "null")
    )

    components.html(html, height=height)

    return model_json is not None
