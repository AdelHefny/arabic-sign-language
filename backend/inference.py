"""
backend/inference.py
====================
ONNX-based inference engine for both static and dynamic gesture recognition.

Architecture
------------
- StaticInferenceEngine  : single-frame hand-landmark classifier
- DynamicInferenceEngine : temporal transformer over a sliding window
- GestureRouter          : selects the right engine per frame and smooths results
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labels(path: Path) -> Dict[int, str]:
    """Load integer-to-label mapping from labels.json."""
    if not path.exists():
        logger.warning("labels.json not found at %s — using empty map", path)
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    # Support both {str_idx: label} and {label: str_idx}
    if raw and isinstance(next(iter(raw.keys())), str) and next(iter(raw.keys())).isdigit():
        return {int(k): v for k, v in raw.items()}
    # Inverted: {label: idx} → flip
    return {int(v): k for k, v in raw.items()}


def _build_session(model_path: Path) -> Optional[ort.InferenceSession]:
    """Create an ONNX Runtime InferenceSession with sensible defaults."""
    if not model_path.exists():
        logger.warning("Model not found: %s", model_path)
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    logger.info("Loaded model: %s (inputs: %s)", model_path.name,
                [i.name for i in session.get_inputs()])
    return session


# ---------------------------------------------------------------------------
# Static engine
# ---------------------------------------------------------------------------

class StaticInferenceEngine:
    """
    Recognises single-frame signs using the medium ONNX model.

    Even though this is called 'static', the model expects shape
    (1, 30, 126) — so we maintain a short ring buffer and replicate
    the latest frame to fill it until the buffer is populated.

    Output : (label: str, confidence: float)
    """

    def __init__(self, model_path: Path, labels: Dict[int, str]) -> None:
        self.session = _build_session(model_path)
        self.labels  = labels
        self._input_name: Optional[str] = (
            self.session.get_inputs()[0].name if self.session else None
        )
        self._window: deque[np.ndarray] = deque(maxlen=config.SEQUENCE_LENGTH)

    @property
    def ready(self) -> bool:
        return self.session is not None

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Parameters
        ----------
        landmarks : shape (126,) — raw landmark coords.

        Returns
        -------
        (label, confidence)
        """
        if not self.ready:
            return "", 0.0

        self._window.append(landmarks.astype(np.float32))

        # Pad by repeating the latest frame until window is full
        buf = list(self._window)
        while len(buf) < config.SEQUENCE_LENGTH:
            buf.insert(0, buf[0])

        seq = np.stack(buf, axis=0)[np.newaxis, ...]   # (1, 30, 126)
        outputs = self.session.run(None, {self._input_name: seq})

        logits = outputs[0][0]          # shape: (num_classes,)
        probs  = _softmax(logits)
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        label  = self.labels.get(idx, f"class_{idx}")
        return label, conf

    def reset(self) -> None:
        self._window.clear()


# ---------------------------------------------------------------------------
# Dynamic engine
# ---------------------------------------------------------------------------

class DynamicInferenceEngine:
    """
    Recognises continuous / dynamic gestures using a temporal transformer.

    Maintains an internal sliding window of SEQUENCE_LENGTH frames.

    Input  : landmark vector for ONE frame — engine buffers internally.
    Output : (label: str, confidence: float) once window is full.
    """

    def __init__(self, model_path: Path, labels: Dict[int, str]) -> None:
        self.session = _build_session(model_path)
        self.labels  = labels
        self._input_name: Optional[str] = (
            self.session.get_inputs()[0].name if self.session else None
        )
        self._window: deque[np.ndarray] = deque(maxlen=config.SEQUENCE_LENGTH)

    @property
    def ready(self) -> bool:
        return self.session is not None

    @property
    def window_full(self) -> bool:
        return len(self._window) == config.SEQUENCE_LENGTH

    def push_frame(self, landmarks: np.ndarray) -> None:
        """Add one frame of landmarks to the sliding window."""
        self._window.append(landmarks.astype(np.float32))

    def predict(self) -> Tuple[str, float]:
        """
        Run inference on the current window.
        Returns ("", 0.0) if the window is not yet full.
        """
        if not self.ready or not self.window_full:
            return "", 0.0

        seq = np.stack(list(self._window), axis=0)  # (T, D)
        x   = seq[np.newaxis, ...]                   # (1, T, D)
        outputs = self.session.run(None, {self._input_name: x})

        logits = outputs[0][0]
        probs  = _softmax(logits)
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        label  = self.labels.get(idx, f"class_{idx}")
        return label, conf

    def reset(self) -> None:
        self._window.clear()


# ---------------------------------------------------------------------------
# Gesture router
# ---------------------------------------------------------------------------

class GestureRouter:
    """
    Top-level inference controller.

    Per-frame workflow
    ------------------
    1. Compute motion magnitude between consecutive frames.
    2. If motion > MOTION_THRESHOLD → route to DynamicInferenceEngine.
    3. Else → route to StaticInferenceEngine.
    4. Apply exponential smoothing on confidence scores.
    5. Commit prediction only when streak count is reached.
    """

    def __init__(self) -> None:
        labels = _load_labels(config.LABELS_FILE)

        self.static_engine  = StaticInferenceEngine(config.STATIC_MODEL_PATH,  labels)
        self.dynamic_engine = DynamicInferenceEngine(config.DYNAMIC_MODEL_PATH, labels)

        self._prev_landmarks: Optional[np.ndarray] = None
        self._smoothed_conf: float  = 0.0
        self._last_label: str       = ""
        self._streak: int           = 0
        self._committed_label: str  = ""

        # Latency tracking
        self._inference_times: deque[float] = deque(maxlen=100)

    # ------------------------------------------------------------------
    def process_frame(self, landmarks: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        landmarks : 1-D float32 array of hand landmarks for the current frame.

        Returns
        -------
        dict with keys:
            raw_label, raw_conf, committed_label,
            mode, motion, avg_inference_ms
        """
        t0 = time.perf_counter()

        motion = self._compute_motion(landmarks)
        self._prev_landmarks = landmarks.copy()

        # Route
        if motion > config.MOTION_THRESHOLD:
            self.dynamic_engine.push_frame(landmarks)
            label, conf = self.dynamic_engine.predict()
            mode = "dynamic"
        else:
            label, conf = self.static_engine.predict(landmarks)
            mode = "static"

        # Smoothing
        if label:
            self._smoothed_conf = (
                config.SMOOTHING_ALPHA * conf
                + (1 - config.SMOOTHING_ALPHA) * self._smoothed_conf
            )
        else:
            self._smoothed_conf *= (1 - config.SMOOTHING_ALPHA)

        # Streak / commit logic
        threshold = (config.STATIC_CONFIDENCE_THRESHOLD
                     if mode == "static"
                     else config.DYNAMIC_CONFIDENCE_THRESHOLD)

        if label == self._last_label and self._smoothed_conf >= threshold:
            self._streak += 1
        else:
            self._streak = 0
            self._last_label = label

        committed = ""
        if self._streak >= config.COMMIT_STREAK and label:
            committed = label
            self._committed_label = label
            self._streak = 0          # reset after commit

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._inference_times.append(elapsed_ms)
        avg_ms = float(np.mean(self._inference_times))

        return {
            "raw_label":       label,
            "raw_conf":        round(float(self._smoothed_conf), 4),
            "committed_label": committed,
            "mode":            mode,
            "motion":          round(float(motion), 5),
            "avg_inference_ms": round(avg_ms, 2),
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state (e.g., between participants)."""
        self._prev_landmarks  = None
        self._smoothed_conf   = 0.0
        self._last_label      = ""
        self._streak          = 0
        self._committed_label = ""
        self.static_engine.reset()
        self.dynamic_engine.reset()

    # ------------------------------------------------------------------
    def _compute_motion(self, landmarks: np.ndarray) -> float:
        if self._prev_landmarks is None:
            return 0.0
        diff = landmarks - self._prev_landmarks
        return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
