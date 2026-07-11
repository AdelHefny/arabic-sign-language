"""
backend/config.py
=================
Central configuration for the Arabic Sign Language backend.
All paths, thresholds, and runtime settings live here.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"

# ONNX model paths (relative to project root)
STATIC_MODEL_PATH  = MODELS_DIR / "arabic_sign_model_medium.onnx"
DYNAMIC_MODEL_PATH = MODELS_DIR / "arabic_sign_video_model_large.onnx"
DYNAMIC_MODEL_SMALL_PATH = MODELS_DIR / "arabic_sign_video_model_small.onnx"

LABELS_FILE = ROOT_DIR / "labels.json"

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

# Number of hand landmarks per frame (21 keypoints × 3 coords)
HAND_LANDMARK_DIM: int = 63          # 21 × 3
BOTH_HANDS_DIM: int    = 126         # left + right

# All three ONNX models use the same temporal input: (batch, 30, 126)
SEQUENCE_LENGTH: int = 30            # frames per gesture window

# Number of output classes (from ONNX model inspection)
NUM_CLASSES: int = 502

# Gesture type thresholds
# All models are treated as temporal (30-frame window)
# The 'static' engine uses the medium model with a short window
STATIC_CONFIDENCE_THRESHOLD:  float = 0.65
DYNAMIC_CONFIDENCE_THRESHOLD: float = 0.60

# Smoothing window (exponential moving average α)
SMOOTHING_ALPHA: float = 0.4

# How many consecutive identical predictions before committing
COMMIT_STREAK: int = 5

# ---------------------------------------------------------------------------
# Model selection heuristic
# ---------------------------------------------------------------------------
# If motion magnitude between consecutive frames exceeds this, use dynamic
MOTION_THRESHOLD: float = 0.015

# ---------------------------------------------------------------------------
# Backend server
# ---------------------------------------------------------------------------

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
WORKERS: int = int(os.getenv("WORKERS", "1"))

# CORS origins allowed for the Chrome Extension
ALLOWED_ORIGINS: list[str] = [
    "chrome-extension://*",
    "http://localhost:*",
    "https://meet.google.com",
]

# ---------------------------------------------------------------------------
# NLP settings
# ---------------------------------------------------------------------------

# Rule-based correction is always enabled
USE_ARABERT: bool = os.getenv("USE_ARABERT", "false").lower() == "true"
ARABERT_MODEL: str = "aubmindlab/bert-base-arabertv02"

# Max sentence word count before flushing
MAX_SENTENCE_WORDS: int = 20

# How long (seconds) of silence resets the sentence buffer
SILENCE_TIMEOUT_SECS: float = 3.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
