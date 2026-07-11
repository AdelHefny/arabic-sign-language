# Arabic Sign Language Translation System — Project Summary

## Goal

Build a production-ready, end-to-end Arabic Sign Language (ArSL) translation system that:

- Recognises **static signs** (alphabet, numbers) from single frames
- Recognises **dynamic / continuous gestures** from temporal sequences
- Constructs **grammatically correct Arabic sentences** in real time
- Renders **live subtitles** inside Google Meet via a Chrome Extension

---

## Architecture

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                              │
│  Google Meet Video ──► Frame Capture ──► MediaPipe Hands        │
└────────────────────────────┬────────────────────────────────────┘
                             │  Landmark vector (21×3 per hand)
┌────────────────────────────▼────────────────────────────────────┐
│                     Motion Analysis                             │
│   motion < threshold → Static Engine                           │
│   motion ≥ threshold → Dynamic Engine (sliding window)         │
└────────────────────────────┬────────────────────────────────────┘
                             │  (label, confidence)
┌────────────────────────────▼────────────────────────────────────┐
│                Confidence Smoothing (EMA)                       │
│   Streak counter → commit after N consecutive identical labels  │
└────────────────────────────┬────────────────────────────────────┘
                             │  committed token
┌────────────────────────────▼────────────────────────────────────┐
│                    NLP Post-Processing                          │
│   SentenceBuffer → rule-based correction → punctuation         │
│   Optional: AraBERT grammar refinement                         │
└────────────────────────────┬────────────────────────────────────┘
                             │  Arabic sentence
┌────────────────────────────▼────────────────────────────────────┐
│               Chrome Extension Overlay                          │
│   RTL subtitle div ──► Google Meet participant video tile       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Models

### Static Classifier (`arabic_sign_model_medium.onnx`)

- **Input**: `(1, D)` where D = 63 (21 landmarks × 3 coords) or 126 (both hands)
- **Backbone**: MobileNetV3 / EfficientNet-B0 (configurable in notebook 02)
- **Training techniques**: Transfer learning, label smoothing, cosine annealing, EMA, early stopping, mixed precision

### Dynamic Transformer (`arabic_sign_video_model_large.onnx`)

- **Input**: `(1, T, D)` where T = 30 frames, D = landmark dimension
- **Architecture**: Temporal Transformer with positional encoding
- **Features**: motion, speed, hand trajectory, temporal relationships
- **Training techniques**: Progressive layer unfreezing, gradient clipping, warmup scheduler, cosine annealing, EMA

---

## Data Augmentation

Applied during training on landmark sequences:

| Technique | Description |
|---|---|
| Gaussian Noise | Random noise on coordinates |
| Coordinate Jitter | Small random perturbations |
| Rotation | 2D rotation of hand landmark cloud |
| Scaling | Uniform scale variation ±20% |
| Translation | Random shift of landmark centroid |
| Temporal Stretching | Resample sequence at variable speed |
| Speed Variation | Slow / fast playback simulation |
| Random Frame Drop | Remove up to 20% of frames |
| Landmark Interpolation | Fill missing landmarks smoothly |
| Sequence Shift | Temporal start-point randomisation |

---

## Evaluation

### Classification Metrics

- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- Top-5 Accuracy

### Translation Metrics

- BLEU-1 to BLEU-4
- ROUGE-L
- Character Error Rate (CER)
- Word Error Rate (WER)

### Performance Metrics

- Inference FPS (target ≥ 15 FPS on CPU)
- End-to-end latency (target < 100 ms)
- Memory footprint
- CPU / GPU utilisation

---

## Deployment Topology

```
┌───────────────┐     WebSocket / REST     ┌──────────────────────┐
│ Chrome Ext.   │ ◄──────────────────────► │  FastAPI Backend     │
│ (Google Meet) │                          │  :8000               │
└───────────────┘                          │  ONNX Runtime (CPU)  │
                                           └──────────────────────┘
                                                    │
                                           ┌────────┴──────────┐
                                           │  models/  *.onnx  │
                                           └───────────────────┘
```

- **Training**: GPU (CUDA) via Jupyter notebooks
- **Inference**: CPU-only (ONNX Runtime, no GPU required at runtime)
- **Containerisation**: Docker + docker-compose

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ONNX Runtime inference | Platform-agnostic, CPU-compatible, no PyTorch at runtime |
| MediaPipe in the Extension | Offloads landmark extraction to the client; backend receives compact vectors |
| WebSocket transport | Low latency, full-duplex; ideal for real-time subtitle streaming |
| Per-client GestureRouter | Supports multiple simultaneous participants in one Meet call |
| EMA smoothing + streak commit | Reduces flickering from single-frame noise |
| Rule-based NLP first | Zero external dependency; AraBERT is an optional enhancement |
| Multi-stage Docker build | Lean runtime image (no build tools or training libs) |

---

## Notebooks Overview

| # | Notebook | GPU | Duration (est.) |
|---|---|---|---|
| 01 | Dataset Preparation | Optional | 1–4 h |
| 02 | Static Model Training | **Required** | 2–6 h |
| 03 | Dynamic Model Training | **Required** | 4–12 h |
| 04 | Model Optimisation | Optional | 30–60 min |
