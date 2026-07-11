"""
backend/app.py
==============
Main FastAPI application.

Endpoints
---------
REST
  POST /predict          — single-frame static inference
  POST /predict/sequence — multi-frame dynamic inference
  GET  /health           — liveness probe
  GET  /metrics          — inference statistics

WebSocket
  WS /ws/{client_id}    — real-time streaming inference

All responses include CORS headers compatible with the Chrome Extension.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config
from .inference import GestureRouter
from .nlp import SentenceBuffer, format_for_subtitle

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Arabic Sign Language Translation API",
    description=(
        "Real-time Arabic Sign Language (ArSL) translation backend. "
        "Accepts MediaPipe hand landmarks and returns Arabic text predictions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Per-client state (in-memory; replace with Redis for multi-process deploy)
# ---------------------------------------------------------------------------
_routers: Dict[str, GestureRouter]     = {}
_buffers: Dict[str, SentenceBuffer]    = {}
_sentences: Dict[str, List[str]]       = {}


def _get_router(client_id: str) -> GestureRouter:
    if client_id not in _routers:
        _routers[client_id]   = GestureRouter()
        _buffers[client_id]   = SentenceBuffer()
        _sentences[client_id] = []
    return _routers[client_id]


def _get_buffer(client_id: str) -> SentenceBuffer:
    _get_router(client_id)          # ensure initialised
    return _buffers[client_id]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class LandmarkFrame(BaseModel):
    """A single frame of hand landmarks."""
    client_id:  str            = Field(default_factory=lambda: str(uuid.uuid4()))
    landmarks:  List[float]    = Field(..., description="Flat list of landmark coordinates (x,y,z per point)")
    timestamp:  float          = Field(0.0, description="Client-side timestamp (ms)")


class LandmarkSequence(BaseModel):
    """A pre-built temporal sequence (T × D)."""
    client_id:  str
    sequence:   List[List[float]]
    timestamp:  float = 0.0


class ResetRequest(BaseModel):
    client_id: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "1.0.0"})


@app.get("/metrics", tags=["System"])
async def metrics() -> JSONResponse:
    data: Dict[str, object] = {}
    for cid, router in _routers.items():
        data[cid] = {
            "avg_inference_ms": round(
                float(np.mean(list(router._inference_times))) if router._inference_times else 0, 2
            ),
            "pending_words": _buffers[cid].pending_words,
            "committed_sentences": len(_sentences.get(cid, [])),
        }
    return JSONResponse(data)


@app.post("/predict", tags=["Inference"])
async def predict_frame(frame: LandmarkFrame) -> JSONResponse:
    """
    Single-frame inference.
    Routes to static or dynamic engine based on motion magnitude.
    """
    landmarks = np.array(frame.landmarks, dtype=np.float32)
    router    = _get_router(frame.client_id)
    result    = router.process_frame(landmarks)

    # Feed committed label into sentence buffer
    sentence_completed = None
    if result["committed_label"]:
        buf = _get_buffer(frame.client_id)
        sentence_completed = buf.push(result["committed_label"])
        if sentence_completed:
            _sentences.setdefault(frame.client_id, []).append(sentence_completed)

    result["sentence_completed"] = sentence_completed
    result["buffer_preview"]     = _get_buffer(frame.client_id).preview
    return JSONResponse(result)


@app.post("/predict/sequence", tags=["Inference"])
async def predict_sequence(seq: LandmarkSequence) -> JSONResponse:
    """
    Direct dynamic inference on a caller-provided temporal sequence.
    Bypasses the sliding window; useful for offline batch testing.
    """
    router = _get_router(seq.client_id)
    arr    = np.array(seq.sequence, dtype=np.float32)   # (T, D)
    for frame in arr:
        router.dynamic_engine.push_frame(frame)
    label, conf = router.dynamic_engine.predict()
    return JSONResponse({"label": label, "confidence": round(conf, 4)})


@app.post("/reset", tags=["Session"])
async def reset_session(req: ResetRequest) -> JSONResponse:
    cid = req.client_id
    if cid in _routers:
        _routers[cid].reset()
        _buffers[cid]   = SentenceBuffer()
        _sentences[cid] = []
    return JSONResponse({"status": "reset", "client_id": cid})


@app.get("/sentences/{client_id}", tags=["Session"])
async def get_sentences(client_id: str) -> JSONResponse:
    sentences = _sentences.get(client_id, [])
    return JSONResponse({"client_id": client_id, "sentences": sentences})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    """
    Real-time streaming inference over WebSocket.

    Client sends JSON messages:
      { "landmarks": [x0,y0,z0, ...], "timestamp": 1234.5 }

    Server responds with JSON:
      { "raw_label": "...", "raw_conf": 0.9, "committed_label": "...",
        "sentence_completed": "...", "buffer_preview": "...",
        "mode": "static", "avg_inference_ms": 3.2 }
    """
    await websocket.accept()
    router = _get_router(client_id)
    buf    = _get_buffer(client_id)
    logger.info("WebSocket connected: %s", client_id)

    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)

            landmarks = np.array(msg["landmarks"], dtype=np.float32)
            result    = router.process_frame(landmarks)

            sentence_completed = None
            if result["committed_label"]:
                sentence_completed = buf.push(result["committed_label"])
                if sentence_completed:
                    _sentences.setdefault(client_id, []).append(sentence_completed)

            result["sentence_completed"] = sentence_completed
            result["buffer_preview"]     = buf.preview

            if sentence_completed:
                result["subtitle"] = format_for_subtitle(sentence_completed)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client_id)
    except Exception as exc:
        logger.error("WebSocket error for %s: %s", client_id, exc)
        await websocket.close(code=1011)


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        reload=False,
    )
