import os
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist

app = FastAPI(title="ASL Temporal Transformer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "arabic_sign_video_model_large.onnx")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json")

ort_session = None
labels = []

@app.on_event("startup")
def load_resources():
    global ort_session, labels
    if not os.path.exists(MODEL_PATH):
        alt_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "arabic_sign_video_model_large.onnx")
        if os.path.exists(alt_model):
            MODEL_PATH = alt_model
        else:
            raise RuntimeError(f"Model file not found at {MODEL_PATH} or {alt_model}")
            
    print(f"Loading model from {MODEL_PATH}")
    ort_session = ort.InferenceSession(MODEL_PATH)
    
    if not os.path.exists(LABELS_PATH):
        alt_labels = os.path.join(os.path.dirname(os.path.dirname(__file__)), "labels.json")
        if os.path.exists(alt_labels):
            LABELS_PATH = alt_labels
        else:
            raise RuntimeError(f"Labels file not found at {LABELS_PATH} or {alt_labels}")
            
    print(f"Loading labels from {LABELS_PATH}")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

class PredictionRequest(BaseModel):
    frames: conlist(conlist(float, min_length=126, max_length=126), min_length=30, max_length=30)

@app.get("/")
def read_root():
    return {"status": "healthy", "model": "ASL Temporal Transformer"}

@app.post("/predict")
def predict(req: PredictionRequest):
    if ort_session is None or not labels:
        raise HTTPException(status_code=503, detail="Model or labels not loaded yet.")
    try:
        input_data = np.array(req.frames, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_name = ort_session.get_inputs()[0].name
        raw_output = ort_session.run(None, {input_name: input_data})
        probs = raw_output[0][0] 
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_label = labels[top_idx] if top_idx < len(labels) else "Unknown"
        return {
            "prediction": top_label,
            "confidence": top_prob,
            "class_index": top_idx
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
