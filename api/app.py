from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_TITLE = "Diabetes Ridge Predictor"
MODEL_PATH = Path("models/model.joblib")

FEATURES_NAMES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]

class PredictRequest(BaseModel):
    features: Dict[str, float]

class PredictBatchRequest(BaseModel):
    records: List[Dict[str, float]]


app = FastAPI(title=APP_TITLE)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run 'dvc repro' to train first"
        )
    return joblib.load(MODEL_PATH)

MODEL = None

@app.on_event("startup")
def startup_event():
    global MODEL
    MODEL = load_model()

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None: 
        raise HTTPException(status_code=500, detail= "Model Not loaded")

    # Ensuring feature order and presence
    missing = [f for f in FEATURES_NAMES if f not in req.features]
    extra = [k for k in req.features.keys() if k not in FEATURES_NAMES]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected features: {extra}")

    X = pd.DataFrame([[req.features[f] for f in FEATURES_NAMES]], columns=FEATURES_NAMES)
    pred = float(MODEL.predict(X)[0])
    return {"prediction": pred}


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    for i, r in enumerate(req.records):
        missing = [f for f in FEATURES_NAMES if f not in r]
        extra = [k for k in r.keys() if k not in FEATURES_NAMES]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Record {i} missing features: {missing}"
            )
        if extra:
            raise HTTPException(
                status_code=400, detail=f"Record {i} unexpected features: {extra}"
            )

    X = pd.DataFrame([[r[f] for f in FEATURES_NAMES] for r in req.records], columns=FEATURES_NAMES)
    preds = MODEL.predict(X).astype(float).tolist()
    return {"predictions": preds}

@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(MODEL_PATH)}
