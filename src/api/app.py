"""
FastAPI serving surface for the Music Virality system.

Endpoints:
  GET  /health          — liveness check
  POST /predict/detection   — score a video for current virality
  POST /predict/prediction    — score a video for future virality potential
"""
import os
from datetime import datetime
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.inference.feature_builder import build_feature_vector

app = FastAPI(title="Music Virality API", version="1.0.0")


MODEL_PATHS = {
    "detection": ("models/trained/detection_model.pkl", "models/trained/detection_threshold.pkl"),
    "prediction": ("models/trained/prediction_model.pkl", "models/trained/prediction_threshold.pkl"),
}


def _load_artifacts(model_type: str):
    model_path, threshold_path = MODEL_PATHS[model_type]
    if not os.path.exists(model_path) or not os.path.exists(threshold_path):
        raise HTTPException(status_code=503, detail=f"Artifacts for {model_type} not found. Train the model first.")
    return joblib.load(model_path), joblib.load(threshold_path)


class PredictRequest(BaseModel):
    views: int = Field(..., ge=0, description="Current view count")
    likes: int = Field(..., ge=0, description="Current like count")
    comments: int = Field(..., ge=0, description="Current comment count")
    subscribers: int = Field(..., ge=0, description="Channel subscriber count")
    days_old: int = Field(..., ge=0, description="Days since publication")
    duration_seconds: int = Field(..., ge=1, description="Video duration in seconds")
    title: str = Field(default="", description="Video title")
    published_at: datetime | None = Field(default=None, description="ISO 8601 publication timestamp")

    @field_validator("published_at", mode="before")
    @classmethod
    def _parse_published_at(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        try:
            return pd.to_datetime(value)
        except Exception as exc:
            raise ValueError(f"Invalid published_at: {value}") from exc


class PredictResponse(BaseModel):
    model_type: Literal["detection", "prediction"]
    probability: float
    threshold: float
    is_viral: bool
    features_used: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


def _predict(model_type: Literal["detection", "prediction"], payload: PredictRequest) -> PredictResponse:
    model, threshold_info = _load_artifacts(model_type)
    threshold = float(threshold_info.get("threshold", 0.5))

    features = build_feature_vector(
        views=payload.views,
        likes=payload.likes,
        comments=payload.comments,
        subscribers=payload.subscribers,
        days_old=payload.days_old,
        duration_seconds=payload.duration_seconds,
        title=payload.title,
        published_at=payload.published_at,
        model_type=model_type,
    )

    proba = float(model.predict_proba(features)[0, 1])
    is_viral = proba >= threshold

    return PredictResponse(
        model_type=model_type,
        probability=proba,
        threshold=threshold,
        is_viral=is_viral,
        features_used=features.columns.tolist(),
    )


@app.post("/predict/detection", response_model=PredictResponse)
def predict_detection(payload: PredictRequest):
    return _predict("detection", payload)


@app.post("/predict/prediction", response_model=PredictResponse)
def predict_prediction(payload: PredictRequest):
    return _predict("prediction", payload)
