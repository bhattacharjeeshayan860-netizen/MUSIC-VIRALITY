"""
FastAPI backend for the Music Virality System.

Serves both the detection ("is this video viral right now?") and the prediction
("will this video go viral?") models over a clean REST API, plus a convenience
endpoint that fetches live stats straight from YouTube.

Run (from project root):
    uvicorn src.api.app:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import build_feature_vector


def _iso_duration_to_seconds(s) -> int:
    """Parse an ISO-8601 duration (e.g. 'PT3M5S') to whole seconds.

    Mirrors the parser in src/data_processing/cleaning_data.py (which is a
    nested function and therefore not importable) so serving matches training.
    """
    if s is None:
        return 0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(s))
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + sec

DETECTION_MODEL_PATH = "models/trained/detection_model.pkl"
DETECTION_THRESHOLD_PATH = "models/trained/detection_threshold.pkl"
PREDICTION_MODEL_PATH = "models/trained/prediction_model.pkl"
PREDICTION_THRESHOLD_PATH = "models/trained/prediction_threshold.pkl"


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class VideoStats(BaseModel):
    views: int = Field(..., ge=0, description="Current view count")
    likes: int = Field(..., ge=0, description="Current like count")
    comments: int = Field(..., ge=0, description="Current comment count")
    subscribers: int = Field(..., ge=0, description="Channel subscriber count")
    days_old: int = Field(..., ge=1, description="Days since the video was published")
    duration_seconds: int = Field(..., ge=1, description="Video duration in seconds")


class PredictionResponse(BaseModel):
    model: str
    probability: float
    threshold: float
    is_viral: bool
    label: str


class YouTubeRequest(BaseModel):
    video_id_or_url: str = Field(..., description="A YouTube video ID or full URL")
    youtube_api_key: Optional[str] = Field(
        None, description="Override YOUTUBE_API_KEY env var (optional)"
    )


class YouTubePredictionResponse(BaseModel):
    video_id: str
    fetched_stats: VideoStats
    detection: PredictionResponse
    prediction: PredictionResponse


# ─────────────────────────────────────────────
# MODEL LOADING (cached at startup)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_bundle(kind: str):
    if kind == "detection":
        model_path, th_path = DETECTION_MODEL_PATH, DETECTION_THRESHOLD_PATH
    else:
        model_path, th_path = PREDICTION_MODEL_PATH, PREDICTION_THRESHOLD_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{kind} model not found at {model_path}. Run `python main.py` first."
        )
    model = joblib.load(model_path)
    threshold_info = joblib.load(th_path)
    return model, float(threshold_info["threshold"]), str(threshold_info.get("model_name", "XGBoost"))


def _predict_kind(kind: str, stats: VideoStats) -> PredictionResponse:
    model, threshold, model_name = _load_bundle(kind)
    features = build_feature_vector(
        views=stats.views, likes=stats.likes, comments=stats.comments,
        subscribers=stats.subscribers, days_old=stats.days_old,
        duration_seconds=stats.duration_seconds,
    )
    probability = float(model.predict_proba(features)[0, 1])
    is_viral = bool(probability >= threshold)
    return PredictionResponse(
        model=model_name,
        probability=round(probability, 4),
        threshold=round(threshold, 4),
        is_viral=is_viral,
        label="VIRAL" if is_viral else "Not Viral",
    )


# ─────────────────────────────────────────────
# YOUTUBE HELPERS
# ─────────────────────────────────────────────

_VIDEO_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
)


def _extract_video_id(video_id_or_url: str) -> str:
    s = video_id_or_url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID_RE.search(s)
    if m:
        return m.group(1)
    raise HTTPException(
        status_code=400,
        detail=f"Could not extract an 11-char YouTube video ID from: {video_id_or_url!r}",
    )


def _fetch_youtube_stats(video_id: str, api_key: Optional[str]) -> VideoStats:
    """Fetch live stats for one video and map them to the model's input schema."""
    if api_key:
        os.environ["YOUTUBE_API_KEY"] = api_key

    try:
        from src.api.youtube_client import YouTubeClient
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"YouTube client import failed: {e}")

    try:
        client = YouTubeClient()
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{e}. Set YOUTUBE_API_KEY in .env or pass youtube_api_key in the request.",
        )

    videos = client.fetch_video_details([video_id])
    if not videos:
        raise HTTPException(status_code=404, detail=f"YouTube returned no stats for video {video_id}.")
    v = videos[0]

    channel_id = v.get("channel_id")
    subscribers = 0
    if channel_id:
        chans = client.fetch_channel_details([channel_id])
        if chans:
            subscribers = int(chans[0].get("subscriber_count", 0))

    duration_seconds = int(_iso_duration_to_seconds(v.get("duration")) or 0)
    if duration_seconds < 1:
        duration_seconds = 1

    # published_at isn't returned by fetch_video_details; derive age from
    # a separate lightweight search call is overkill, so require days_old via
    # the videos.list snippet if available. We fetch it here for accuracy.
    days_old = _fetch_days_old(client, video_id)

    return VideoStats(
        views=int(v.get("view_count", 0)),
        likes=int(v.get("like_count", 0)),
        comments=int(v.get("comment_count", 0)),
        subscribers=subscribers,
        days_old=max(days_old, 1),
        duration_seconds=duration_seconds,
    )


def _fetch_days_old(client, video_id: str) -> int:
    """Best-effort: read publishedAt from the videos.list snippet part."""
    import requests
    url = client.base_url + "/videos"
    params = {"part": "snippet", "id": video_id, "key": client.api_key}
    try:
        resp = client._get_with_retry("videos", url, params)
        if resp is None:
            return 1
        items = resp.json().get("items", [])
        if not items:
            return 1
        published = items[0].get("snippet", {}).get("publishedAt")
        if not published:
            return 1
        pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - pub_dt).days
        return max(days, 1)
    except Exception:
        return 1


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Music Virality System API",
    description="Detect current virality and predict future virality for YouTube music videos.",
    version="1.0.0",
)


@app.get("/health")
def health():
    ready = os.path.exists(DETECTION_MODEL_PATH) and os.path.exists(PREDICTION_MODEL_PATH)
    return {"status": "ok", "models_loaded": ready}


@app.post("/predict/detection", response_model=PredictionResponse)
def predict_detection(stats: VideoStats):
    """Is this video already viral right now?"""
    try:
        return _predict_kind("detection", stats)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/prediction", response_model=PredictionResponse)
def predict_prediction(stats: VideoStats):
    """Will this video become viral at a later snapshot? (short-horizon momentum model)"""
    try:
        return _predict_kind("prediction", stats)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/from-youtube", response_model=YouTubePredictionResponse)
def predict_from_youtube(req: YouTubeRequest):
    """Fetch live stats from YouTube by URL/ID and run BOTH models."""
    video_id = _extract_video_id(req.video_id_or_url)
    stats = _fetch_youtube_stats(video_id, req.youtube_api_key)
    detection = _predict_kind("detection", stats)
    prediction = _predict_kind("prediction", stats)
    return YouTubePredictionResponse(
        video_id=video_id,
        fetched_stats=stats,
        detection=detection,
        prediction=prediction,
    )
