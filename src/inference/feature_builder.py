"""
Single source of truth for inference-time feature construction.

WHY THIS EXISTS:
Earlier the Streamlit dashboards hardcoded their own FEATURE_COLUMNS lists that
drifted from the trimmed training feature builders. Both dashboards ended up
sending extra columns (view_count_log, days_log, views_per_day, ...) that the
trained models never saw, crashing model.predict_proba() with a feature-name
mismatch ValueError.

This module builds EXACTLY the columns the models were trained on, in order,
imported directly from the training feature builders so it can never drift again.
It is shared by the dashboards and the FastAPI backend.
"""
import numpy as np
import pandas as pd

from src.features.build_detections_features import FEATURE_COLS as _DETECTION_FEATURES
from src.features.build_predictions_features import FEATURE_COLUMNS as _PREDICTION_FEATURES


def _channel_tier(subscribers: float) -> float:
    if subscribers < 10_000:
        return 0.0
    if subscribers < 100_000:
        return 1.0
    if subscribers < 1_000_000:
        return 2.0
    if subscribers < 10_000_000:
        return 3.0
    return 4.0


def _age_bucket(days_old: float) -> float:
    if days_old <= 7:
        return 0.0
    if days_old <= 30:
        return 1.0
    if days_old <= 90:
        return 2.0
    if days_old <= 365:
        return 3.0
    return 4.0


# Both trained models use the identical 19-feature set. Verify at import time so
# a future divergence is caught immediately instead of silently breaking inference.
assert set(_DETECTION_FEATURES) == set(_PREDICTION_FEATURES), (
    "Detection and prediction feature sets have diverged — inference builder "
    "must handle them separately."
)
INFERENCE_FEATURES = list(_DETECTION_FEATURES)


def build_feature_vector(
    views: int,
    likes: int,
    comments: int,
    subscribers: int,
    days_old: int,
    duration_seconds: int,
) -> pd.DataFrame:
    """
    Build the 19-feature single-snapshot row the trained models expect.

    Mirrors src/features/feature_engineering.py exactly:
      - ratios normalized by views (NaN-safe -> 0.0 for zero denominators)
      - per-day velocity with days clipped to >=1 (matches training's .clip(lower=1))
      - log1p transforms for subscriber_count and duration
      - is_short_video = duration_seconds < 60  (NOT 180 — matches training)
      - channel_tier / age_bucket ordinal bins match training's pd.cut edges
      - momentum diff features are NaN for a single snapshot; the trained
        pipeline's SimpleImputer(fill_value=0) handles them, so we pass NaN
        to let the imputer do its job (identical to how test rows are scored).

    Returns a 1-row DataFrame with columns in INFERENCE_FEATURES order.
    """
    views = float(views)
    likes = float(likes)
    comments = float(comments)
    subscribers = float(subscribers)
    days = float(max(days_old, 1))  # training clips day_since_published to >=1
    duration = float(duration_seconds)

    like_rate = likes / views if views > 0 else 0.0
    comment_rate = comments / views if views > 0 else 0.0
    engagement_rate = (likes + 2 * comments) / views if views > 0 else 0.0
    comment_like_ratio = comments / likes if likes > 0 else 0.0

    likes_per_day = likes / days
    comments_per_day = comments / days
    views_per_day = views / days  # intermediate only; not a model feature

    # Single snapshot -> no prior row, so diffs are NaN (imputer fills 0 at score time).
    nan = float("nan")
    views_growth_rate = nan
    views_acceleration = nan
    views_diff = nan
    likes_diff = nan
    engagement_diff = nan

    subscriber_count_log = np.log1p(subscribers)
    subs_safe = subscribers if subscribers > 0 else nan
    views_to_subs_ratio = views / subs_safe if subscribers > 0 else nan
    views_per_day_per_sub = (views_per_day / subs_safe) if subscribers > 0 else nan

    channel_tier = _channel_tier(subscribers)
    age_bucket = _age_bucket(days_old)
    duration_log = np.log1p(duration)
    is_short_video = 1.0 if duration < 60 else 0.0
    snapshot_rank = 1.0

    row = {
        "like_rate": like_rate,
        "comment_rate": comment_rate,
        "engagement_rate": engagement_rate,
        "comment_like_ratio": comment_like_ratio,
        "likes_per_day": likes_per_day,
        "comments_per_day": comments_per_day,
        "views_growth_rate": views_growth_rate,
        "views_acceleration": views_acceleration,
        "views_diff": views_diff,
        "likes_diff": likes_diff,
        "engagement_diff": engagement_diff,
        "subscriber_count_log": subscriber_count_log,
        "views_to_subs_ratio": views_to_subs_ratio,
        "views_per_day_per_sub": views_per_day_per_sub,
        "channel_tier": channel_tier,
        "age_bucket": age_bucket,
        "duration_log": duration_log,
        "is_short_video": is_short_video,
        "snapshot_rank": snapshot_rank,
    }

    return pd.DataFrame([{c: row.get(c, 0.0) for c in INFERENCE_FEATURES}])[INFERENCE_FEATURES]
