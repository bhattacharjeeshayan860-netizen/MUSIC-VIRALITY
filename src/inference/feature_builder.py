"""
Single source of truth for inference-time feature construction.

WHY THIS EXISTS:
Earlier the Streamlit dashboards hardcoded their own FEATURE_COLUMNS lists that
drifted from the trimmed training feature builders. Both dashboards ended up
sending extra columns (view_count_log, days_log, views_per_day, ...) that the
trained models never saw, crashing model.predict_proba() with a feature-name
mismatch ValueError.

This module builds EXACTLY the columns each model was trained on, in order,
imported directly from the training feature builders so it can never drift again.
It is shared by the dashboards and the FastAPI backend.

NEW BEHAVIOUR:
Detection and prediction now intentionally use slightly different feature sets
(view_count_log and views_ratio_to_first are allowed for prediction because the
label is a future snapshot, but excluded from detection where raw view scale
would trivialize the current-virality label). Callers choose the target model with
`model_type="detection"` or `model_type="prediction"`.
"""
import re
from datetime import datetime

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


def _publish_features(published_at) -> dict:
    """Extract safe publish-time features from a datetime string/object."""
    pub = None
    if published_at is not None:
        try:
            pub = pd.to_datetime(published_at, errors="coerce").tz_localize(None)
        except Exception:
            pub = None

    if pd.isna(pub):
        return {
            "publish_hour": 12.0,
            "publish_day_of_week": 3.0,
            "publish_is_weekend": 0.0,
            "publish_is_prime_time": 0.0,
            "publish_month": 6.0,
        }

    return {
        "publish_hour": float(pub.hour),
        "publish_day_of_week": float(pub.dayofweek),
        "publish_is_weekend": 1.0 if pub.dayofweek in (5, 6) else 0.0,
        "publish_is_prime_time": 1.0 if 18 <= pub.hour <= 22 else 0.0,
        "publish_month": float(pub.month),
    }


def _title_features(title: str) -> dict:
    """Extract safe title-text features from the video title."""
    title = title or ""

    title_length = float(len(title))
    words = title.split()
    title_word_count = float(len(words))

    letters = [c for c in title if c.isalpha()]
    title_caps_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters)
        if letters else 0.0
    )

    viral_keywords = [
        "official", "music video", "mv", "remix", "cover", "live", "acoustic",
        "ft", "feat", "ft.", "feat.", "new song", "latest", "trending", "viral",
    ]
    pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in viral_keywords) + r")\b"

    return {
        "title_length": title_length,
        "title_word_count": title_word_count,
        "title_caps_ratio": title_caps_ratio,
        "title_has_emoji": float(bool(re.search(r"[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]", title))),
        "title_has_number": float(bool(re.search(r"\d", title))),
        "title_has_special_char": float(bool(re.search(r"[!?*#]", title))),
        "title_has_hashtag": float("#" in title),
        "title_has_viral_keyword": float(bool(re.search(pattern, title, re.IGNORECASE))),
    }


def build_feature_vector(
    views: int,
    likes: int,
    comments: int,
    subscribers: int,
    days_old: int,
    duration_seconds: int,
    title: str = "",
    published_at=None,
    model_type: str = "detection",
) -> pd.DataFrame:
    """
    Build a single-snapshot row that matches the requested model's training columns.

    Mirrors src/features/feature_engineering.py:
      - ratios normalized by views (NaN-safe -> 0.0 for zero denominators)
      - per-day velocity with days clipped to >=1
      - log1p transforms for subscriber_count and duration
      - is_short_video = duration_seconds < 60
      - channel_tier / age_bucket ordinal bins match training's pd.cut edges
      - momentum diff features are NaN for a single snapshot; the trained
        pipeline's SimpleImputer(fill_value=0) handles them.

    Parameters
    ----------
    model_type : "detection" or "prediction"
        Selects which trained model the row is intended for.
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
    views_per_day = views / days  # intermediate only
    engagement_per_day = (likes + 2 * comments) / days

    # Single snapshot -> no prior row, so diffs and trajectory are NaN/0
    nan = float("nan")
    views_growth_rate = nan
    views_acceleration = nan
    views_diff = nan
    likes_diff = nan
    engagement_diff = nan

    days_since_first_snapshot = 0.0
    views_ratio_to_first = nan

    subscriber_count_log = np.log1p(subscribers)
    subs_safe = subscribers if subscribers > 0 else nan
    views_to_subs_ratio = views / subs_safe if subscribers > 0 else nan
    views_per_day_per_sub = (views_per_day / subs_safe) if subscribers > 0 else nan
    likes_to_subs_ratio = likes / subs_safe if subscribers > 0 else nan
    comments_to_subs_ratio = comments / subs_safe if subscribers > 0 else nan
    engagement_to_subs_ratio = (likes + 2 * comments) / subs_safe if subscribers > 0 else nan

    like_to_comment_ratio = likes / comments if comments > 0 else 0.0

    channel_tier = _channel_tier(subscribers)
    age_bucket = _age_bucket(days_old)
    duration_log = np.log1p(duration)
    is_short_video = 1.0 if duration < 60 else 0.0
    snapshot_rank = 1.0

    view_count_log = np.log1p(views)
    likes_per_day_log = np.log1p(likes_per_day)
    comments_per_day_log = np.log1p(comments_per_day)
    engagement_per_day_log = np.log1p(engagement_per_day)

    row = {
        "like_rate": like_rate,
        "comment_rate": comment_rate,
        "engagement_rate": engagement_rate,
        "comment_like_ratio": comment_like_ratio,
        "likes_per_day": likes_per_day,
        "comments_per_day": comments_per_day,
        "engagement_per_day": engagement_per_day,
        "like_to_comment_ratio": like_to_comment_ratio,
        "comments_to_subs_ratio": comments_to_subs_ratio,
        "engagement_to_subs_ratio": engagement_to_subs_ratio,
        "likes_to_subs_ratio": likes_to_subs_ratio,
        "likes_per_day_log": likes_per_day_log,
        "comments_per_day_log": comments_per_day_log,
        "engagement_per_day_log": engagement_per_day_log,
        "views_growth_rate": views_growth_rate,
        "views_acceleration": views_acceleration,
        "views_diff": views_diff,
        "likes_diff": likes_diff,
        "engagement_diff": engagement_diff,
        "days_since_first_snapshot": days_since_first_snapshot,
        "views_ratio_to_first": views_ratio_to_first,
        "subscriber_count_log": subscriber_count_log,
        "views_to_subs_ratio": views_to_subs_ratio,
        "views_per_day_per_sub": views_per_day_per_sub,
        "channel_tier": channel_tier,
        "age_bucket": age_bucket,
        "duration_log": duration_log,
        "is_short_video": is_short_video,
        "snapshot_rank": snapshot_rank,
        "view_count_log": view_count_log,
        "publish_hour": _publish_features(published_at)["publish_hour"],
        "publish_day_of_week": _publish_features(published_at)["publish_day_of_week"],
        "publish_is_weekend": _publish_features(published_at)["publish_is_weekend"],
        "publish_is_prime_time": _publish_features(published_at)["publish_is_prime_time"],
        "publish_month": _publish_features(published_at)["publish_month"],
        **_title_features(title),
    }

    if model_type == "detection":
        feature_cols = list(_DETECTION_FEATURES)
    elif model_type == "prediction":
        feature_cols = list(_PREDICTION_FEATURES)
    else:
        raise ValueError(f"model_type must be 'detection' or 'prediction', got {model_type!r}")

    return pd.DataFrame([{c: row.get(c, 0.0) for c in feature_cols}])[feature_cols]
