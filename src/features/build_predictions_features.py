import os

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# FEATURE COLUMNS — PREDICTION MODEL
# ─────────────────────────────────────────────
#
# Prediction label: future_is_viral = (last_snapshot_views >= THRESHOLD)
# Training row: an earlier snapshot — NOT the last snapshot.
#
# view_count_log IS included here because:
#   - The feature comes from snapshot T (current)
#   - The label comes from snapshot T+N (future, last available)
#   - These are different timestamps, so current views != future label directly
#
# NOTE: With the current ~6-day collection window, many future-viral videos are
# already above threshold at snapshot-1. This is documented in training output.
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    # --- Engagement ratios ---
    "like_rate",
    "comment_rate",
    "engagement_rate",
    "comment_like_ratio",

    # --- Velocity ---
    "views_per_day",
    "likes_per_day",
    "comments_per_day",

    # --- Momentum ---
    "views_growth_rate",
    "views_acceleration",
    "views_diff",
    "likes_diff",
    "engagement_diff",

    # --- Scale (safe here because label = FUTURE snapshot views) ---
    "view_count_log",
    "like_count_log",
    "comment_count_log",

    # --- Channel authority ---
    "subscriber_count_log",
    "views_to_subs_ratio",
    "views_per_day_per_sub",
    "channel_tier",

    # --- Age context ---
    "days_log",
    "age_bucket",

    # --- Video format ---
    "duration_log",
    "is_short_video",

    # --- Snapshot metadata ---
    "snapshot_rank",
    "snapshot_count",
]


def load_future_labeled_data(file_path="data/processed/future_labeled_music_virality_data.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path).copy()
    print(f"File {file_path} does not exist.")
    return pd.DataFrame()


def get_X_y(file_path="data/processed/future_labeled_music_virality_data.csv"):
    """
    Returns (df, X, y) so training script can do GroupShuffleSplit on video_id.
    Same pattern as build_detections_features.get_X_y for consistency.
    """
    df = load_future_labeled_data(file_path=file_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype="int64")

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"[Warning] Missing features will be skipped: {missing_features}")

    data = df[
        available_features
        + ["future_is_viral"]
        + (["video_id"] if "video_id" in df.columns else [])
    ].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["future_is_viral"])

    X = data[available_features]
    y = data["future_is_viral"].astype(int)

    return data, X, y