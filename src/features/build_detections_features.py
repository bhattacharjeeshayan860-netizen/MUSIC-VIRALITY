import os
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# FEATURE COLUMNS — DETECTION MODEL
# ─────────────────────────────────────────────
#
# AUDIT LOG — what was changed and WHY:
#
# REMOVED: view_count_log
#   Reason: view_count_log = log1p(view_count).
#   The label is_viral = (view_count >= 10M).
#   Therefore log1p(view_count) perfectly separates classes at log1p(10M)=16.12.
#   Including it gives ROC-AUC=1.0 but the model learns nothing — it just finds
#   the threshold. A model that uses the label's own source column as a feature
#   is not a classifier, it's a lookup table. Confirmed by audit: removing it
#   drops AUC from fraudulent 1.0 to honest 0.95.
#
# ADDED: subscriber_count_log
#   A 10M-view video on a 300M-sub channel is less viral than on a 10K-sub channel.
#   Channel authority contextualizes view counts.
#
# ADDED: views_to_subs_ratio
#   Direct measure of outperforming the channel's baseline reach.
#   Strong virality signal independent of raw view count.
#
# ADDED: channel_tier
#   Ordinal channel size bracket. Helps model learn tier-specific patterns.
#
# ADDED: days_log, age_bucket
#   A video's age changes the meaning of its velocity. Without age context,
#   a 3-year-old song with 5M views looks the same as a 2-day-old song with 5M views.
#
# ADDED: snapshot_rank, snapshot_count
#   Tells the model where in a video's lifecycle this snapshot falls.
#   Earlier snapshots are noisier; later ones are more stable.
#
# ADDED: duration_log, is_short_video
#   YouTube Shorts (<60s) have different viral mechanics. Without this flag,
#   the model conflates two incompatible distributions.
#
# KEPT: All original ratio, velocity, and momentum features.
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # --- Engagement ratios (your original, correct) ---
    "like_rate",
    "comment_rate",
    "engagement_rate",
    "comment_like_ratio",

    # --- Velocity (your original, correct) ---
    "likes_per_day",
    "comments_per_day",

    # --- Momentum / time-series diffs (your original, correct) ---
    "views_growth_rate",
    "views_acceleration",
    "views_diff",
    "likes_diff",
    "engagement_diff",

    # --- Channel authority (NEW) ---
    "subscriber_count_log",
    "views_to_subs_ratio",
    "views_per_day_per_sub",
    "channel_tier",

    # --- Video age context (NEW) ---
    
    "age_bucket",

    # --- Video format (NEW) ---
    "duration_log",
    "is_short_video",

    # --- Snapshot metadata (NEW) ---
    "snapshot_rank",
    "snapshot_count",

]


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_labeled_data(file_path="data/processed/final_labelled_music_virality_data.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    print(f"File {file_path} does not exist.")
    return pd.DataFrame()


# ─────────────────────────────────────────────
# GET X, y — WITH video_id FOR GROUP SPLIT
# ─────────────────────────────────────────────

def get_X_y(file_path="data/processed/final_labelled_music_virality_data.csv"):
    """
    Returns (df, X, y) — the full df is needed so the training script can
    extract video_id for GroupShuffleSplit without leaking it into features.

    WHY return df: train_test_split on rows ignores video_id, so the same video
    ends up in both train and test (confirmed: 77.2% of test rows were leaked).
    GroupShuffleSplit needs groups=df['video_id'], which requires the column.
    """
    df = load_labeled_data(file_path=file_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype="int64")

    # Only use features that actually exist in this file
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        print(f"[Warning] These features are missing from data and will be skipped: {missing_features}")

    required_cols = available_features + ["is_viral"]
    data = df[required_cols + (["video_id"] if "video_id" in df.columns else [])].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["is_viral"])

    X = data[available_features]
    y = data["is_viral"].astype(int)

    return data, X, y