"""
Tests for the inference-time feature builder (src/inference/feature_builder.py).

This module is the single source of truth shared by both Streamlit dashboards
and the FastAPI backend. The critical contract it must uphold:

  - The returned DataFrame has EXACTLY the columns the trained models expect,
    in the same order (otherwise model.predict_proba raises a feature-name
    mismatch at serving time — the exact bug this module was created to fix).
  - Zero-safe arithmetic mirrors training (no inf/nan from zero views/likes).
  - Channel tier / age bucket / is_short_video match training's bin edges.
"""
import numpy as np
import pandas as pd
import pytest

from src.inference.feature_builder import (
    build_feature_vector,
    INFERENCE_FEATURES,
)
from src.features.build_detections_features import FEATURE_COLS as DETECTION_FEATURES
from src.features.build_predictions_features import FEATURE_COLUMNS as PREDICTION_FEATURES


def test_inference_features_match_training():
    assert set(INFERENCE_FEATURES) == set(DETECTION_FEATURES)
    assert set(INFERENCE_FEATURES) == set(PREDICTION_FEATURES)


def test_build_feature_vector_columns_and_order():
    row = build_feature_vector(
        views=5_000_000, likes=200_000, comments=15_000,
        subscribers=1_200_000, days_old=10, duration_seconds=210,
    )
    assert list(row.columns) == list(INFERENCE_FEATURES)
    assert len(row) == 1


def test_build_feature_vector_values():
    row = build_feature_vector(
        views=1_000_000, likes=50_000, comments=5_000,
        subscribers=2_000_000, days_old=20, duration_seconds=300,
    ).iloc[0]
    assert row["like_rate"] == pytest.approx(0.05)
    assert row["comment_rate"] == pytest.approx(0.005)
    assert row["engagement_rate"] == pytest.approx((50_000 + 2 * 5_000) / 1_000_000)
    # 2M subs -> macro tier (3)
    assert row["channel_tier"] == 3.0
    # 20 days -> age_bucket 1 (8-30)
    assert row["age_bucket"] == 1.0
    # 300s -> not short
    assert row["is_short_video"] == 0.0


def test_build_feature_vector_zero_views_safe():
    row = build_feature_vector(
        views=0, likes=0, comments=0,
        subscribers=0, days_old=0, duration_seconds=30,
    ).iloc[0]
    # No inf/nan from zero denominators.
    assert np.isfinite(row["like_rate"])
    assert np.isfinite(row["comment_rate"])
    assert np.isfinite(row["engagement_rate"])
    assert np.isfinite(row["comment_like_ratio"])
    # day_since_published clipped to >=1 in training; 0 views / 1 day = 0
    assert np.isfinite(row["likes_per_day"])
    # 0 subs -> tier 0
    assert row["channel_tier"] == 0.0
    # 30s -> short
    assert row["is_short_video"] == 1.0


def test_build_feature_vector_short_video_boundary():
    row_below = build_feature_vector(
        views=100, likes=10, comments=1, subscribers=100,
        days_old=1, duration_seconds=59,
    ).iloc[0]
    row_at = build_feature_vector(
        views=100, likes=10, comments=1, subscribers=100,
        days_old=1, duration_seconds=60,
    ).iloc[0]
    assert row_below["is_short_video"] == 1.0
    assert row_at["is_short_video"] == 0.0
