"""
Tests for the inference-time feature builder (src/inference/feature_builder.py).

This module is the single source of truth shared by both Streamlit dashboards
and the FastAPI backend. The critical contract it must uphold:

  - The returned DataFrame has EXACTLY the columns the requested model was
    trained on, in the same order.
  - Zero-safe arithmetic mirrors training (no inf/nan from zero views/likes).
  - Channel tier / age bucket / is_short_video match training's bin edges.
  - Title and publish-time features are computed correctly from casual inputs.
"""
import numpy as np
import pandas as pd
import pytest

from src.inference.feature_builder import (
    build_feature_vector,
)
from src.features.build_detections_features import FEATURE_COLS as DETECTION_FEATURES
from src.features.build_predictions_features import FEATURE_COLUMNS as PREDICTION_FEATURES


def test_detection_feature_vector_matches_training():
    row = build_feature_vector(
        views=5_000_000, likes=200_000, comments=15_000,
        subscribers=1_200_000, days_old=10, duration_seconds=210,
        title="Official Music Video #trending",
        published_at="2024-03-15T19:00:00Z",
        model_type="detection",
    )
    assert set(row.columns) == set(DETECTION_FEATURES)
    assert list(row.columns) == list(DETECTION_FEATURES)
    assert len(row) == 1


def test_prediction_feature_vector_matches_training():
    row = build_feature_vector(
        views=5_000_000, likes=200_000, comments=15_000,
        subscribers=1_200_000, days_old=10, duration_seconds=210,
        title="Official Music Video #trending",
        published_at="2024-03-15T19:00:00Z",
        model_type="prediction",
    )
    assert set(row.columns) == set(PREDICTION_FEATURES)
    assert list(row.columns) == list(PREDICTION_FEATURES)
    assert "view_count_log" in row.columns
    assert "views_ratio_to_first" in row.columns


def test_prediction_has_more_view_scale_features():
    pred = build_feature_vector(
        views=5_000_000, likes=200_000, comments=15_000,
        subscribers=1_200_000, days_old=10, duration_seconds=210,
        title="Official Music Video",
        published_at="2024-03-15T19:00:00Z",
        model_type="prediction",
    )
    det = build_feature_vector(
        views=5_000_000, likes=200_000, comments=15_000,
        subscribers=1_200_000, days_old=10, duration_seconds=210,
        title="Official Music Video",
        published_at="2024-03-15T19:00:00Z",
        model_type="detection",
    )
    # Detection must never see raw view scale features
    assert "view_count_log" not in det.columns
    assert "views_ratio_to_first" not in det.columns
    # Prediction is allowed to use them because its label is a future snapshot
    assert "view_count_log" in pred.columns
    assert "views_ratio_to_first" in pred.columns


def test_build_feature_vector_values():
    row = build_feature_vector(
        views=1_000_000, likes=50_000, comments=5_000,
        subscribers=2_000_000, days_old=20, duration_seconds=300,
        title="Official Music Video #trending",
        published_at="2024-03-15T19:00:00Z",
        model_type="detection",
    ).iloc[0]
    assert row["like_rate"] == pytest.approx(0.05)
    assert row["comment_rate"] == pytest.approx(0.005)
    assert row["engagement_rate"] == pytest.approx((50_000 + 2 * 5_000) / 1_000_000)
    assert row["engagement_per_day"] == pytest.approx((50_000 + 2 * 5_000) / 20)
    # 2M subs -> macro tier (3)
    assert row["channel_tier"] == 3.0
    # 20 days -> age_bucket 1 (8-30)
    assert row["age_bucket"] == 1.0
    # 300s -> not short
    assert row["is_short_video"] == 0.0
    # 19:00 is prime time
    assert row["publish_is_prime_time"] == 1.0
    assert row["publish_hour"] == 19.0
    assert row["title_has_hashtag"] == 1.0
    assert row["title_has_viral_keyword"] == 1.0


def test_build_feature_vector_zero_views_safe():
    row = build_feature_vector(
        views=0, likes=0, comments=0,
        subscribers=0, days_old=0, duration_seconds=30,
        title="",
        model_type="detection",
    ).iloc[0]
    # No inf/nan from zero denominators.
    assert np.isfinite(row["like_rate"])
    assert np.isfinite(row["comment_rate"])
    assert np.isfinite(row["engagement_rate"])
    assert np.isfinite(row["comment_like_ratio"])
    assert np.isfinite(row["like_to_comment_ratio"])
    assert np.isfinite(row["likes_per_day"])
    # 0 subs -> tier 0
    assert row["channel_tier"] == 0.0
    # 30s -> short
    assert row["is_short_video"] == 1.0


def test_build_feature_vector_short_video_boundary():
    row_below = build_feature_vector(
        views=100, likes=10, comments=1, subscribers=100,
        days_old=1, duration_seconds=59,
        title="short",
        model_type="detection",
    ).iloc[0]
    row_at = build_feature_vector(
        views=100, likes=10, comments=1, subscribers=100,
        days_old=1, duration_seconds=60,
        title="borderline",
        model_type="detection",
    ).iloc[0]
    assert row_below["is_short_video"] == 1.0
    assert row_at["is_short_video"] == 0.0


def test_invalid_model_type_raises():
    with pytest.raises(ValueError, match="model_type"):
        build_feature_vector(views=1, likes=1, comments=1, subscribers=1, days_old=1, duration_seconds=60, model_type="unknown")
