"""
Tests for feature engineering (src/features/feature_engineering.py).

Focuses on the invariants that, if broken, would silently corrupt the model:
  - engagement ratios are NaN-safe (no inf from zero views/likes),
  - velocity divides by a clipped day count (never /0),
  - channel tier / age bucket binning matches documented edges,
  - momentum features produce NaN for the first snapshot of each video,
  - view_count_log is produced (used by prediction, intentionally NOT detection).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    create_ratio_features,
    create_scale_features,
    create_velocity_features,
    create_channel_authority_features,
    create_duration_features,
    create_time_context_features,
    create_momentum_features,
    create_eng_features,
)


def test_ratio_features_nan_safe_on_zero_views():
    df = pd.DataFrame({
        "view_count": [0, 1000],
        "like_count": [0, 100],
        "comment_count": [0, 10],
    })
    out = create_ratio_features(df)
    assert np.isnan(out.loc[0, "like_rate"])
    assert np.isnan(out.loc[0, "comment_rate"])
    assert np.isnan(out.loc[0, "engagement_rate"])
    # Non-zero row should compute cleanly.
    assert out.loc[1, "like_rate"] == pytest.approx(0.1)


def test_ratio_features_comment_like_ratio_zero_likes():
    df = pd.DataFrame({
        "view_count": [100, 200],
        "like_count": [0, 50],
        "comment_count": [5, 10],
    })
    out = create_ratio_features(df)
    assert np.isnan(out.loc[0, "comment_like_ratio"])
    assert out.loc[1, "comment_like_ratio"] == pytest.approx(0.2)


def test_velocity_features_no_div_by_zero():
    # day_since_published = 0 must be clipped to 1, not produce inf.
    df = pd.DataFrame({
        "view_count": [1000, 2000],
        "like_count": [50, 60],
        "comment_count": [5, 8],
        "day_since_published": [0, 2],
        "subscriber_count": [1000, 2000],
    })
    out = create_velocity_features(df)
    assert np.isfinite(out.loc[0, "views_per_day"])
    assert out.loc[0, "views_per_day"] == pytest.approx(1000.0)  # 1000/1
    assert out.loc[1, "views_per_day"] == pytest.approx(1000.0)  # 2000/2


def test_channel_tier_bins():
    df = pd.DataFrame({
        "view_count": [1, 1, 1, 1, 1],
        "like_count": [0, 0, 0, 0, 0],
        "comment_count": [0, 0, 0, 0, 0],
        "subscriber_count": [5_000, 50_000, 500_000, 5_000_000, 50_000_000],
    })
    out = create_channel_authority_features(df)
    # nano=0, micro=1, mid=2, macro=3, mega=4
    assert list(out["channel_tier"]) == [0, 1, 2, 3, 4]


def test_duration_features_short_flag():
    df = pd.DataFrame({
        "duration_seconds": [30, 200, 400],
    })
    out = create_duration_features(df)
    assert out.loc[0, "is_short_video"] == 1.0
    assert out.loc[1, "is_short_video"] == 0.0
    # 120-600 is "standard music video"
    assert out.loc[1, "is_standard_music_video"] == 1.0


def test_age_bucket_bins():
    df = pd.DataFrame({"day_since_published": [3, 20, 60, 200, 500]})
    out = create_time_context_features(df)
    assert list(out["age_bucket"]) == [0, 1, 2, 3, 4]


def test_momentum_first_snapshot_is_nan(sample_snapshot_df):
    df = create_momentum_features(sample_snapshot_df)
    # First chronological snapshot of each video has NaN diffs.
    v1_first = df[(df["video_id"] == "vid_1")].iloc[0]
    assert np.isnan(v1_first["views_diff"])
    assert np.isnan(v1_first["views_growth_rate"])
    # Second snapshot should have a finite diff.
    v1_second = df[(df["video_id"] == "vid_1")].iloc[1]
    assert np.isfinite(v1_second["views_diff"])
    assert v1_second["views_diff"] > 0


def test_momentum_snapshot_rank_and_count(sample_snapshot_df):
    df = create_momentum_features(sample_snapshot_df)
    v1 = df[df["video_id"] == "vid_1"]
    assert list(v1["snapshot_rank"]) == [1, 2, 3]
    assert (v1["snapshot_count"] == 3).all()


def test_eng_features_preserves_row_count(sample_snapshot_df):
    out = create_eng_features(sample_snapshot_df)
    assert len(out) == len(sample_snapshot_df)


def test_view_count_log_present_for_prediction(sample_snapshot_df):
    """view_count_log must exist (prediction model uses it); detection just
    must not select it as a feature — that's enforced in build_detections_features."""
    out = create_scale_features(sample_snapshot_df.copy())
    assert "view_count_log" in out.columns
    expected = np.log1p(sample_snapshot_df["view_count"])
    assert np.allclose(out["view_count_log"].values, expected.values)
