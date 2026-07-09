"""
Tests for labeling logic (src/features/labeling.py).

Covers the two guarantees the README leans on:
  - Detection labels: is_viral is a pure function of the same-row view_count
    (no temporal leakage).
  - Future labels: the LAST snapshot per video is dropped (otherwise the label
    would be derived from the row's own view_count = direct target leakage),
    and the future label is driven by the video's final snapshot.
"""
import pandas as pd
import pytest

from src.features.labeling import (
    VIRALITY_THRESHOLD,
    create_detection_labels,
    create_future_labels,
)


def test_detection_label_threshold(sample_snapshot_df):
    df = create_detection_labels(sample_snapshot_df, threshold=VIRALITY_THRESHOLD)
    assert "is_viral" in df.columns
    # vid_3 is above 10M in every snapshot -> all viral.
    assert (df.loc[df["video_id"] == "vid_3", "is_viral"] == 1).all()
    # vid_2 never reaches 10M -> all non-viral.
    assert (df.loc[df["video_id"] == "vid_2", "is_viral"] == 0).all()


def test_detection_label_no_leakage(sample_snapshot_df):
    """is_viral must equal (view_count >= threshold) row-by-row, nothing more."""
    df = create_detection_labels(sample_snapshot_df, threshold=VIRALITY_THRESHOLD)
    expected = (df["view_count"] >= VIRALITY_THRESHOLD).astype(int)
    assert (df["is_viral"] == expected).all()


def test_detection_label_custom_threshold(sample_snapshot_df):
    df = create_detection_labels(sample_snapshot_df, threshold=1_000_000)
    # At 1M, vid_1 and vid_3 have viral rows; vid_2 stays under.
    assert df.loc[df["video_id"] == "vid_2", "is_viral"].sum() == 0


def test_future_label_drops_last_snapshot(sample_snapshot_df):
    df = create_future_labels(sample_snapshot_df, threshold=VIRALITY_THRESHOLD)
    # Each video's last snapshot must be removed (no future to predict).
    # vid_1 had 3 snapshots -> 2 remain; vid_2 had 2 -> 1 remains; etc.
    counts = df.groupby("video_id").size().to_dict()
    assert counts.get("vid_1") == 2
    assert counts.get("vid_2") == 1
    assert counts.get("vid_3") == 1
    assert "vid_4" not in counts  # single snapshot dropped entirely


def test_future_label_uses_final_snapshot(sample_snapshot_df):
    df = create_future_labels(sample_snapshot_df, threshold=VIRALITY_THRESHOLD)
    # vid_1's final snapshot is 12M -> viral. All non-final rows of vid_1 labelled 1.
    vid1 = df[df["video_id"] == "vid_1"]
    assert (vid1["future_is_viral"] == 1).all()
    # vid_2's final snapshot is 120k -> not viral.
    vid2 = df[df["video_id"] == "vid_2"]
    assert (vid2["future_is_viral"] == 0).all()


def test_future_label_no_self_reference_column(sample_snapshot_df):
    """The helper column used to derive the label must not leak into the output."""
    df = create_future_labels(sample_snapshot_df, threshold=VIRALITY_THRESHOLD)
    assert "last_snapshot_views" not in df.columns
