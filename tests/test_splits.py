"""
Tests for the stratified group-aware split (src/utils/splits.py).

This is the single most important correctness property in the project: a leak
across train/val/test would invalidate every reported metric. These tests pin
down the two guarantees the README advertises:

  1. No video_id appears in more than one split (group-aware).
  2. The positive-class rate is preserved across all three splits (stratified),
     which is what keeps PR-AUC stable on a skewed dataset.
"""
import numpy as np
import pandas as pd
import pytest

from src.utils.splits import stratified_group_three_way_split


def _build_grouped_dataset(n_videos=200, snapshots_per_video=3, pos_rate=0.2, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        is_viral = int(rng.random() < pos_rate)
        base_views = rng.integers(15_000_000, 30_000_000) if is_viral else rng.integers(10_000, 5_000_000)
        for s in range(snapshots_per_video):
            views = int(base_views * (1 + 0.1 * s) + rng.integers(-50_000, 50_000))
            rows.append({
                "video_id": f"vid_{v}",
                "collected_at": f"2024-01-{s+1:02d}",
                "view_count": max(views, 0),
                "like_count": int(views * 0.05),
                "comment_count": int(views * 0.003),
                "subscriber_count": int(rng.integers(1_000, 5_000_000)),
                "day_since_published": int(rng.integers(1, 60)),
                "duration_seconds": int(rng.integers(30, 400)),
            })
    df = pd.DataFrame(rows)
    df["y"] = (df["view_count"] >= 10_000_000).astype(int)
    return df


@pytest.fixture
def grouped():
    return _build_grouped_dataset()


def _video_sets(df, x_train, x_val, x_test):
    """Map each split's row indices back to the source df and extract video_ids."""
    tr = set(df.loc[x_train.index, "video_id"])
    va = set(df.loc[x_val.index, "video_id"])
    te = set(df.loc[x_test.index, "video_id"])
    return tr, va, te


def test_no_group_leakage_across_splits(grouped):
    df = grouped
    X = df.drop(columns=["y"])
    y = df["y"]

    x_tr, x_va, x_te, y_tr, y_va, y_te = stratified_group_three_way_split(
        df, X, y, val_size=0.15, test_size=0.15, random_state=42
    )

    tr, va, te = _video_sets(df, x_tr, x_va, x_te)

    assert tr.isdisjoint(va), "train/val video overlap (leakage!)"
    assert tr.isdisjoint(te), "train/test video overlap (leakage!)"
    assert va.isdisjoint(te), "val/test video overlap (leakage!)"


def test_all_rows_accounted_for(grouped):
    df = grouped
    X = df.drop(columns=["y"])
    y = df["y"]

    x_tr, x_va, x_te, _, _, _ = stratified_group_three_way_split(
        df, X, y, val_size=0.15, test_size=0.15, random_state=42
    )

    total = len(x_tr) + len(x_va) + len(x_te)
    assert total == len(df), f"Row count mismatch: {total} vs {len(df)}"
    # No row index should be duplicated across splits.
    all_idx = list(x_tr.index) + list(x_va.index) + list(x_te.index)
    assert len(all_idx) == len(set(all_idx)), "Duplicate row index across splits"


def test_positive_rate_preserved(grouped):
    """The whole point of stratification: positive rate stays close across splits."""
    df = grouped
    X = df.drop(columns=["y"])
    y = df["y"]

    x_tr, x_va, x_te, y_tr, y_va, y_te = stratified_group_three_way_split(
        df, X, y, val_size=0.15, test_size=0.15, random_state=42
    )

    overall = y.mean()
    for name, y_split in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        rate = y_split.mean()
        # Allow a tolerance band; stratification preserves rate within a few points
        # even on small/skewed splits.
        assert abs(rate - overall) < 0.08, (
            f"{name} positive rate {rate:.3f} drifted from overall {overall:.3f}"
        )


def test_split_sizes_match_request(grouped):
    df = grouped
    X = df.drop(columns=["y"])
    y = df["y"]

    x_tr, x_va, x_te, _, _, _ = stratified_group_three_way_split(
        df, X, y, val_size=0.15, test_size=0.15, random_state=42
    )

    n = len(df)
    # Train ~70%, val ~15%, test ~15%. Allow slack for group boundaries.
    assert len(x_tr) / n > 0.60
    assert 0.05 < len(x_va) / n < 0.25
    assert 0.05 < len(x_te) / n < 0.25


def test_fallback_when_no_video_id():
    """When video_id is missing the split should gracefully fall back, not crash."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.random(100),
        "y": rng.integers(0, 2, 100),
    })
    X = df.drop(columns=["y"])
    y = df["y"]
    x_tr, x_va, x_te, _, _, _ = stratified_group_three_way_split(
        df, X, y, val_size=0.15, test_size=0.15, random_state=42
    )
    assert len(x_tr) > 0 and len(x_va) > 0 and len(x_te) > 0
