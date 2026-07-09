"""
Shared fixtures for the Music Virality test suite.

Tests run against small synthetic datasets so they are fast, deterministic, and
do not depend on the raw YouTube snapshots living in data/. The synthetic data
mirrors the real schema (video_id, collected_at, view_count, like_count,
comment_count, subscriber_count, day_since_published, duration_seconds) so the
feature engineering and labeling code under test operates on the columns it
expects in production.
"""
import pandas as pd
import pytest


def _snapshot_rows():
    rows = [
        ("vid_1", "2024-01-01", 2_000_000, 50_000, 4_000, 500_000, 5, 240),
        ("vid_1", "2024-01-03", 9_000_000, 180_000, 12_000, 500_000, 7, 240),
        ("vid_1", "2024-01-05", 12_000_000, 250_000, 20_000, 500_000, 9, 240),
        ("vid_2", "2024-01-01", 50_000, 1_500, 120, 20_000, 10, 200),
        ("vid_2", "2024-01-05", 120_000, 3_000, 200, 20_000, 14, 200),
        ("vid_3", "2024-01-01", 25_000_000, 800_000, 60_000, 8_000_000, 30, 300),
        ("vid_3", "2024-01-05", 40_000_000, 1_200_000, 90_000, 8_000_000, 34, 300),
        ("vid_4", "2024-01-01", 800_000, 20_000, 1_500, 90_000, 3, 45),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "video_id", "collected_at", "view_count", "like_count",
            "comment_count", "subscriber_count", "day_since_published",
            "duration_seconds",
        ],
    )


@pytest.fixture
def sample_snapshot_df():
    """
    Multi-snapshot dataset with a mix of viral / non-viral videos.

    - vid_1: crosses 10M on its 2nd snapshot (good detection + future test case)
    - vid_2: stays below threshold the whole time
    - vid_3: already viral at every snapshot
    - vid_4: single-snapshot video (exercises momentum NaN handling)
    """
    return _snapshot_rows()
