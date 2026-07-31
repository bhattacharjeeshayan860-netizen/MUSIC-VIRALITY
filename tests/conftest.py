"""
Shared fixtures for the Music Virality test suite.

Tests run against small synthetic datasets so they are fast, deterministic, and
do not depend on the raw YouTube snapshots living in data/. The synthetic data
mirrors the real schema (video_id, collected_at, view_count, like_count,
comment_count, subscriber_count, day_since_published, duration_seconds) so the
feature engineering and labeling code under test operates on the columns it
expects in production.
"""
import os
import sys

# Ensure the repository root is on sys.path so `from src...` imports resolve
# everywhere (local runs, CI, pytest from any cwd).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import pytest


def _snapshot_rows(title_prefix="Song"):
    rows = [
        ("vid_1", "2024-01-01", 2_000_000, 50_000, 4_000, 500_000, 5, 240, f"{title_prefix} One - Official Music Video #trending", "2023-12-27T18:00:00Z"),
        ("vid_1", "2024-01-03", 9_000_000, 180_000, 12_000, 500_000, 7, 240, f"{title_prefix} One - Official Music Video #trending", "2023-12-27T18:00:00Z"),
        ("vid_1", "2024-01-05", 12_000_000, 250_000, 20_000, 500_000, 9, 240, f"{title_prefix} One - Official Music Video #trending", "2023-12-27T18:00:00Z"),
        ("vid_2", "2024-01-01", 50_000, 1_500, 120, 20_000, 10, 200, f"{title_prefix} Two - freestyle", "2023-12-22T10:00:00Z"),
        ("vid_2", "2024-01-05", 120_000, 3_000, 200, 20_000, 14, 200, f"{title_prefix} Two - freestyle", "2023-12-22T10:00:00Z"),
        ("vid_3", "2024-01-01", 25_000_000, 800_000, 60_000, 8_000_000, 30, 300, f"{title_prefix} Three (Official)", "2023-12-02T14:00:00Z"),
        ("vid_3", "2024-01-05", 40_000_000, 1_200_000, 90_000, 8_000_000, 34, 300, f"{title_prefix} Three (Official)", "2023-12-02T14:00:00Z"),
        ("vid_4", "2024-01-01", 800_000, 20_000, 1_500, 90_000, 3, 45, f"{title_prefix} Four", "2023-12-29T20:00:00Z"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "video_id", "collected_at", "view_count", "like_count",
            "comment_count", "subscriber_count", "day_since_published",
            "duration_seconds", "title", "published_at",
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
