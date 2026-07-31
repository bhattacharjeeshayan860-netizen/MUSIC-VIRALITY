"""Data collection pipeline for music virality system.

Run from the repo root with:
    python -m src.pipelines.data_collection

This module collects YouTube search results for the configured queries,
enriches them with video and channel statistics, and appends the rows to
`data/raw/music_virality_data.csv`. It skips videos already collected today
to avoid expensive duplicate API calls.
"""
import sys
import os

# Ensure repo root is importable when running this module directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.api.youtube_client import YouTubeClient
from config.config import QUERIES
import pandas as pd
import time
from datetime import datetime, date


def get_already_collected_today(existing_csv="data/raw/music_virality_data.csv"):
    """Returns set of video_ids already collected today — skip these."""
    try:
        df = pd.read_csv(existing_csv)
        df["collected_at"] = pd.to_datetime(df["collected_at"])
        collected_today = df[df["collected_at"].dt.date == date.today()]
        return set(collected_today["video_id"].tolist())
    except FileNotFoundError:
        return set()


def collect_for_queries(
    queries,
    client,
    existing_csv="data/raw/music_virality_data.csv",
    max_results=50,
    max_pages=3,
):
    """Collect videos for the given queries and return a DataFrame."""
    blocks = []
    total_input_count = len(queries)
    total_videos_collected = 0
    already_collected_today = get_already_collected_today(existing_csv)

    print(f"Starting data collection for {total_input_count} queries...\n")

    for query_idx, query in enumerate(queries, 1):
        print(f"[{query_idx}/{total_input_count}] Searching for {query}...")
        videos = client.fetch_music_videos(query=query, max_results=max_results, max_pages=max_pages)
        if not videos:
            continue

        # Skip videos already collected today (from prior runs and earlier queries in this run)
        videos = [v for v in videos if v.get("video_id") and v["video_id"] not in already_collected_today]
        if not videos:
            print(f"No new videos to collect for query: {query} (all already collected today)")
            continue

        time.sleep(1)  # Respect API rate limits
        video_ids = [v["video_id"] for v in videos if v["video_id"]]
        video_stats = client.fetch_video_details(video_ids=video_ids)
        stats_dict = {s["video_id"]: s for s in video_stats}
        time.sleep(1)  # Respect API rate limits
        channel_ids = list(set(v["channel_id"] for v in videos if v["channel_id"]))
        channel_stats = client.fetch_channel_details(channel_ids=channel_ids)
        channel_dict = {ch["channel_id"]: ch for ch in channel_stats}

        time.sleep(1)  # Respect API rate limits
        # merge data
        for video in videos:
            video_id = video["video_id"]
            video_stat = stats_dict.get(video_id, {})
            channel_stat = channel_dict.get(video["channel_id"], {})
            blocks.append({
                "query": query,
                "video_id": video_id,
                "title": video["title"],
                "published_at": video["published_at"],
                "channel_id": video["channel_id"],
                "channel_title": video["channel_title"],
                "view_count": video_stat.get("view_count", 0),
                "like_count": video_stat.get("like_count", 0),
                "comment_count": video_stat.get("comment_count", 0),
                "subscriber_count": channel_stat.get("subscriber_count", 0),
                "duration": video_stat.get("duration"),
                "collected_at": datetime.now().isoformat(),
            })
            total_videos_collected += 1
            already_collected_today.add(video_id)
        print(f"Collected {len(videos)} videos for query: {query}")

    df = pd.DataFrame(blocks)
    print(f"\n{'=' * 50}")
    print(f"Data Collection Summary:")
    print(f"Input queries processed: {total_input_count}")
    print(f"Total videos collected: {total_videos_collected}")
    print(f"{'=' * 50}\n")
    return df


def save_raw_data(df, existing_csv="data/raw/music_virality_data.csv"):
    """Append new rows to the raw CSV, creating it if it does not exist."""
    os.makedirs(os.path.dirname(existing_csv), exist_ok=True)
    df.to_csv(
        existing_csv,
        index=False,
        mode="a",
        header=not pd.io.common.file_exists(existing_csv),
    )
    print(f"Data saved to {existing_csv}")


def main():
    client = YouTubeClient()
    df = collect_for_queries(QUERIES, client)
    if not df.empty:
        save_raw_data(df)
    else:
        print("No new data collected.")


if __name__ == "__main__":
    main()
