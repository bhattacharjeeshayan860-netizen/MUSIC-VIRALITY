"""Data collection pipeline for music virality system."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.youtube_client import YouTubeClient
from config.config import QUERIES
import pandas as pd
import time
from datetime import datetime 

client = YouTubeClient()
blocks=[] #to store data for each queries
total_input_count = len(QUERIES)
total_videos_collected = 0

print(f"Starting data collection for {total_input_count} queries...\n")

for query_idx, query in enumerate(QUERIES, 1):
    print(f"[{query_idx}/{total_input_count}] Searching for {query}...")
    videos = client.fetch_music_videos(query=query, max_results=20, order="date", max_pages=3)
    if not videos:
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
    #merge data
    for video in videos:
        video_id=video["video_id"]
        video_stat=stats_dict.get(video_id, {})
        channel_stat=channel_dict.get(video["channel_id"], {})
        blocks.append({"query": query,
                       "video_id": video_id,
                       "title": video["title"],
                       "published_at": video["published_at"],
                       "channel_id": video["channel_id"],
                       "channel_title": video["channel_title"],
                       "view_count": video_stat.get("view_count",0),
                       "like_count": video_stat.get("like_count",0),
                       "comment_count": video_stat.get("comment_count",0),
                       "subscriber_count": channel_stat.get("subscriber_count",0),
                       "duration": video_stat.get("duration"),
                       "collected_at": datetime.now().isoformat()
                       })
        total_videos_collected += 1
    print(f"Collected {len(videos)} videos for query: {query}")


#save to csv
df=pd.DataFrame(blocks)

print(f"\n{'='*50}")
print(f"Data Collection Summary:")
print(f"Input queries processed: {total_input_count}")
print(f"Total videos collected: {total_videos_collected}")
print(f"{'='*50}\n")

df.to_csv("data/raw/music_virality_data.csv",
          index=False,
          mode="a",
          header=not pd.io.common.file_exists("data/raw/music_virality_data.csv"),
        )
print(f"Data saved to data/raw/music_virality_data.csv")
        