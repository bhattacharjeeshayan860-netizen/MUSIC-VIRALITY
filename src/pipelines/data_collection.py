"""Data collection pipeline for music virality system."""
from src.api.youtube_client import YouTubeClient
from config.config import QUERIES
import pandas as pd
import time
client = YouTubeClient()
blocks=[] #to store data for each queries
for query in QUERIES:
    print(f"searching for {query}...")
    videos= client.fetch_music_videos(query=query)
    if not videos:
        continue
    time.sleep(0.3) #to respect API rate limits
    video_ids=[v["video_id"] for v in videos]
    video_stats=client.fetch_video_details(video_ids=video_ids)
    stats_dict={s["video_id"]: s for s in video_stats}
    time.sleep(0.3) #to respect API rate limits
    channel_ids=list(set(v["channel_id"] for v in videos))
    channel_stats=client.fetch_channel_details(channel_ids=channel_ids)
    channel_dict={ch["channel_id"]: ch for ch in channel_stats}
    time.sleep(0.3) #to respect API rate limits
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
                       "subscriber_count": channel_stat.get("subscriber_count",0)
                       })
#save to csv
df=pd.DataFrame(blocks)
df.to_csv("data/raw/music_virality_data.csv",
          index=False,
          mode="a",
          header=not pd.io.common.file_exists("data/raw/music_virality_data.csv"),
        )
        