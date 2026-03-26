"""YouTube API client for fetching music data."""
import os
from dotenv import load_dotenv
import requests

BASE_URL="https://www.googleapis.com/youtube/v3"

class YouTubeClient:
    def __init__(self):
        load_dotenv()
        self.base_url = BASE_URL
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("Youtube API key not found.please set YOUTUBE_API_KEY in .env")
        
        
    def fetch_music_videos(self, query, max_results=10, order="date", max_pages=1):
        """Search videos and return CLEAN list of dicts"""
        next_page_token = None
        videos = []
        page_counter = 0
        try:
            url = self.base_url + "/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "order": order,
                "key": self.api_key,
            }
            
            while True:
                if next_page_token:
                    params["pageToken"] = next_page_token
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                page_counter += 1
                
                for item in data.get("items", []):
                    videos.append({
                        "video_id": item.get("id", {}).get("videoId"),
                        "title": item.get("snippet", {}).get("title"),
                        "channel_id": item.get("snippet", {}).get("channelId"),
                        "channel_title": item.get("snippet", {}).get("channelTitle"),
                        "published_at": item.get("snippet", {}).get("publishedAt"),
                    })
                
                next_page_token = data.get("nextPageToken")
                if not next_page_token or page_counter >= max_pages:
                    break
                    
        except requests.RequestException as e:
            print(f"Error fetching music videos: {e}")
            return []
        return videos

    def fetch_video_details(self, video_ids):
        """Get views, likes, comments - batched to avoid URL limits"""
        if not video_ids:
            return []
        
        stats = []
        batch_size = 50  # YouTube API limit is ~50 IDs per request
        
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]
            url = self.base_url + "/videos"
            params = {
                "part": "statistics,contentDetails",
                "id": ",".join(batch),
                "key": self.api_key,
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                for item in data.get("items", []):
                    stats.append({
                        "video_id": item.get("id"),
                        "view_count": int(item.get("statistics", {}).get("viewCount", 0)),
                        "like_count": int(item.get("statistics", {}).get("likeCount", 0)),
                        "comment_count": int(item.get("statistics", {}).get("commentCount", 0)),
                        "duration": item.get("contentDetails", {}).get("duration")
                    })
            except requests.RequestException as e:
                print(f"Error fetching video details batch: {e}")
                continue
        
        return stats

    def fetch_channel_details(self, channel_ids):
        """Get subscriber count - batched to avoid URL limits"""
        if not channel_ids:
            return []
        
        channels = []
        batch_size = 50  # YouTube API limit is ~50 IDs per request
        
        for i in range(0, len(channel_ids), batch_size):
            batch = channel_ids[i:i + batch_size]
            url = self.base_url + "/channels"
            params = {
                "part": "statistics",
                "id": ",".join(batch),
                "key": self.api_key,
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                for item in data.get("items", []):
                    channels.append({
                        "channel_id": item.get("id"),
                        "subscriber_count": int(item.get("statistics", {}).get("subscriberCount", 0))
                    })
            except requests.RequestException as e:
                print(f"Error fetching channel details batch: {e}")
                continue
        
        return channels


    
    