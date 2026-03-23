"""YouTube API client for fetching music data."""
import os
from dotenv import load_dotenv
import requests

BASE_URL="https://www.googleapis.com/youtube/v3"
load_dotenv()
api_key=os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("Youtube API key not found.please set YOUTUBE_API_KEY in .env")
class YouTubeClient:
    def __init__(self):
        self.base_url = BASE_URL
        self.api_key = api_key

    def fetch_music_videos(self, query,max_results=10):
        """Search videos and return CLEAN list of dicts"""
        try:
            url = self.base_url + "/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": self.api_key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            videos=[]
            for item in data.get("items",[]):
                videos.append({
                    "video_id":item["id"]["videoId"],
                    "title":item["snippet"]["title"],
                    "channel_id":item["snippet"]["channelId"],
                    "channel_title":item["snippet"]["channelTitle"],
                    "published_at":item["snippet"]["publishedAt"]
                })
            return videos
        except requests.RequestException as e:
            print(f"Error fetching music videos: {e}")
            return None
    def fetch_video_details(self,video_id):
        return 0
    def fetch_channel_details(self,channel_id):
        return 0


    
    