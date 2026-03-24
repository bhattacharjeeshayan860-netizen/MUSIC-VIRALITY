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
                    "video_id":item.get("id",{}).get("videoId"),
                    "title":item.get("snippet",{}).get("title"),
                    "channel_id":item.get("snippet",{}).get("channelId"),
                    "channel_title":item.get("snippet",{}).get("channelTitle"),
                    "published_at":item.get("snippet",{}).get("publishedAt")
                })
            return videos
        except requests.RequestException as e:
            print(f"Error fetching music videos: {e}")
            return []
        

    def fetch_video_details(self,video_ids):
        """Get views, likes, comments"""
        url=self.base_url+"/videos"
        params={"part":"statistics",
                "id":",".join(video_ids),
                "key":self.api_key,
                }
        try:
            response=requests.get(url,params=params)
            response.raise_for_status()
            data=response.json()
            stats=[]
            for item in data.get("items",[]):
                stats.append({
                    "video_id":item.get("id",{}).get("videoId"),
                    "view_count":int(item.get("statistics",{}).get("viewCount",0)),
                    "like_count":int(item.get("statistics",{}).get("likeCount",0)),
                    "comment_count":int(item.get("statistics",{}).get("commentCount",0))
                })
            
            return stats
        except requests.RequestException as e:
            print(f"Error fetching video details: {e}")
            return []
    def fetch_channel_details(self,channel_ids):
        """Get subscriber count"""
        url=self.base_url+"/channels"
        params={"part":"statistics",
                "id":",".join(channel_ids),
                "key":self.api_key,}
        try:
            response=requests.get(url,params=params)
            response.raise_for_status()
            data=response.json()
            channels=[]
            for item in data.get("items",[]):
                channels.append({
                    "channel_id":item.get("id"),
                    "subscriber_count":int(item.get("statistics",{}).get("subscriberCount",0))
                })
            return channels
        except requests.RequestException as e:
            print(f"Error fetching channel details:{e}")
        return []


    
    