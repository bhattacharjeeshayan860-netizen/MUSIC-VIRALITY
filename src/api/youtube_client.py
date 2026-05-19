"""YouTube API client for fetching music data."""
import os
from dotenv import load_dotenv
import requests
import json

BASE_URL="https://www.googleapis.com/youtube/v3"


def _mask_key(params: dict) -> dict:
    """Return a shallow-copied params dict with API key masked for safe logging."""
    if not isinstance(params, dict):
        return {}
    masked = dict(params)
    if "key" in masked and masked["key"]:
        masked["key"] = "***"
    return masked

class YouTubeClient:
    def __init__(self):
        load_dotenv()
        self.base_url = BASE_URL
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("Youtube API key not found.please set YOUTUBE_API_KEY in .env")
    
    def _handle_api_error(self, response, endpoint):
        """Handle YouTube API errors with detailed diagnostics"""
        try:
            error_data = response.json()
            error = error_data.get("error", {})
            error_code = error.get("code")
            error_message = error.get("message", "Unknown error")
            
            # Parse error details
            errors = error.get("errors", [])
            reason = errors[0].get("reason", "unknown") if errors else "unknown"
            domain = errors[0].get("domain", "unknown") if errors else "unknown"
            print(f"\n[Warning] YouTube API Error on {endpoint}:")
            print(f"   HTTP Status: {response.status_code}")
            if error_code is not None:
                print(f"   Code: {error_code}")
            print(f"   Reason: {reason}")
            print(f"   Domain: {domain}")
            print(f"   Message: {error_message}\n")
            return reason
        except:
            pass
        return None
        
        
    def fetch_music_videos(
        self,
        query,
        max_results=10,
        max_pages=1,
        order="relevance",
        region_code=None,
        published_after="2026-04-01T00:00:00Z",
    ):
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
                "order": order,  # Added: relevance, date, rating, title, videoCount, viewCount
                "publishedAfter": published_after,
                "key": self.api_key,
            }

            if not published_after:
                params.pop("publishedAfter", None)
            
            # Add optional region code if provided
            if region_code:
                params["regionCode"] = region_code
            
            while True:
                if next_page_token:
                    params["pageToken"] = next_page_token
                
                response = requests.get(url, params=params)
                
                # Handle 400 errors before raise_for_status()
                if response.status_code == 400:
                    self._handle_api_error(response, "search")
                    print(f"Request params: {_mask_key(params)}")
                    return []
                    
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
            # Avoid printing raw exception text; it may contain the full URL with API key.
            resp = getattr(e, "response", None)
            if resp is not None:
                self._handle_api_error(resp, "search")
                print(f"Error fetching music videos: HTTP {resp.status_code} on search")
            else:
                print(f"Error fetching music videos: {type(e).__name__}")
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
                
                # Handle 400 errors before raise_for_status()
                if response.status_code == 400:
                    self._handle_api_error(response, "videos")
                    print(f"Request params: {_mask_key(params)}")
                    continue
                    
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
                resp = getattr(e, "response", None)
                if resp is not None:
                    self._handle_api_error(resp, "videos")
                    print(f"Error fetching video details batch: HTTP {resp.status_code} on videos")
                else:
                    print(f"Error fetching video details batch: {type(e).__name__}")
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
                
                # Handle 400 errors before raise_for_status()
                if response.status_code == 400:
                    self._handle_api_error(response, "channels")
                    print(f"Request params: {_mask_key(params)}")
                    continue
                    
                response.raise_for_status()
                data = response.json()
                for item in data.get("items", []):
                    channels.append({
                        "channel_id": item.get("id"),
                        "subscriber_count": int(item.get("statistics", {}).get("subscriberCount", 0))
                    })
            except requests.RequestException as e:
                resp = getattr(e, "response", None)
                if resp is not None:
                    self._handle_api_error(resp, "channels")
                    print(f"Error fetching channel details batch: HTTP {resp.status_code} on channels")
                else:
                    print(f"Error fetching channel details batch: {type(e).__name__}")
                continue
        
        return channels


    
    