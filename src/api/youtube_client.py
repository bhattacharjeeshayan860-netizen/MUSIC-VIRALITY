"""YouTube API client for fetching music data."""
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import json
import random
import time
from typing import Optional

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

        # Throttling / retry configuration (tunable via env vars)
        self.search_min_interval_seconds = float(os.getenv("YOUTUBE_SEARCH_MIN_INTERVAL_SECONDS", "7"))
        self.videos_min_interval_seconds = float(os.getenv("YOUTUBE_VIDEOS_MIN_INTERVAL_SECONDS", "1"))
        self.channels_min_interval_seconds = float(os.getenv("YOUTUBE_CHANNELS_MIN_INTERVAL_SECONDS", "1"))
        self.max_retries = int(os.getenv("YOUTUBE_MAX_RETRIES", "6"))
        self.backoff_base_seconds = float(os.getenv("YOUTUBE_BACKOFF_BASE_SECONDS", "2"))
        self.backoff_max_seconds = float(os.getenv("YOUTUBE_BACKOFF_MAX_SECONDS", "60"))

        self._last_request_monotonic_by_endpoint = {
            "search": 0.0,
            "videos": 0.0,
            "channels": 0.0,
        }
    
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

    def _get_min_interval(self, endpoint: str) -> float:
        if endpoint == "search":
            return self.search_min_interval_seconds
        if endpoint == "videos":
            return self.videos_min_interval_seconds
        if endpoint == "channels":
            return self.channels_min_interval_seconds
        return 0.0

    def _throttle(self, endpoint: str) -> None:
        min_interval = self._get_min_interval(endpoint)
        if min_interval <= 0:
            return
        last_t = self._last_request_monotonic_by_endpoint.get(endpoint, 0.0)
        now = time.monotonic()
        elapsed = now - last_t
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_monotonic_by_endpoint[endpoint] = time.monotonic()

    def _parse_error_reason(self, response) -> Optional[str]:
        try:
            data = response.json()
            error = data.get("error", {})
            errors = error.get("errors", [])
            if errors:
                return errors[0].get("reason")
        except Exception:
            return None
        return None

    def _should_retry(self, response, endpoint: str) -> bool:
        if response is None:
            return True

        if response.status_code in (500, 502, 503, 504, 429):
            return True

        # YouTube sometimes uses 403 for quota/rate-limit issues.
        if response.status_code == 403:
            reason = self._parse_error_reason(response)
            if reason in {
                "rateLimitExceeded",
                "userRateLimitExceeded",
                "quotaExceeded",
                "dailyLimitExceeded",
                "accessNotConfigured",
            }:
                return True
        return False

    def _retry_sleep_seconds(self, response, attempt: int) -> float:
        retry_after = None
        try:
            if response is not None:
                raw = response.headers.get("Retry-After")
                if raw:
                    retry_after = float(raw)
        except Exception:
            retry_after = None

        if retry_after is not None and retry_after > 0:
            base = retry_after
        else:
            base = self.backoff_base_seconds * (2 ** max(0, attempt - 1))

        base = min(base, self.backoff_max_seconds)
        jitter = random.uniform(0, max(0.1, base * 0.25))
        return min(base + jitter, self.backoff_max_seconds)

    def _get_with_retry(self, endpoint: str, url: str, params: dict):
        """GET with per-endpoint throttling and retry/backoff on rate-limit/transient errors."""
        last_response = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle(endpoint)
            try:
                response = requests.get(url, params=params, timeout=30)
                last_response = response

                # Non-retryable client error
                if response.status_code == 400:
                    self._handle_api_error(response, endpoint)
                    print(f"Request params: {_mask_key(params)}")
                    return None

                if response.ok:
                    return response

                if self._should_retry(response, endpoint) and attempt < self.max_retries:
                    # Print diagnostics once per failed attempt
                    self._handle_api_error(response, endpoint)
                    sleep_s = self._retry_sleep_seconds(response, attempt)
                    print(f"[Info] Retrying {endpoint} after {sleep_s:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(sleep_s)
                    continue

                # Not retryable or out of retries
                self._handle_api_error(response, endpoint)
                return None

            except requests.RequestException:
                if attempt < self.max_retries:
                    sleep_s = self._retry_sleep_seconds(last_response, attempt)
                    print(f"[Info] Network error; retrying {endpoint} after {sleep_s:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(sleep_s)
                    continue
                return None

        return None
        
        
    def fetch_music_videos(
        self,
        query,
        max_results=10,
        max_pages=1,
        order="relevance",
        region_code=None,
        published_after=None,
    ):
        """Search videos and return CLEAN list of dicts"""
        next_page_token = None
        videos = []
        page_counter = 0
        if published_after is None:
            published_after = (datetime.utcnow() - timedelta(days=14)).strftime('%Y-%m-%dT%H:%M:%SZ')

        url = self.base_url + "/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "order": order,  # relevance, date, rating, title, videoCount, viewCount
            "publishedAfter": published_after,
            "key": self.api_key,
        }

        if not published_after:
            params.pop("publishedAfter", None)

        if region_code:
            params["regionCode"] = region_code

        while True:
            if next_page_token:
                params["pageToken"] = next_page_token

            response = self._get_with_retry("search", url, params)
            if response is None:
                print("Error fetching music videos: failed request on search")
                return []

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

            response = self._get_with_retry("videos", url, params)
            if response is None:
                print("Error fetching video details batch: failed request on videos")
                continue

            data = response.json()
            for item in data.get("items", []):
                stats.append({
                    "video_id": item.get("id"),
                    "view_count": int(item.get("statistics", {}).get("viewCount", 0)),
                    "like_count": int(item.get("statistics", {}).get("likeCount", 0)),
                    "comment_count": int(item.get("statistics", {}).get("commentCount", 0)),
                    "duration": item.get("contentDetails", {}).get("duration")
                })
        
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

            response = self._get_with_retry("channels", url, params)
            if response is None:
                print("Error fetching channel details batch: failed request on channels")
                continue

            data = response.json()
            for item in data.get("items", []):
                channels.append({
                    "channel_id": item.get("id"),
                    "subscriber_count": int(item.get("statistics", {}).get("subscriberCount", 0))
                })
        
        return channels


    
    