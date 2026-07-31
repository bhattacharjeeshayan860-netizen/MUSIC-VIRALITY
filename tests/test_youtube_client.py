"""
Tests for src/api/youtube_client.py using mocked HTTP responses.

These tests verify request construction, batching, throttling logic, and error
handling without ever calling the real YouTube Data API. This keeps the test
suite fast, deterministic, and free of API quota usage.
"""
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.api.youtube_client import YouTubeClient


@pytest.fixture
def client(monkeypatch):
    """A YouTubeClient with a dummy API key."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "TEST_KEY")
    return YouTubeClient()


def _make_response(json_data, status_code=200, headers=None):
    resp = Mock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_data
    resp.ok = 200 <= status_code < 300
    return resp


def test_fetch_music_videos_basic(client):
    """Search should parse items and return clean video dicts."""
    search_response = {
        "items": [
            {
                "id": {"videoId": "vid123"},
                "snippet": {
                    "title": "Test Song",
                    "channelId": "chan123",
                    "channelTitle": "Test Channel",
                    "publishedAt": "2024-01-01T00:00:00Z",
                },
            }
        ],
        "nextPageToken": None,
    }

    with patch("requests.get", return_value=_make_response(search_response)) as mock_get:
        videos = client.fetch_music_videos("test query", max_results=10, max_pages=1)

    assert len(videos) == 1
    assert videos[0]["video_id"] == "vid123"
    assert videos[0]["channel_id"] == "chan123"
    assert mock_get.call_count == 1
    called_params = mock_get.call_args[1]["params"]
    assert called_params["q"] == "test query"
    assert called_params["type"] == "video"


def test_fetch_video_details_batched(client):
    """Video details should be batched to respect the 50-id API limit."""
    ids = [f"v{i}" for i in range(55)]
    first_batch = {
        "items": [
            {"id": f"v{i}", "statistics": {"viewCount": str(i * 1000), "likeCount": str(i * 100), "commentCount": str(i * 10)}, "contentDetails": {"duration": "PT3M"}}
            for i in range(50)
        ]
    }
    second_batch = {
        "items": [
            {"id": f"v{i}", "statistics": {"viewCount": str(i * 1000), "likeCount": str(i * 100), "commentCount": str(i * 10)}, "contentDetails": {"duration": "PT3M"}}
            for i in range(50, 55)
        ]
    }

    responses = [_make_response(first_batch), _make_response(second_batch)]
    with patch("requests.get", side_effect=responses) as mock_get:
        details = client.fetch_video_details(ids)

    assert len(details) == 55
    assert mock_get.call_count == 2


def test_fetch_channel_details_batched(client):
    """Channel details should be batched to respect the 50-id API limit."""
    ids = [f"c{i}" for i in range(55)]
    first_batch = {"items": [{"id": f"c{i}", "statistics": {"subscriberCount": str(i * 1000)}} for i in range(50)]}
    second_batch = {"items": [{"id": f"c{i}", "statistics": {"subscriberCount": str(i * 1000)}} for i in range(50, 55)]}

    responses = [_make_response(first_batch), _make_response(second_batch)]
    with patch("requests.get", side_effect=responses) as mock_get:
        channels = client.fetch_channel_details(ids)

    assert len(channels) == 55
    assert mock_get.call_count == 2


def test_client_raises_without_api_key(monkeypatch):
    """A missing API key should raise a clear error on construction.

    We patch load_dotenv so a local .env file cannot mask the missing env var.
    """
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr("src.api.youtube_client.load_dotenv", lambda: None)
    with pytest.raises(ValueError, match="Youtube API key"):
        YouTubeClient()


def test_retry_on_rate_limit(client):
    """Requests that get a 429 should be retried with backoff."""
    error_response = _make_response({"error": {"errors": [{"reason": "rateLimitExceeded"}]}}, status_code=429, headers={"Retry-After": "1"})
    ok_response = _make_response({"items": []})

    with patch("requests.get", side_effect=[error_response, ok_response]) as mock_get:
        videos = client.fetch_music_videos("x", max_results=1, max_pages=1)

    assert videos == []
    assert mock_get.call_count == 2


def test_default_published_after_is_recent(client):
    """If no published_after is supplied, search should default to roughly 14 days ago."""
    search_response = {"items": [], "nextPageToken": None}
    with patch("requests.get", return_value=_make_response(search_response)) as mock_get:
        client.fetch_music_videos("x")

    called_params = mock_get.call_args[1]["params"]
    published_after = datetime.strptime(called_params["publishedAfter"], "%Y-%m-%dT%H:%M:%SZ")
    now = datetime.utcnow()
    delta = now - published_after
    assert timedelta(days=13) <= delta <= timedelta(days=15), f"delta was {delta}"
