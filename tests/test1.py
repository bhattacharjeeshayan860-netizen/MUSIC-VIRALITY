from src.api.youtube_client import YouTubeClient
client=YouTubeClient()
videos=client.fetch_music_videos("Taylor Swift")
print(videos[:5])
print(len(videos))
