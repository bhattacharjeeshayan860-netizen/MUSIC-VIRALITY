from src.api.youtube_client import YouTubeClient
client=YouTubeClient()
videos=client.fetch_music_videos("Taylor Swift")
print(videos[:])
print(len(videos))
