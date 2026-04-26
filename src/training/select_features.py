from data.processed import final_labelled_music_virality_data

data=final_labelled_music_virality_data
df=data.copy()
FEATURE_COLUMNS = [
    df["views_growth_rate"],
    df["views_per_day"],
    df["engagement_rate"],
    df["like_rate"],
    df["comment_rate"],
    df["comment_like_ratio"],
    df["views_acceleration"],
    df["view_count_log"]
]

TARGET = df["is_viral"]