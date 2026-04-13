import pandas as pd
import numpy as np
import os

def load_data(file_path="data/processed/clean_music_virality_data.csv"):
    if os.path.exists(file_path):
        new_df = pd.read_csv(file_path)
        df = new_df.copy()
        return df
    else:
        print(f"file {file_path} does not exist.")
        return pd.DataFrame()


def create_eng_features(df):

    views = df["view_count"]
    likes = df["like_count"]
    comments = df["comment_count"]
    days = df["day_since_published"]

    # -------------------- Ratio Features --------------------

    df["like_rate"] = likes / views
    df["comment_rate"] = comments / views
    df["engagement_rate"] = (likes + 2 * comments) / views
    df["comment_like_ratio"] = comments / likes

    # handle edge cases
    df.loc[views == 0, ["like_rate", "comment_rate", "engagement_rate"]] = np.nan
    df.loc[likes == 0, "comment_like_ratio"] = np.nan

    # -------------------- Scale Features --------------------

    df["view_count_log"] = np.log1p(views)
    df["like_count_log"] = np.log1p(likes)
    df["comment_count_log"] = np.log1p(comments)

    df["engagement_log"] = np.log1p(likes + comments) / np.log1p(views)
    df.loc[views == 0, "engagement_log"] = np.nan

    # -------------------- Velocity Features --------------------

    df["likes_per_day"] = likes / days
    df["comments_per_day"] = comments / days
    df["views_per_day"] = views / days

    df.loc[days == 0, ["likes_per_day", "comments_per_day", "views_per_day"]] = np.nan

    # -------------------- Sort for Time Series --------------------

    df = df.sort_values(by=["video_id", "collected_at"]).reset_index(drop=True)

    # -------------------- Growth Features --------------------

    df["views_diff"] = df.groupby("video_id")["view_count"].diff()
    df["likes_diff"] = df.groupby("video_id")["like_count"].diff()
    df["comments_diff"] = df.groupby("video_id")["comment_count"].diff()

    df["previous_views"] = df.groupby("video_id")["view_count"].shift(1)

    df["views_growth_rate"] = df["views_diff"] / df["previous_views"]
    df.loc[df["previous_views"] == 0, "views_growth_rate"] = np.nan

    # optional safety (recommended)
    df["views_growth_rate"] = df["views_growth_rate"].replace([np.inf, -np.inf], np.nan)

    # -------------------- Trend Momentum --------------------

    df["views_acceleration"] = df.groupby("video_id")["views_diff"].diff()

    return df
def save_feature_engineered_data(df):
    df.to_csv("data/processed/feature_engineered_music_virality_data.csv",index=False)

def run_feature_engineering_pipeline():
    df=load_data()
    if df.empty:
        print("no data to process.")
        return pd.DataFrame()
    else:
        df=create_eng_features(df)
        save_feature_engineered_data(df)