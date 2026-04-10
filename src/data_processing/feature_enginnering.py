import pandas as pd
import numpy as np
import os
def load_data(file_path="data/processed/clean_music_virality_data.csv"):
    if os.path.exists(file_path):
        new_df=pd.read_csv(file_path)
        df=new_df.copy()
        return df
    else:
        print(f"file {file_path} does not exist.")
        return pd.DataFrame()
def create_eng_features(df):

    #Ratio Features

    df["like_rate"]=df["like_count"]/(df["view_count"]+1)
    df["comment_rate"]=df["comment_count"]/(df["view_count"]+1)
    df["engagement_rate"]=(df["like_count"]+2*df["comment_count"])/(df["view_count"]+1)
    df["comment_like_ratio"]=df["comment_count"]/(df["like_count"]+1)

    #Scale Features

    df["views_count_log"]=np.log1p(df["view_count"])
    df["likes_count_log"]=np.log1p(df["like_count"])
    df["comment_count_log"]=np.log1p(df["comment_count"])
    df["engagement_log"]=np.log1p(df["like_count"]+df["comment_count"])/np.log1p(df["view_count"])


    #Velocity Features


    df["likes_per_day"]=df["like_count"]/(df["day_since_published"]+1)
    df["comments_per_day"]=df["comment_count"]/(df["day_since_published"]+1)
    df["views_per_day"]=df["view_count"]/(df["day_since_published"]+1)

    #sorting values for growth features

    df=df.sort_values(by=["video_id","collected_at"]).reset_index(drop=True)

    #Ggrowth features


    df["views_diff"]=df.groupby("video_id")["view_count"].diff()
    df["likes_diff"]=df.groupby("video_id")["like_count"].diff()
    df["comments_diff"]=df.groupby("video_id")["comment_count"].diff()
    df[["views_diff","likes_diff","comments_diff"]]=df[["views_diff","likes_diff","comments_diff"]].fillna(0)
    df["views_growth_rate"]=df["views_diff"]/(df.groupby("video_id")["view_count"].shift(1) + 1)

    #Trend momentum

    df["views_acceleration"]=df.groupby("video_id")["views_diff"].diff()
    df["views_acceleration"]=df["views_acceleration"].fillna(0)


    return df