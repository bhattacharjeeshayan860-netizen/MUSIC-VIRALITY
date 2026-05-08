import pandas as pd
import os
def load_data(file_path="data/processed/laelen_music_virality_data.csv"):
    if os.path.exists(file_path):
        new_df = pd.read_csv(file_path)
        df = new_df.copy()
        df["collected_at"]=pd.to_datetime(df["collected_at"],errors="coerce")
        return df
    else:
        print(f"file {file_path} does not exist.")
        return pd.DataFrame()
def create_future_labels(df,shift_steps=3):
    df=df.sort_values(by=["video_id","collected_at"])
    df["future_is_viral"]=df.groupby("video_id")["is_viral"].shift(-shift_steps)
    df=df.dropna(subset=["future_is_viral"])
    df["future_is_viral"]=df["future_is_viral"].astype(int)
    return df