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
def create_future_labels(df,future_steps=3):
    if df.empty:
        print("DataFrame is empty. Cannot create future labels.")
        return pd.DataFrame()

    df=df.sort_values(
        by=["video_id","collected_at"]
        ).reset_index(drop=True)
    
    df["future_is_viral"]=df.groupby("video_id")["is_viral"].shift(-future_steps)

    df=df.dropna(subset=["future_is_viral"])

    df["future_is_viral"]=df["future_is_viral"].astype(int)

    return df
def save_future_labeled_data(
        df,
        file_path="data/processed/future_labeled_music_virality_data.csv"):
    if df.empty:
        print("DataFrame is empty. Nothing to save.")
        return pd.DataFrame()
    df.to_csv(file_path,index=False)
    print(f"Future labeled data saved.")