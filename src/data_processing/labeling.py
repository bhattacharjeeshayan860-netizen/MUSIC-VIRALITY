import os
import pandas as pd

def load_data(file_path="data/processed/feature_engineered_music_virality_data.csv"):
    if os.path.exists(file_path):
        new_df = pd.read_csv(file_path)
        df = new_df.copy()
        return df
    else:
        print(f"file {file_path} does not exist.")
        return pd.DataFrame()
