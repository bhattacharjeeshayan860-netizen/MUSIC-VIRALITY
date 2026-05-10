import pandas as pd
import os
import numpy as np
FEATURE_COLUMNS=[
    "views_growth_rate",
    "views_per_day",
    "engagement_rate",
    "views_acceleration",
    "like_rate",
    "comment_rate",
    "comment_like_ratio",
]
def load_future_labeled_data(file_path: str = "data/processed/future_labeled_music_virality_data.csv") -> pd.DataFrame:
    if os.path.exists(file_path=file_path):
        new_df=pd.read_csv(file_path)
        df=new_df.copy()
        return df
    else:
        print(f"file {file_path} does not exist.")
        return pd.DataFrame()
def get_X_y(file_path: str= "data/processed/future_labeled_music_virality_data.csv") -> tuple[pd.DataFrame,pd.Series]:
    df=load_future_labeled_data(file_path=file_path)
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype="int64")
    missing=[c for c in FEATURE_COLUMNS + ["future_is_viral"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data=df[FEATURE_COLUMNS + ["future_is_viral"]].replace([np.inf, -np.inf], np.nan).dropna()
    X=data[FEATURE_COLUMNS]
    y=data["future_is_viral"].astype("int")
    return X, y