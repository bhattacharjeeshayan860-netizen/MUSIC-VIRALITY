import os

import pandas as pd
import numpy as np


FEATURE_COLS = [
    "views_growth_rate",
    "views_per_day",
    "engagement_rate",
    "like_rate",
    "comment_rate",
    "comment_like_ratio",
    "views_acceleration",
    "view_count_log",
]


def load_labeled_data(file_path: str = "data/processed/final_labelled_music_virality_data.csv") -> pd.DataFrame:
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    print(f"file {file_path} does not exist.")
    return pd.DataFrame()


def get_X_y(file_path: str = "data/processed/final_labelled_music_virality_data.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = load_labeled_data(file_path=file_path)
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype="int64")

    missing = [c for c in FEATURE_COLS + ["is_viral"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[FEATURE_COLS + ["is_viral"]].replace([np.inf, -np.inf], np.nan).dropna()
    X = data[FEATURE_COLS]
    y = data["is_viral"].astype(int)
    return X, y


# Backwards-compatible names (kept to avoid breaking existing imports)
FEATURE_COLUMNS = pd.DataFrame(columns=FEATURE_COLS)
TARGET = pd.Series(dtype="int64")