import pandas as pd
import numpy as np
import os
import re


# ─────────────────────────────────────────────
# RAW LOAD
# ─────────────────────────────────────────────

def load_raw_data(file_path="data/raw/music_virality_data.csv"):
    data = pd.read_csv(file_path)
    new_df = data.copy()
    return new_df


# ─────────────────────────────────────────────
# CLEANING STEPS
# ─────────────────────────────────────────────

def handle_missing_values(new_df):
    print("NA counts for each column:")
    print(new_df.isna().sum())
    new_df = new_df.dropna(subset=["video_id", "collected_at", "view_count"])
    return new_df


def convert_datetime(new_df):
    new_df["collected_at"] = pd.to_datetime(new_df["collected_at"], errors="coerce").dt.tz_localize(None)
    new_df["published_at"] = pd.to_datetime(new_df["published_at"], errors="coerce").dt.tz_localize(None)
    return new_df


def create_date_column(new_df):
    new_df["date"] = new_df["collected_at"].dt.date
    new_df["day_since_published"] = (new_df["collected_at"] - new_df["published_at"]).dt.days
    return new_df


def parse_duration_seconds(new_df):
    """
    Parse ISO 8601 duration strings (e.g. 'PT3M5S') into total seconds.
    WHY: Duration is a raw signal — short videos (reels/shorts < 60s) behave
    very differently from full music videos (3-5 min). Raw string is useless to ML.
    """
    def _iso_to_seconds(s):
        if pd.isna(s):
            return np.nan
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(s))
        if not match:
            return np.nan
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        sec = int(match.group(3) or 0)
        return h * 3600 + m * 60 + sec

    new_df["duration_seconds"] = new_df["duration"].apply(_iso_to_seconds)
    return new_df


def remove_noise_duplicates(new_df):
    new_df = new_df.sort_values(by=["video_id", "collected_at"])
    new_df = new_df.groupby(["video_id", "date"]).last().reset_index()
    return new_df


def sort_date(new_df):
    new_df = new_df.sort_values(by=["date", "video_id"])
    return new_df


# ─────────────────────────────────────────────
# EXISTING DATA MERGE (incremental collection)
# ─────────────────────────────────────────────

def load_existing_data(file_path="data/processed/clean_music_virality_data.csv"):
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        existing_df = convert_datetime(existing_df)
        existing_df = create_date_column(existing_df)
        existing_df = remove_noise_duplicates(existing_df)
        existing_df = sort_date(existing_df)
        return existing_df
    else:
        return pd.DataFrame()


def merge_existing_data(new_df, existing_df):
    if not existing_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
        df = convert_datetime(df)
        df = create_date_column(df)
        df = df.sort_values(by=["video_id", "collected_at"])
        df = df.groupby(["video_id", "date"]).last().reset_index()
        df = sort_date(df)
        return df
    else:
        return new_df


def save_cleaned_data(df, file_path="data/processed/clean_music_virality_data.csv"):
    df.to_csv(file_path, index=False)
    print(f"Saved cleaned data -> {file_path} ({len(df)} rows)")


# ─────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────

def run_cleaning_pipeline(
    raw_path="data/raw/music_virality_data.csv",
    output_path="data/processed/clean_music_virality_data.csv",
):
    new_df = load_raw_data(raw_path)
    new_df = handle_missing_values(new_df)
    new_df = convert_datetime(new_df)
    new_df = create_date_column(new_df)
    new_df = parse_duration_seconds(new_df)
    new_df = remove_noise_duplicates(new_df)
    new_df = sort_date(new_df)
    existing_df = load_existing_data(output_path)
    df = merge_existing_data(new_df, existing_df)
    save_cleaned_data(df, output_path)
    return df