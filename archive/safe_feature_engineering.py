import os

import numpy as np
import pandas as pd


def load_data(file_path: str = "data/processed/clean_music_virality_data.csv") -> pd.DataFrame:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["collected_at"] = pd.to_datetime(df.get("collected_at"), errors="coerce")
        df["published_at"] = pd.to_datetime(df.get("published_at"), errors="coerce")
        return df

    print(f"File {file_path} does not exist.")
    return pd.DataFrame()


def create_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features that DON'T leak the target.

    These are features available BEFORE a video goes viral.
    """

    # ==================== TEMPORAL FEATURES (SAFE) ====================
    df["hour_published"] = df["published_at"].dt.hour
    df["day_of_week"] = df["published_at"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["published_at"].dt.month

    # Publishing strategy (prime time = 6pm-10pm)
    df["is_prime_time"] = df["hour_published"].between(18, 22).astype(int)

    # ==================== CHANNEL FEATURES (SAFE) ====================
    df["subscriber_count_log"] = np.log1p(df["subscriber_count"])

    # Channel authority score
    df["channel_authority"] = df["subscriber_count"] / (df["subscriber_count"].max() + 1)

    # ==================== VIDEO METADATA (SAFE) ====================
    df["duration_minutes"] = df["duration"] / 60.0
    df["duration_log"] = np.log1p(df["duration"])

    # Optimal duration bins (music videos typically 3-5 min)
    df["is_optimal_duration"] = df["duration_minutes"].between(3, 5).astype(int)
    df["is_short_form"] = (df["duration_minutes"] < 3).astype(int)
    df["is_long_form"] = (df["duration_minutes"] > 6).astype(int)

    # ==================== TITLE FEATURES (SAFE) ====================
    df["title_length"] = df["title"].astype(str).str.len()
    df["title_word_count"] = df["title"].astype(str).str.split().str.len()
    df["avg_word_length"] = df["title_length"] / (df["title_word_count"] + 1)

    # Attention-grabbing patterns
    df["has_emoji"] = df["title"].astype(str).str.contains(r"[😀-🙏🌀-🗿🚀-🛿]", regex=True, na=False).astype(int)
    df["has_numbers"] = df["title"].astype(str).str.contains(r"\d", regex=True, na=False).astype(int)
    df["has_special_chars"] = df["title"].astype(str).str.contains(r"[!?*#]", regex=True, na=False).astype(int)
    df["all_caps_ratio"] = df["title"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )

    # Common viral keywords (customize for music)
    viral_keywords = ["official", "mv", "music video", "ft", "feat", "remix", "cover", "live", "acoustic"]
    df["has_viral_keyword"] = df["title"].astype(str).str.lower().str.contains("|".join(viral_keywords), na=False).astype(int)

    # ==================== EARLY ENGAGEMENT (SAFE - First 24hrs only) ====================
    # Only use metrics from first snapshot (day 0-1)
    df_sorted = df.sort_values(["video_id", "collected_at"]).reset_index(drop=True)

    # Get FIRST snapshot per video (early signal)
    first_snapshot = df_sorted.groupby("video_id").first().reset_index()
    first_snapshot = first_snapshot.rename(
        columns={
            "view_count": "early_views",
            "like_count": "early_likes",
            "comment_count": "early_comments",
        }
    )

    df = df.merge(
        first_snapshot[["video_id", "early_views", "early_likes", "early_comments"]],
        on="video_id",
        how="left",
    )

    # Early engagement ratios (first 24hr signal)
    df["early_like_rate"] = df["early_likes"] / (df["early_views"] + 1)
    df["early_comment_rate"] = df["early_comments"] / (df["early_views"] + 1)
    df["early_engagement_score"] = (df["early_likes"] + 2 * df["early_comments"]) / (df["early_views"] + 1)

    # Log scale early metrics
    df["early_views_log"] = np.log1p(df["early_views"])
    df["early_likes_log"] = np.log1p(df["early_likes"])

    # ==================== INTERACTION PATTERNS (SAFE) ====================
    df["early_comment_like_ratio"] = df["early_comments"] / (df["early_likes"] + 1)

    # Virality indicator: high comment-to-like ratio = discussion = viral potential
    df["is_high_discussion"] = (df["early_comment_like_ratio"] > df["early_comment_like_ratio"].median()).astype(int)

    return df


def save_safe_features(df: pd.DataFrame, output_path: str = "data/processed/safe_featured_data.csv"):
    df.to_csv(output_path, index=False)
    print(f"Safe features saved to {output_path}")


def run_safe_feature_engineering_pipeline(
    input_path: str = "data/processed/clean_music_virality_data.csv",
    output_path: str = "data/processed/safe_featured_data.csv",
) -> pd.DataFrame:
    df = load_data(file_path=input_path)
    if df.empty:
        print("No data to process.")
        return pd.DataFrame()

    df = create_safe_features(df)
    save_safe_features(df, output_path=output_path)
    print(f"✅ Safe feature engineering complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_safe_feature_engineering_pipeline()
