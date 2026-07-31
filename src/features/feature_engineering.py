import pandas as pd
import numpy as np
import os
import re
from datetime import datetime


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_data(file_path="data/processed/clean_music_virality_data.csv"):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.copy()
    else:
        print(f"File {file_path} does not exist.")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────

def create_ratio_features(df):
    """
    Engagement ratios — normalized by views so they're scale-invariant.
    Safe for both detection and prediction: they don't encode view_count directly.
    """
    views = df["view_count"]
    likes = df["like_count"]
    comments = df["comment_count"]

    df["like_rate"] = likes / views
    df["comment_rate"] = comments / views
    df["engagement_rate"] = (likes + 2 * comments) / views
    df["comment_like_ratio"] = comments / likes

    df.loc[views == 0, ["like_rate", "comment_rate", "engagement_rate"]] = np.nan
    df.loc[likes == 0, "comment_like_ratio"] = np.nan

    return df


def create_scale_features(df):
    """
    Log-transformed counts for heavy-tailed distributions.

    CRITICAL NOTE — view_count_log MUST NOT be used in the detection model.
    Reason: if is_viral = (view_count >= THRESHOLD), then log1p(view_count)
    perfectly separates the two classes at log1p(THRESHOLD). The model learns
    a trivial threshold, not real virality patterns.

    view_count_log is kept here for the prediction model, where the label
    is based on a FUTURE snapshot's views — not the current snapshot's views.
    Verify this is true before including it in prediction features.
    """
    views = df["view_count"]
    likes = df["like_count"]
    comments = df["comment_count"]

    # view_count_log: DO NOT include in detection FEATURE_COLS
    df["view_count_log"] = np.log1p(views)
    df["like_count_log"] = np.log1p(likes)
    df["comment_count_log"] = np.log1p(comments)

    # subscriber scale — useful channel authority signal
    if "subscriber_count" in df.columns:
        df["subscriber_count_log"] = np.log1p(df["subscriber_count"])

    return df


def create_velocity_features(df):
    """
    Per-day velocity features.
    These are legitimate signals for both detection and prediction.
    They capture momentum speed, not just cumulative size.

    WHY days+1: day_since_published=0 on upload day; avoid division by zero
    and artificially infinite velocity for day-0 uploads.
    """
    views = df["view_count"]
    likes = df["like_count"]
    comments = df["comment_count"]
    days = df["day_since_published"].clip(lower=1)  # clip at 1 to avoid /0

    df["views_per_day"] = views / days
    df["likes_per_day"] = likes / days
    df["comments_per_day"] = comments / days

    # Views per day relative to channel subscriber base — normalizes for artist size
    # A 1000-sub channel getting 500K views/day is more viral than a 50M-sub channel
    if "subscriber_count" in df.columns:
        subs = df["subscriber_count"].replace(0, np.nan)
        df["views_per_day_per_sub"] = df["views_per_day"] / subs

    return df


def create_channel_authority_features(df):
    """
    Channel-level features. Strong ML signal because:
    - Same view count means different things for a 300M-sub channel vs a 10K-sub channel
    - Viral = outperforming your channel's baseline, not just hitting a raw threshold
    """
    if "subscriber_count" not in df.columns:
        return df

    subs = df["subscriber_count"].replace(0, np.nan)
    df["views_to_subs_ratio"] = df["view_count"] / subs
    df["likes_to_subs_ratio"] = df["like_count"] / subs

    # Channel tier — interpretable categorical encoded as ordinal
    # WHY: helps model learn that virality thresholds differ by channel size
    bins = [0, 10_000, 100_000, 1_000_000, 10_000_000, float("inf")]
    labels = [0, 1, 2, 3, 4]  # nano, micro, mid, macro, mega
    df["channel_tier"] = pd.cut(
        df["subscriber_count"], bins=bins, labels=labels, right=True
    ).astype(float)

    return df


def create_duration_features(df):
    """
    Video duration signal.
    WHY: Shorts (<60s) have completely different viral mechanics than full music videos.
    Mixing them without a flag confuses the model.
    """
    if "duration_seconds" not in df.columns:
        return df

    df["is_short_video"] = (df["duration_seconds"] < 60).astype(float)
    df["is_standard_music_video"] = (
        (df["duration_seconds"] >= 120) & (df["duration_seconds"] <= 600)
    ).astype(float)
    # Log duration to handle wide range (1s to 10hr+ for livestreams)
    df["duration_log"] = np.log1p(df["duration_seconds"])

    return df


def create_time_context_features(df):
    """
    Video age and temporal context.
    WHY: A 3-year-old video with 5M views behaves differently from a 2-day-old video
    with 5M views. Age contextualizes all velocity signals.
    """
    if "day_since_published" not in df.columns:
        return df

    days = df["day_since_published"]
    df["age_bucket"] = pd.cut(
        days,
        bins=[-1, 7, 30, 90, 365, float("inf")],
        labels=[0, 1, 2, 3, 4],  # new, recent, growing, established, catalog
    ).astype(float)

    df["days_log"] = np.log1p(days)

    return df


def create_momentum_features(df):
    """
    Time-series diff features — require multi-snapshot data per video.
    Rows with only one snapshot will have NaN here — that is expected and correct.
    The imputer in the training pipeline handles NaN via fill_value=0.

    WHY sort first: diffs are meaningless unless rows are chronologically ordered.
    """
    df = df.sort_values(by=["video_id", "collected_at"]).reset_index(drop=True)

    df["views_diff"] = df.groupby("video_id")["view_count"].diff()
    df["likes_diff"] = df.groupby("video_id")["like_count"].diff()
    df["comments_diff"] = df.groupby("video_id")["comment_count"].diff()

    df["previous_views"] = df.groupby("video_id")["view_count"].shift(1)
    df["views_growth_rate"] = df["views_diff"] / df["previous_views"].replace(0, np.nan)
    df["views_growth_rate"] = df["views_growth_rate"].replace([np.inf, -np.inf], np.nan)

    df["views_acceleration"] = df.groupby("video_id")["views_diff"].diff()

    # Engagement momentum — are people engaging more with each snapshot?
    df["engagement_diff"] = df.groupby("video_id")["like_count"].diff() + \
                            2 * df.groupby("video_id")["comment_count"].diff()

    # Snapshot rank — which snapshot is this for the video (1st, 2nd, 3rd...)
    # WHY: early snapshots are noisier; later ones have more stable signal
    df["snapshot_rank"] = df.groupby("video_id").cumcount() + 1
    df["snapshot_count"] = df.groupby("video_id")["video_id"].transform("count")

    return df


def create_publish_time_features(df):
    """
    Publish-time metadata signals.

    WHY THESE ARE SAFE:
    - They are fixed at upload time and cannot leak future information.
    - The model can learn whether certain upload windows correlate with
      eventual virality (e.g., prime-time uploads, weekend music drops).

    WHY ADD THEM: the raw data has `published_at` but the pipeline was not
    using it at all. These give the model another views-agnostic signal that is
    available in a real early-warning setting.
    """
    if "published_at" not in df.columns:
        return df

    pub = pd.to_datetime(df["published_at"], errors="coerce").dt.tz_localize(None)

    # Hour / day-of-week are intuitive for release strategy
    df["publish_hour"] = pub.dt.hour
    df["publish_day_of_week"] = pub.dt.dayofweek  # 0=Mon ... 6=Sun
    df["publish_is_weekend"] = df["publish_day_of_week"].isin([5, 6]).astype(float)

    # Prime time for Indian/global music drops (18:00-22:00 local)
    df["publish_is_prime_time"] = pub.dt.hour.between(18, 22).astype(float)

    # Month can capture seasonal release campaigns / festivals
    df["publish_month"] = pub.dt.month.astype(float)

    return df


def create_title_features(df):
    """
    Title-text features.

    WHY THESE ARE SAFE:
    - A video's title is known at upload time; it cannot leak future views.
    - Title style (hashtags, emojis, caps, viral keywords) is a genuine
      marketing signal that correlates with how the artist/promoter expects
      the content to perform.

    WHY ADD THEM: the raw data has `title` but it was completely unused.
    Short-form and aggressively-titled content often has different viral
    mechanics than traditional music videos.
    """
    if "title" not in df.columns:
        return df

    title = df["title"].fillna("").astype(str)

    df["title_length"] = title.str.len().astype(float)
    df["title_word_count"] = title.str.split().str.len().astype(float)

    # Percentage of uppercase letters in the title (0-1, robust to empty titles)
    def _upper_ratio(t):
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    df["title_caps_ratio"] = title.apply(_upper_ratio).astype(float)

    # Attention-grabbing patterns
    # df["title_has_emoji"] = title.str.contains(r"[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]", regex=True, na=False).astype(float)
    df["title_has_number"] = title.str.contains(r"\d", regex=True, na=False).astype(float)
    df["title_has_special_char"] = title.str.contains(r"[!?*#]", regex=True, na=False).astype(float)
    df["title_has_hashtag"] = title.str.contains(r"#", regex=True, na=False).astype(float)

    # Common music-promotion keywords
    viral_keywords = [
        "official", "music video", "mv", "remix", "cover", "live", "acoustic",
        "ft", "feat", "ft.", "feat.", "new song", "latest", "trending", "viral",
    ]
    # Non-capturing group avoids the pandas .str.contains warning about regex groups.
    pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in viral_keywords) + r")\b"
    df["title_has_viral_keyword"] = (
        title.str.lower().str.contains(pattern, regex=True, na=False).astype(float)
    )

    return df


def create_extra_engagement_features(df):
    """
    Extra engagement/audience ratio features.

    WHY THESE ARE SAFE:
    - All are ratios or per-day rates derived from the same snapshot.
    - They do not include raw `view_count_log` (which would trivialize
      detection), but they do normalize views/likes/comments by channel size
    and age.

    WHY ADD THEM: the pre-viral slice needs stronger signals that are not
    dominated by absolute view count. Ratios like `comments_to_subs_ratio` and
    `engagement_per_day` measure *intensity* of audience reaction.
    """
    views = df["view_count"]
    likes = df["like_count"]
    comments = df["comment_count"]
    days = df["day_since_published"].clip(lower=1)

    # Engagement per day — total engagement activity normalised by age
    df["engagement_per_day"] = (likes + 2 * comments) / days

    # Inverse ratio: likes per comment (high = lots of likes, little discussion)
    comment_safe = comments.replace(0, np.nan)
    df["like_to_comment_ratio"] = (likes / comment_safe).replace([np.inf, -np.inf], np.nan)

    # Comments relative to subscriber base (channel-independent discussion intensity)
    if "subscriber_count" in df.columns:
        subs = df["subscriber_count"].replace(0, np.nan)
        df["comments_to_subs_ratio"] = comments / subs
        df["likes_to_subs_ratio"] = likes / subs
        df["engagement_to_subs_ratio"] = (likes + 2 * comments) / subs

    # Robust log versions of velocities (prediction-friendly; detection excludes raw views)
    df["likes_per_day_log"] = np.log1p(df["likes_per_day"])
    df["comments_per_day_log"] = np.log1p(df["comments_per_day"])
    df["engagement_per_day_log"] = np.log1p(df["engagement_per_day"])

    return df


def create_trajectory_features(df):
    """
    Historical trajectory features for videos with multiple snapshots.

    WHY THESE ARE SAFE:
    - They only use *previous* snapshots of the same video.
    - The first snapshot gets NaN/0 values, which the imputer fills.

    WHY ADD THEM: the current diff features only compare adjacent snapshots.
    These give a longer-horizon view of how the video has grown since it was
    first observed, which is especially useful for the prediction task.
    """
    if "collected_at" not in df.columns or "video_id" not in df.columns:
        return df

    df = df.sort_values(by=["video_id", "collected_at"]).reset_index(drop=True)

    first_snapshot = df.groupby("video_id").first()[["collected_at", "view_count"]]
    first_snapshot = first_snapshot.rename(
        columns={"collected_at": "first_snapshot_at", "view_count": "first_snapshot_views"}
    )
    df = df.merge(first_snapshot, on="video_id", how="left")

    df["days_since_first_snapshot"] = (
        (df["collected_at"] - df["first_snapshot_at"]).dt.total_seconds() / 86_400.0
    )

    first_views_safe = df["first_snapshot_views"].replace(0, np.nan)
    df["views_ratio_to_first"] = (df["view_count"] / first_views_safe).replace([np.inf, -np.inf], np.nan)

    df = df.drop(columns=["first_snapshot_at", "first_snapshot_views"])

    return df


# ─────────────────────────────────────────────
# MAIN FEATURE BUILDER
# ─────────────────────────────────────────────

def create_eng_features(df):
    """
    Run all feature groups in order.
    Preserves your original function name and return contract.
    """
    # Normalize timestamp columns once so downstream functions can rely on them
    # being datetime even when create_eng_features is called directly on raw data.
    for col in ["collected_at", "published_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

    df = create_ratio_features(df)
    df = create_scale_features(df)
    df = create_velocity_features(df)
    df = create_channel_authority_features(df)
    df = create_duration_features(df)
    df = create_time_context_features(df)
    df = create_extra_engagement_features(df)
    df = create_trajectory_features(df)
    df = create_publish_time_features(df)
    df = create_title_features(df)
    df = create_momentum_features(df)
    return df


def save_feature_engineered_data(df, file_path="data/processed/feature_engineered_music_virality_data.csv"):
    df.to_csv(file_path, index=False)
    print(f"Saved feature-engineered data -> {file_path} ({len(df)} rows)")


def run_feature_engineering_pipeline(
    input_path="data/processed/clean_music_virality_data.csv",
    output_path="data/processed/feature_engineered_music_virality_data.csv",
):
    df = load_data(input_path)
    if df.empty:
        print("No data to process.")
        return pd.DataFrame()

    df = create_eng_features(df)
    save_feature_engineered_data(df, output_path)
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df