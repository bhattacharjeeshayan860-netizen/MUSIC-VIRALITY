import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# VIRALITY THRESHOLD
# ─────────────────────────────────────────────

# WHY 10M: At the 10M threshold, class balance is ~28% viral — workable for ML.
# Lower thresholds (1M) give 65% viral — not a useful signal in Indian music context
# where even mid-tier artists cross 1M easily. 10M filters for genuine breakout hits.
VIRALITY_THRESHOLD = 10_000_000

# Future label: how many snapshots ahead defines "future"?
# With 4 collection dates over 6 days, "future" means the LAST available snapshot
# for a given video. This is an honest constraint of the dataset.
FUTURE_SNAPSHOT_OFFSET = 1  # use next available snapshot as "future"


# ─────────────────────────────────────────────
# DETECTION LABEL
# is_viral: Is this video already viral AT THIS SNAPSHOT?
# No temporal leakage here — we're labeling the present state.
# ─────────────────────────────────────────────

def create_detection_labels(df, threshold=VIRALITY_THRESHOLD):
    """
    Label each snapshot row: is this video viral right now?

    No leakage risk: we label based on view_count at the same timestamp
    as the features. The model's job is to detect from ENGAGEMENT RATIOS,
    not from view_count itself.

    IMPORTANT: view_count_log must NOT appear in detection FEATURE_COLS.
    This label function is fine — the feature selection is where to be careful.
    """
    df = df.copy()
    df["is_viral"] = (df["view_count"] >= threshold).astype(int)
    viral_pct = df["is_viral"].mean()
    print(f"Detection label created. Threshold: {threshold:,}")
    print(f"Class balance: {viral_pct:.1%} viral ({df['is_viral'].sum():,} / {len(df):,} rows)")
    return df


# ─────────────────────────────────────────────
# FUTURE / PREDICTION LABEL
# future_is_viral: Will this video be viral at a LATER snapshot?
# ─────────────────────────────────────────────

def create_future_labels(df, threshold=VIRALITY_THRESHOLD):
    """
    For each video snapshot, look up what the LAST snapshot's view_count was.
    Label it future_is_viral if that final view_count >= threshold.

    Design rationale:
    - We use the LAST snapshot as the "future" target because data spans only 6 days.
    - A row is only included if it is NOT the last snapshot (i.e., there IS a future to predict).
    - Rows that are already the last snapshot are excluded — they have no future to label.

    Known limitation: 6 days is a short prediction horizon. This is documented honestly
    in the README. The model learns momentum signals that correlate with eventual size,
    not true early virality prediction. Adequate for a portfolio dataset.
    """
    df = df.copy()
    df = df.sort_values(["video_id", "collected_at"]).reset_index(drop=True)

    # For each video, get the LAST snapshot's view count as the future target
    last_views = df.groupby("video_id")["view_count"].last().rename("last_snapshot_views")
    df = df.merge(last_views, on="video_id", how="left")

    # future label: will this video eventually exceed the threshold?
    df["future_is_viral"] = (df["last_snapshot_views"] >= threshold).astype(int)

    # CRITICAL: remove rows that ARE the last snapshot — they have no "future"
    # WHY: if we train on the last snapshot with a label derived from itself,
    # that is direct target leakage for the prediction model.
    is_last = df.groupby("video_id").cumcount(ascending=False) == 0
    df = df[~is_last].copy()
    df = df.drop(columns=["last_snapshot_views"])

    viral_pct = df["future_is_viral"].mean()
    print(f"Future label created. Threshold: {threshold:,}")
    print(f"Rows after removing last snapshots: {len(df):,}")
    print(f"Class balance: {viral_pct:.1%} future viral")

    # Honest limitation check
    already_viral = (
        (df["view_count"] >= threshold) & (df["future_is_viral"] == 1)
    ).sum()
    total_future_viral = df["future_is_viral"].sum()
    if total_future_viral > 0:
        already_pct = already_viral / total_future_viral
        print(f"\n[NOTE] Prediction dataset note:")
        print(f"  {already_pct:.1%} of 'future viral' videos were ALREADY above threshold")
        print(f"  at the time of the training snapshot.")
        print(f"  This reflects the 6-day collection window constraint.")
        print(f"  Model learns stability/momentum signals, not pre-viral detection.")

    return df


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_labeled_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved -> {file_path} ({len(df)} rows)")


# ─────────────────────────────────────────────
# PIPELINE ENTRY POINTS
# ─────────────────────────────────────────────

def run_detection_labeling_pipeline(
    input_path="data/processed/feature_engineered_music_virality_data.csv",
    output_path="data/processed/final_labelled_music_virality_data.csv",
    threshold=VIRALITY_THRESHOLD,
):
    df = pd.read_csv(input_path)
    df = create_detection_labels(df, threshold=threshold)
    save_labeled_data(df, output_path)
    return df


def run_future_labeling_pipeline(
    input_path="data/processed/feature_engineered_music_virality_data.csv",
    output_path="data/processed/future_labeled_music_virality_data.csv",
    threshold=VIRALITY_THRESHOLD,
):
    df = pd.read_csv(input_path)
    df = create_future_labels(df, threshold=threshold)
    save_labeled_data(df, output_path)
    return df