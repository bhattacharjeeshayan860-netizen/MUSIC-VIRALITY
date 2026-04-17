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
    

def is_viral(df):
    required_cols = ["view_count_log", "views_growth_rate", "engagement_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    quantiles = [0.4, 0.5, 0.6,0.65, 0.7, 0.8]
    summary = {}

    for q in quantiles:
        view_threshold = df["view_count_log"].quantile(q)
        growth_threshold = df["views_growth_rate"].quantile(q)
        engagement_threshold = df["engagement_rate"].quantile(q)

        views_flag = df["view_count_log"] >= view_threshold
        growth_flag = df["views_growth_rate"] >= growth_threshold
        engagement_flag = df["engagement_rate"] >= engagement_threshold

        score = views_flag.astype(int) + growth_flag.astype(int) + engagement_flag.astype(int)

        col = f"is_viral_q{int(q * 100):02d}"
        df[col] = (score >= 2).astype(int)

        # Store normalized distribution (0/1) for this quantile
        summary[q] = df[col].value_counts(normalize=True).reindex([0, 1], fill_value=0.0)

    # Keep a single 'is_viral' column for downstream compatibility (use the strictest quantile).
    df["is_viral"] = df["is_viral_q80"]

    # Print a compact normalization table: rows=quantiles, cols=label(0/1)
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "quantile"
    summary_df.columns = ["not_viral(0)", "viral(1)"]
    print(summary_df)

    return df
def save_labeled_data(df):
    df.to_csv("data/processed/labeled_music_virality_data.csv",index=False)
    
def run_labeling_pipeline():
    df=load_data()
    df=is_viral(df)
    save_labeled_data(df)
    return df

