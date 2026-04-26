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
    
print("Labeling pipeline loaded successfully.")
def is_viral(df):
    required_cols = ["view_count_log", "views_growth_rate", "engagement_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    view_threshold = df["view_count_log"].quantile(0.65)
    growth_threshold = df["views_growth_rate"].quantile(0.65)
    engagement_threshold = df["engagement_rate"].quantile(0.65)

    views_flag = df["view_count_log"] >= view_threshold
    growth_flag = df["views_growth_rate"] >= growth_threshold
    engagement_flag = df["engagement_rate"] >= engagement_threshold

    score = (views_flag.astype(int) + growth_flag.astype(int) + engagement_flag.astype(int))

    df["is_viral"]= (score>=2).astype(int)
    print(df["is_viral"].value_counts(normalize=True))


    return df
def save_labeled_data(df):
    print("Saving labeled data to data/processed/final_labelled_music_virality_data.csv")
    df.to_csv("data/processed/final_labelled_music_virality_data.csv",index=False)
    
def run_labeling_pipeline():
    df=load_data()
    df=is_viral(df)
    save_labeled_data(df)
    return df

