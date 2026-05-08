import pandas as pd
import os

def load_raw_data(file_path="data/raw/music_virality_data.csv"):
    data=pd.read_csv(file_path)
    new_df=data.copy()
    return new_df
def handle_missing_values(new_df):
    print("NA counts for each column:")
    print(new_df.isna().sum())
    new_df=new_df.dropna(subset=["video_id","collected_at","view_count"])
    return new_df

def convert_datetime(new_df):
    new_df["collected_at"]=pd.to_datetime(new_df["collected_at"],errors="coerce").dt.tz_localize(None)
    new_df["published_at"]=pd.to_datetime(new_df["published_at"],errors="coerce").dt.tz_localize(None)
    return new_df

def create_date_column(new_df):
    new_df["date"]=new_df["collected_at"].dt.date
    new_df["day_since_published"]=(new_df["collected_at"]-new_df["published_at"]).dt.days
    return new_df

def remove_noise_duplicates(new_df):
    new_df=new_df.sort_values(by=["video_id","collected_at"])
    new_df=new_df.groupby(["video_id","date"]).last().reset_index()
    return new_df

def sort_date(new_df):
    new_df=new_df.sort_values(by=["date","video_id"])
    return new_df

def load_existing_data(file_path="data/processed/clean_music_virality_data.csv"):
    if os.path.exists(file_path):
        existing_df=pd.read_csv(file_path)
        existing_df=convert_datetime(existing_df)
        existing_df=create_date_column(existing_df)
        existing_df=remove_noise_duplicates(existing_df)
        existing_df=sort_date(existing_df)
        return existing_df
    else:
        existing_df=pd.DataFrame()
        return existing_df

def merge_existing_data(new_df,existing_df):
        if not existing_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True)

            df = convert_datetime(df)
            df = create_date_column(df)

            df = df.sort_values(by=["video_id", "collected_at"])

            df = (
                df.groupby(["video_id", "date"])
                .last()
                .reset_index()
            )

            df = sort_date(df)
            return df
        else:
            df=new_df    
            return df

def save_cleaned_data(df):
        df.to_csv("data/processed/clean_music_virality_data.csv",index=False,)
def run_cleaning_pipeline():
    new_df=load_raw_data()
    new_df=handle_missing_values(new_df)
    new_df=convert_datetime(new_df)
    new_df=create_date_column(new_df)
    new_df=remove_noise_duplicates(new_df)
    new_df=sort_date(new_df)
    existing_df=load_existing_data()
    df=merge_existing_data(new_df,existing_df)
    save_cleaned_data(df)
    