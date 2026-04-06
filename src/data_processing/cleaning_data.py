import pandas as pd
import os

def load_raw_data(file_path="music_virality_data.csv"):
    data=pd.read_csv(file_path)
    new_df=data.copy()
    return new_df
def convert_datetime(new_df):
    new_df["collected_at"]=pd.to_datetime(new_df["collected_at"])
    return new_df

def create_date_column(new_df):
    new_df["date"]=new_df["collected_at"].dt.date
    return new_df

def remove_noise_duplicates(new_df):
    new_df=new_df.sort_values(by=["video_id","collected_at"])
    new_df=new_df.groupby(["video_id","date"]).last().reset_index()
    return new_df

def sort_date(new_df):
    new_df=new_df.sort_values(by=["date","video_id"])
    return new_df

def load_existing_data(file_path="clean_music_virality_data.csv"):
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
            df=pd.concat([existing_df,new_df],ignore_index=True)
            df=convert_datetime(df)
            df=create_date_column(df)
            df=remove_noise_duplicates(df)
            df=sort_date(df)
            return df
        else:
            df=new_df    
            return df

def save_cleaned_data(df):
        df.to_csv("clean_music_virality_data.csv",index=False,)
    