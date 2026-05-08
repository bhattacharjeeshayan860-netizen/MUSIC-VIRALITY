import pandas as pd
import os
    
def load_data(file_path:str = "data/processed/clean_music_virality_data.csv") -> pd.DataFrame:
    if os.path.exists(file_path):
        df=pd.read_csv(file_path)
        return df
    print(f"file {file_path} does not exist.")
    return pd.DataFrame()