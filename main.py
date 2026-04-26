from src.data_processing.cleaning_data import run_cleaning_pipeline
from src.data_processing.feature_engineering import run_feature_engineering_pipeline
from src.data_processing.labeling import run_labeling_pipeline
from src.training.train_model import run_training_pipeline
import sys
import os

sys.path.append(os.path.abspath("."))

def main():
    print("🔹 Step 1: Cleaning")
    run_cleaning_pipeline()
    print("🔹 Step 2: Feature Engineering")
    run_feature_engineering_pipeline()
    print("🔹 Step 3: Labeling")
    run_labeling_pipeline()
    print("🔹 Step 4: Training")
    run_training_pipeline()

if __name__ == "__main__":
    main()