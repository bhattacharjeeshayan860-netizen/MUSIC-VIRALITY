"""
main.py — Master pipeline orchestrator
======================================
Run the full system end to end:
    python main.py

Or run individual stages:
    python main.py --stage clean
    python main.py --stage features
    python main.py --stage label
    python main.py --stage train_detection
    python main.py --stage train_prediction
    python main.py --stage explain
    python main.py --stage all
"""

import argparse
import sys
import os


def run_clean():
    print("\n[Stage 1] Cleaning...")
    from src.data_processing.cleaning_data import run_cleaning_pipeline
    run_cleaning_pipeline()


def run_features():
    print("\n[Stage 2] Feature Engineering...")
    from src.features.feature_engineering import run_feature_engineering_pipeline
    run_feature_engineering_pipeline()


def run_label():
    print("\n[Stage 3] Labeling...")
    from src.features.labeling import run_detection_labeling_pipeline, run_future_labeling_pipeline
    run_detection_labeling_pipeline()
    run_future_labeling_pipeline()


def run_train_detection():
    print("\n[Stage 4a] Training Detection Model...")
    from src.training.training_detection import run_detection_training_pipeline
    run_detection_training_pipeline()


def run_train_prediction():
    print("\n[Stage 4b] Training Prediction Model...")
    from src.training.training_prediction import run_prediction_training_pipeline
    run_prediction_training_pipeline()


def run_explain():
    print("\n[Stage 5] SHAP Explainability...")
    import joblib
    import pandas as pd
    from src.utils.shap_explainer import run_shap_analysis
    from src.features.build_detections_features import get_X_y as det_get_X_y, FEATURE_COLS
    from src.features.build_predictions_features import get_X_y as pred_get_X_y, FEATURE_COLUMNS
    from sklearn.model_selection import GroupShuffleSplit

    # Detection model explainability
    det_model_path = "models/trained/detection_model.pkl"
    if os.path.exists(det_model_path):
        model = joblib.load(det_model_path)
        df, X, y = det_get_X_y()
        if not X.empty:
            groups = df["video_id"].astype(str).values if "video_id" in df.columns else None
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            _, test_idx = next(gss.split(X, y, groups=groups))
            x_test = X.iloc[test_idx]
            run_shap_analysis(model, x_test, feature_names=FEATURE_COLS, label="detection")
    else:
        print(f"Detection model not found at {det_model_path}. Run train_detection first.")

    # Prediction model explainability
    pred_model_path = "models/trained/prediction_model.pkl"
    if os.path.exists(pred_model_path):
        model = joblib.load(pred_model_path)
        df, X, y = pred_get_X_y()
        if not X.empty:
            groups = df["video_id"].astype(str).values if "video_id" in df.columns else None
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            _, test_idx = next(gss.split(X, y, groups=groups))
            x_test = X.iloc[test_idx]
            run_shap_analysis(model, x_test, feature_names=list(X.columns), label="prediction")
    else:
        print(f"Prediction model not found at {pred_model_path}. Run train_prediction first.")


def main():
    parser = argparse.ArgumentParser(description="Music Virality ML Pipeline")
    parser.add_argument(
        "--stage",
        choices=["clean", "features", "label", "train_detection", "train_prediction", "explain", "all"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    stage = args.stage

    if stage == "all":
        run_clean()
        run_features()
        run_label()
        run_train_detection()
        run_train_prediction()
        run_explain()
    elif stage == "clean":
        run_clean()
    elif stage == "features":
        run_features()
    elif stage == "label":
        run_label()
    elif stage == "train_detection":
        run_train_detection()
    elif stage == "train_prediction":
        run_train_prediction()
    elif stage == "explain":
        run_explain()
    else:
        print(f"Unknown stage: {stage}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()