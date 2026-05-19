import os
import joblib
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_predictions_features import FEATURE_COLUMNS, get_X_y
from src.utils.evaluation import evaluate_classifier, find_best_threshold, business_impact_summary

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


# ─────────────────────────────────────────────
# GROUP-AWARE SPLIT (same logic as detection)
# ─────────────────────────────────────────────

def _train_test_split_no_leakage(df, X, y, test_size=0.2, random_state=42):
    """
    Your original function name preserved. Logic upgraded to GroupShuffleSplit.
    The prediction pipeline already used this correctly — keeping it intact.
    """
    if "video_id" in df.columns and df["video_id"].notna().any():
        groups = df["video_id"].astype(str).values
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        train_vids = set(groups[train_idx])
        test_vids = set(groups[test_idx])
        overlap = train_vids & test_vids
        print(f"\n[Split] Train videos: {len(train_vids):,}  |  Test videos: {len(test_vids):,}  |  Overlap: {len(overlap):,}")
        print("[OK] No group leakage - clean split confirmed." if not overlap else "[WARN] Leakage detected.")

        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    from sklearn.model_selection import train_test_split
    stratify = y if y.nunique() > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def _build_prediction_models(class_ratio):
    imputer = lambda: SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)

    models = [
        ("Dummy (baseline)", DummyClassifier(strategy="most_frequent")),

        ("Random Forest", Pipeline([
            ("imputer", imputer()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])),

        ("Logistic Regression", Pipeline([
            ("imputer", imputer()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                random_state=42,
            )),
        ])),
    ]

    if _HAS_XGBOOST:
        models.append(("XGBoost", Pipeline([
            ("imputer", imputer()),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                min_child_weight=3,
                reg_alpha=0.5,
                reg_lambda=1,
                scale_pos_weight=class_ratio,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])))
    else:
        print("Note: xgboost not installed, skipping XGBoost.")

    return models


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_prediction_training_pipeline(
    file_path="data/processed/future_labeled_music_virality_data.csv",
    artifact_path="models/trained/prediction_model.pkl",
    threshold_path="models/trained/prediction_threshold.pkl",
):
    """
    Train prediction model — will this video become viral?

    Key improvements vs original:
    1. get_X_y now returns (df, X, y) instead of (df, X, y) — consistent with detection
    2. All new features from feature_engineering.py are available
    3. Threshold optimization added
    4. Business impact summary added
    """
    print("\n" + "="*60)
    print("  PREDICTION MODEL TRAINING")
    print("="*60)

    df, X, y = get_X_y(file_path=file_path)
    if X.empty or y.empty:
        print("No labeled data available for training.")
        return

    print(f"\nDataset: {len(X):,} rows | {y.mean():.1%} future viral | {X.shape[1]} features")
    print("\n[NOTE] Dataset note: 'Future' label spans ~6 days only.")
    print("  Model is a momentum stability classifier, not true pre-viral predictor.")
    print("  This is documented in the README. Real-world use needs longer collection window.")

    class_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"\nClass ratio (neg/pos): {class_ratio:.2f}")

    x_train, x_test, y_train, y_test = _train_test_split_no_leakage(df, X, y)

    models = _build_prediction_models(class_ratio)
    best = None
    best_score = -1.0

    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        y_pred, y_proba, metrics = evaluate_classifier(model, x_test, y_test, model_name=name)
        business_impact_summary(y_test, y_pred, y_proba, model_name=name)

        score = metrics.get("pr_auc", metrics.get("roc_auc", metrics["f1"]))
        if score > best_score:
            best_score = score
            best = (name, model, metrics, y_proba)

    if best is None:
        print("No model trained.")
        return

    best_name, best_model, best_metrics, best_proba = best
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_name}")
    print(f"  Score (PR-AUC): {best_score:.4f}")
    print(f"{'='*60}")

    best_threshold = 0.5
    if best_proba is not None:
        best_threshold, _ = find_best_threshold(y_test, best_proba, metric="f1")

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(best_model, artifact_path)
    joblib.dump({"threshold": best_threshold, "model_name": best_name}, threshold_path)

    print(f"\nModel saved     -> {artifact_path}")
    print(f"Threshold saved -> {threshold_path}")
    print(f"Optimal threshold: {best_threshold:.4f}")

    return best_model, best_metrics, best_threshold


# Backwards-compatible alias
def run_future_training_pipeline():
    return run_prediction_training_pipeline()