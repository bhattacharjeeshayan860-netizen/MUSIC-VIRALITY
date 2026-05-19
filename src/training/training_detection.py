import os
import joblib
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_detections_features import FEATURE_COLS, get_X_y
from src.utils.evaluation import evaluate_classifier, find_best_threshold, business_impact_summary

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


# ─────────────────────────────────────────────
# GROUP-AWARE TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

def _split_by_group(df, X, y, test_size=0.2, random_state=42):
    """
    Split by video_id so no video appears in both train and test.

    WHY: With multiple snapshots per video, a random split puts snapshot #1
    (train) and snapshot #3 (test) of the same video on opposite sides.
    The model memorizes video-level patterns and reports inflated metrics.
    Confirmed in audit: 77.2% of test rows were from leaked videos.

    GroupShuffleSplit ensures complete video_id isolation between splits.
    """
    if "video_id" not in df.columns or df["video_id"].isna().all():
        print("[Warning] video_id not available. Falling back to stratified random split.")
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    groups = df["video_id"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Diagnostic: confirm no group leakage
    train_vids = set(groups[train_idx])
    test_vids = set(groups[test_idx])
    overlap = train_vids & test_vids
    print(f"\n[Split] Train videos: {len(train_vids):,}  |  Test videos: {len(test_vids):,}  |  Overlap: {len(overlap):,}")
    if overlap:
        print("[WARN] Group leakage detected. Check video_id column.")
    else:
        print("[OK] No group leakage - clean split confirmed.")

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def _build_models(class_ratio):
    """
    class_ratio: n_negative / n_positive — used for XGBoost scale_pos_weight.
    WHY: class_weight='balanced' and scale_pos_weight compensate for imbalance.
    Without this, the model learns to always predict the majority class.
    """
    imputer = lambda: SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)

    models = [
        ("Dummy (baseline)", DummyClassifier(strategy="most_frequent")),

        ("Random Forest", Pipeline([
            ("imputer", imputer()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
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
                C=0.5,
                solver="lbfgs",
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
                min_child_weight=5,
                reg_alpha=0.5,
                reg_lambda=2,
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

def run_detection_training_pipeline(
    file_path="data/processed/final_labelled_music_virality_data.csv",
    artifact_path="models/trained/detection_model.pkl",
    threshold_path="models/trained/detection_threshold.pkl",
):
    """
    Train detection model — is this video already viral?

    Key fixes vs original:
    1. GroupShuffleSplit instead of random train_test_split (leakage fix)
    2. view_count_log excluded from FEATURE_COLS (label-feature overlap fix)
    3. Threshold tuning added
    4. Business impact summary added
    """
    print("\n" + "="*60)
    print("  DETECTION MODEL TRAINING")
    print("="*60)

    df, X, y = get_X_y(file_path=file_path)
    if X.empty or y.empty:
        print("No labeled data available for training.")
        return

    print(f"\nDataset: {len(X):,} rows | {y.mean():.1%} viral | {X.shape[1]} features")

    class_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"Class ratio (neg/pos): {class_ratio:.2f}")

    x_train, x_test, y_train, y_test = _split_by_group(df, X, y)

    models = _build_models(class_ratio)
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

    # Threshold optimization on the best model
    best_threshold = 0.5
    if best_proba is not None:
        best_threshold, _ = find_best_threshold(y_test, best_proba, metric="f1")

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(best_model, artifact_path)
    joblib.dump({"threshold": best_threshold, "model_name": best_name}, threshold_path)

    print(f"\nModel saved    -> {artifact_path}")
    print(f"Threshold saved -> {threshold_path}")
    print(f"Optimal threshold: {best_threshold:.4f} (use this at inference, not default 0.5)")

    return best_model, best_metrics, best_threshold