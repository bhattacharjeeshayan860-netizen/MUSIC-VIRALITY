import os
import joblib
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.features.build_detections_features import FEATURE_COLS, get_X_y
from src.utils.evaluation import evaluate_classifier, find_best_threshold, business_impact_summary
from src.utils.splits import stratified_group_three_way_split as _split_by_group

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


# ─────────────────────────────────────────────
# GROUP-AWARE STRATIFIED TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────
# Implemented in src/utils/splits.py and imported as _split_by_group.
# Stratified by per-video label so the positive rate is preserved across
# splits (important for this skewed ~11% viral dataset) while keeping every
# video in exactly one split.


# ─────────────────────────────────────────────
# OVERFITTING DIAGNOSTIC (NEW)
# ─────────────────────────────────────────────

def _diagnose_overfitting(name, model, x_train, y_train, x_test, y_test):
    """
    Compares train F1 vs test F1, and runs 5-fold CV on the training set.

    WHY: A high test score alone can't tell you if hyperparameters are too
    aggressive (overfitting) or if the leakage-removal already explains the
    score you're seeing. This isolates that question with evidence:

    - Train F1 >> Test F1 (gap > ~0.08-0.10)  -> overfitting, tune params
    - Train F1 ≈ Test F1, both modest          -> model is honest, params are fine
    - CV fold scores vary wildly                -> unstable, possibly too little data
      or too much variance in features (also a max_depth/n_estimators signal)
    """
    train_pred = model.predict(x_train)
    train_f1 = f1_score(y_train, train_pred)

    test_pred = model.predict(x_test)
    test_f1 = f1_score(y_test, test_pred)

    gap = train_f1 - test_f1

    print(f"\n--- Overfitting Diagnostic: {name} ---")
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    print(f"  Gap:      {gap:.4f}", end="  ")

    if gap > 0.10:
        print("<- LARGE gap: model likely overfitting. Consider reducing max_depth, ")
        print("   n_estimators, or increasing reg_alpha/reg_lambda/min_child_weight.")
    elif gap > 0.05:
        print("<- moderate gap, worth watching but not alarming on its own.")
    else:
        print("<- small gap: no strong evidence of overfitting from this alone.")

    # 5-fold CV on training set only (test set stays untouched)
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        print(f"  5-fold CV F1 (train set): mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")
        if cv_scores.std() > 0.05:
            print("   <- high variance across folds: model/features may be unstable.")
    except Exception as e:
        print(f"  [CV skipped: {e}]")

    return {"train_f1": train_f1, "test_f1": test_f1, "gap": gap}


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def _build_models(class_ratio):
    """
    class_ratio: n_negative / n_positive — used for XGBoost scale_pos_weight.
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

    LEAKAGE FIX:
    - 3-way group-aware split (train / val / test).
    - Best model selected by VALIDATION PR-AUC.
    - Classification threshold tuned on VALIDATION set.
    - Test set touched exactly once for the final honest report.
    (Previously model choice + threshold were both made on the test set.)
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

    x_train, x_val, x_test, y_train, y_val, y_test = _split_by_group(df, X, y)

    models = _build_models(class_ratio)
    best = None
    best_score = -1.0
    overfit_results = {}

    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        # Evaluate on VALIDATION set only -> drives model selection (no test touch)
        y_pred, y_proba, metrics = evaluate_classifier(model, x_val, y_val, model_name=name)
        business_impact_summary(y_val, y_pred, y_proba, model_name=name)

        if name != "Dummy (baseline)":
            overfit_results[name] = _diagnose_overfitting(name, model, x_train, y_train, x_val, y_val)

        score = metrics.get("pr_auc", metrics.get("roc_auc", metrics["f1"]))
        if score > best_score:
            best_score = score
            best = (name, model, metrics, y_proba)

    if best is None:
        print("No model trained.")
        return

    best_name, best_model, best_metrics, best_proba_val = best
    print(f"\n{'='*60}")
    print(f"  BEST MODEL (by validation PR-AUC): {best_name}")
    print(f"  Val Score (PR-AUC): {best_score:.4f}")
    if best_name in overfit_results:
        g = overfit_results[best_name]
        print(f"  Train/Val F1 gap: {g['gap']:.4f}  (train={g['train_f1']:.4f}, val={g['test_f1']:.4f})")
    print(f"{'='*60}")

    # Tune threshold on VALIDATION set (test stays untouched)
    best_threshold = 0.5
    if best_proba_val is not None:
        best_threshold, _ = find_best_threshold(y_val, best_proba_val, metric="f1")

    # ── FINAL HONEST EVALUATION ON UNTOUCHED TEST SET ──
    print("\n" + "="*60)
    print("  FINAL HELD-OUT TEST EVALUATION (touched once, no selection bias)")
    print("="*60)
    y_pred_test, y_proba_test, test_metrics = evaluate_classifier(
        best_model, x_test, y_test, model_name=best_name
    )
    if y_proba_test is not None:
        y_pred_test_tuned = (y_proba_test >= best_threshold).astype(int)
        print(f"\nTest F1 at tuned threshold ({best_threshold:.4f}): "
              f"{f1_score(y_test, y_pred_test_tuned, zero_division=0):.4f}")
        business_impact_summary(y_test, y_pred_test_tuned, y_proba_test,
                                model_name=best_name + " (test@tuned)")

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(best_model, artifact_path)
    joblib.dump({"threshold": best_threshold, "model_name": best_name}, threshold_path)

    print(f"\nModel saved    -> {artifact_path}")
    print(f"Threshold saved -> {threshold_path}")
    print(f"Optimal threshold: {best_threshold:.4f} (use this at inference, not default 0.5)")

    return best_model, test_metrics, best_threshold