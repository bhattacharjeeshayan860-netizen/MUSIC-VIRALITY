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

from src.features.build_predictions_features import FEATURE_COLUMNS, get_X_y
from src.utils.evaluation import evaluate_classifier, find_best_threshold, business_impact_summary
from src.utils.splits import stratified_group_three_way_split as _train_test_split_no_leakage

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


# ─────────────────────────────────────────────
# GROUP-AWARE STRATIFIED SPLIT
# ─────────────────────────────────────────────
# Implemented in src/utils/splits.py and imported as _train_test_split_no_leakage.
# Name kept for backwards compatibility. Stratified by per-video label so the
# positive rate is preserved across splits (this dataset is ~19% future-viral)
# while keeping every video in exactly one split.


# ─────────────────────────────────────────────
# OVERFITTING DIAGNOSTIC (NEW — same logic as detection script)
# ─────────────────────────────────────────────

def _diagnose_overfitting(name, model, x_train, y_train, x_test, y_test):
    """
    Compares train F1 vs test F1, and runs 5-fold CV on the training set.
    See detection script for full rationale — identical logic here so both
    models are judged on the same standard.
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

    LEAKAGE FIX:
    - 3-way group-aware split (train / val / test).
    - Best model selected by VALIDATION PR-AUC.
    - Classification threshold tuned on VALIDATION set.
    - Test set touched exactly once for the final honest report.
    (Previously model choice + threshold were both made on the test set.)
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

    x_train, x_val, x_test, y_train, y_val, y_test = _train_test_split_no_leakage(df, X, y)

    models = _build_prediction_models(class_ratio)
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

    # ── PRE-VIRAL HONEST EVALUATION SLICE ──
    # Most "future viral" rows are already above the 10M threshold at the
    # training snapshot (~96%), so the full-test score mostly measures "is an
    # already-big video going to stay big". The meaningful question for an
    # early-warning system is: can the model flag a video that is NOT YET viral?
    # This slice evaluates only on test rows whose CURRENT view_count is below
    # the threshold — the honest, improvable score.
    try:
        from src.features.labeling import VIRALITY_THRESHOLD
        # get_X_y() returns a trimmed frame without view_count, so read the full
        # labeled CSV (indices are preserved through dropna) to recover it.
        import pandas as _pd
        full_df = _pd.read_csv(file_path)
        test_full = full_df.loc[x_test.index]
        if "view_count" in test_full.columns and y_proba_test is not None:
            pre_viral_mask = (test_full["view_count"].values < VIRALITY_THRESHOLD)
            if pre_viral_mask.sum() > 0 and y_test[pre_viral_mask].sum() > 0:
                pv_y = y_test[pre_viral_mask]
                pv_proba = y_proba_test[pre_viral_mask]
                pv_pred = (pv_proba >= best_threshold).astype(int)
                pv_f1 = f1_score(pv_y, pv_pred, zero_division=0)
                from sklearn.metrics import roc_auc_score, average_precision_score
                pv_roc = roc_auc_score(pv_y, pv_proba)
                pv_pr = average_precision_score(pv_y, pv_proba)
                print("\n" + "="*60)
                print("  PRE-VIRAL HONEST SLICE (current views < 10M at snapshot)")
                print("="*60)
                print(f"  Rows: {int(pre_viral_mask.sum()):,} | "
                      f"future-viral among them: {int(pv_y.sum()):,} ({pv_y.mean():.1%})")
                print(f"  F1: {pv_f1:.4f} | PR-AUC: {pv_pr:.4f} | ROC-AUC: {pv_roc:.4f}")
                print("  ^ This is the true early-warning difficulty, not the inflated full score.")
            else:
                print("\n[Pre-viral slice skipped: no future-viral rows below threshold in test.]")
        else:
            print("\n[Pre-viral slice skipped: view_count unavailable in labeled data.]")
    except Exception as e:
        print(f"\n[Pre-viral slice skipped: {e}]")

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(best_model, artifact_path)
    joblib.dump({"threshold": best_threshold, "model_name": best_name}, threshold_path)

    print(f"\nModel saved     -> {artifact_path}")
    print(f"Threshold saved -> {threshold_path}")
    print(f"Optimal threshold: {best_threshold:.4f}")

    return best_model, test_metrics, best_threshold


# Backwards-compatible alias
def run_future_training_pipeline():
    return run_prediction_training_pipeline()