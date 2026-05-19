"""
SHAP Explainability Module
==========================
WHY SHAP:
- Standard feature importance in Random Forest / XGBoost tells you which features
  were used most across ALL trees, averaged. It doesn't tell you HOW a feature
  affects a specific prediction, or whether its effect is positive or negative.
- SHAP (SHapley Additive exPlanations) gives you a per-prediction explanation:
  "This video's views_per_day of 45,000 pushed the viral probability UP by 0.23."
- At 15-25 LPA interviews, SHAP is the standard explainability tool. Knowing it
  shows you can build systems that are auditable, not just accurate.

Usage:
    from src.utils.shap_explainer import run_shap_analysis
    run_shap_analysis(model, x_test, feature_names, model_type="tree")
"""

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[Warning] SHAP not installed. Run: pip install shap")


def _is_xgboost_model(model) -> bool:
    """Return True for xgboost sklearn models/boosters (best-effort)."""
    if model is None:
        return False
    mod = getattr(model.__class__, "__module__", "")
    return mod.startswith("xgboost") or hasattr(model, "get_booster")


def _xgboost_pred_contribs(model, X):
    """Compute SHAP-like contributions using XGBoost's native pred_contribs.

    Works around SHAP <-> XGBoost JSON compatibility issues (e.g., base_score as list).
    Returns array shape (n_samples, n_features).
    """
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmatrix = xgb.DMatrix(X)
    contrib = booster.predict(dmatrix, pred_contribs=True)
    # last column is the bias term
    return contrib[:, :-1]


def _coerce_shap_values_2d(shap_values, X_sample=None, positive_class: int = 1):
    """Coerce SHAP outputs into a 2D array (n_samples, n_features).

    SHAP output formats vary by model + shap version:
    - list[class] of arrays
    - np.ndarray with shape (n, features, outputs)
    - np.ndarray with shape (outputs, n, features)
    - shap.Explanation with .values
    """
    if shap_values is None:
        return None

    # shap.Explanation
    if hasattr(shap_values, "values") and not isinstance(shap_values, (np.ndarray, list)):
        shap_values = shap_values.values

    # Old API: list per class
    if isinstance(shap_values, list):
        if len(shap_values) > positive_class:
            shap_values = shap_values[positive_class]
        else:
            shap_values = shap_values[-1]

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and X_sample is not None and hasattr(X_sample, "shape"):
        n_samples, n_features = int(X_sample.shape[0]), int(X_sample.shape[1])

        # (n_samples, n_features, n_outputs)
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            out_dim = arr.shape[2]
            idx = positive_class if out_dim > positive_class else 0
            return arr[:, :, idx]

        # (n_outputs, n_samples, n_features)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            out_dim = arr.shape[0]
            idx = positive_class if out_dim > positive_class else 0
            return arr[idx, :, :]

    # Fallback: if last dim looks like outputs, slice it
    if arr.ndim == 3:
        idx = positive_class if arr.shape[-1] > positive_class else 0
        return arr[:, :, idx]

    # Unexpected shape
    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")


def _get_underlying_clf(model):
    """Extract the classifier from a sklearn Pipeline if needed."""
    if hasattr(model, "named_steps"):
        # It's a Pipeline — get the last step
        last_step_name = list(model.named_steps.keys())[-1]
        return model.named_steps[last_step_name]
    return model


def _get_transformed_X(model, X):
    """If model is a Pipeline, transform X through all steps except the last classifier."""
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        X_transformed = X.copy()
        for name, step in steps[:-1]:  # all except the last (clf) step
            X_transformed = step.transform(X_transformed)
        return X_transformed
    return X


def run_shap_analysis(
    model,
    x_test,
    feature_names=None,
    model_type="tree",
    max_display=15,
    sample_size=200,
    output_dir="models/explainability",
    label="detection",
):
    """
    Run SHAP analysis and save results.

    Parameters
    ----------
    model : fitted sklearn model or Pipeline
    x_test : pd.DataFrame or np.ndarray — test set features
    feature_names : list of str — if x_test is a DataFrame, will be inferred
    model_type : "tree" for RF/XGBoost, "linear" for LogReg
    max_display : how many features to show in summary plot
    sample_size : SHAP is O(N*features) — subsample for speed
    output_dir : where to save plots
    label : "detection" or "prediction" — used in filenames
    """
    if not _HAS_SHAP:
        print("SHAP not available. Install with: pip install shap")
        return None

    import os
    os.makedirs(output_dir, exist_ok=True)

    if feature_names is None and hasattr(x_test, "columns"):
        feature_names = x_test.columns.tolist()

    # Extract the underlying classifier if it's a Pipeline
    clf = _get_underlying_clf(model)
    X_transformed = _get_transformed_X(model, x_test)

    # Subsample for speed — SHAP on 5000+ rows is slow
    n_samples = min(sample_size, len(X_transformed))
    if hasattr(X_transformed, "iloc"):
        X_sample = X_transformed.iloc[:n_samples]
    else:
        X_sample = X_transformed[:n_samples]

    print(f"\n--- SHAP Analysis: {label} model ({n_samples} samples) ---")

    # Choose explainer based on model type
    try:
        if model_type == "tree":
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_sample)
            except Exception as tree_err:
                # SHAP sometimes breaks on newer XGBoost model formats.
                # In that case we can still get exact TreeSHAP values via XGBoost itself.
                if _is_xgboost_model(clf):
                    print(f"[Info] TreeExplainer failed for XGBoost ({tree_err}); using xgboost pred_contribs.")
                    explainer = None
                    shap_values = _xgboost_pred_contribs(clf, X_sample)
                else:
                    raise
        else:
            explainer = shap.LinearExplainer(clf, X_sample)
            shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[Warning] SHAP explainer failed: {e}")
        print("Falling back to permutation-based feature importance.")
        return _fallback_feature_importance(model, x_test, feature_names, label)

    # Normalize to 2D so pandas + plots work reliably
    shap_values = _coerce_shap_values_2d(shap_values, X_sample=X_sample, positive_class=1)

    # Convert to DataFrame for easy inspection
    if feature_names is not None:
        if hasattr(X_sample, "values"):
            X_sample_arr = X_sample.values
        else:
            X_sample_arr = X_sample

        if shap_values.shape[1] != len(feature_names):
            print(
                f"[Warning] SHAP values/features mismatch: {shap_values.shape[1]} vs {len(feature_names)}. "
                "Using generic feature names for the saved importance table."
            )
            feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

        shap_df = pd.DataFrame(shap_values, columns=feature_names)

        # Mean absolute SHAP value = global feature importance
        importance = shap_df.abs().mean().sort_values(ascending=False)
        print("\nTop features by mean |SHAP|:")
        print(importance.head(max_display).to_string())

        # Save importance CSV
        importance_path = os.path.join(output_dir, f"{label}_shap_importance.csv")
        importance.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}).to_csv(
            importance_path, index=False
        )
        print(f"\nSHAP importance saved -> {importance_path}")

    # Save plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        bar_path = os.path.join(output_dir, f"{label}_shap_bar.png")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"SHAP bar plot saved -> {bar_path}")

        # Beeswarm plot (shows direction of effect)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        beeswarm_path = os.path.join(output_dir, f"{label}_shap_beeswarm.png")
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"SHAP beeswarm plot saved -> {beeswarm_path}")

    except Exception as e:
        print(f"[Info] Could not save SHAP plots: {e}")

    return shap_values, explainer


def explain_single_prediction(model, single_row, feature_names=None, model_type="tree"):
    """
    Explain one prediction. Use this in the dashboard or for debugging.

    WHY USEFUL: When a label team asks "why did we flag this video as viral?"
    you can show them the top 3 contributing features, not just a probability score.
    """
    if not _HAS_SHAP:
        print("SHAP not available.")
        return

    clf = _get_underlying_clf(model)
    X_transformed = _get_transformed_X(model, single_row)

    if feature_names is None and hasattr(single_row, "columns"):
        feature_names = single_row.columns.tolist()

    try:
        if model_type == "tree":
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_transformed)
            except Exception as tree_err:
                if _is_xgboost_model(clf):
                    print(f"[Info] TreeExplainer failed for XGBoost ({tree_err}); using xgboost pred_contribs.")
                    explainer = None
                    shap_values = _xgboost_pred_contribs(clf, X_transformed)
                else:
                    raise
        else:
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_values = explainer.shap_values(X_transformed)

        shap_values = _coerce_shap_values_2d(shap_values, X_sample=X_transformed, positive_class=1)

        shap_series = pd.Series(shap_values[0], index=feature_names).sort_values(key=abs, ascending=False)
        print("\n--- Single Prediction Explanation ---")
        print("Top contributing features:")
        print(shap_series.head(10).to_string())
        return shap_series

    except Exception as e:
        print(f"Explanation failed: {e}")
        return None


def _fallback_feature_importance(model, x_test, feature_names, label):
    """
    When SHAP fails, fall back to built-in feature importance.
    Less informative than SHAP but better than nothing.
    """
    clf = _get_underlying_clf(model)
    if hasattr(clf, "feature_importances_"):
        importance = pd.Series(clf.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=False)
        print("\nFallback: Built-in feature importance (mean impurity decrease):")
        print(importance.head(15).to_string())
        return importance
    print("No feature importance available for this model type.")
    return None