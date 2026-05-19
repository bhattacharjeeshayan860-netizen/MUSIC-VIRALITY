import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ─────────────────────────────────────────────
# CORE EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, x_test, y_test, model_name="Model"):
    """
    Full evaluation suite. Returns metrics dict and predictions.
    Used identically by detection and prediction training pipelines.
    """
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(x_test)

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nSummary Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return y_pred, y_proba, metrics


# ─────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────

def find_best_threshold(y_test, y_proba, metric="f1"):
    """
    Find the classification threshold that maximizes a given metric.

    WHY THIS MATTERS:
    Default threshold = 0.5 is almost never optimal with class imbalance.
    A model that outputs 0.35 probability for viral videos gets penalized by
    the default threshold. Tuning threshold is free performance — no retraining.

    In production: this threshold gets saved alongside the model and used at
    inference time (model.predict_proba > threshold, not model.predict).

    Returns the best threshold and the F1 at that threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    best_threshold = 0.5
    best_score = 0.0

    for i, threshold in enumerate(thresholds):
        if precisions[i] + recalls[i] == 0:
            continue
        if metric == "f1":
            score = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
        elif metric == "precision":
            score = precisions[i]
        elif metric == "recall":
            score = recalls[i]
        else:
            score = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])

        if score > best_score:
            best_score = score
            best_threshold = threshold

    y_pred_tuned = (y_proba >= best_threshold).astype(int)
    print(f"\n--- Threshold Optimization (metric={metric}) ---")
    print(f"  Default threshold (0.5) F1:  {f1_score(y_test, (y_proba >= 0.5).astype(int), zero_division=0):.4f}")
    print(f"  Optimal threshold ({best_threshold:.3f}) F1: {best_score:.4f}")
    print(f"\nClassification report at optimal threshold:")
    print(classification_report(y_test, y_pred_tuned, zero_division=0))

    return best_threshold, best_score


# ─────────────────────────────────────────────
# BUSINESS METRICS
# ─────────────────────────────────────────────

def business_impact_summary(y_test, y_pred, y_proba=None, model_name="Model"):
    """
    Translate ML metrics into business language.

    WHY: Interviewers at 15-25 LPA level ask 'so what?' after you show F1 scores.
    You need to articulate what a false positive / false negative costs in the
    context of music virality prediction.

    Example framing:
    - False Positive: We promote a non-viral song → wasted marketing budget
    - False Negative: We miss a viral song → lost early opportunity
    """
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total = len(y_test)
    actual_viral = y_test.sum()
    predicted_viral = y_pred.sum()

    print(f"\n--- Business Impact Summary: {model_name} ---")
    print(f"  Total videos evaluated:          {total:,}")
    print(f"  Actually viral:                  {actual_viral:,} ({actual_viral/total:.1%})")
    print(f"  Predicted viral:                 {predicted_viral:,} ({predicted_viral/total:.1%})")
    print(f"\n  True Positives (viral, caught):  {tp:,}")
    print(f"  False Positives (wrong alarm):   {fp:,}  <- wasted promotion budget")
    print(f"  False Negatives (missed viral):  {fn:,}  <- missed opportunity")
    print(f"  True Negatives (correctly skipped): {tn:,}")

    if actual_viral > 0:
        recall = tp / actual_viral
        print(f"\n  Viral recall (coverage):         {recall:.1%}")
        print(f"  -> We catch {recall:.0%} of all viral songs before they peak.")

    if predicted_viral > 0:
        precision = tp / predicted_viral
        print(f"  Prediction precision:            {precision:.1%}")
        print(f"  -> {precision:.0%} of songs we flag actually go viral.")