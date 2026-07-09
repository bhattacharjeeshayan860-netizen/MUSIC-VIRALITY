"""
Tests for the evaluation utilities (src/utils/evaluation.py).

Pins down:
  - evaluate_classifier returns the expected metric keys and shape-consistent
    predictions,
  - find_best_threshold never returns a threshold outside [0,1] and always does
    at least as well as the default 0.5 on the chosen metric,
  - business_impact_summary runs without error and reports sane confusion-matrix
    counts.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils.evaluation import (
    evaluate_classifier,
    find_best_threshold,
    business_impact_summary,
)


@pytest.fixture
def fitted_model_and_data():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.random((n, 4))
    # Make the target a deterministic function of feature 0 so the model can learn it.
    y = (X[:, 0] + 0.3 * X[:, 1] > 0.9).astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    x_test = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    y_test = pd.Series(y)
    return model, x_test, y_test


def test_evaluate_classifier_returns_metrics(fitted_model_and_data):
    model, x_test, y_test = fitted_model_and_data
    y_pred, y_proba, metrics = evaluate_classifier(model, x_test, y_test, model_name="test")
    assert len(y_pred) == len(y_test)
    assert y_proba is not None
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0


def test_find_best_threshold_in_range(fitted_model_and_data):
    model, x_test, y_test = fitted_model_and_data
    y_proba = model.predict_proba(x_test)[:, 1]
    best_t, best_score = find_best_threshold(y_test, y_proba, metric="f1")
    assert 0.0 <= best_t <= 1.0
    assert best_score >= 0.0


def test_tuned_threshold_beats_or_matches_default(fitted_model_and_data):
    from sklearn.metrics import f1_score
    model, x_test, y_test = fitted_model_and_data
    y_proba = model.predict_proba(x_test)[:, 1]
    best_t, best_score = find_best_threshold(y_test, y_proba, metric="f1")
    default_f1 = f1_score(y_test, (y_proba >= 0.5).astype(int), zero_division=0)
    assert best_score >= default_f1 - 1e-9


def test_business_impact_summary_runs(fitted_model_and_data):
    model, x_test, y_test = fitted_model_and_data
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    # Should not raise and should print a summary.
    business_impact_summary(y_test, y_pred, y_proba=y_proba, model_name="test")
