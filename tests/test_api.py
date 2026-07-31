"""
Tests for the FastAPI serving surface (src/api/app.py).

Uses FastAPI's TestClient so no actual server is started, and mocks the model
artifacts so the tests do not depend on trained models being present.
"""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.api.app import app


try:
    from fastapi.testclient import TestClient
    client = TestClient(app)
    _HAS_TESTCLIENT = True
except ImportError:
    client = None
    _HAS_TESTCLIENT = False


def _fake_model(proba: float):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, proba]])
    return model


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi testclient not installed")
@patch("src.api.app._load_artifacts")
def test_predict_detection_endpoint(mock_load_artifacts):
    mock_load_artifacts.return_value = (_fake_model(0.85), {"threshold": 0.75})
    response = client.post(
        "/predict/detection",
        json={
            "views": 1_000_000,
            "likes": 50_000,
            "comments": 5_000,
            "subscribers": 500_000,
            "days_old": 5,
            "duration_seconds": 240,
            "title": "Official Music Video",
            "published_at": "2024-03-15T19:00:00Z",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "detection"
    assert data["probability"] == pytest.approx(0.85)
    assert data["is_viral"] is True
    assert "features_used" in data
    assert "view_count_log" not in data["features_used"]


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi testclient not installed")
@patch("src.api.app._load_artifacts")
def test_predict_prediction_endpoint(mock_load_artifacts):
    mock_load_artifacts.return_value = (_fake_model(0.65), {"threshold": 0.70})
    response = client.post(
        "/predict/prediction",
        json={
            "views": 800_000,
            "likes": 40_000,
            "comments": 4_000,
            "subscribers": 200_000,
            "days_old": 3,
            "duration_seconds": 180,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "prediction"
    assert data["is_viral"] is False
    assert "view_count_log" in data["features_used"]


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi testclient not installed")
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi testclient not installed")
def test_prediction_rejects_negative_views():
    response = client.post(
        "/predict/detection",
        json={
            "views": -1,
            "likes": 0,
            "comments": 0,
            "subscribers": 0,
            "days_old": 1,
            "duration_seconds": 60,
        },
    )
    assert response.status_code == 422
