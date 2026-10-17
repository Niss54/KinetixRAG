"""Unit & Integration tests for GraphSAGE Risk Intelligence API endpoints."""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.graph_risk import reset_inference_engine
from ml.graph_risk.model import GraphSAGENet
from ml.graph_risk.schemas import FEATURE_DIM, CheckpointMetadata

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_engine_state():
    """Reset service engine singleton before and after each test."""
    reset_inference_engine()
    yield
    reset_inference_engine()


def test_status_disabled_by_default(monkeypatch):
    monkeypatch.setenv("ENABLE_GRAPH_RISK_MODEL", "false")
    reset_inference_engine()

    response = client.get("/api/graph-risk/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "disabled"
    assert data["model_available"] is False
    assert data["schema_version"] == "1.0.0"
    assert data["feature_dim"] == FEATURE_DIM


def test_predict_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_GRAPH_RISK_MODEL", "false")
    reset_inference_engine()

    payload = {
        "sample_id": "OBS-001",
        "equipment_id": "P-101A",
        "features": [1.85, 2.62, 1.42, 2.95, 52.4, 0.05, 33.3],
        "explain": False,
        "deterministic_risk": 0.42,
    }

    response = client.post("/api/graph-risk/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "disabled"
    assert data["advisory_risk_score"] is None
    assert data["deterministic_risk_score"] == 0.42
    assert data["is_advisory"] is True
    assert "ENABLE_GRAPH_RISK_MODEL is false" in data["limitations"]


def test_status_and_predict_when_enabled_with_checkpoint(monkeypatch, tmp_path):
    # Create and save a valid checkpoint
    model = GraphSAGENet(random_state=42)
    checkpoint_file = tmp_path / "valid_checkpoint.json"
    meta = CheckpointMetadata(val_pr_auc=0.91)
    model.save_checkpoint(checkpoint_file, metadata=meta)

    monkeypatch.setenv("ENABLE_GRAPH_RISK_MODEL", "true")
    monkeypatch.setenv("GRAPH_RISK_CHECKPOINT_PATH", str(checkpoint_file))
    reset_inference_engine()

    # Test status endpoint
    status_resp = client.get("/api/graph-risk/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "available"
    assert status_data["model_available"] is True
    assert status_data["val_pr_auc"] == 0.91

    # Test predict endpoint
    payload = {
        "sample_id": "OBS-001",
        "equipment_id": "P-101A",
        "features": [1.85, 2.62, 1.42, 2.95, 52.4, 0.05, 33.3],
        "explain": False,
        "deterministic_risk": 0.35,
    }

    predict_resp = client.post("/api/graph-risk/predict", json=payload)
    assert predict_resp.status_code == 200
    predict_data = predict_resp.json()
    assert predict_data["status"] == "available"
    assert predict_data["advisory_risk_score"] is not None
    assert 0.0 <= predict_data["advisory_risk_score"] <= 1.0
    assert predict_data["calibrated"] is False
    assert predict_data["is_advisory"] is True
    assert predict_data["deterministic_risk_score"] == 0.35


def test_predict_validation_errors(monkeypatch):
    monkeypatch.setenv("ENABLE_GRAPH_RISK_MODEL", "true")
    reset_inference_engine()

    # Mismatched dimension (5 features instead of 7)
    payload_wrong_dim = {
        "sample_id": "OBS-001",
        "equipment_id": "P-101A",
        "features": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    resp = client.post("/api/graph-risk/predict", json=payload_wrong_dim)
    assert resp.status_code == 422
    assert "incompatible_feature_dimension" in resp.text

    # Non-numeric / malformed feature values
    payload_invalid = {
        "sample_id": "OBS-001",
        "equipment_id": "P-101A",
        "features": ["not_a_number", 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
    resp_invalid = client.post("/api/graph-risk/predict", json=payload_invalid)
    assert resp_invalid.status_code == 422


def test_root_and_existing_routes_unaffected():
    # Verify core root and health check remain completely operational
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
