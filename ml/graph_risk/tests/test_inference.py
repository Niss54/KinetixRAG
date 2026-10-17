"""Unit tests for Phase 4: GraphRiskInferenceEngine lifecycle and status handling."""

from pathlib import Path
import pytest

from ml.graph_risk.inference import GraphRiskInferenceEngine
from ml.graph_risk.model import GraphSAGENet
from ml.graph_risk.schemas import (
    FEATURE_DIM,
    ModelOperationalStatus,
)


def test_inference_engine_disabled():
    engine = GraphRiskInferenceEngine(enabled=False)
    status = engine.get_status()
    assert status["status"] == ModelOperationalStatus.DISABLED.value
    assert status["model_available"] is False

    pred = engine.predict(
        sample_id="SAMPLE-1",
        equipment_id="P-101A",
        features=[1.0] * FEATURE_DIM,
    )
    assert pred.status == ModelOperationalStatus.DISABLED
    assert pred.advisory_risk_score is None


def test_inference_engine_model_not_trained():
    engine = GraphRiskInferenceEngine(checkpoint_path=None, enabled=True)
    status = engine.get_status()
    assert status["status"] == ModelOperationalStatus.MODEL_NOT_TRAINED.value
    assert status["model_available"] is False

    pred = engine.predict(
        sample_id="SAMPLE-1",
        equipment_id="P-101A",
        features=[1.0] * FEATURE_DIM,
    )
    assert pred.status == ModelOperationalStatus.MODEL_NOT_TRAINED
    assert pred.advisory_risk_score is None


def test_inference_engine_available_with_loaded_checkpoint(tmp_path):
    # Create and save a valid checkpoint
    model = GraphSAGENet(random_state=42)
    checkpoint_file = tmp_path / "valid_checkpoint.json"
    model.save_checkpoint(checkpoint_file)

    engine = GraphRiskInferenceEngine(checkpoint_path=checkpoint_file, enabled=True)
    status = engine.get_status()
    assert status["status"] == ModelOperationalStatus.AVAILABLE.value
    assert status["model_available"] is True

    # Test single isolated node prediction
    valid_features = [1.85, 2.62, 1.42, 2.95, 52.4, 0.05, 33.3]
    pred = engine.predict(
        sample_id="SAMPLE-1",
        equipment_id="P-101A",
        features=valid_features,
        deterministic_risk=0.35,
    )
    assert pred.status == ModelOperationalStatus.AVAILABLE
    assert pred.advisory_risk_score is not None
    assert 0.0 <= pred.advisory_risk_score <= 1.0
    assert pred.calibrated is False
    assert pred.is_advisory is True
    assert pred.deterministic_risk_score == 0.35


def test_inference_engine_incompatible_features(tmp_path):
    model = GraphSAGENet(random_state=42)
    checkpoint_file = tmp_path / "valid_checkpoint.json"
    model.save_checkpoint(checkpoint_file)

    engine = GraphRiskInferenceEngine(checkpoint_path=checkpoint_file, enabled=True)
    # Pass 5 features instead of 7
    pred = engine.predict(
        sample_id="SAMPLE-1",
        equipment_id="P-101A",
        features=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert pred.status == ModelOperationalStatus.INCOMPATIBLE_FEATURES
    assert pred.advisory_risk_score is None
