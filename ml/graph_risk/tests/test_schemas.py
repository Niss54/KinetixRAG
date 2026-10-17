"""Unit tests for Phase 1: Data Contracts, Schemas, and Industrial Fixtures."""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from ml.graph_risk.schemas import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    AdvisoryExplanation,
    AdvisoryPrediction,
    CheckpointMetadata,
    DatasetSplit,
    EdgeType,
    ExplanationItem,
    FailureStage,
    GraphEdge,
    GraphNode,
    ModelOperationalStatus,
    NodeType,
    RawObservation,
    RiskLabel,
    SplitStrategy,
    WindowFeatures,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "industrial_telemetry_fixture.json"


def test_schema_constants():
    assert SCHEMA_VERSION == "1.0.0"
    assert len(FEATURE_NAMES) == 7
    assert FEATURE_DIM == 7
    assert "vibration_rms_mm_s" in FEATURE_NAMES
    assert "bearing_temp_c" in FEATURE_NAMES


def test_raw_observation_valid():
    obs = RawObservation(
        equipment_id="P-101A",
        sensor_id="VIB-01",
        timestamp=datetime(2004, 2, 12, 10, 30, tzinfo=timezone.utc),
        vibration_mm_s=2.45,
        bearing_temp_c=55.0,
        frequency_peak_hz=33.3,
        iso_zone="Zone A/B",
        stage="HEALTHY_BASELINE",
    )
    assert obs.equipment_id == "P-101A"
    assert obs.vibration_mm_s == 2.45


def test_raw_observation_invalid():
    with pytest.raises(ValidationError):
        # Negative vibration
        RawObservation(
            equipment_id="P-101A",
            sensor_id="VIB-01",
            timestamp=datetime(2004, 2, 12, 10, 30, tzinfo=timezone.utc),
            vibration_mm_s=-1.0,
            bearing_temp_c=55.0,
        )

    with pytest.raises(ValidationError):
        # Out-of-bounds temperature (>250 C)
        RawObservation(
            equipment_id="P-101A",
            sensor_id="VIB-01",
            timestamp=datetime(2004, 2, 12, 10, 30, tzinfo=timezone.utc),
            vibration_mm_s=2.0,
            bearing_temp_c=300.0,
        )


def test_window_features_vector_and_validation():
    wf = WindowFeatures(
        vibration_rms_mm_s=1.85,
        vibration_peak_mm_s=2.62,
        vibration_crest_factor=1.42,
        vibration_kurtosis=2.95,
        bearing_temp_c=52.4,
        temp_rate_of_change_c_per_hr=0.05,
        frequency_dominant_hz=33.3,
    )
    vec = wf.to_vector()
    assert len(vec) == FEATURE_DIM
    assert vec[0] == 1.85
    assert vec[4] == 52.4

    # Test non-finite rejection (NaN / Inf)
    with pytest.raises(ValidationError):
        WindowFeatures(
            vibration_rms_mm_s=float("nan"),
            vibration_peak_mm_s=2.62,
            vibration_crest_factor=1.42,
            vibration_kurtosis=2.95,
            bearing_temp_c=52.4,
            temp_rate_of_change_c_per_hr=0.05,
            frequency_dominant_hz=33.3,
        )


def test_graph_node_dimension_enforcement():
    valid_features = [1.85, 2.62, 1.42, 2.95, 52.4, 0.05, 33.3]
    node = GraphNode(
        node_id="NODE-1",
        node_index=0,
        node_type=NodeType.SENSOR_OBSERVATION,
        equipment_id="P-101A",
        timestamp=datetime(2004, 2, 12, 10, 30, tzinfo=timezone.utc),
        features=valid_features,
    )
    assert node.node_index == 0
    assert len(node.features) == 7

    # Reject mismatched dimensions (e.g. 5 instead of 7)
    with pytest.raises(ValidationError):
        GraphNode(
            node_id="NODE-1",
            node_index=0,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=datetime(2004, 2, 12, 10, 30, tzinfo=timezone.utc),
            features=[1.0, 2.0, 3.0, 4.0, 5.0],
        )


def test_graph_edge_causal_anti_leakage():
    t1 = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2004, 2, 13, 10, 0, tzinfo=timezone.utc)

    # Valid causal edge: t1 -> t2
    edge = GraphEdge(
        source_id="N1",
        target_id="N2",
        source_index=0,
        target_index=1,
        edge_type=EdgeType.TEMPORAL_PRECEDES,
        source_timestamp=t1,
        target_timestamp=t2,
    )
    assert edge.source_index == 0
    assert edge.target_index == 1

    # Invalid edge: t2 -> t1 (future leakage backwards into past)
    with pytest.raises(ValidationError, match="Temporal leakage detected"):
        GraphEdge(
            source_id="N2",
            target_id="N1",
            source_index=1,
            target_index=0,
            edge_type=EdgeType.TEMPORAL_PRECEDES,
            source_timestamp=t2,
            target_timestamp=t1,
        )


def test_dataset_split_overlap_rejection():
    # Valid split
    split = DatasetSplit(
        train_indices=[0, 1, 2, 3],
        val_indices=[4, 5],
        test_indices=[6, 7],
        split_strategy=SplitStrategy.TEMPORAL_HOLDOUT,
    )
    assert len(split.train_indices) == 4

    # Overlapping train and val
    with pytest.raises(ValidationError, match="Train and validation splits overlap"):
        DatasetSplit(
            train_indices=[0, 1, 2, 3],
            val_indices=[3, 4],  # 3 is duplicated
            test_indices=[5, 6],
        )

    # Overlapping train and test
    with pytest.raises(ValidationError, match="Train and test splits overlap"):
        DatasetSplit(
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
            test_indices=[2, 5],  # 2 is duplicated
        )


def test_advisory_prediction_contract():
    pred = AdvisoryPrediction(
        sample_id="OBS-001",
        equipment_id="P-101A",
        timestamp="2004-02-12T10:30:00Z",
        status=ModelOperationalStatus.AVAILABLE,
        advisory_risk_score=0.742,
        calibrated=False,
        is_advisory=True,
        deterministic_risk_score=0.680,
        explanation=AdvisoryExplanation(
            status="available",
            hops_analyzed=2,
            influential_nodes=[
                ExplanationItem(
                    node_id="OBS-005",
                    entity_type="SENSOR_OBSERVATION",
                    influence_score=0.88,
                    timestamp="2004-02-17T18:45:00Z",
                    dominant_feature="vibration_rms_mm_s",
                )
            ],
        ),
    )
    assert pred.advisory_risk_score == 0.742
    assert pred.calibrated is False
    assert pred.is_advisory is True
    assert pred.explanation.status == "available"


def test_checkpoint_metadata_contract():
    meta = CheckpointMetadata(
        architecture="GraphSAGE-2L-Mean",
        model_version="1.0.0",
        schema_version="1.0.0",
        input_dim=7,
        hidden_dim=64,
        num_layers=2,
        dataset_name="NASA IMS Bearing Run-to-Failure",
        train_pr_auc=0.945,
        val_pr_auc=0.912,
        test_pr_auc=0.897,
    )
    assert meta.input_dim == 7
    assert meta.hidden_dim == 64
    assert meta.dataset_name == "NASA IMS Bearing Run-to-Failure"


def test_industrial_telemetry_fixture_parsing():
    assert FIXTURE_PATH.exists(), f"Fixture file not found at {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify metadata
    assert data["metadata"]["feature_schema_version"] == SCHEMA_VERSION
    assert data["metadata"]["equipment_id"] == "P-101A"

    # Validate each node through GraphNode and RiskLabel schemas
    parsed_nodes = []
    for node_raw in data["nodes"]:
        node = GraphNode(
            node_id=node_raw["node_id"],
            node_index=node_raw["node_index"],
            node_type=NodeType(node_raw["node_type"]),
            equipment_id=node_raw["equipment_id"],
            timestamp=datetime.fromisoformat(node_raw["timestamp"].replace("Z", "+00:00")),
            features=node_raw["features"],
        )
        parsed_nodes.append(node)

        # Validate label schema
        label = RiskLabel(
            sample_id=node_raw["node_id"],
            timestamp=datetime.fromisoformat(node_raw["timestamp"].replace("Z", "+00:00")),
            is_impending_failure=node_raw["label"],
            horizon_hours=data["metadata"]["lookahead_horizon_hours"],
            stage=FailureStage(node_raw["stage"]),
            iso_zone=node_raw.get("iso_zone"),
        )
        assert label.is_impending_failure in (0, 1)

    assert len(parsed_nodes) == 7

    # Validate edges through GraphEdge schema
    parsed_edges = []
    for edge_raw in data["edges"]:
        edge = GraphEdge(
            source_id=edge_raw["source_id"],
            target_id=edge_raw["target_id"],
            source_index=edge_raw["source_index"],
            target_index=edge_raw["target_index"],
            edge_type=EdgeType(edge_raw["edge_type"]),
            weight=edge_raw.get("weight", 1.0),
            source_timestamp=datetime.fromisoformat(edge_raw["source_timestamp"].replace("Z", "+00:00")),
            target_timestamp=datetime.fromisoformat(edge_raw["target_timestamp"].replace("Z", "+00:00")),
        )
        parsed_edges.append(edge)
    assert len(parsed_edges) == 6

    # Validate split through DatasetSplit schema
    split = DatasetSplit(
        train_indices=data["split"]["train_indices"],
        val_indices=data["split"]["val_indices"],
        test_indices=data["split"]["test_indices"],
        split_strategy=SplitStrategy(data["split"]["split_strategy"]),
    )
    assert len(split.train_indices) == 4
    assert len(split.val_indices) == 2
    assert len(split.test_indices) == 1
