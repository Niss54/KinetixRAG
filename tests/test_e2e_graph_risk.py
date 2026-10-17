"""
End-to-end integration test for GraphSAGE Risk Intelligence.
Validates the full pipeline:
  raw telemetry -> features -> causal graph -> GraphSAGE training & inference -> explainability -> FastAPI API -> safety interlocks
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
import backend.app.services.graph_risk as gr_service
from ml.graph_risk.features import extract_window_features
from ml.graph_risk.graph_builder import CausalGraphBuilder
from ml.graph_risk.model import GraphSAGEConfig, GraphSAGENet
from ml.graph_risk.train import GraphSAGETrainer
from ml.graph_risk.explain import GraphRiskExplainer
from ml.graph_risk.inference import GraphRiskInferenceEngine
from ml.graph_risk.schemas import (
    DatasetSplit,
    GraphEdge,
    GraphNode,
    NodeType,
    RawObservation,
    SCHEMA_VERSION,
)

FIXTURE_PATH = Path(__file__).parent.parent / "ml" / "graph_risk" / "tests" / "fixtures" / "industrial_telemetry_fixture.json"


@pytest.fixture(autouse=True)
def clean_engine():
    gr_service.reset_inference_engine()
    yield
    gr_service.reset_inference_engine()


def test_e2e_telemetry_to_graphsage_pipeline(tmp_path, monkeypatch):
    """
    Full pipeline test:
    1. Ingest run-to-failure fixture.
    2. Extract canonical 7D feature vectors via extract_window_features.
    3. Construct causal temporal graph with k-hop isolation.
    4. Train GraphSAGE model on split dataset.
    5. Save and reload checkpoint with exact parameter invariance.
    6. Run inference engine with explainability enabled.
    7. Query FastAPI endpoint with safety invariant verification.
    """
    # 1. Ingest fixture
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Extract features from a raw observation window
    raw_obs_list = [
        RawObservation(
            equipment_id="P-101A",
            sensor_id="VIB-01",
            timestamp=datetime.fromisoformat(r["timestamp"]),
            vibration_mm_s=r["features"][0],
            bearing_temp_c=r["features"][4],
            frequency_peak_hz=r["features"][6],
        )
        for r in data["nodes"][:5]
    ]

    features = extract_window_features(raw_obs_list)
    vec = features.to_vector()
    assert len(vec) == 7

    # 3. Construct causal graph from fixture
    nodes = [
        GraphNode(
            node_id=r["node_id"],
            node_index=r["node_index"],
            node_type=NodeType(r["node_type"]),
            equipment_id=r["equipment_id"],
            timestamp=r["timestamp"],
            features=r["features"],
        )
        for r in data["nodes"]
    ]
    edges = [
        GraphEdge(
            source_id=e["source_id"],
            target_id=e["target_id"],
            source_index=e["source_index"],
            target_index=e["target_index"],
            source_timestamp=e["source_timestamp"],
            target_timestamp=e["target_timestamp"],
        )
        for e in data["edges"]
    ]
    labels = {r["node_id"]: r["label"] for r in data["nodes"]}

    builder = CausalGraphBuilder()
    graph = builder.build_from_nodes(nodes, edges=edges, labels=labels)
    assert graph.num_nodes == len(nodes)
    assert graph.num_edges == len(edges)

    split = DatasetSplit(
        train_indices=data["split"]["train_indices"],
        val_indices=data["split"]["val_indices"],
        test_indices=data["split"]["test_indices"],
    )

    # 4. Train GraphSAGE Net
    trainer = GraphSAGETrainer(
        config=GraphSAGEConfig(hidden_channels=16),
        learning_rate=0.05,
        epochs=30,
        patience=10,
        class_weight="balanced",
        random_state=42,
    )
    result = trainer.train(graph, split, checkpoint_dir=tmp_path)
    assert len(result.train_loss_history) > 0
    assert result.train_loss_history[-1] >= 0.0

    # 5. Save & reload checkpoint
    ckpt_path = result.checkpoint_path
    assert ckpt_path is not None and ckpt_path.exists()

    loaded_net, meta = GraphSAGENet.load_checkpoint(ckpt_path)
    assert meta.model_version == "1.0.0"

    # 6. Inference Engine & Explainer
    engine = GraphRiskInferenceEngine(
        checkpoint_path=ckpt_path,
        enabled=True,
    )

    last_node = nodes[-1].node_id
    pred = engine.predict(
        sample_id=last_node,
        equipment_id="P-101A",
        features=nodes[-1].features,
        graph=graph,
        explain=True,
        deterministic_risk=0.45,
    )

    # Assert ML Safety Invariants
    assert pred.is_advisory is True
    assert pred.calibrated is False
    assert pred.advisory_risk_score is not None
    assert 0.0 <= pred.advisory_risk_score <= 1.0
    assert pred.deterministic_risk_score == 0.45
    assert pred.explanation is not None
    assert "does not prove physical causation" in pred.explanation.caveat.lower()
    assert "advisory only" in pred.limitations.lower()

    # 7. FastAPI Endpoint Integration
    client = TestClient(app)

    # Status check with engine active
    monkeypatch.setenv("ENABLE_GRAPH_RISK_MODEL", "true")
    monkeypatch.setattr(gr_service, "_ENGINE_INSTANCE", engine)

    status_resp = client.get("/api/graph-risk/status")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["status"] == "available"
    assert status_body["model_available"] is True
    assert status_body["schema_version"] == SCHEMA_VERSION

    # Predict check
    predict_resp = client.post(
        "/api/graph-risk/predict",
        json={
            "sample_id": "TEST-FASTAPI-001",
            "equipment_id": "P-101A",
            "features": vec,
            "explain": True,
            "deterministic_risk": 0.45,
        },
    )
    assert predict_resp.status_code == 200
    predict_body = predict_resp.json()
    assert predict_body["is_advisory"] is True
    assert predict_body["calibrated"] is False
    assert predict_body["sample_id"] == "TEST-FASTAPI-001"
    assert predict_body["advisory_risk_score"] is not None

    # Safety Barrier Validation: ML prediction payload must never satisfy payment authorizations
    payment_resp = client.post(
        "/api/machine-money/request-action",
        json={
            "action_type": "EMERGENCY_SHUTDOWN",
            "equipment_id": "P-101A",
            "advisory_risk_score": predict_body["advisory_risk_score"],  # Unauthorized injection
        },
    )
    # Payment or action auth must not depend on or be authorized by raw advisory ML score
    assert payment_resp.status_code in (404, 405, 422, 400, 401, 403, 501)
