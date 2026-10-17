"""Unit tests for Phase 5: Graph explainability, bounded attribution, and result provenance."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pytest

from ml.graph_risk.explain import GraphRiskExplainer
from ml.graph_risk.graph_builder import CausalGraphBuilder
from ml.graph_risk.inference import GraphRiskInferenceEngine
from ml.graph_risk.model import GraphSAGENet
from ml.graph_risk.schemas import FEATURE_DIM, GraphNode, NodeType


def test_explain_node_with_predecessors():
    model = GraphSAGENet(random_state=42)
    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)

    # 3 nodes in sequence: OBS-0 -> OBS-1 -> OBS-2
    nodes = [
        GraphNode(
            node_id="OBS-0",
            node_index=0,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time,
            features=[10.0, 15.0, 1.8, 5.0, 95.0, 1.2, 236.4],  # Highly anomalous vibration
        ),
        GraphNode(
            node_id="OBS-1",
            node_index=1,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time + timedelta(hours=6),
            features=[6.0, 9.0, 1.6, 4.0, 80.0, 0.5, 236.4],
        ),
        GraphNode(
            node_id="OBS-2",
            node_index=2,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time + timedelta(hours=12),
            features=[4.0, 6.0, 1.5, 3.5, 70.0, 0.2, 66.6],
        ),
    ]

    builder = CausalGraphBuilder(max_temporal_window_hours=24.0)
    graph = builder.build_from_nodes(nodes)

    explainer = GraphRiskExplainer(model=model, max_hops=2, max_influential_nodes=3)
    explanation = explainer.explain_node(graph, target_node_id="OBS-2")

    assert explanation.status == "available"
    assert explanation.hops_analyzed == 2
    assert len(explanation.influential_nodes) > 0
    assert "does not prove physical causation" in explanation.caveat

    # Check returned entities: must strictly be real predecessor node IDs
    returned_ids = [item.node_id for item in explanation.influential_nodes]
    for nid in returned_ids:
        assert nid in {"OBS-0", "OBS-1"}
        assert nid != "OBS-2"  # Target cannot explain itself

    for item in explanation.influential_nodes:
        assert 0.0 <= item.influence_score <= 1.0
        assert item.dominant_feature is not None


def test_explain_isolated_node():
    model = GraphSAGENet(random_state=42)
    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)

    # Single isolated node
    nodes = [
        GraphNode(
            node_id="SOLO-NODE",
            node_index=0,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time,
            features=[2.0, 3.0, 1.4, 3.0, 55.0, 0.05, 33.3],
        )
    ]
    builder = CausalGraphBuilder()
    graph = builder.build_from_nodes(nodes)

    explainer = GraphRiskExplainer(model=model)
    explanation = explainer.explain_node(graph, target_node_id="SOLO-NODE")

    assert explanation.status == "isolated_node"
    assert len(explanation.influential_nodes) == 0
    assert "does not prove physical causation" in explanation.caveat


def test_explain_unknown_node_raises_key_error():
    model = GraphSAGENet(random_state=42)
    nodes = [
        GraphNode(
            node_id="OBS-0",
            node_index=0,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc),
            features=[2.0] * FEATURE_DIM,
        )
    ]
    graph = CausalGraphBuilder().build_from_nodes(nodes)
    explainer = GraphRiskExplainer(model=model)

    with pytest.raises(KeyError, match="Target node 'NONEXISTENT' not found"):
        explainer.explain_node(graph, target_node_id="NONEXISTENT")


def test_inference_engine_with_explain_flag(tmp_path):
    model = GraphSAGENet(random_state=42)
    checkpoint_file = tmp_path / "valid_checkpoint.json"
    model.save_checkpoint(checkpoint_file)

    engine = GraphRiskInferenceEngine(checkpoint_path=checkpoint_file, enabled=True)

    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    nodes = [
        GraphNode(
            node_id=f"OBS-{i}",
            node_index=i,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time + timedelta(hours=i * 6),
            features=[2.0 + i] * FEATURE_DIM,
        )
        for i in range(3)
    ]
    graph = CausalGraphBuilder().build_from_nodes(nodes)

    pred = engine.predict(
        sample_id="OBS-2",
        equipment_id="P-101A",
        features=[4.0] * FEATURE_DIM,
        graph=graph,
        explain=True,
    )

    assert pred.status.value == "available"
    assert pred.advisory_risk_score is not None
    assert pred.explanation is not None
    assert pred.explanation.status == "available"
    assert len(pred.explanation.influential_nodes) > 0
