"""Unit tests for Phase 4: GraphSAGE training loop, class weighting, and early stopping."""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.graph_risk.graph_builder import CausalGraphBuilder
from ml.graph_risk.model import GraphSAGEConfig
from ml.graph_risk.schemas import DatasetSplit, GraphEdge, GraphNode, NodeType
from ml.graph_risk.train import GraphSAGETrainer

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "industrial_telemetry_fixture.json"


def test_graphsage_training_smoke_test(tmp_path):
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    split = DatasetSplit(
        train_indices=data["split"]["train_indices"],
        val_indices=data["split"]["val_indices"],
        test_indices=data["split"]["test_indices"],
    )

    trainer = GraphSAGETrainer(
        config=GraphSAGEConfig(hidden_channels=16),
        learning_rate=0.05,
        epochs=40,
        patience=10,
        class_weight="balanced",
        random_state=42,
    )

    result = trainer.train(graph, split, checkpoint_dir=tmp_path)

    assert result.best_epoch > 0
    assert len(result.train_loss_history) > 0
    assert result.val_metrics is not None
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_graphsage_training_reproducibility():
    # Synthetic graph (10 nodes)
    np.random.seed(42)
    X = np.random.normal(size=(10, 7)).astype(np.float32)
    edge_index = np.array([list(range(9)), list(range(1, 10))], dtype=np.int64)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    from ml.graph_risk.graph_builder import IndustrialGraph
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    graph = IndustrialGraph(
        node_ids=[f"N-{i}" for i in range(10)],
        node_map={f"N-{i}": i for i in range(10)},
        features=X,
        edge_index=edge_index,
        edge_types=["TEMPORAL_PRECEDES"] * 9,
        edge_weights=np.ones(9, dtype=np.float32),
        timestamps=[t0 + timedelta(hours=i) for i in range(10)],
        equipment_ids=["P-101A"] * 10,
        labels=y,
    ).validate()

    split = DatasetSplit(
        train_indices=[0, 1, 2, 3, 4, 6, 7],
        val_indices=[5, 8],
        test_indices=[9],
    )

    t1 = GraphSAGETrainer(epochs=10, random_state=42)
    r1 = t1.train(graph, split)

    t2 = GraphSAGETrainer(epochs=10, random_state=42)
    r2 = t2.train(graph, split)

    # Both runs should yield identical validation losses
    np.testing.assert_allclose(r1.val_loss_history, r2.val_loss_history, atol=1e-5)
