"""Unit tests for Phase 4: GraphSAGENet architecture and checkpointing."""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.graph_risk.model import GraphSAGEConfig, GraphSAGENet
from ml.graph_risk.schemas import FEATURE_DIM, CheckpointMetadata


def test_graphsage_initialization_and_forward():
    config = GraphSAGEConfig(in_channels=FEATURE_DIM, hidden_channels=16, num_layers=2)
    model = GraphSAGENet(config=config, random_state=42)

    # 4 nodes, 3 directed edges (0->1, 1->2, 2->3)
    X = np.random.normal(size=(4, FEATURE_DIM)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    scores = model.forward(X, edge_index)
    assert scores.shape == (4,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_inductive_neighborhood_aggregation_impact():
    """Verify that changing a predecessor's features modifies the target's output."""
    model = GraphSAGENet(random_state=42)

    # 2 nodes: 0 -> 1
    X1 = np.ones((2, FEATURE_DIM), dtype=np.float32)
    edge_index = np.array([[0], [1]], dtype=np.int64)

    out1 = model.forward(X1, edge_index)

    # Modify predecessor (node 0) while keeping node 1 unchanged
    X2 = X1.copy()
    X2[0] += 5.0

    out2 = model.forward(X2, edge_index)

    # Node 1's score MUST change because it aggregates node 0's representation!
    assert out1[1] != out2[1]


def test_checkpoint_save_and_reload_invariance(tmp_path):
    model = GraphSAGENet(random_state=42)
    X = np.random.normal(size=(5, FEATURE_DIM)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    original_scores = model.forward(X, edge_index)

    checkpoint_file = tmp_path / "model_checkpoint.json"
    meta = CheckpointMetadata(val_pr_auc=0.92)
    model.save_checkpoint(checkpoint_file, metadata=meta)

    # Reload model from file
    reloaded_model, reloaded_meta = GraphSAGENet.load_checkpoint(checkpoint_file)
    assert reloaded_meta.val_pr_auc == 0.92
    assert reloaded_meta.schema_version == "1.0.0"

    reloaded_scores = reloaded_model.forward(X, edge_index)

    # Exact bit-for-bit mathematical parity
    np.testing.assert_allclose(original_scores, reloaded_scores, atol=1e-6)


def test_checkpoint_incompatible_rejection(tmp_path):
    checkpoint_file = tmp_path / "corrupted_checkpoint.json"
    data = {
        "metadata": {
            "schema_version": "99.0.0",  # Incompatible version!
            "input_dim": FEATURE_DIM,
            "architecture": "GraphSAGE",
        },
        "config": {},
        "weights": {},
    }
    with open(checkpoint_file, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Incompatible schema version"):
        GraphSAGENet.load_checkpoint(checkpoint_file)
