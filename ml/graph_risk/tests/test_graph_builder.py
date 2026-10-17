"""Unit tests for Phase 2: Causal graph building and topological anti-leakage."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pytest

from ml.graph_risk.graph_builder import CausalGraphBuilder, IndustrialGraph
from ml.graph_risk.schemas import EdgeType, GraphEdge, GraphNode, NodeType

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "industrial_telemetry_fixture.json"


def test_industrial_graph_empty_rejection():
    with pytest.raises(ValueError, match="0 nodes"):
        IndustrialGraph(
            node_ids=[],
            node_map={},
            features=np.empty((0, 7), dtype=np.float32),
            edge_index=np.empty((2, 0), dtype=np.int64),
            edge_types=[],
            edge_weights=np.empty((0,), dtype=np.float32),
            timestamps=[],
            equipment_ids=[],
        ).validate()


def test_industrial_graph_single_node_valid():
    graph = IndustrialGraph(
        node_ids=["OBS-1"],
        node_map={"OBS-1": 0},
        features=np.ones((1, 7), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_types=[],
        edge_weights=np.empty((0,), dtype=np.float32),
        timestamps=[datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)],
        equipment_ids=["P-101A"],
    ).validate()

    assert graph.num_nodes == 1
    assert graph.num_edges == 0


def test_causal_graph_builder_automatic_sequential_edges():
    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    nodes = [
        GraphNode(
            node_id=f"OBS-{i}",
            node_index=i,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time + timedelta(hours=i * 6),
            features=[1.0, 2.0, 1.5, 3.0, 50.0 + i, 0.1, 33.3],
        )
        for i in range(4)
    ]

    builder = CausalGraphBuilder(max_temporal_window_hours=24.0)
    graph = builder.build_from_nodes(nodes)

    assert graph.num_nodes == 4
    # Sequential edges: 0->1, 1->2, 2->3
    assert graph.num_edges == 3
    assert np.array_equal(graph.edge_index[0], [0, 1, 2])
    assert np.array_equal(graph.edge_index[1], [1, 2, 3])
    assert all(t == EdgeType.TEMPORAL_PRECEDES.value for t in graph.edge_types)


def test_causal_graph_builder_anti_leakage_enforcement():
    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    t1 = base_time
    t2 = base_time + timedelta(hours=6)

    # Manually configure backward edge: node 1 (later) -> node 0 (earlier)
    graph = IndustrialGraph(
        node_ids=["OBS-0", "OBS-1"],
        node_map={"OBS-0": 0, "OBS-1": 1},
        features=np.ones((2, 7), dtype=np.float32),
        edge_index=np.array([[1], [0]], dtype=np.int64),  # 1 -> 0 backward!
        edge_types=[EdgeType.TEMPORAL_PRECEDES.value],
        edge_weights=np.array([1.0], dtype=np.float32),
        timestamps=[t1, t2],
        equipment_ids=["P-101A", "P-101A"],
    )

    with pytest.raises(ValueError, match="Temporal leakage detected"):
        graph.validate()


def test_extract_k_hop_subgraph_no_future_leakage():
    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    nodes = [
        GraphNode(
            node_id=f"OBS-{i}",
            node_index=i,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time + timedelta(hours=i * 12),
            features=[1.0, 2.0, 1.5, 3.0, 50.0 + i, 0.1, 33.3],
        )
        for i in range(5)  # OBS-0, OBS-1, OBS-2, OBS-3, OBS-4
    ]

    builder = CausalGraphBuilder(max_temporal_window_hours=48.0)
    full_graph = builder.build_from_nodes(nodes)
    assert full_graph.num_nodes == 5

    # Extract 2-hop causal predecessor subgraph for OBS-2 (time = base_time + 24h)
    subgraph = builder.extract_k_hop_subgraph(full_graph, target_node_id="OBS-2", num_hops=2)

    # Strict invariant: OBS-3 and OBS-4 are in the future of OBS-2 and must NOT be in the subgraph!
    assert "OBS-3" not in subgraph.node_ids
    assert "OBS-4" not in subgraph.node_ids
    assert set(subgraph.node_ids) == {"OBS-0", "OBS-1", "OBS-2"}
    assert subgraph.num_nodes == 3

    # The re-indexed edge tensor must have indices strictly in [0, 2]
    assert np.all(subgraph.edge_index < 3)
    assert np.all(subgraph.edge_index >= 0)


def test_to_pyg_dict_format():
    base_time = datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc)
    nodes = [
        GraphNode(
            node_id=f"OBS-{i}",
            node_index=i,
            node_type=NodeType.SENSOR_OBSERVATION,
            equipment_id="P-101A",
            timestamp=base_time + timedelta(hours=i),
            features=[1.0, 2.0, 1.5, 3.0, 50.0, 0.1, 33.3],
        )
        for i in range(2)
    ]
    builder = CausalGraphBuilder()
    graph = builder.build_from_nodes(nodes, labels={"OBS-0": 0, "OBS-1": 1})

    pyg_dict = graph.to_pyg_dict()
    assert "x" in pyg_dict
    assert "edge_index" in pyg_dict
    assert "edge_attr" in pyg_dict
    assert "y" in pyg_dict
    assert pyg_dict["num_nodes"] == 2
    assert pyg_dict["y"].tolist() == [0, 1]


def test_graph_building_from_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = [
        GraphNode(
            node_id=r["node_id"],
            node_index=r["node_index"],
            node_type=NodeType(r["node_type"]),
            equipment_id=r["equipment_id"],
            timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
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
            edge_type=EdgeType(e["edge_type"]),
            weight=e.get("weight", 1.0),
            source_timestamp=datetime.fromisoformat(e["source_timestamp"].replace("Z", "+00:00")),
            target_timestamp=datetime.fromisoformat(e["target_timestamp"].replace("Z", "+00:00")),
        )
        for e in data["edges"]
    ]

    labels = {r["node_id"]: r["label"] for r in data["nodes"]}

    builder = CausalGraphBuilder()
    graph = builder.build_from_nodes(nodes, edges=edges, labels=labels)

    assert graph.num_nodes == 7
    assert graph.num_edges == 6
    assert graph.labels is not None
    assert np.array_equal(graph.labels, [0, 0, 0, 0, 1, 1, 1])
