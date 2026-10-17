"""Causal Graph Builder for Industrial Telemetry in KinetixRAG.

Constructs leakage-aware graph representations from timestamped equipment observations,
enforcing chronological edge direction (past -> present) and dense index validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ml.graph_risk.schemas import (
    FEATURE_DIM,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    RawObservation,
    WindowFeatures,
)


@dataclass
class IndustrialGraph:
    """In-memory tensor/index representation of an industrial condition graph."""
    node_ids: List[str]
    node_map: Dict[str, int]
    features: np.ndarray                       # Shape: (N, FEATURE_DIM)
    edge_index: np.ndarray                     # Shape: (2, M), dtype: int64
    edge_types: List[str]                      # Length: M
    edge_weights: np.ndarray                   # Shape: (M,)
    timestamps: List[datetime]                 # Length: N
    equipment_ids: List[str]                   # Length: N
    labels: Optional[np.ndarray] = None        # Shape: (N,), binary {0, 1}

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.ndim == 2 else 0

    def validate(self) -> "IndustrialGraph":
        """Strict structural integrity validation.

        Raises:
            ValueError: On dimension mismatches, index out-of-bounds, or temporal leakage.
        """
        N = self.num_nodes
        if N == 0:
            raise ValueError("Graph has 0 nodes; valid IndustrialGraph must contain at least 1 node")

        if self.features.ndim != 2 or self.features.shape[0] != N or self.features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Feature matrix shape mismatch: expected ({N}, {FEATURE_DIM}), got {self.features.shape}"
            )

        if len(self.timestamps) != N:
            raise ValueError(f"Timestamps length ({len(self.timestamps)}) does not match num_nodes ({N})")
        if len(self.equipment_ids) != N:
            raise ValueError(f"Equipment IDs length ({len(self.equipment_ids)}) does not match num_nodes ({N})")
        if len(self.node_map) != N:
            raise ValueError(f"Node map size ({len(self.node_map)}) does not match num_nodes ({N})")

        # Validate edge tensor
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"Edge index shape must be (2, M), got {self.edge_index.shape}")

        M = self.edge_index.shape[1]
        if len(self.edge_types) != M:
            raise ValueError(f"Edge types length ({len(self.edge_types)}) does not match num_edges ({M})")
        if len(self.edge_weights) != M:
            raise ValueError(f"Edge weights length ({len(self.edge_weights)}) does not match num_edges ({M})")

        if M > 0:
            src_indices = self.edge_index[0]
            dst_indices = self.edge_index[1]

            if np.any(src_indices < 0) or np.any(src_indices >= N):
                raise ValueError(f"Source edge index out of range [0, {N-1}]")
            if np.any(dst_indices < 0) or np.any(dst_indices >= N):
                raise ValueError(f"Target edge index out of range [0, {N-1}]")

            # Check temporal anti-leakage for all TEMPORAL_PRECEDES edges
            for i in range(M):
                if self.edge_types[i] == EdgeType.TEMPORAL_PRECEDES.value:
                    u = int(src_indices[i])
                    v = int(dst_indices[i])
                    t_u = self.timestamps[u]
                    t_v = self.timestamps[v]
                    if t_u > t_v:
                        raise ValueError(
                            f"Temporal leakage detected across edge {i} ({self.node_ids[u]} -> {self.node_ids[v]}): "
                            f"source timestamp {t_u} is after target timestamp {t_v}"
                        )

        if self.labels is not None:
            if len(self.labels) != N:
                raise ValueError(f"Labels length ({len(self.labels)}) does not match num_nodes ({N})")

        return self

    def to_pyg_dict(self) -> Dict[str, Any]:
        """Convert to standard PyTorch Geometric compatible dictionary without requiring torch."""
        return {
            "x": self.features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_weights,
            "y": self.labels,
            "num_nodes": self.num_nodes,
        }


class CausalGraphBuilder:
    """Builder for causal condition-monitoring graphs from telemetry observations."""

    def __init__(self, max_temporal_window_hours: float = 72.0, max_preceding_hops: int = 5):
        self.max_temporal_window_hours = max_temporal_window_hours
        self.max_preceding_hops = max_preceding_hops

    def build_from_nodes(
        self,
        nodes: List[GraphNode],
        edges: Optional[List[GraphEdge]] = None,
        labels: Optional[Dict[str, int]] = None,
    ) -> IndustrialGraph:
        """Construct an IndustrialGraph from a validated list of GraphNodes and optional GraphEdges."""
        if not nodes:
            raise ValueError("Cannot build graph from empty nodes list")

        # Sort nodes chronologically
        sorted_nodes = sorted(nodes, key=lambda n: n.timestamp)
        node_ids = [n.node_id for n in sorted_nodes]
        node_map = {n.node_id: idx for idx, n in enumerate(sorted_nodes)}

        features = np.array([n.features for n in sorted_nodes], dtype=np.float32)
        timestamps = [n.timestamp for n in sorted_nodes]
        equipment_ids = [n.equipment_id for n in sorted_nodes]

        src_list: List[int] = []
        dst_list: List[int] = []
        edge_types: List[str] = []
        edge_weights: List[float] = []

        if edges:
            for e in edges:
                if e.source_id not in node_map or e.target_id not in node_map:
                    continue
                u = node_map[e.source_id]
                v = node_map[e.target_id]
                src_list.append(u)
                dst_list.append(v)
                edge_types.append(e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type))
                edge_weights.append(float(e.weight))
        else:
            # Generate sequential temporal edges (u -> v where u precedes v)
            for i in range(len(sorted_nodes) - 1):
                t_i = timestamps[i]
                t_next = timestamps[i + 1]
                delta_h = (t_next - t_i).total_seconds() / 3600.0
                if 0.0 <= delta_h <= self.max_temporal_window_hours:
                    src_list.append(i)
                    dst_list.append(i + 1)
                    edge_types.append(EdgeType.TEMPORAL_PRECEDES.value)
                    # Exponential decay weight based on temporal proximity
                    decay_weight = float(np.exp(-delta_h / max(self.max_temporal_window_hours, 1.0)))
                    edge_weights.append(decay_weight)

        edge_index = np.array([src_list, dst_list], dtype=np.int64) if src_list else np.empty((2, 0), dtype=np.int64)
        edge_weights_arr = np.array(edge_weights, dtype=np.float32) if edge_weights else np.empty((0,), dtype=np.float32)

        label_arr = None
        if labels is not None:
            label_arr = np.array([labels.get(nid, 0) for nid in node_ids], dtype=np.int64)

        graph = IndustrialGraph(
            node_ids=node_ids,
            node_map=node_map,
            features=features,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_weights=edge_weights_arr,
            timestamps=timestamps,
            equipment_ids=equipment_ids,
            labels=label_arr,
        )
        return graph.validate()

    def extract_k_hop_subgraph(
        self,
        graph: IndustrialGraph,
        target_node_id: str,
        num_hops: int = 2,
    ) -> IndustrialGraph:
        """Extract bounded preceding causal k-hop subgraph ending at target_node_id.

        Guarantees that no nodes or edges with timestamps AFTER the target node are included.

        Args:
            graph: Full IndustrialGraph.
            target_node_id: Center node of interest (e.g. current evaluation sample).
            num_hops: Number of preceding hops to traverse.

        Returns:
            Validated re-indexed IndustrialGraph containing target node and its causal predecessors.
        """
        if target_node_id not in graph.node_map:
            raise KeyError(f"Target node '{target_node_id}' not found in graph")

        target_idx = graph.node_map[target_node_id]
        target_time = graph.timestamps[target_idx]

        # Reverse adjacency list (target <- sources) to find predecessors
        rev_adj: Dict[int, List[int]] = {i: [] for i in range(graph.num_nodes)}
        if graph.num_edges > 0:
            for e_idx in range(graph.num_edges):
                src = int(graph.edge_index[0, e_idx])
                dst = int(graph.edge_index[1, e_idx])
                rev_adj[dst].append(src)

        # BFS backwards up to num_hops
        included_nodes: Set[int] = {target_idx}
        frontier: Set[int] = {target_idx}

        for _ in range(num_hops):
            next_frontier: Set[int] = set()
            for curr in frontier:
                for pred in rev_adj.get(curr, []):
                    # Strict causal anti-leakage: pred timestamp must be <= target timestamp
                    if graph.timestamps[pred] <= target_time:
                        if pred not in included_nodes:
                            included_nodes.add(pred)
                            next_frontier.add(pred)
            frontier = next_frontier
            if not frontier:
                break

        # Sort selected nodes chronologically
        sorted_sub_nodes = sorted(list(included_nodes), key=lambda idx: graph.timestamps[idx])
        sub_node_map = {graph.node_ids[old_idx]: new_idx for new_idx, old_idx in enumerate(sorted_sub_nodes)}

        sub_node_ids = [graph.node_ids[i] for i in sorted_sub_nodes]
        sub_features = graph.features[sorted_sub_nodes]
        sub_timestamps = [graph.timestamps[i] for i in sorted_sub_nodes]
        sub_equipment_ids = [graph.equipment_ids[i] for i in sorted_sub_nodes]
        sub_labels = graph.labels[sorted_sub_nodes] if graph.labels is not None else None

        # Re-index edges among included nodes
        sub_src: List[int] = []
        sub_dst: List[int] = []
        sub_types: List[str] = []
        sub_weights: List[float] = []

        if graph.num_edges > 0:
            old_set = set(sorted_sub_nodes)
            for e_idx in range(graph.num_edges):
                old_u = int(graph.edge_index[0, e_idx])
                old_v = int(graph.edge_index[1, e_idx])
                if old_u in old_set and old_v in old_set:
                    sub_src.append(sub_node_map[graph.node_ids[old_u]])
                    sub_dst.append(sub_node_map[graph.node_ids[old_v]])
                    sub_types.append(graph.edge_types[e_idx])
                    sub_weights.append(float(graph.edge_weights[e_idx]))

        sub_edge_index = (
            np.array([sub_src, sub_dst], dtype=np.int64) if sub_src else np.empty((2, 0), dtype=np.int64)
        )
        sub_edge_weights = (
            np.array(sub_weights, dtype=np.float32) if sub_weights else np.empty((0,), dtype=np.float32)
        )

        subgraph = IndustrialGraph(
            node_ids=sub_node_ids,
            node_map=sub_node_map,
            features=sub_features,
            edge_index=sub_edge_index,
            edge_types=sub_types,
            edge_weights=sub_edge_weights,
            timestamps=sub_timestamps,
            equipment_ids=sub_equipment_ids,
            labels=sub_labels,
        )
        return subgraph.validate()
