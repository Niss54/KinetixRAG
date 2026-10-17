"""Graph Explainability and Attribution Engine for Industrial Risk Intelligence.

Implements bounded k-hop subgraph extraction, leave-one-out node attribution,
and dominant sensor feature identification without asserting physical causality.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ml.graph_risk.graph_builder import CausalGraphBuilder, IndustrialGraph
from ml.graph_risk.model import GraphSAGENet
from ml.graph_risk.schemas import (
    FEATURE_NAMES,
    AdvisoryExplanation,
    ExplanationItem,
)


class GraphRiskExplainer:
    """Bounded explainability engine computing predecessor influence on risk scores."""

    def __init__(
        self,
        model: GraphSAGENet,
        max_hops: int = 2,
        max_influential_nodes: int = 5,
    ):
        self.model = model
        self.max_hops = max_hops
        self.max_influential_nodes = max_influential_nodes
        self.graph_builder = CausalGraphBuilder(max_preceding_hops=max_hops)

    def explain_node(
        self,
        graph: IndustrialGraph,
        target_node_id: str,
    ) -> AdvisoryExplanation:
        """Compute bounded causal explanation for a target node in an IndustrialGraph.

        Args:
            graph: Full IndustrialGraph.
            target_node_id: Canonical ID of node being explained.

        Returns:
            AdvisoryExplanation object conforming to schema contract.
        """
        if target_node_id not in graph.node_map:
            raise KeyError(f"Target node '{target_node_id}' not found in graph")

        try:
            # Extract bounded predecessor subgraph terminating at target
            subgraph = self.graph_builder.extract_k_hop_subgraph(
                graph=graph,
                target_node_id=target_node_id,
                num_hops=self.max_hops,
            )
        except Exception:
            return AdvisoryExplanation(
                status="insufficient_neighborhood",
                hops_analyzed=self.max_hops,
                influential_nodes=[],
                caveat="Explanation reflects graph model attribution only; does not prove physical causation.",
            )

        if subgraph.num_nodes <= 1:
            # Isolated node with no predecessors
            return AdvisoryExplanation(
                status="isolated_node",
                hops_analyzed=self.max_hops,
                influential_nodes=[],
                caveat="Explanation reflects graph model attribution only; does not prove physical causation.",
            )

        target_sub_idx = subgraph.node_map[target_node_id]

        # Base forward pass score on the intact subgraph
        base_scores = self.model.forward(
            subgraph.features,
            subgraph.edge_index,
            subgraph.edge_weights,
        )
        base_target_score = float(base_scores[target_sub_idx])

        # Compute perturbation attribution for each predecessor node
        node_deltas: List[Tuple[str, float, str, str]] = []

        for sub_idx, node_id in enumerate(subgraph.node_ids):
            if sub_idx == target_sub_idx:
                continue  # Explain predecessors only

            # Zero-out node features to measure counterfactual impact
            perturbed_features = subgraph.features.copy()
            perturbed_features[sub_idx] = 0.0

            perturbed_scores = self.model.forward(
                perturbed_features,
                subgraph.edge_index,
                subgraph.edge_weights,
            )
            perturbed_target_score = float(perturbed_scores[target_sub_idx])

            delta = abs(base_target_score - perturbed_target_score)

            # Identify dominant feature on this predecessor node
            node_feat = subgraph.features[sub_idx]
            dom_idx = int(np.argmax(np.abs(node_feat)))
            dom_feature_name = FEATURE_NAMES[dom_idx] if dom_idx < len(FEATURE_NAMES) else "unknown"

            node_time_str = subgraph.timestamps[sub_idx].isoformat()
            node_deltas.append((node_id, delta, node_time_str, dom_feature_name))

        if not node_deltas:
            return AdvisoryExplanation(
                status="insufficient_neighborhood",
                hops_analyzed=self.max_hops,
                influential_nodes=[],
                caveat="Explanation reflects graph model attribution only; does not prove physical causation.",
            )

        # Normalize deltas to influence scores in [0.0, 1.0]
        total_delta = sum(d[1] for d in node_deltas)
        items: List[ExplanationItem] = []

        # Sort predecessors by delta descending
        node_deltas.sort(key=lambda x: x[1], reverse=True)

        for node_id, delta, ts_str, dom_feat in node_deltas[: self.max_influential_nodes]:
            normalized_score = float(delta / total_delta) if total_delta > 1e-6 else 1.0 / len(node_deltas)
            items.append(
                ExplanationItem(
                    node_id=node_id,
                    entity_type="SENSOR_OBSERVATION",
                    influence_score=round(min(1.0, max(0.0, normalized_score)), 3),
                    timestamp=ts_str,
                    dominant_feature=dom_feat,
                )
            )

        return AdvisoryExplanation(
            status="available",
            hops_analyzed=self.max_hops,
            influential_nodes=items,
            caveat="Explanation reflects graph model attribution only; does not prove physical causation.",
        )
