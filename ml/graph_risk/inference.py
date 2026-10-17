"""Inference engine for GraphSAGE Risk Intelligence in KinetixRAG.

Provides safe, lazy-loaded runtime execution, checkpoint compatibility validation,
and graceful degradation when models are disabled or missing.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ml.graph_risk.graph_builder import IndustrialGraph
from ml.graph_risk.model import GraphSAGENet
from ml.graph_risk.schemas import (
    FEATURE_DIM,
    SCHEMA_VERSION,
    AdvisoryPrediction,
    CheckpointMetadata,
    ModelOperationalStatus,
)


class GraphRiskInferenceEngine:
    """Safe inference manager for advisory GraphSAGE scores."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model: Optional[GraphSAGENet] = None
        self.metadata: Optional[CheckpointMetadata] = None
        self.load_error: Optional[str] = None

        if self.enabled and self.checkpoint_path and self.checkpoint_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        try:
            self.model, self.metadata = GraphSAGENet.load_checkpoint(self.checkpoint_path)
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.metadata = None
            self.load_error = str(exc)

    def get_status(self) -> Dict[str, Any]:
        """Return operational readiness and model provenance."""
        if not self.enabled:
            return {
                "status": ModelOperationalStatus.DISABLED.value,
                "model_available": False,
                "model_version": None,
                "schema_version": SCHEMA_VERSION,
                "detail": "GraphSAGE model is disabled by configuration.",
            }

        if self.load_error:
            return {
                "status": ModelOperationalStatus.INCOMPATIBLE_FEATURES.value,
                "model_available": False,
                "model_version": None,
                "schema_version": SCHEMA_VERSION,
                "detail": f"Model checkpoint loading failed: {self.load_error}",
            }

        if self.model is None:
            return {
                "status": ModelOperationalStatus.MODEL_NOT_TRAINED.value,
                "model_available": False,
                "model_version": None,
                "schema_version": SCHEMA_VERSION,
                "detail": "No model checkpoint loaded. Model is in uninitialized state.",
            }

        return {
            "status": ModelOperationalStatus.AVAILABLE.value,
            "model_available": True,
            "model_version": self.metadata.model_version if self.metadata else "1.0.0",
            "schema_version": SCHEMA_VERSION,
            "architecture": self.metadata.architecture if self.metadata else "GraphSAGE-2L-Mean",
            "val_pr_auc": self.metadata.val_pr_auc if self.metadata else None,
            "detail": "GraphSAGE advisory model loaded and verified.",
        }

    def predict(
        self,
        sample_id: str,
        equipment_id: str,
        features: List[float],
        graph: Optional[IndustrialGraph] = None,
        deterministic_risk: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> AdvisoryPrediction:
        """Score an observation or graph node, returning standardized AdvisoryPrediction."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Handle disabled or unavailable states
        if not self.enabled:
            return AdvisoryPrediction(
                sample_id=sample_id,
                equipment_id=equipment_id,
                timestamp=ts,
                status=ModelOperationalStatus.DISABLED,
                advisory_risk_score=None,
                deterministic_risk_score=deterministic_risk,
                limitations="Advisory ML score unavailable: feature flag ENABLE_GRAPH_RISK_MODEL is false.",
            )

        if self.model is None:
            return AdvisoryPrediction(
                sample_id=sample_id,
                equipment_id=equipment_id,
                timestamp=ts,
                status=ModelOperationalStatus.MODEL_NOT_TRAINED,
                advisory_risk_score=None,
                deterministic_risk_score=deterministic_risk,
                limitations="Advisory ML score unavailable: no trained checkpoint loaded.",
            )

        # Validate input features
        feat_arr = np.asarray(features, dtype=np.float32)
        if len(feat_arr) != FEATURE_DIM:
            return AdvisoryPrediction(
                sample_id=sample_id,
                equipment_id=equipment_id,
                timestamp=ts,
                status=ModelOperationalStatus.INCOMPATIBLE_FEATURES,
                advisory_risk_score=None,
                deterministic_risk_score=deterministic_risk,
                limitations=f"Expected {FEATURE_DIM} features, got {len(feat_arr)}.",
            )

        try:
            if graph is not None:
                # Predict over provided graph topology
                scores = self.model.forward(graph.features, graph.edge_index, graph.edge_weights)
                if sample_id in graph.node_map:
                    target_idx = graph.node_map[sample_id]
                    risk_score = float(scores[target_idx])
                else:
                    risk_score = float(scores[-1])
            else:
                # Isolated node inference (empty edges)
                X = feat_arr.reshape(1, -1)
                E = np.empty((2, 0), dtype=np.int64)
                scores = self.model.forward(X, E)
                risk_score = float(scores[0])

            return AdvisoryPrediction(
                sample_id=sample_id,
                equipment_id=equipment_id,
                timestamp=ts,
                status=ModelOperationalStatus.AVAILABLE,
                advisory_risk_score=round(risk_score, 4),
                calibrated=False,
                is_advisory=True,
                deterministic_risk_score=deterministic_risk,
                model_version=self.metadata.model_version if self.metadata else "1.0.0",
                schema_version=SCHEMA_VERSION,
            )
        except Exception as exc:
            return AdvisoryPrediction(
                sample_id=sample_id,
                equipment_id=equipment_id,
                timestamp=ts,
                status=ModelOperationalStatus.INFERENCE_ERROR,
                advisory_risk_score=None,
                deterministic_risk_score=deterministic_risk,
                limitations=f"Inference computation error: {str(exc)}",
            )
