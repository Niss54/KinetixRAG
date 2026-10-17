"""Backend service for GraphSAGE Risk Intelligence.

Provides lazy-loaded inference engine initialization, feature flag management,
and operational safety barriers ensuring ML scores cannot bypass governance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.graph_risk.inference import GraphRiskInferenceEngine
from ml.graph_risk.schemas import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    ModelOperationalStatus,
)

_ENGINE_INSTANCE: Optional[GraphRiskInferenceEngine] = None


def is_graph_risk_enabled() -> bool:
    """Return whether GraphSAGE ML intelligence is enabled in configuration.

    Default: False (PRD FR-07 & FR-09 safe deployment baseline).
    """
    val = os.environ.get("ENABLE_GRAPH_RISK_MODEL", "false").strip().lower()
    return val in ("true", "1", "yes", "on")


def get_checkpoint_path() -> Optional[Path]:
    """Resolve configured model checkpoint path if present."""
    custom_path = os.environ.get("GRAPH_RISK_CHECKPOINT_PATH")
    if custom_path:
        return Path(custom_path)

    # Default checkpoint location
    default_path = (
        Path(__file__).resolve().parents[2]
        / "ml"
        / "graph_risk"
        / "checkpoints"
        / "graphsage_best_checkpoint.json"
    )
    return default_path if default_path.exists() else None


def get_inference_engine(force_reload: bool = False) -> GraphRiskInferenceEngine:
    """Lazy-load and return the singleton GraphRiskInferenceEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None or force_reload:
        enabled = is_graph_risk_enabled()
        chk = get_checkpoint_path()
        _ENGINE_INSTANCE = GraphRiskInferenceEngine(
            checkpoint_path=chk,
            enabled=enabled,
        )
    return _ENGINE_INSTANCE


def reset_inference_engine() -> None:
    """Reset singleton engine instance for testing."""
    global _ENGINE_INSTANCE
    _ENGINE_INSTANCE = None


def get_graph_risk_status() -> Dict[str, Any]:
    """Retrieve current operational status of the GraphSAGE risk service."""
    engine = get_inference_engine()
    status = engine.get_status()
    status["feature_flag_enabled"] = is_graph_risk_enabled()
    status["feature_names"] = list(FEATURE_NAMES)
    status["feature_dim"] = FEATURE_DIM
    return status


def predict_graph_risk(
    sample_id: str,
    equipment_id: str,
    features: List[float],
    explain: bool = False,
    deterministic_risk: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute risk scoring with strict governance barrier.

    ML score is strictly advisory. It cannot trigger payments or bypass approvals.
    """
    engine = get_inference_engine()
    prediction = engine.predict(
        sample_id=sample_id,
        equipment_id=equipment_id,
        features=features,
        deterministic_risk=deterministic_risk,
        explain=explain,
    )
    res = prediction.model_dump()
    # Enforce advisory safety flag
    res["is_advisory"] = True
    res["calibrated"] = False
    return res
