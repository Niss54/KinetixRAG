"""FastAPI API Router for GraphSAGE Risk Intelligence.

Exposes status and advisory prediction endpoints adhering to strict governance,
lazy runtime loading, and safe degradation when models are disabled.
"""

import math
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services.graph_risk import (
    get_graph_risk_status,
    predict_graph_risk,
)
from ml.graph_risk.schemas import FEATURE_DIM, FEATURE_NAMES

router = APIRouter(tags=["graph-risk"])


class GraphRiskPredictRequest(BaseModel):
    """Payload for requesting an advisory GraphSAGE risk prediction."""
    sample_id: str = Field(..., description="Unique observation/sample identifier")
    equipment_id: str = Field(..., description="Target equipment identifier (e.g. P-101A)")
    features: List[float] = Field(
        ...,
        description=f"Ordered {FEATURE_DIM}-dimensional feature vector matching: {', '.join(FEATURE_NAMES)}",
    )
    explain: bool = Field(default=False, description="Whether to compute bounded k-hop attribution")
    deterministic_risk: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Parallel deterministic baseline risk score"
    )


@router.get("/graph-risk/status")
def status_endpoint() -> Dict[str, Any]:
    """Retrieve the operational readiness, schema version, and feature flag state."""
    try:
        return get_graph_risk_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "status_check_failed", "detail": "Failed to query model status."},
        ) from exc


@router.post("/graph-risk/predict")
def predict_endpoint(payload: GraphRiskPredictRequest) -> Dict[str, Any]:
    """Calculate advisory risk score for a single observation or equipment window.

    Advisory only. Score cannot authorize payments, shut down equipment, or bypass approvals.
    """
    # Strict validation of input feature length
    if len(payload.features) != FEATURE_DIM:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "incompatible_feature_dimension",
                "detail": f"Expected exactly {FEATURE_DIM} features, got {len(payload.features)}.",
                "canonical_features": list(FEATURE_NAMES),
            },
        )

    # Validate finite values
    for idx, val in enumerate(payload.features):
        if math.isnan(val) or math.isinf(val):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "non_finite_feature_value",
                    "detail": f"Feature at index {idx} ({FEATURE_NAMES[idx]}) must be finite, got {val}.",
                },
            )

    try:
        result = predict_graph_risk(
            sample_id=payload.sample_id,
            equipment_id=payload.equipment_id,
            features=payload.features,
            explain=payload.explain,
            deterministic_risk=payload.deterministic_risk,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "inference_execution_error", "detail": "Safe degraded response: inference error occurred."},
        ) from exc
