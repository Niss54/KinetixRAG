"""Data contracts and schemas for GraphSAGE Risk Intelligence in KinetixRAG.

Defines versioned schemas for telemetry observations, feature vectors, causal graph
topology, failure labels, dataset splits, checkpoint metadata, and advisory predictions.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"

# Ordered canonical feature contract for industrial rotating equipment (pumps/bearings)
FEATURE_NAMES: Tuple[str, ...] = (
    "vibration_rms_mm_s",
    "vibration_peak_mm_s",
    "vibration_crest_factor",
    "vibration_kurtosis",
    "bearing_temp_c",
    "temp_rate_of_change_c_per_hr",
    "frequency_dominant_hz",
)
FEATURE_DIM = len(FEATURE_NAMES)  # 7 features


class NodeType(str, Enum):
    """Types of nodes supported in the industrial risk graph."""
    SENSOR_OBSERVATION = "SENSOR_OBSERVATION"
    EQUIPMENT = "EQUIPMENT"
    FAILURE_EVENT = "FAILURE_EVENT"


class EdgeType(str, Enum):
    """Types of directed edges in the industrial risk graph."""
    TEMPORAL_PRECEDES = "TEMPORAL_PRECEDES"  # Chronological edge (t1 -> t2 where t1 < t2)
    CO_LOCATED = "CO_LOCATED"                # Multi-sensor spatial connection on same equipment
    MONITORS = "MONITORS"                    # Sensor -> Equipment relationship


class FailureStage(str, Enum):
    """Industrial failure progression stages based on ISO 10816-3."""
    HEALTHY_BASELINE = "HEALTHY_BASELINE"
    NOMINAL_STABLE = "NOMINAL_STABLE"
    INITIAL_DEGRADATION = "INITIAL_DEGRADATION"
    ZONE_C_ANOMALY = "ZONE_C_ANOMALY"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


class SplitStrategy(str, Enum):
    """Leakage-aware dataset splitting strategies."""
    TEMPORAL_HOLDOUT = "temporal_holdout"
    RUN_TO_FAILURE_HOLD_OUT = "run_to_failure_hold_out"


class ModelOperationalStatus(str, Enum):
    """Operational availability status of the graph risk model."""
    AVAILABLE = "available"
    DISABLED = "disabled"
    MODEL_NOT_TRAINED = "model_not_trained"
    INCOMPATIBLE_FEATURES = "incompatible_features"
    INFERENCE_ERROR = "inference_error"


class RawObservation(BaseModel):
    """Single empirical industrial telemetry point."""
    equipment_id: str = Field(..., description="Canonical equipment identifier (e.g. P-101A)")
    sensor_id: str = Field(..., description="Unique sensor tag (e.g. REPLAY-SENSOR-BEARING-01)")
    timestamp: datetime = Field(..., description="Observation UTC timestamp")
    vibration_mm_s: float = Field(..., ge=0.0, description="Vibration velocity RMS in mm/s")
    bearing_temp_c: float = Field(..., ge=-40.0, le=250.0, description="Bearing temperature in Celsius")
    frequency_peak_hz: float = Field(default=0.0, ge=0.0, description="Dominant vibration frequency in Hz")
    vibration_rms_g: Optional[float] = Field(default=None, description="Raw acceleration in g (optional)")
    iso_zone: Optional[str] = Field(default=None, description="ISO 10816 vibration zone classification")
    stage: Optional[str] = Field(default=None, description="Failure progression stage if labeled")


class WindowFeatures(BaseModel):
    """Normalized statistical feature vector extracted from a temporal window."""
    vibration_rms_mm_s: float = Field(..., ge=0.0)
    vibration_peak_mm_s: float = Field(..., ge=0.0)
    vibration_crest_factor: float = Field(..., ge=1.0)
    vibration_kurtosis: float = Field(...)
    bearing_temp_c: float = Field(..., ge=-40.0, le=250.0)
    temp_rate_of_change_c_per_hr: float = Field(...)
    frequency_dominant_hz: float = Field(..., ge=0.0)

    def to_vector(self) -> List[float]:
        """Return features strictly adhering to canonical FEATURE_NAMES ordering."""
        return [getattr(self, name) for name in FEATURE_NAMES]

    @field_validator(
        "vibration_rms_mm_s",
        "vibration_peak_mm_s",
        "vibration_crest_factor",
        "vibration_kurtosis",
        "bearing_temp_c",
        "temp_rate_of_change_c_per_hr",
        "frequency_dominant_hz",
    )
    @classmethod
    def validate_finite(cls, v: float) -> float:
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Feature values must be finite (no NaN or Inf)")
        return float(v)


class GraphNode(BaseModel):
    """Node in the industrial risk graph."""
    node_id: str = Field(..., description="Globally unique node identifier")
    node_index: int = Field(..., ge=0, description="Dense 0-indexed integer for tensor mapping")
    node_type: NodeType = Field(default=NodeType.SENSOR_OBSERVATION)
    equipment_id: str = Field(..., description="Associated equipment identifier")
    timestamp: datetime = Field(..., description="Timestamp of observation")
    features: List[float] = Field(..., description="Dense feature vector matching FEATURE_DIM")

    @field_validator("features")
    @classmethod
    def validate_features_shape(cls, v: List[float]) -> List[float]:
        if len(v) != FEATURE_DIM:
            raise ValueError(f"Feature vector must have exactly {FEATURE_DIM} dimensions, got {len(v)}")
        import math
        for idx, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature at index {idx} must be finite, got {val}")
        return v


class GraphEdge(BaseModel):
    """Directed edge in the industrial risk graph enforcing causal chronology."""
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    source_index: int = Field(..., ge=0, description="Dense source tensor index")
    target_index: int = Field(..., ge=0, description="Dense target tensor index")
    edge_type: EdgeType = Field(default=EdgeType.TEMPORAL_PRECEDES)
    weight: float = Field(default=1.0, ge=0.0)
    source_timestamp: Optional[datetime] = None
    target_timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_causal_direction(self) -> "GraphEdge":
        """Anti-leakage validator: temporal edges must strictly flow past -> present."""
        if self.edge_type == EdgeType.TEMPORAL_PRECEDES:
            if self.source_timestamp and self.target_timestamp:
                if self.source_timestamp > self.target_timestamp:
                    raise ValueError(
                        f"Temporal leakage detected: source_timestamp ({self.source_timestamp}) "
                        f"is after target_timestamp ({self.target_timestamp})."
                    )
        return self


class RiskLabel(BaseModel):
    """Ground truth risk label for supervised GNN and baseline models."""
    sample_id: str = Field(..., description="Identifier of the evaluated observation/node")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    is_impending_failure: int = Field(..., ge=0, le=1, description="Binary target: 1 = failure within horizon, 0 = normal")
    horizon_hours: float = Field(..., gt=0.0, description="Look-ahead prediction horizon in hours")
    stage: FailureStage = Field(default=FailureStage.HEALTHY_BASELINE)
    iso_zone: Optional[str] = None


class DatasetSplit(BaseModel):
    """Leakage-aware dataset partitioning."""
    train_indices: List[int] = Field(..., description="Dense node indices for training")
    val_indices: List[int] = Field(..., description="Dense node indices for validation")
    test_indices: List[int] = Field(..., description="Dense node indices for test (held out)")
    split_strategy: SplitStrategy = Field(default=SplitStrategy.TEMPORAL_HOLDOUT)

    @model_validator(mode="after")
    def validate_no_overlap(self) -> "DatasetSplit":
        train_set = set(self.train_indices)
        val_set = set(self.val_indices)
        test_set = set(self.test_indices)

        overlap_tv = train_set.intersection(val_set)
        if overlap_tv:
            raise ValueError(f"Train and validation splits overlap on indices: {overlap_tv}")

        overlap_tt = train_set.intersection(test_set)
        if overlap_tt:
            raise ValueError(f"Train and test splits overlap on indices: {overlap_tt}")

        overlap_vt = val_set.intersection(test_set)
        if overlap_vt:
            raise ValueError(f"Validation and test splits overlap on indices: {overlap_vt}")

        return self


class ExplanationItem(BaseModel):
    """Single contributing entity/sensor in an explainability response."""
    node_id: str
    entity_type: str
    influence_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: str
    dominant_feature: Optional[str] = None


class AdvisoryExplanation(BaseModel):
    """Graph explanation summary."""
    status: str = Field(..., description="available, timeout, or insufficient_neighborhood")
    hops_analyzed: int = Field(default=2, ge=1)
    influential_nodes: List[ExplanationItem] = Field(default_factory=list)
    caveat: str = Field(
        default="Explanation reflects graph model attribution only; does not prove physical causation.",
        description="Mandatory scientific limitation disclosure"
    )


class AdvisoryPrediction(BaseModel):
    """Inference output schema for FastAPI and operator dashboard."""
    sample_id: str
    equipment_id: str
    timestamp: str
    status: ModelOperationalStatus = ModelOperationalStatus.AVAILABLE
    advisory_risk_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="GraphSAGE predicted risk score"
    )
    calibrated: bool = Field(default=False, description="Whether probability calibration has been performed")
    is_advisory: bool = Field(default=True, description="Strict indicator that score is non-operational advisory")
    deterministic_risk_score: Optional[float] = Field(
        default=None, description="Parallel deterministic baseline risk score for comparison"
    )
    model_version: str = Field(default="1.0.0")
    schema_version: str = Field(default=SCHEMA_VERSION)
    explanation: Optional[AdvisoryExplanation] = None
    limitations: str = Field(
        default="Advisory only. Must not bypass human review or automate payment disbursement."
    )


class CheckpointMetadata(BaseModel):
    """Metadata envelope embedded in model checkpoints for safety verification."""
    architecture: str = Field(default="GraphSAGE-2L-Mean")
    model_version: str = Field(default="1.0.0")
    schema_version: str = Field(default=SCHEMA_VERSION)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_dim: int = Field(default=FEATURE_DIM)
    hidden_dim: int = Field(default=64)
    num_layers: int = Field(default=2)
    random_seed: int = Field(default=42)
    dataset_name: str = Field(default="NASA IMS Bearing Run-to-Failure")
    train_pr_auc: Optional[float] = None
    val_pr_auc: Optional[float] = None
    test_pr_auc: Optional[float] = None
    hash_sha256: Optional[str] = None
