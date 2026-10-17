"""Deterministic industrial feature extraction and scaling pipeline.

Extracts canonical 7-dimensional statistical and kinematic features from
rotating machinery telemetry windows, and provides leakage-free scaling fit
strictly on training splits.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ml.graph_risk.schemas import (
    FEATURE_DIM,
    FEATURE_NAMES,
    RawObservation,
    WindowFeatures,
)


def calculate_rms(values: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) of a signal array."""
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(values ** 2)))


def calculate_peak(values: np.ndarray) -> float:
    """Calculate peak absolute amplitude of a signal array."""
    if len(values) == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def calculate_crest_factor(peak: float, rms: float, epsilon: float = 1e-6) -> float:
    """Calculate crest factor (Peak / RMS) with numerical safety."""
    if rms < epsilon:
        return 1.0
    return float(max(1.0, peak / rms))


def calculate_kurtosis(values: np.ndarray, epsilon: float = 1e-6) -> float:
    """Calculate sample kurtosis (4th normalized moment).

    Returns 3.0 for constant or near-zero variance signal (normal distribution equivalent).
    """
    if len(values) < 2:
        return 3.0
    variance = np.var(values)
    if variance < epsilon:
        return 3.0
    mean_val = np.mean(values)
    fourth_moment = np.mean((values - mean_val) ** 4)
    return float(fourth_moment / (variance ** 2))


def extract_window_features(readings: List[RawObservation]) -> WindowFeatures:
    """Extract canonical WindowFeatures from a list of timestamped sensor readings.

    Args:
        readings: Chronologically ordered observations from one sensor/equipment window.

    Raises:
        ValueError: If readings is empty or contains non-finite values.
    """
    if not readings:
        raise ValueError("Cannot extract features from an empty list of readings")

    # Sort strictly by timestamp to maintain temporal ordering
    sorted_readings = sorted(readings, key=lambda r: r.timestamp)

    vibrations = np.array([r.vibration_mm_s for r in sorted_readings], dtype=np.float64)
    temperatures = np.array([r.bearing_temp_c for r in sorted_readings], dtype=np.float64)
    frequencies = np.array([r.frequency_peak_hz for r in sorted_readings], dtype=np.float64)

    # Check for NaN / Inf in inputs
    if not np.all(np.isfinite(vibrations)):
        raise ValueError("Non-finite values detected in vibration readings")
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("Non-finite values detected in temperature readings")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Non-finite values detected in frequency readings")

    rms = calculate_rms(vibrations)
    peak = calculate_peak(vibrations)
    crest_factor = calculate_crest_factor(peak, rms)
    kurt = calculate_kurtosis(vibrations)

    mean_temp = float(np.mean(temperatures))

    # Calculate temperature rate of change (degrees C per hour)
    if len(sorted_readings) > 1:
        t_start = sorted_readings[0].timestamp
        t_end = sorted_readings[-1].timestamp
        delta_hours = max((t_end - t_start).total_seconds() / 3600.0, 1e-4)
        delta_temp = temperatures[-1] - temperatures[0]
        temp_rate = float(delta_temp / delta_hours)
    else:
        temp_rate = 0.0

    dom_freq = float(np.median(frequencies)) if len(frequencies) > 0 else 0.0

    return WindowFeatures(
        vibration_rms_mm_s=rms,
        vibration_peak_mm_s=peak,
        vibration_crest_factor=crest_factor,
        vibration_kurtosis=kurt,
        bearing_temp_c=mean_temp,
        temp_rate_of_change_c_per_hr=temp_rate,
        frequency_dominant_hz=dom_freq,
    )


class FeatureScaler:
    """Deterministic feature scaler with zero-leakage training fit discipline.

    Stores mean and standard deviation learned exclusively from training samples,
    and applies standard z-score normalization with bounds clipping to prevent outlier explosion.
    """

    def __init__(self, clip_range: Tuple[float, float] = (-10.0, 10.0)):
        self.clip_min, self.clip_max = clip_range
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.feature_names: Tuple[str, ...] = FEATURE_NAMES
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """Fit scaler parameters on training data matrix X (N x D).

        Raises:
            ValueError: If X has wrong shape or non-finite entries.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Feature matrix X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} columns matching FEATURE_NAMES, got {X_arr.shape[1]}")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Feature matrix contains non-finite entries (NaN or Inf)")

        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        # Avoid division by zero for constant features
        self.scale_ = np.where(std < 1e-6, 1.0, std)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected shape (*, {FEATURE_DIM}), got {X_arr.shape}")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Transform input contains non-finite entries")

        normalized = (X_arr - self.mean_) / self.scale_
        return np.clip(normalized, self.clip_min, self.clip_max)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler on X and return transformed matrix."""
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler parameters for checkpoint metadata envelope."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted FeatureScaler")
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "clip_range": [self.clip_min, self.clip_max],
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureScaler":
        """Deserialize scaler parameters from checkpoint metadata."""
        scaler = cls(clip_range=tuple(data.get("clip_range", (-10.0, 10.0))))
        scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        scaler.scale_ = np.array(data["scale"], dtype=np.float64)
        scaler.feature_names = tuple(data.get("feature_names", FEATURE_NAMES))
        scaler.is_fitted = True
        return scaler
