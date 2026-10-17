"""Unit tests for Phase 2: Feature extraction and leakage-aware scaling."""

import math
from datetime import datetime, timezone
import numpy as np
import pytest

from ml.graph_risk.features import (
    FeatureScaler,
    calculate_crest_factor,
    calculate_kurtosis,
    calculate_peak,
    calculate_rms,
    extract_window_features,
)
from ml.graph_risk.schemas import FEATURE_DIM, RawObservation


def test_signal_statistics_primitives():
    # Symmetric square-wave-like signal
    signal = np.array([2.0, -2.0, 2.0, -2.0], dtype=np.float64)
    assert calculate_rms(signal) == pytest.approx(2.0)
    assert calculate_peak(signal) == 2.0
    assert calculate_crest_factor(2.0, 2.0) == 1.0

    # Sine wave (peak = 1, rms = 1/sqrt(2) ~ 0.707, crest factor ~ sqrt(2) ~ 1.414)
    t = np.linspace(0, 2 * np.pi, 1000)
    sine = np.sin(t)
    rms_sine = calculate_rms(sine)
    peak_sine = calculate_peak(sine)
    assert rms_sine == pytest.approx(1.0 / np.sqrt(2), rel=1e-2)
    assert calculate_crest_factor(peak_sine, rms_sine) == pytest.approx(np.sqrt(2), rel=1e-2)

    # Constant signal kurtosis returns 3.0
    const_signal = np.array([5.0, 5.0, 5.0, 5.0])
    assert calculate_kurtosis(const_signal) == 3.0


def test_extract_window_features_valid():
    readings = [
        RawObservation(
            equipment_id="P-101A",
            sensor_id="VIB-01",
            timestamp=datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc),
            vibration_mm_s=2.0,
            bearing_temp_c=50.0,
            frequency_peak_hz=33.3,
        ),
        RawObservation(
            equipment_id="P-101A",
            sensor_id="VIB-01",
            timestamp=datetime(2004, 2, 12, 11, 0, tzinfo=timezone.utc),
            vibration_mm_s=3.0,
            bearing_temp_c=52.0,
            frequency_peak_hz=33.3,
        ),
        RawObservation(
            equipment_id="P-101A",
            sensor_id="VIB-01",
            timestamp=datetime(2004, 2, 12, 12, 0, tzinfo=timezone.utc),
            vibration_mm_s=4.0,
            bearing_temp_c=54.0,
            frequency_peak_hz=33.3,
        ),
    ]

    wf = extract_window_features(readings)
    assert wf.vibration_peak_mm_s == 4.0
    assert wf.bearing_temp_c == pytest.approx(52.0)
    # Rate of change: (54 - 50) / 2 hours = 2.0 C/hr
    assert wf.temp_rate_of_change_c_per_hr == pytest.approx(2.0, rel=1e-2)
    assert wf.frequency_dominant_hz == 33.3
    assert len(wf.to_vector()) == FEATURE_DIM


def test_extract_window_features_negative_cases():
    # Empty readings
    with pytest.raises(ValueError, match="empty list"):
        extract_window_features([])

    # NaN in readings
    with pytest.raises(ValueError):
        extract_window_features([
            RawObservation(
                equipment_id="P-101A",
                sensor_id="VIB-01",
                timestamp=datetime(2004, 2, 12, 10, 0, tzinfo=timezone.utc),
                vibration_mm_s=float("nan"),
                bearing_temp_c=50.0,
            )
        ])


def test_feature_scaler_lifecycle():
    # Generate synthetic training matrix (100 samples, 7 features)
    np.random.seed(42)
    train_matrix = np.random.normal(loc=[2.0, 3.0, 1.5, 3.0, 60.0, 0.5, 33.0], scale=[0.5, 0.8, 0.2, 0.5, 5.0, 0.1, 2.0], size=(100, 7))

    scaler = FeatureScaler()
    assert scaler.is_fitted is False

    # Calling transform before fit raises RuntimeError
    with pytest.raises(RuntimeError):
        scaler.transform(train_matrix)

    # Fit and transform
    train_transformed = scaler.fit_transform(train_matrix)
    assert scaler.is_fitted is True
    assert train_transformed.shape == (100, 7)

    # Transformed training data should have approximately mean 0 and std 1
    np.testing.assert_allclose(np.mean(train_transformed, axis=0), np.zeros(7), atol=1e-5)
    np.testing.assert_allclose(np.std(train_transformed, axis=0), np.ones(7), atol=1e-5)

    # Test data transformed with TRAIN parameters (no test leakage into parameters)
    test_matrix = np.array([[2.0, 3.0, 1.5, 3.0, 60.0, 0.5, 33.0]])
    test_transformed = scaler.transform(test_matrix)
    assert test_transformed.shape == (1, 7)

    # Serialization roundtrip
    state = scaler.to_dict()
    reloaded_scaler = FeatureScaler.from_dict(state)
    assert reloaded_scaler.is_fitted is True
    reloaded_transformed = reloaded_scaler.transform(test_matrix)
    np.testing.assert_allclose(test_transformed, reloaded_transformed)


def test_feature_scaler_shape_validation():
    scaler = FeatureScaler()
    with pytest.raises(ValueError, match="Expected 7 columns"):
        scaler.fit(np.zeros((10, 5)))  # Only 5 columns
