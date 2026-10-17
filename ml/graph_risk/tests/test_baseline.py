"""Unit tests for Phase 3: Tabular baseline model and split evaluation."""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.graph_risk.baseline import BaselineAlgorithm, TabularBaselineModel
from ml.graph_risk.schemas import FEATURE_DIM, FEATURE_NAMES

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "industrial_telemetry_fixture.json"


@pytest.fixture
def synthetic_industrial_data():
    np.random.seed(42)
    # 80 normal samples (low vib, low temp)
    normal_x = np.random.normal(loc=[2.0, 3.0, 1.4, 3.0, 55.0, 0.05, 33.3], scale=[0.3, 0.5, 0.1, 0.2, 3.0, 0.02, 1.0], size=(80, 7))
    normal_y = np.zeros(80, dtype=np.int64)

    # 20 failure samples (high vib, high temp, elevated kurtosis)
    failure_x = np.random.normal(loc=[6.5, 11.0, 1.8, 5.0, 85.0, 0.8, 236.0], scale=[1.0, 2.0, 0.2, 0.5, 5.0, 0.2, 5.0], size=(20, 7))
    failure_y = np.ones(20, dtype=np.int64)

    X = np.vstack([normal_x, failure_x])
    y = np.concatenate([normal_y, failure_y])

    # Shuffle indices
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split: 70 train, 30 test
    return X[:70], y[:70], X[70:], y[70:]


def test_random_forest_baseline_lifecycle(synthetic_industrial_data):
    X_train, y_train, X_test, y_test = synthetic_industrial_data

    baseline = TabularBaselineModel(algorithm=BaselineAlgorithm.RANDOM_FOREST, random_state=42)
    assert baseline.is_fitted is False

    with pytest.raises(RuntimeError):
        baseline.predict_proba(X_test)

    baseline.fit(X_train, y_train)
    assert baseline.is_fitted is True

    probs = baseline.predict_proba(X_test)
    assert len(probs) == len(X_test)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = baseline.predict(X_test, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})

    metrics = baseline.evaluate(X_test, y_test, split_name="test")
    assert metrics.pr_auc > 0.80
    assert metrics.f1 > 0.70
    assert metrics.total_samples == len(X_test)


def test_logistic_regression_baseline_lifecycle(synthetic_industrial_data):
    X_train, y_train, X_test, y_test = synthetic_industrial_data

    baseline = TabularBaselineModel(algorithm=BaselineAlgorithm.LOGISTIC_REGRESSION, random_state=42)
    baseline.fit(X_train, y_train)

    probs = baseline.predict_proba(X_test)
    assert len(probs) == len(X_test)
    metrics = baseline.evaluate(X_test, y_test, split_name="test")
    assert metrics.pr_auc > 0.70


def test_baseline_reproducibility(synthetic_industrial_data):
    X_train, y_train, X_test, _ = synthetic_industrial_data

    m1 = TabularBaselineModel(algorithm=BaselineAlgorithm.RANDOM_FOREST, random_state=42).fit(X_train, y_train)
    m2 = TabularBaselineModel(algorithm=BaselineAlgorithm.RANDOM_FOREST, random_state=42).fit(X_train, y_train)

    p1 = m1.predict_proba(X_test)
    p2 = m2.predict_proba(X_test)
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_baseline_feature_importances(synthetic_industrial_data):
    X_train, y_train, _, _ = synthetic_industrial_data
    baseline = TabularBaselineModel(algorithm=BaselineAlgorithm.RANDOM_FOREST, random_state=42).fit(X_train, y_train)

    importances = baseline.get_feature_importances()
    assert len(importances) == FEATURE_DIM
    assert set(importances.keys()) == set(FEATURE_NAMES)
    total_weight = sum(importances.values())
    assert total_weight == pytest.approx(1.0, rel=1e-2)


def test_baseline_evaluation_on_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    X = np.array([n["features"] for n in nodes], dtype=np.float64)
    y = np.array([n["label"] for n in nodes], dtype=np.int64)

    # Fixture split: train on nodes 0..3 (normal), val on 4..5 (degraded), test on 6 (critical)
    train_idx = data["split"]["train_indices"]
    val_idx = data["split"]["val_indices"]

    baseline = TabularBaselineModel(algorithm=BaselineAlgorithm.RANDOM_FOREST, random_state=42)
    # Train model on train split
    baseline.fit(X[train_idx], y[train_idx])

    # Evaluate on val split
    metrics = baseline.evaluate(X[val_idx], y[val_idx], split_name="val")
    assert metrics.total_samples == len(val_idx)
