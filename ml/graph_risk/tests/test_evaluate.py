"""Unit tests for Phase 3: Evaluation harness and metric calculation."""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.graph_risk.evaluate import (
    EvaluationMetrics,
    calculate_metrics,
    format_evaluation_report,
    save_evaluation_report,
)


def test_calculate_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8]

    metrics = calculate_metrics(y_true, y_score, threshold=0.5, model_name="TestModel", split_name="test")
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]
    assert metrics.total_samples == 4
    assert metrics.positive_count == 2
    assert metrics.negative_count == 2
    assert metrics.positive_prevalence == 0.5


def test_calculate_metrics_imbalanced():
    # 90 negatives, 10 positives
    np.random.seed(42)
    y_true = np.array([0] * 90 + [1] * 10)
    # Good classifier gives higher scores to positives
    y_score = np.concatenate([
        np.random.uniform(0.0, 0.4, size=90),
        np.random.uniform(0.6, 1.0, size=10),
    ])

    metrics = calculate_metrics(y_true, y_score, threshold=0.5, model_name="ImbalanceTest", split_name="val")
    assert metrics.pr_auc > 0.8
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.9
    assert metrics.f1 > 0.8
    assert metrics.positive_count == 10
    assert metrics.negative_count == 90
    assert metrics.positive_prevalence == 0.1


def test_calculate_metrics_negative_cases():
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_metrics([0, 1], [0.5])

    with pytest.raises(ValueError, match="empty arrays"):
        calculate_metrics([], [])


def test_save_and_format_evaluation_report(tmp_path):
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8]
    metrics = calculate_metrics(y_true, y_score, model_name="Baseline-RF", split_name="test")

    # Format markdown report
    md = format_evaluation_report(metrics)
    assert "### Evaluation Report: Baseline-RF (test split)" in md
    assert "PR-AUC:" in md
    assert "Confusion Matrix:" in md

    # Save JSON report
    report_file = tmp_path / "baseline_report.json"
    saved_path = save_evaluation_report(metrics, report_file)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_name"] == "Baseline-RF"
    assert data["pr_auc"] == 1.0
    assert data["total_samples"] == 4
