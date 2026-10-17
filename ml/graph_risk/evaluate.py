"""Evaluation harness for industrial risk intelligence models.

Calculates standardized classification metrics prioritising PR-AUC for imbalanced
industrial failure forecasting, produces machine-readable JSON reports, and formats
comparative assessment summaries.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.graph_risk.schemas import SCHEMA_VERSION


@dataclass
class EvaluationMetrics:
    """Standardized metric report for risk intelligence models."""
    model_name: str
    split_name: str
    pr_auc: float
    roc_auc: Optional[float]
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    total_samples: int
    positive_count: int
    negative_count: int
    positive_prevalence: float
    threshold: float
    feature_schema_version: str = SCHEMA_VERSION
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["timestamp"]:
            d["timestamp"] = datetime.now(timezone.utc).isoformat()
        return d


def calculate_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_score: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
    model_name: str = "Model",
    split_name: str = "test",
) -> EvaluationMetrics:
    """Calculate classification and ranking metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_score: Continuous predicted risk scores in [0.0, 1.0].
        threshold: Operating decision threshold for binary classification.
        model_name: Name of the evaluated architecture.
        split_name: Name of the evaluated dataset split.

    Returns:
        EvaluationMetrics object.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(f"Length mismatch: y_true has {len(y_true_arr)}, y_score has {len(y_score_arr)}")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays")

    y_pred_arr = (y_score_arr >= threshold).astype(np.int64)

    total = int(len(y_true_arr))
    positives = int(np.sum(y_true_arr == 1))
    negatives = int(np.sum(y_true_arr == 0))
    prevalence = float(positives / total) if total > 0 else 0.0

    # Handle single-class edge cases in small splits/fixtures
    unique_classes = set(y_true_arr)
    if len(unique_classes) > 1:
        try:
            pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
        except Exception:
            pr_auc = 0.0
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
        except Exception:
            roc_auc = None
    else:
        # Default for homogeneous splits
        pr_auc = 1.0 if 1 in unique_classes and np.all(y_pred_arr == 1) else 0.0
        roc_auc = None

    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()

    return EvaluationMetrics(
        model_name=model_name,
        split_name=split_name,
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        confusion_matrix=cm,
        total_samples=total,
        positive_count=positives,
        negative_count=negatives,
        positive_prevalence=round(prevalence, 4),
        threshold=round(float(threshold), 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_evaluation_report(metrics: EvaluationMetrics, output_file: Union[str, Path]) -> Path:
    """Save metrics to machine-readable JSON file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    return path


def format_evaluation_report(metrics: EvaluationMetrics) -> str:
    """Format metrics into a human-readable markdown table."""
    roc_str = f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A (single-class split)"
    cm = metrics.confusion_matrix
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    return f"""### Evaluation Report: {metrics.model_name} ({metrics.split_name} split)
- **Evaluation Date:** {metrics.timestamp}
- **PR-AUC:** `{metrics.pr_auc:.4f}` (Primary imbalanced metric)
- **ROC-AUC:** `{roc_str}`
- **F1-Score:** `{metrics.f1:.4f}` (Operating threshold = {metrics.threshold})
- **Precision:** `{metrics.precision:.4f}`
- **Recall:** `{metrics.recall:.4f}`
- **Sample Count:** {metrics.total_samples} (Positives: {metrics.positive_count}, Negatives: {metrics.negative_count}, Prevalence: {metrics.positive_prevalence*100:.1f}%)
- **Confusion Matrix:**
  | Actual \\ Predicted | Negative (0) | Positive (1) |
  |---|---|---|
  | **Normal (0)** | TN = {tn} | FP = {fp} |
  | **Impending Failure (1)** | FN = {fn} | TP = {tp} |
"""
