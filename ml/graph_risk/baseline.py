"""Non-graph tabular baseline model for industrial risk intelligence.

Provides reproducible tabular models (RandomForest and LogisticRegression) trained
on identical industrial feature matrices and splits as GraphSAGE, handling class
imbalance with rigorous split isolation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.graph_risk.evaluate import EvaluationMetrics, calculate_metrics
from ml.graph_risk.features import FeatureScaler
from ml.graph_risk.schemas import FEATURE_DIM, FEATURE_NAMES


class BaselineAlgorithm(str, Enum):
    """Supported tabular baseline architectures."""
    RANDOM_FOREST = "RandomForest"
    LOGISTIC_REGRESSION = "LogisticRegression"


class TabularBaselineModel:
    """Tabular classifier baseline operating on non-graph node feature matrices."""

    def __init__(
        self,
        algorithm: BaselineAlgorithm = BaselineAlgorithm.RANDOM_FOREST,
        random_state: int = 42,
        class_weight: str = "balanced",
        n_estimators: int = 50,
        max_depth: Optional[int] = 5,
        c_reg: float = 1.0,
    ):
        self.algorithm = algorithm
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.c_reg = c_reg
        self.is_fitted: bool = False
        self.model: Optional[Union[RandomForestClassifier, LogisticRegression]] = None
        self.scaler: Optional[FeatureScaler] = None
        self._init_model()

    def _init_model(self) -> None:
        if self.algorithm == BaselineAlgorithm.RANDOM_FOREST:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
        elif self.algorithm == BaselineAlgorithm.LOGISTIC_REGRESSION:
            self.model = LogisticRegression(
                C=self.c_reg,
                random_state=self.random_state,
                class_weight=self.class_weight,
                max_iter=500,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale_inputs: bool = True,
    ) -> "TabularBaselineModel":
        """Fit baseline exclusively on training split data.

        Args:
            X_train: Dense feature matrix of shape (N_train, FEATURE_DIM).
            y_train: Binary labels of shape (N_train,).
            scale_inputs: If True, fits and applies FeatureScaler.
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.int64)

        if X_arr.ndim != 2 or X_arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected X_train shape (*, {FEATURE_DIM}), got {X_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Mismatch: len(X_train)={len(X_arr)}, len(y_train)={len(y_arr)}")
        if len(X_arr) == 0:
            raise ValueError("Cannot train on empty dataset")

        if scale_inputs:
            self.scaler = FeatureScaler().fit(X_arr)
            X_scaled = self.scaler.transform(X_arr)
        else:
            X_scaled = X_arr

        # Handle edge case where training set contains only a single class
        unique_classes = np.unique(y_arr)
        if len(unique_classes) == 1:
            self.is_fitted = True
            self._single_class = int(unique_classes[0])
            return self

        self._single_class = None
        self.model.fit(X_scaled, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class (impending failure) probabilities."""
        if not self.is_fitted:
            raise RuntimeError("TabularBaselineModel must be fitted before predict_proba")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} columns, got {X_arr.shape[1]}")

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_arr)
        else:
            X_scaled = X_arr

        if self._single_class is not None:
            return np.full(len(X_arr), float(self._single_class), dtype=np.float64)

        probabilities = self.model.predict_proba(X_scaled)
        # Return probability of positive class (index 1)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities[:, 0]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict discrete binary labels based on operating threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def evaluate(
        self,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        split_name: str = "test",
        threshold: float = 0.5,
    ) -> EvaluationMetrics:
        """Evaluate model performance on a specified split."""
        probs = self.predict_proba(X_eval)
        model_tag = f"Baseline-{self.algorithm.value}"
        return calculate_metrics(
            y_true=y_eval,
            y_score=probs,
            threshold=threshold,
            model_name=model_tag,
            split_name=split_name,
        )

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance or normalized weight coefficients."""
        if not self.is_fitted or self._single_class is not None:
            return {name: 1.0 / FEATURE_DIM for name in FEATURE_NAMES}

        if hasattr(self.model, "feature_importances_"):
            raw_weights = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            raw_weights = np.abs(self.model.coef_[0])
            total = np.sum(raw_weights)
            raw_weights = raw_weights / (total if total > 1e-6 else 1.0)
        else:
            raw_weights = np.ones(FEATURE_DIM) / FEATURE_DIM

        return {FEATURE_NAMES[i]: round(float(raw_weights[i]), 4) for i in range(FEATURE_DIM)}
