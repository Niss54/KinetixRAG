"""Inductive GraphSAGE Training Pipeline with Validation-Driven Early Stopping.

Implements reproducible gradient-based training with class imbalance weighting,
masked node evaluation (no test leakage), validation early stopping, and checkpoint saving.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ml.graph_risk.evaluate import EvaluationMetrics, calculate_metrics
from ml.graph_risk.graph_builder import IndustrialGraph
from ml.graph_risk.model import GraphSAGEConfig, GraphSAGENet, relu, sigmoid
from ml.graph_risk.schemas import CheckpointMetadata, DatasetSplit


@dataclass
class TrainingResult:
    """Artifact of a GraphSAGE training run."""
    model: GraphSAGENet
    best_epoch: int
    train_loss_history: List[float]
    val_loss_history: List[float]
    val_metrics: EvaluationMetrics
    test_metrics: Optional[EvaluationMetrics] = None
    checkpoint_path: Optional[Path] = None


def compute_neighbor_aggregation(X: np.ndarray, edge_index: np.ndarray, edge_weights: Optional[np.ndarray]) -> np.ndarray:
    """Compute mean neighborhood feature aggregation for each node."""
    N = X.shape[0]
    M = edge_index.shape[1] if edge_index.ndim == 2 and edge_index.size > 0 else 0

    neigh_sum = np.zeros_like(X)
    degrees = np.zeros((N, 1), dtype=np.float32)

    if M > 0:
        src = edge_index[0]
        dst = edge_index[1]
        weights = edge_weights if edge_weights is not None else np.ones(M, dtype=np.float32)

        for i in range(M):
            u = src[i]
            v = dst[i]
            w = weights[i]
            neigh_sum[v] += w * X[u]
            degrees[v] += w

    deg_safe = np.where(degrees > 0, degrees, 1.0)
    return neigh_sum / deg_safe


class GraphSAGETrainer:
    """Trainer for GraphSAGENet implementing class-weighted BCE and early stopping."""

    def __init__(
        self,
        config: Optional[GraphSAGEConfig] = None,
        learning_rate: float = 0.02,
        epochs: int = 150,
        patience: int = 20,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.config = config or GraphSAGEConfig()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.class_weight = class_weight
        self.random_state = random_state

    def train(
        self,
        graph: IndustrialGraph,
        split: DatasetSplit,
        checkpoint_dir: Optional[Path] = None,
    ) -> TrainingResult:
        """Execute supervised training on graph nodes partitioned by split.

        Args:
            graph: Full IndustrialGraph containing features, edge_index, and labels.
            split: DatasetSplit with disjoint train, val, and test indices.
            checkpoint_dir: Optional directory to save best validation checkpoint.

        Returns:
            TrainingResult with trained model and evaluation histories.
        """
        if graph.labels is None:
            raise ValueError("Cannot train GraphSAGE: graph.labels is None")

        train_idx = np.array(split.train_indices, dtype=np.int64)
        val_idx = np.array(split.val_indices, dtype=np.int64)
        test_idx = np.array(split.test_indices, dtype=np.int64) if split.test_indices else None

        X = graph.features
        E = graph.edge_index
        W = graph.edge_weights
        y = graph.labels.astype(np.float32)

        # Calculate positive class imbalance weight on TRAINING data only
        y_train = y[train_idx]
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)

        if self.class_weight == "balanced" and pos_count > 0 and neg_count > 0:
            pos_weight = float(neg_count / pos_count)
        else:
            pos_weight = 1.0

        model = GraphSAGENet(config=self.config, random_state=self.random_state)

        # Precompute neighbor aggregation for Layer 1
        M_X = compute_neighbor_aggregation(X, E, W)

        # Adam optimizer state variables
        params = [
            model.layer1.W_self, model.layer1.W_neigh, model.layer1.bias,
            model.layer2.W_self, model.layer2.W_neigh, model.layer2.bias,
        ]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_weights = None

        train_loss_history: List[float] = []
        val_loss_history: List[float] = []

        N_train = len(train_idx)
        weight_mask = np.ones(N_train, dtype=np.float32)
        weight_mask[y_train == 1] = pos_weight

        for epoch in range(1, self.epochs + 1):
            # Forward pass
            z1 = np.matmul(X, model.layer1.W_self) + np.matmul(M_X, model.layer1.W_neigh) + model.layer1.bias
            h1 = relu(z1)

            M_h1 = compute_neighbor_aggregation(h1, E, W)
            z2 = np.matmul(h1, model.layer2.W_self) + np.matmul(M_h1, model.layer2.W_neigh) + model.layer2.bias
            p = sigmoid(z2).squeeze(-1)

            # Training loss on train_idx
            p_train = p[train_idx]
            p_clip = np.clip(p_train, 1e-7, 1.0 - 1e-7)
            bce_loss = - (weight_mask * (y_train * np.log(p_clip) + (1.0 - y_train) * np.log(1.0 - p_clip)))
            train_loss = float(np.mean(bce_loss))
            train_loss_history.append(train_loss)

            # Backpropagation gradients
            g_z2 = np.zeros(graph.num_nodes, dtype=np.float32)
            g_z2[train_idx] = (weight_mask / N_train) * (p_train - y_train)
            g_z2 = g_z2.reshape(-1, 1)

            grad_W2_self = np.matmul(h1.T, g_z2)
            grad_W2_neigh = np.matmul(M_h1.T, g_z2)
            grad_b2 = np.sum(g_z2, axis=0)

            # Backward through Layer 1
            g_h1 = np.matmul(g_z2, model.layer2.W_self.T)
            g_z1 = g_h1 * (z1 > 0).astype(np.float32)

            grad_W1_self = np.matmul(X.T, g_z1)
            grad_W1_neigh = np.matmul(M_X.T, g_z1)
            grad_b1 = np.sum(g_z1, axis=0)

            grads = [
                grad_W1_self, grad_W1_neigh, grad_b1,
                grad_W2_self, grad_W2_neigh, grad_b2,
            ]

            # Adam parameter update
            t = epoch
            for i in range(len(params)):
                m_adam[i] = beta1 * m_adam[i] + (1.0 - beta1) * grads[i]
                v_adam[i] = beta2 * v_adam[i] + (1.0 - beta2) * (grads[i] ** 2)
                m_hat = m_adam[i] / (1.0 - beta1 ** t)
                v_hat = v_adam[i] / (1.0 - beta2 ** t)
                params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # Validation loss evaluation
            p_val = p[val_idx]
            p_val_clip = np.clip(p_val, 1e-7, 1.0 - 1e-7)
            y_val = y[val_idx]
            val_loss = float(-np.mean(y_val * np.log(p_val_clip) + (1.0 - y_val) * np.log(1.0 - p_val_clip)))
            val_loss_history.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_weights = [p.copy() for p in params]
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best weights
        if best_weights is not None:
            model.layer1.W_self = best_weights[0]
            model.layer1.W_neigh = best_weights[1]
            model.layer1.bias = best_weights[2]
            model.layer2.W_self = best_weights[3]
            model.layer2.W_neigh = best_weights[4]
            model.layer2.bias = best_weights[5]

        # Final evaluation on validation set
        final_p = model.forward(X, E, W)
        val_metrics = calculate_metrics(
            y_true=y[val_idx],
            y_score=final_p[val_idx],
            model_name="GraphSAGE",
            split_name="validation",
        )

        test_metrics = None
        if test_idx is not None and len(test_idx) > 0:
            test_metrics = calculate_metrics(
                y_true=y[test_idx],
                y_score=final_p[test_idx],
                model_name="GraphSAGE",
                split_name="test",
            )

        checkpoint_path = None
        if checkpoint_dir:
            chk_dir = Path(checkpoint_dir)
            chk_dir.mkdir(parents=True, exist_ok=True)
            chk_file = chk_dir / "graphsage_best_checkpoint.json"
            meta = CheckpointMetadata(
                architecture=self.config.architecture_name,
                model_version="1.0.0",
                train_pr_auc=float(val_metrics.pr_auc),
                val_pr_auc=float(val_metrics.pr_auc),
                test_pr_auc=float(test_metrics.pr_auc) if test_metrics else None,
            )
            checkpoint_path = model.save_checkpoint(chk_file, metadata=meta)

        return TrainingResult(
            model=model,
            best_epoch=best_epoch,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            checkpoint_path=checkpoint_path,
        )
