"""GraphSAGE Neural Network Architecture for Industrial Risk Intelligence.

Implements multi-layer inductive GraphSAGE with explicit configuration, mean
neighborhood aggregation, numerical stability safeguards, and checkpointing discipline.
Adapted from transferable engineering patterns in fraud-detection-gnn.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ml.graph_risk.schemas import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    CheckpointMetadata,
)


@dataclass
class GraphSAGEConfig:
    """Explicit hyperparameters for GraphSAGE architecture."""
    in_channels: int = FEATURE_DIM
    hidden_channels: int = 32
    out_channels: int = 1
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    schema_version: str = SCHEMA_VERSION
    architecture_name: str = "GraphSAGE-2L-Mean"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphSAGEConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))),
        np.exp(np.clip(x, -50.0, 50.0)) / (1.0 + np.exp(np.clip(x, -50.0, 50.0)))
    )


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0.0, x)


class SAGEConvLayer:
    """Single inductive GraphSAGE layer with Mean neighborhood aggregation.

    Computes:
        h_v = activation( W_self * h_v + W_neigh * MEAN_{u in N(v)}(h_u) + b )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Optional[str] = "relu",
        random_state: int = 42,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        rng = np.random.RandomState(random_state)

        # Xavier / Glorot uniform initialization
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W_self = rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.W_neigh = rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.bias = np.zeros(out_dim, dtype=np.float32)

    def forward(
        self,
        X: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward inductive propagation over graph nodes.

        Args:
            X: Node representations (N, in_dim).
            edge_index: Directed graph edges (2, M), where row 0 = src, row 1 = dst.
            edge_weights: Optional weights for each edge (M,).

        Returns:
            Updated node representations (N, out_dim).
        """
        N = X.shape[0]
        M = edge_index.shape[1] if edge_index.ndim == 2 and edge_index.size > 0 else 0

        # Mean aggregation: accumulate sum of incoming neighbor features and count degrees
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

        # Safe element-wise division for degree normalization
        deg_safe = np.where(degrees > 0, degrees, 1.0)
        neigh_mean = neigh_sum / deg_safe

        # Linear projections
        out = np.matmul(X, self.W_self) + np.matmul(neigh_mean, self.W_neigh) + self.bias

        if self.activation == "relu":
            return relu(out)
        elif self.activation == "sigmoid":
            return sigmoid(out)
        return out


class GraphSAGENet:
    """Multi-layer GraphSAGE network for rotating machinery failure risk scoring."""

    def __init__(self, config: Optional[GraphSAGEConfig] = None, random_state: int = 42):
        self.config = config or GraphSAGEConfig()
        self.random_state = random_state

        if self.config.in_channels != FEATURE_DIM:
            raise ValueError(
                f"Config in_channels ({self.config.in_channels}) must match canonical FEATURE_DIM ({FEATURE_DIM})"
            )

        # Layer 1: in_channels -> hidden_channels (ReLU)
        self.layer1 = SAGEConvLayer(
            in_dim=self.config.in_channels,
            out_dim=self.config.hidden_channels,
            activation="relu",
            random_state=random_state,
        )

        # Layer 2: hidden_channels -> out_channels (Sigmoid output)
        self.layer2 = SAGEConvLayer(
            in_dim=self.config.hidden_channels,
            out_dim=self.config.out_channels,
            activation="sigmoid",
            random_state=random_state + 1,
        )

    def forward(
        self,
        X: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute inductive risk probabilities for each node in graph.

        Args:
            X: Node features (N, FEATURE_DIM).
            edge_index: Directed edge indices (2, M).
            edge_weights: Edge affinity weights (M,).

        Returns:
            1D array of predicted risk probabilities (N,) in [0.0, 1.0].
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected node features shape (*, {FEATURE_DIM}), got {X_arr.shape}")

        h1 = self.layer1.forward(X_arr, edge_index, edge_weights)
        h2 = self.layer2.forward(h1, edge_index, edge_weights)
        return h2.squeeze(-1)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        metadata: Optional[CheckpointMetadata] = None,
    ) -> Path:
        """Save network weights and metadata envelope strictly into a JSON bundle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = metadata or CheckpointMetadata(
            architecture=self.config.architecture_name,
            model_version=self.config.schema_version,
            schema_version=SCHEMA_VERSION,
            input_dim=self.config.in_channels,
            hidden_dim=self.config.hidden_channels,
            num_layers=self.config.num_layers,
            random_seed=self.random_state,
        )

        checkpoint_data = {
            "metadata": meta.model_dump(),
            "config": self.config.to_dict(),
            "weights": {
                "layer1_W_self": self.layer1.W_self.tolist(),
                "layer1_W_neigh": self.layer1.W_neigh.tolist(),
                "layer1_bias": self.layer1.bias.tolist(),
                "layer2_W_self": self.layer2.W_self.tolist(),
                "layer2_W_neigh": self.layer2.W_neigh.tolist(),
                "layer2_bias": self.layer2.bias.tolist(),
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        return path

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path]) -> Tuple["GraphSAGENet", CheckpointMetadata]:
        """Load and strictly validate a persisted GraphSAGE checkpoint bundle."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint bundle not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        meta_dict = bundle.get("metadata", {})
        if meta_dict.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible schema version: checkpoint has {meta_dict.get('schema_version')}, "
                f"expected {SCHEMA_VERSION}"
            )
        if meta_dict.get("input_dim") != FEATURE_DIM:
            raise ValueError(
                f"Incompatible feature dimension: checkpoint has {meta_dict.get('input_dim')}, "
                f"expected {FEATURE_DIM}"
            )

        metadata = CheckpointMetadata(**meta_dict)
        config = GraphSAGEConfig.from_dict(bundle.get("config", {}))

        model = cls(config=config, random_state=config.to_dict().get("random_seed", 42))

        # Restore weights
        w = bundle["weights"]
        model.layer1.W_self = np.array(w["layer1_W_self"], dtype=np.float32)
        model.layer1.W_neigh = np.array(w["layer1_W_neigh"], dtype=np.float32)
        model.layer1.bias = np.array(w["layer1_bias"], dtype=np.float32)
        model.layer2.W_self = np.array(w["layer2_W_self"], dtype=np.float32)
        model.layer2.W_neigh = np.array(w["layer2_W_neigh"], dtype=np.float32)
        model.layer2.bias = np.array(w["layer2_bias"], dtype=np.float32)

        return model, metadata
