from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime

from gnn.shared.utils.graph_utils import EDGE_FEATURE_SCHEMA, NODE_FEATURE_SCHEMA

current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Derived from the shared node schema so it stays in sync with feature changes automatically.
NATIVE_NODE_FEATURE_COUNT = len(NODE_FEATURE_SCHEMA)
NATIVE_EDGE_FEATURE_COUNT = len(EDGE_FEATURE_SCHEMA)
# 8 global state scalars (solver removed — it is the network's action, not a feature).
NATIVE_GLOBAL_FEATURE_COUNT = 8
PADDED_NODE_FEATURE_COUNT = NATIVE_NODE_FEATURE_COUNT
PADDED_EDGE_FEATURE_COUNT = NATIVE_EDGE_FEATURE_COUNT
PADDED_GLOBAL_FEATURE_COUNT = NATIVE_GLOBAL_FEATURE_COUNT

NODE_INPUT_DIM_CHOICES = (4, 5)
EDGE_INPUT_DIM_CHOICES = (4, 8)
GLOBAL_INPUT_DIM_CHOICES = (6, 8)

# Suffix for Optuna study/DB names. Bump (or change choices) when categorical search
# spaces change so load_if_exists does not reuse incompatible distributions.
OPTUNA_SEARCH_SPACE_SUFFIX = (
    f"n{''.join(str(choice) for choice in NODE_INPUT_DIM_CHOICES)}"
    f"e{''.join(str(choice) for choice in EDGE_INPUT_DIM_CHOICES)}"
    f"g{''.join(str(choice) for choice in GLOBAL_INPUT_DIM_CHOICES)}"
    f"_{current_timestamp}"
)

HIDDEN_DIM_CHOICES = (64, 128, 256, 512)
GNN_ARCHITECTURE_CHOICES = (
    "gatv2_stack",
    "gine_stack",
)
GNN_LAYER_COUNT_CHOICES = (1, 2, 3)
GAT_HEAD_COUNT_CHOICES = (2, 4, 8)
GNN_ACTIVATION_CHOICES = (
    "prelu",
    "relu",
    "leaky_relu",
    "elu",
    "tanh",
    "gelu",
)


@dataclass(frozen=True)
class FeatureLayout:
    node_input_dim: int
    global_input_dim: int
    edge_input_dim: int = EDGE_INPUT_DIM_CHOICES[0]
    padded_node_feature_count: int = PADDED_NODE_FEATURE_COUNT
    padded_edge_feature_count: int = PADDED_EDGE_FEATURE_COUNT
    padded_global_feature_count: int = PADDED_GLOBAL_FEATURE_COUNT
    # Ordered node-feature names present in x (active subset). None => full NODE_FEATURE_SCHEMA.
    # Lets the node encoder locate categorical columns by name under any subset/reorder.
    active_feature_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.node_input_dim not in NODE_INPUT_DIM_CHOICES:
            raise ValueError(
                f"node_input_dim must be one of {NODE_INPUT_DIM_CHOICES}, "
                f"got {self.node_input_dim}"
            )
        if self.edge_input_dim not in EDGE_INPUT_DIM_CHOICES:
            raise ValueError(
                f"edge_input_dim must be one of {EDGE_INPUT_DIM_CHOICES}, "
                f"got {self.edge_input_dim}"
            )
        if self.global_input_dim not in GLOBAL_INPUT_DIM_CHOICES:
            raise ValueError(
                f"global_input_dim must be one of {GLOBAL_INPUT_DIM_CHOICES}, "
                f"got {self.global_input_dim}"
            )
