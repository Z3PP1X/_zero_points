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

# Fixed architecture — NOT Optuna-searched. Set once here and applied identically in
# both the search builder (ppo_optuna_search) and the final-training builder
# (train_best). dim_inner == the GNN hidden width.
FIXED_DIM_INNER = 32            # GNN hidden width ("dim_inner")
FIXED_GNN_LAYER_COUNT = 1       # number of message-passing layers
FIXED_DROPOUT = 0.2             # per-graph feature-column dropout (ExpressionGNN)
FIXED_GNN_ACTIVATION = "prelu"  # sole activation function

# Suffix for Optuna study/DB names. Bump (or change choices) when categorical search
# spaces change so load_if_exists does not reuse incompatible distributions.
# edge_input_dim is no longer searched (GIN convs ignore edge features), so it is
# absent from both the search space and this suffix.
OPTUNA_SEARCH_SPACE_SUFFIX = (
    f"n{''.join(str(choice) for choice in NODE_INPUT_DIM_CHOICES)}"
    f"g{''.join(str(choice) for choice in GLOBAL_INPUT_DIM_CHOICES)}"
    f"_{current_timestamp}"
)

# Activations the GNN backbone supports. The RL pipeline is fixed to
# FIXED_GNN_ACTIVATION; this tuple is kept for the backbone unit test that builds
# every variant.
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
