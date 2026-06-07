from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime

current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

NATIVE_NODE_FEATURE_COUNT = 22
NATIVE_GLOBAL_FEATURE_COUNT = 9
PADDED_NODE_FEATURE_COUNT = NATIVE_NODE_FEATURE_COUNT
PADDED_GLOBAL_FEATURE_COUNT = NATIVE_GLOBAL_FEATURE_COUNT

NODE_INPUT_DIM_CHOICES = (4, 5)
GLOBAL_INPUT_DIM_CHOICES = (6, 9)

# Suffix for Optuna study/DB names. Bump (or change choices) when categorical search
# spaces change so load_if_exists does not reuse incompatible distributions.
OPTUNA_SEARCH_SPACE_SUFFIX = (
    f"n{''.join(str(choice) for choice in NODE_INPUT_DIM_CHOICES)}"
    f"g{''.join(str(choice) for choice in GLOBAL_INPUT_DIM_CHOICES)}"
    f"_{current_timestamp}"
)

HIDDEN_DIM_CHOICES = (64, 128, 256, 512)
GNN_ARCHITECTURE_CHOICES = (
    "gin_stack",
)
GNN_LAYER_COUNT_CHOICES = (1, 2, 3)
GAT_HEAD_COUNT_CHOICES = (2, 4, 8)
GNN_ACTIVATION_CHOICES = (
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
    padded_node_feature_count: int = PADDED_NODE_FEATURE_COUNT
    padded_global_feature_count: int = PADDED_GLOBAL_FEATURE_COUNT

    def __post_init__(self) -> None:
        if self.node_input_dim not in NODE_INPUT_DIM_CHOICES:
            raise ValueError(
                f"node_input_dim must be one of {NODE_INPUT_DIM_CHOICES}, "
                f"got {self.node_input_dim}"
            )
        if self.global_input_dim not in GLOBAL_INPUT_DIM_CHOICES:
            raise ValueError(
                f"global_input_dim must be one of {GLOBAL_INPUT_DIM_CHOICES}, "
                f"got {self.global_input_dim}"
            )
