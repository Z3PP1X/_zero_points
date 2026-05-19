from __future__ import annotations

from dataclasses import dataclass

NATIVE_NODE_FEATURE_COUNT = 5
NATIVE_GLOBAL_FEATURE_COUNT = 9
PADDED_NODE_FEATURE_COUNT = NATIVE_NODE_FEATURE_COUNT
PADDED_GLOBAL_FEATURE_COUNT = NATIVE_GLOBAL_FEATURE_COUNT

NODE_INPUT_DIM_CHOICES = (4, 5)
GLOBAL_INPUT_DIM_CHOICES = (6, 9)
HIDDEN_DIM_CHOICES = (64, 128, 256)
GNN_ARCHITECTURE_CHOICES = (
    "gatv2_stack",
    "gcn_stack",
    "sage_stack",
    "gin_stack",
)
GNN_LAYER_COUNT_CHOICES = (2, 3, 4)
GAT_HEAD_COUNT_CHOICES = (2, 4, 8)


@dataclass(frozen=True)
class FeatureLayout:
    node_input_dim: int
    global_input_dim: int

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

    @property
    def padded_node_feature_count(self) -> int:
        return PADDED_NODE_FEATURE_COUNT

    @property
    def padded_global_feature_count(self) -> int:
        return PADDED_GLOBAL_FEATURE_COUNT
