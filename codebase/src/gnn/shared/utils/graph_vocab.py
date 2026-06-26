from typing import Any


# Fixed canonical vocabularies (stable across datasets and load order).
CANONICAL_LABELS: tuple[str, ...] = (
    "<UNK>",
    "<CONSTANT>",
    "x",
    "E",
    "Log",
    "Pi",
    "Sin",
    "Cos",
    "Tan",
    "Plus",
    "Minus",
    "Times",
    "Divide",
    "Power",
    "Sqrt",
)
CANONICAL_LABEL_VOCAB: dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
NUM_CANONICAL_LABELS: int = len(CANONICAL_LABELS)

# One-hot column names for each CANONICAL_LABEL entry.
LABEL_ONEHOT_NAMES: tuple[str, ...] = (
    "label_UNK",
    "label_CONSTANT",
    "label_x",
    "label_E",
    "label_Log",
    "label_Pi",
    "label_Sin",
    "label_Cos",
    "label_Tan",
    "label_Plus",
    "label_Minus",
    "label_Times",
    "label_Divide",
    "label_Power",
    "label_Sqrt",
)

# Anchor-based positional encoding. Reduced to 3 semantic groups.
ANCHOR_GROUP_FEATURES: tuple[str, ...] = (
    "anchor_trigonometric",
    "anchor_exponential",
    "anchor_variable",
)
NUM_ANCHOR_GROUPS: int = len(ANCHOR_GROUP_FEATURES)

ANCHOR_GROUP_BY_LABEL: dict[str, int] = {
    "Sin": 1, "Cos": 1, "Tan": 1,
    "E": 2, "Log": 2,
    "x": 3,
}

ANCHOR_EXCLUDED_NODE_IDS: frozenset[str] = frozenset({"global"})

# Identity-aware root-node coloring: each function tree root gets a unique color code.
ROOT_COLOR_VOCAB: dict[str, int] = {"none": 0, "f": 1, "d1": 2, "d2": 3, "kappa": 4}
NUM_ROOT_COLORS: int = len(ROOT_COLOR_VOCAB)  # 5

# Subtree histogram bins — 4 bins.
HISTOGRAM_GROUP_BY_LABEL: dict[str, int] = {
    "Sin": 0, "Cos": 0, "Tan": 0,
    "Log": 1,
}
HISTOGRAM_VARIABLE_BIN: int = 2
HISTOGRAM_CONSTANT_BIN: int = 3
NUM_HISTOGRAM_BINS: int = 4
HISTOGRAM_FEATURES: tuple[str, ...] = (
    "hist_trigonometric",
    "hist_exponential",
    "hist_variables",
    "hist_constants",
)

# node_type integer code → one-hot index (3 distinct codes; supernode omitted — always 0 when add_virtual_supernode=False)
NODE_TYPE_ONEHOT: dict[int, int] = {0: 0, 1: 1, 2: 2}
NODE_TYPE_ONEHOT_DIM: int = 3
ROOT_COLOR_ONEHOT_DIM: int = NUM_ROOT_COLORS  # 5

SUPERNODE_NODE_TYPE: int = 5
SUPERNODE_NODE_ID: str = "virtual_supernode"

# 32-column one-hot node feature schema.
NODE_FEATURE_SCHEMA: list[str] = [
    # node_type one-hot (3 values: global=0, operator=1, function=2)
    "node_type_global",
    "node_type_operator",
    "node_type_function",
    # root_color one-hot (5 values: none=0, f=1, d1=2, d2=3, kappa=4)
    "root_color_none",
    "root_color_f",
    "root_color_d1",
    "root_color_d2",
    "root_color_kappa",
    # label one-hot (15 entries matching CANONICAL_LABELS; GLOBAL omitted — redundant with node_type_global)
    "label_UNK",
    "label_CONSTANT",
    "label_x",
    "label_E",
    "label_Log",
    "label_Pi",
    "label_Sin",
    "label_Cos",
    "label_Tan",
    "label_Plus",
    "label_Minus",
    "label_Times",
    "label_Divide",
    "label_Power",
    "label_Sqrt",
    "subtree_size",
    "subtree_depth",
    # histogram (4 bins)
    "hist_trigonometric",
    "hist_exponential",
    "hist_variables",
    "hist_constants",
    # anchor positional encoding (3 groups)
    "anchor_trigonometric",
    "anchor_exponential",
    "anchor_variable",
]

# Edge attributes are not used; the AST is always encoded top-down (parent->child)
# in edge_index topology only.
EDGE_FEATURE_SCHEMA: list[str] = []


def _is_numeric_label(label: str) -> bool:
    try:
        float(label)
        return True
    except ValueError:
        pass
    if "/" in label:
        parts = label.split("/")
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
                return True
            except ValueError:
                pass
    return False


def encode_label(label: str) -> int:
    if _is_numeric_label(label):
        return CANONICAL_LABEL_VOCAB["<CONSTANT>"]
    return CANONICAL_LABEL_VOCAB.get(label, CANONICAL_LABEL_VOCAB["<UNK>"])


def node_type_onehot(code: int) -> list[float]:
    vec = [0.0] * NODE_TYPE_ONEHOT_DIM
    idx = NODE_TYPE_ONEHOT.get(code, 0)
    vec[idx] = 1.0
    return vec


def root_color_onehot(code: int) -> list[float]:
    vec = [0.0] * ROOT_COLOR_ONEHOT_DIM
    if 0 <= code < ROOT_COLOR_ONEHOT_DIM:
        vec[code] = 1.0
    return vec


def label_onehot(label: str) -> list[float]:
    if _is_numeric_label(label):
        idx = CANONICAL_LABEL_VOCAB["<CONSTANT>"]
    else:
        idx = CANONICAL_LABEL_VOCAB.get(label, 0)
    vec = [0.0] * NUM_CANONICAL_LABELS
    vec[idx] = 1.0
    return vec


def anchor_group_for_node(label: Any, node_type: Any) -> int | None:
    return ANCHOR_GROUP_BY_LABEL.get(label)
