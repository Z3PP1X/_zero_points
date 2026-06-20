from typing import Any


# Fixed canonical vocabularies (stable across datasets and load order).
CANONICAL_LABELS: tuple[str, ...] = (
    "<UNK>",
    "<CONSTANT>",
    "GLOBAL",
    "Plus",
    "Times",
    "Power",
    "x",
    "E",
    "Pi",
    "I",
    "Sin",
    "Cos",
    "Tan",
    "Cot",
    "Sec",
    "Csc",
    "Exp",
    "Log",
    "Sqrt",
    "Abs",
    "ArcSin",
    "ArcCos",
    "ArcTan",
    "Sinh",
    "Cosh",
    "Tanh",
)
CANONICAL_LABEL_VOCAB: dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}

# Anchor-based positional encoding. Each AST node is encoded by its proximity to the
# nearest "anchor" — an operator/function node — of each semantic group. The distance is
# the shortest-path hop count on the undirected AST, measured *within the node's own
# function subgraph*: the global structural connector node is removed before measuring, so
# f, f' and f'' fall into independent components and a node never sees anchors from a
# sibling function (there is no message passing across them anyway). The distance d is
# encoded as 1/(1+d) in (0, 1]: an anchor node scores 1.0 for
# its own group, and a group absent from the node's function scores 0.0. This replaces the
# former Laplacian / random-walk positional encodings (lpe_*/rwpe_*).
ANCHOR_GROUP_FEATURES: tuple[str, ...] = (
    "anchor_additive",        # G1: Plus (addition / subtraction)
    "anchor_scaling",         # G2: Times, Power, Sqrt (multiplicative / algebraic scaling)
    "anchor_periodic",        # G3: Sin, Cos, Tan, Cot, Sec, Csc (trigonometric)
    "anchor_exponential",     # G4: Exp, Log
    "anchor_transcendental",  # G5: Sinh, Cosh, Tanh, Abs, ArcSin/Cos/Tan + any other op/fn
)
NUM_ANCHOR_GROUPS: int = len(ANCHOR_GROUP_FEATURES)

# Operator/function label -> 1-based anchor group index. An operator/function whose label
# is not listed here falls through to the transcendental group (5); variables, constants
# and structural nodes are never anchors.
ANCHOR_GROUP_BY_LABEL: dict[str, int] = {
    "Plus": 1,
    "Times": 2, "Power": 2, "Sqrt": 2,
    "Sin": 3, "Cos": 3, "Tan": 3, "Cot": 3, "Sec": 3, "Csc": 3,
    "Exp": 4, "Log": 4,
    "Sinh": 5, "Cosh": 5, "Tanh": 5, "Abs": 5,
    "ArcSin": 5, "ArcCos": 5, "ArcTan": 5,
}
TRANSCENDENTAL_ANCHOR_GROUP: int = 5

# Only global is excluded from anchor distance; the per-function subgraphs (f / f' / f'')
# become independent connected components naturally once global is excluded from G_ast.
ANCHOR_EXCLUDED_NODE_IDS: frozenset[str] = frozenset({"global"})

# Identity-aware root-node coloring: each function tree root gets a unique color code.
ROOT_COLOR_VOCAB: dict[str, int] = {"none": 0, "f": 1, "d1": 2, "d2": 3, "kappa": 4}
NUM_ROOT_COLORS: int = len(ROOT_COLOR_VOCAB)  # 5

# Subtree histogram bins — count of each operator/function category within a node's subtree.
HISTOGRAM_GROUP_BY_LABEL: dict[str, int] = {
    "Plus": 0,
    "Times": 1, "Power": 1, "Sqrt": 1,
    "Sin": 2, "Cos": 2, "Tan": 2, "Cot": 2, "Sec": 2, "Csc": 2,
    "Exp": 3, "Log": 3,
    "Sinh": 4, "Cosh": 4, "Tanh": 4, "Abs": 4,
    "ArcSin": 4, "ArcCos": 4, "ArcTan": 4,
}
HISTOGRAM_VARIABLE_BIN: int = 5
HISTOGRAM_CONSTANT_BIN: int = 6
NUM_HISTOGRAM_BINS: int = 7
HISTOGRAM_FEATURES: tuple[str, ...] = (
    "hist_additive",
    "hist_multiplicative",
    "hist_trigonometric",
    "hist_exponential",
    "hist_transcendental",
    "hist_variables",
    "hist_constants",
)

CANONICAL_EDGE_TYPES: tuple[str, ...] = (
    "<UNK>",
    "child_of",
    "child_of_reverse",
    "virtual",
    "virtual_reverse",
    "supernode_connection",
    "supernode_connection_reverse",
    "GlobalToKappa",
    "KappaToGlobal",
    # Operand-side relation types for non-commutative binary operators.
    "left_operand",
    "left_operand_reverse",
    "right_operand",
    "right_operand_reverse",
)
CANONICAL_EDGE_TYPE_VOCAB: dict[str, int] = {etype: idx for idx, etype in enumerate(CANONICAL_EDGE_TYPES)}

VIRTUAL_NODE_TYPES: frozenset[str] = frozenset()

# Categorical vocabulary sizes. node_type uses codes: 0=global, 1=operator, 2=root, 5=supernode.
# NUM_NODE_TYPES covers the range 0..5 (6 entries), with gap at 3 and 4.
NUM_NODE_TYPES: int = 6
NUM_LABELS: int = len(CANONICAL_LABEL_VOCAB)
NUM_EDGE_TYPES: int = len(CANONICAL_EDGE_TYPE_VOCAB)

# Optional fully-connected virtual supernode (opt-in via add_virtual_supernode). It is
# given its own node_type code (5) so the model treats it as an ordinary
# message-passing node. NUM_NODE_TYPES already covers code 5.
SUPERNODE_NODE_TYPE: int = 5
SUPERNODE_NODE_ID: str = "virtual_supernode"

NODE_FEATURE_SCHEMA = [
    "node_type",             # 0: global=0, operator=1, root=2, supernode=5
    "root_color",            # 1: none=0, f=1, d1=2, d2=3, kappa=4
    "subtree_size",          # 2: number of nodes in subtree (self + descendants)
    "subtree_depth",         # 3: height = max depth of subtree below this node
    "hist_additive",         # 4: Plus count in subtree
    "hist_multiplicative",   # 5: Times/Power/Sqrt count in subtree
    "hist_trigonometric",    # 6: Sin/Cos/Tan/… count in subtree
    "hist_exponential",      # 7: Exp/Log count in subtree
    "hist_transcendental",   # 8: Sinh/Cosh/Tanh/Abs/ArcSin… count in subtree
    "hist_variables",        # 9: variable node count in subtree
    "hist_constants",        # 10: constant node count in subtree
    "anchor_additive",       # 11: positional encoding
    "anchor_scaling",        # 12
    "anchor_periodic",       # 13
    "anchor_exponential",    # 14
    "anchor_transcendental", # 15
]

# Edge attributes are not used; direction is encoded in edge_index topology only.
EDGE_FEATURE_SCHEMA: list[str] = []

EDGE_DIRECTIONS: tuple[str, ...] = ("top_down", "bottom_up", "bidirectional")


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


def encode_edge_type(etype: str) -> int:
    return CANONICAL_EDGE_TYPE_VOCAB.get(etype, CANONICAL_EDGE_TYPE_VOCAB["<UNK>"])


def anchor_group_for_node(label: Any, node_type: Any) -> int | None:
    """Return the 1-based anchor group for a node, or ``None`` if it is not an anchor.

    Explicitly grouped operator/function labels map to their group; any other
    operator/function falls through to the transcendental group; variables, constants,
    global and root structural nodes are not anchors.
    """
    if label in ANCHOR_GROUP_BY_LABEL:
        return ANCHOR_GROUP_BY_LABEL[label]
    if node_type in ("operator", "function", "root"):
        return TRANSCENDENTAL_ANCHOR_GROUP
    return None


def validate_edge_direction(edge_direction: str) -> str:
    if edge_direction not in EDGE_DIRECTIONS:
        raise ValueError(
            f"Unsupported edge_direction {edge_direction!r}; "
            f"expected one of {list(EDGE_DIRECTIONS)}"
        )
    return edge_direction
