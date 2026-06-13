import json
import logging
import math
from collections import deque
import torch
import networkx as nx
from pathlib import Path
from typing import Union, Any, Dict
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
import numpy as np


logger = logging.getLogger(__name__)


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
# function subgraph*: the structural connector nodes (global / f_root / d1_root / d2_root)
# are removed before measuring, so f, f' and f'' fall into independent components and a
# node never sees anchors from a sibling function (there is no message passing across them
# anyway). The distance d is encoded as 1/(1+d) in (0, 1]: an anchor node scores 1.0 for
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
    "NextUse",
    "NextUseBackward",
    "GlobalToKappa",
    "KappaToGlobal",
) + tuple(f"OuterToInner_Arg{i}" for i in range(10)) + tuple(
    f"InnerToOuter_Arg{i}" for i in range(10)
) + (
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
# given its own node_type code in the otherwise-unused 0..10 gap (5), distinct from the
# f/d1/d2 aggregator virtual types (6/9/10) so the model treats it as an ordinary
# message-passing node rather than a task aggregator. NUM_NODE_TYPES already covers code 5.
SUPERNODE_NODE_TYPE: int = 5
SUPERNODE_NODE_ID: str = "virtual_supernode"


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


def _compute_anchor_positional_encoding(G: nx.DiGraph) -> np.ndarray:
    """Per-node anchor positional encoding, shape ``(num_nodes, NUM_ANCHOR_GROUPS)``.

    Rows are aligned to ``list(G.nodes)`` order (matching the other topology arrays). For
    each node and each anchor group, the value is ``1/(1 + d)`` where ``d`` is the
    shortest-path hop distance (undirected) to the nearest anchor of that group within the
    node's own function subgraph; ``0.0`` if that group is absent from the subgraph.
    Anchor distances are measured after dropping the structural connector nodes so f / f'
    / f'' are isolated.
    """
    node_order = list(G.nodes)
    pe = np.zeros((len(node_order), NUM_ANCHOR_GROUPS))
    if not node_order:
        return pe
    index_of = {node_id: i for i, node_id in enumerate(node_order)}

    G_und = G.to_undirected()
    kept = [n for n in G_und.nodes if str(n) not in ANCHOR_EXCLUDED_NODE_IDS]
    H = G_und.subgraph(kept)

    for component in nx.connected_components(H):
        comp_graph = H.subgraph(component)
        anchors_by_group: dict[int, list] = {}
        for node_id in component:
            attrs = H.nodes[node_id]
            group = anchor_group_for_node(attrs.get("label"), attrs.get("type"))
            if group is not None:
                anchors_by_group.setdefault(group, []).append(node_id)
        for group, sources in anchors_by_group.items():
            distances = _multi_source_bfs(comp_graph, sources)
            col = group - 1
            for node_id, dist in distances.items():
                pe[index_of[node_id], col] = 1.0 / (1.0 + dist)
    return pe


def _multi_source_bfs(graph: nx.Graph, sources: list) -> dict:
    """Shortest-path hop distance from the nearest of ``sources`` to every reachable node.

    A plain breadth-first sweep seeded with all sources at distance 0; equivalent to
    ``nx.multi_source_shortest_path_length`` but available across networkx versions.
    """
    dist: dict = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        next_dist = dist[node] + 1
        for neighbor in graph.neighbors(node):
            if neighbor not in dist:
                dist[neighbor] = next_dist
                queue.append(neighbor)
    return dist


def get_hetero_node_type(raw_type: str) -> str:
    """Map internal node type strings to the heterogeneous node type names.

    All expression-tree nodes collapse to "operator"; root nodes keep their "root"
    type for identity-aware message passing; global and supernode become "global".
    """
    if raw_type == "root":
        return "root"
    elif raw_type in ("global", "supernode"):
        return "global"
    else:
        # operator, function, variable, constant → unified "operator"
        return "operator"


def get_relation_type(parent_label: str, etype: str, child_index: float) -> str:
    is_reverse = etype.endswith("_reverse")
    base_etype = etype[:-8] if is_reverse else etype

    if base_etype == "child_of":
        if parent_label in ("Plus", "Times", "GLOBAL"):
            return "child_of_reverse" if is_reverse else "child_of"
        else:
            if child_index == 0.0:
                return "left_operand_reverse" if is_reverse else "left_operand"
            elif child_index == 1.0:
                return "right_operand_reverse" if is_reverse else "right_operand"
            else:
                return "left_operand_reverse" if is_reverse else "left_operand"
    else:
        return etype



def _histogram_bin_for_node(label: str, orig_type: str) -> int:
    """Return the 0-based histogram bin for a single node (based on its own label/type)."""
    if label in HISTOGRAM_GROUP_BY_LABEL:
        return HISTOGRAM_GROUP_BY_LABEL[label]
    if orig_type == "variable":
        return HISTOGRAM_VARIABLE_BIN
    if orig_type == "constant":
        return HISTOGRAM_CONSTANT_BIN
    # Unknown operator/function → transcendental bin
    return 4


def _compute_subtree_histograms(G: nx.DiGraph) -> dict:
    """Iterative subtree histogram: for each node, count operator types in its subtree.

    Returns a dict mapping node_id -> np.ndarray of shape (NUM_HISTOGRAM_BINS,).
    Each entry counts how many nodes of each category appear in the subtree rooted at
    that node (inclusive). Global, kappa, and supernode nodes are expected to be absent
    from G (pass the pure AST subgraph).
    """
    hists: dict = {
        nid: np.zeros(NUM_HISTOGRAM_BINS, dtype=np.float32)
        for nid in G.nodes
    }

    # Assign the self-contribution of each node
    for nid, attrs in G.nodes(data=True):
        b = _histogram_bin_for_node(attrs.get("label", ""), attrs.get("type", "operator"))
        hists[nid][b] = 1.0

    # Process nodes bottom-up (reverse topological order = leaves first)
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return hists

    for nid in reversed(topo_order):
        for child in G.successors(nid):
            hists[nid] += hists[child]

    return hists


NODE_FEATURE_SCHEMA = [
    "node_type",           # 0: global=0, operator=1, root=2, supernode=5
    "root_color",          # 1: none=0, f=1, d1=2, d2=3, kappa=4
    "subtree_size",        # 2: number of nodes in subtree (self + descendants)
    "subtree_depth",       # 3: height = max depth of subtree below this node
    "hist_additive",       # 4: Plus count in subtree
    "hist_multiplicative", # 5: Times/Power/Sqrt count in subtree
    "hist_trigonometric",  # 6: Sin/Cos/Tan/… count in subtree
    "hist_exponential",    # 7: Exp/Log count in subtree
    "hist_transcendental", # 8: Sinh/Cosh/Tanh/Abs/ArcSin… count in subtree
    "hist_variables",      # 9: variable node count in subtree
    "hist_constants",      # 10: constant node count in subtree
    "anchor_additive",     # 11: positional encoding
    "anchor_scaling",      # 12
    "anchor_periodic",     # 13
    "anchor_exponential",  # 14
    "anchor_transcendental", # 15
]

EDGE_FEATURE_SCHEMA = [
    "child_index",
    "direction",
    "relation_type",
    # kappa (h-function) edge weight: the parsed kappa value on GlobalToKappa/KappaToGlobal
    # edges, 0.0 on all other edges. Continuous column (not in EDGE_CATEGORICAL_REGISTRY).
    "kappa_weight",
]

EDGE_DIRECTIONS: tuple[str, ...] = ("top_down", "bottom_up", "bidirectional")


def validate_edge_direction(edge_direction: str) -> str:
    if edge_direction not in EDGE_DIRECTIONS:
        raise ValueError(
            f"Unsupported edge_direction {edge_direction!r}; "
            f"expected one of {list(EDGE_DIRECTIONS)}"
        )
    return edge_direction


def inject_virtual_supernode(
    G_enriched: nx.DiGraph,
    G_directed: nx.DiGraph,
    node_ids: list,
) -> None:
    """Inject one fully-connected virtual supernode into an already-built graph.

    The supernode is wired with bidirectional edges to every pre-existing node so a
    message can travel between any two nodes in at most two hops, shrinking the
    effective graph diameter and boosting long-range message passing.

    It is injected *after* the AST topology / positional features and the augmented
    (NextUse / function-nesting) edges have been computed, so adding it does not perturb
    those structural node features. The supernode carries its own node_type code
    (``SUPERNODE_NODE_TYPE``), distinct from the f/d1/d2 aggregator virtual types
    (6/9/10); the model therefore treats it as an ordinary message-passing node instead
    of excluding it like the task aggregators.

    Mutates ``G_enriched`` (feature graph), ``G_directed`` (source of the node-type /
    label / belongs tensors and the node/edge counts) and appends the supernode id to
    ``node_ids`` in place. Idempotent: a second call is a no-op.
    """
    if SUPERNODE_NODE_ID in G_enriched:
        return

    existing_nodes = [nid for nid in node_ids if nid != SUPERNODE_NODE_ID]

    supernode_attrs: dict[str, Any] = {
        "node_type": SUPERNODE_NODE_TYPE,
        "root_color": 0.0,
        "label_id": encode_label("GLOBAL"),
        "type": "supernode",
        "label": "GLOBAL",
        "subtree_size": 0.0,
        "subtree_depth": 0.0,
        "anchor_additive": 0.0,
        "anchor_scaling": 0.0,
        "anchor_periodic": 0.0,
        "anchor_exponential": 0.0,
        "anchor_transcendental": 0.0,
        **{name: 0.0 for name in HISTOGRAM_FEATURES},
    }
    G_enriched.add_node(SUPERNODE_NODE_ID, **supernode_attrs)
    G_directed.add_node(SUPERNODE_NODE_ID, **supernode_attrs)

    forward_rel = float(encode_edge_type("supernode_connection"))
    reverse_rel = float(encode_edge_type("supernode_connection_reverse"))

    def _edge_attrs(relation_type: float, direction: float, etype: str) -> dict[str, Any]:
        return {
            "child_index": 0.0,
            "direction": direction,
            "relation_type": relation_type,
            "kappa_weight": 0.0,
            "etype": etype,
        }

    for nid in existing_nodes:
        # Both directions are always added so the supernode shortcut stays bidirectional
        # regardless of the AST edge_direction setting (mirrors virtual/task edges).
        G_enriched.add_edge(
            SUPERNODE_NODE_ID,
            nid,
            **_edge_attrs(forward_rel, 0.0, "supernode_connection"),
        )
        G_enriched.add_edge(
            nid,
            SUPERNODE_NODE_ID,
            **_edge_attrs(reverse_rel, 1.0, "supernode_connection_reverse"),
        )
        # Mirror the structural edges into G_directed so num_edges/num_nodes stay accurate.
        G_directed.add_edge(SUPERNODE_NODE_ID, nid)
        G_directed.add_edge(nid, SUPERNODE_NODE_ID)

    node_ids.append(SUPERNODE_NODE_ID)


def _find_global_node_id(raw: dict) -> str | None:
    for node in raw.get("nodes", []):
        if node.get("type") == "global":
            return node["id"]
    return None


def _mark_function_roots(raw: dict) -> None:
    """Mark AST root nodes with type='root' and root_color, replacing provenance edges.

    For the legacy JSON format that carries ``belongs_to_f/d1/d2`` edges, this rewrites
    those edges as plain ``child_of`` edges from global to each root and annotates the
    root nodes with ``root_color`` so the model can distinguish f / f' / f'' via a
    learnable embedding without needing separate virtual aggregator nodes.
    """
    global_id = _find_global_node_id(raw)
    if global_id is None:
        return

    prov_to_color: dict[str, str] = {
        "belongs_to_f": "f",
        "belongs_to_d1": "d1",
        "belongs_to_d2": "d2",
    }

    # Find root nodes that are direct targets of provenance edges from global.
    root_to_color: dict[str, str] = {}
    for edge in raw.get("edges", []):
        etype = edge.get("type", "")
        if edge.get("source") == global_id and etype in prov_to_color:
            root_to_color[edge["target"]] = prov_to_color[etype]

    # Fallback: if no provenance edges exist, treat all in-degree-0 non-global nodes as
    # roots of f (single-function graphs without explicit provenance markup).
    if not root_to_color:
        non_global = {n["id"] for n in raw.get("nodes", []) if n.get("type") != "global"}
        targets = {e["target"] for e in raw.get("edges", [])}
        for nid in non_global:
            if nid not in targets:
                root_to_color[nid] = "f"

    # Annotate root nodes in raw.
    node_by_id = {n["id"]: n for n in raw.get("nodes", [])}
    for root_id, color in root_to_color.items():
        if root_id in node_by_id:
            node_by_id[root_id]["type"] = "root"
            node_by_id[root_id]["root_color"] = ROOT_COLOR_VOCAB[color]

    # Replace belongs_to_* edges with child_of edges (global → root).
    new_edges = []
    for edge in raw.get("edges", []):
        etype = edge.get("type", "")
        if edge.get("source") == global_id and etype in prov_to_color:
            new_edges.append({"source": global_id, "target": edge["target"], "type": "child_of"})
        else:
            new_edges.append(edge)
    raw["edges"] = new_edges


class ExpressionGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'laplacian':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class ExpressionHeteroData(HeteroData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'laplacian':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class TopologicalFeatureExtractor:
    """Extrahiert topologische Features aus einem NetworkX Graphen."""

    @staticmethod
    def extract_and_annotate(G: nx.DiGraph) -> dict:
        deg_cent = nx.degree_centrality(G)
        nx.set_node_attributes(G, deg_cent, "degree_centrality")

        roots = [n for n, d in G.in_degree() if d == 0]

        # Depths (Depth)
        levels = {}
        if roots:
            for root in roots:
                lengths = nx.single_source_shortest_path_length(G, root)
                for node, length in lengths.items():
                    if node not in levels or length > levels[node]:
                        levels[node] = length
        for node in G.nodes:
            if node not in levels:
                levels[node] = 0

        # Global tree depth
        tree_depth = max(levels.values()) if levels else 0

        # Global tree width
        level_counts = {}
        for lvl in levels.values():
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        tree_width = max(level_counts.values()) if level_counts else 0

        # Basic results
        results = {
            "tree_depth": tree_depth,
            "tree_width": tree_width,
            "depths": levels,
        }

        # Heights (Height)
        heights = {}
        visiting = set()
        def get_height(node):
            if node in heights:
                return heights[node]
            if node in visiting:
                return 0
            visiting.add(node)
            children = list(G.successors(node))
            if not children:
                visiting.remove(node)
                heights[node] = 0
                return 0
            h = 1 + max(get_height(child) for child in children)
            visiting.remove(node)
            heights[node] = h
            return h
        for node in G.nodes:
            get_height(node)

        # Subtree Sizes (SubtreeSize)
        subtree_sizes = {}
        for node in G.nodes:
            subtree_sizes[node] = len(nx.descendants(G, node)) + 1

        # Out-Degrees (Out-Degree)
        out_degrees = {node: G.out_degree(node) for node in G.nodes}

        # Undirected graph representation for the Laplacian and undirected centralities
        # (the anchor PE builds its own undirected view inside its helper).
        G_und = G.to_undirected()
        num_nodes = G_und.number_of_nodes()

        # Betweenness Centrality (computed on undirected graph)
        betweenness = nx.betweenness_centrality(G_und)

        # Laplace-Matrix (Graph)
        if num_nodes > 0:
            laplace_matrix = nx.laplacian_matrix(G_und).toarray()
        else:
            laplace_matrix = np.zeros((0, 0))

        # Anchor-based positional encoding (replaces the former LPE / RWPE). Encodes each
        # node by 1/(1+d) proximity to the nearest anchor of each semantic operator group,
        # measured within the node's own function subgraph. See
        # _compute_anchor_positional_encoding for the full definition.
        anchor_pe = _compute_anchor_positional_encoding(G)

        results.update({
            "heights": heights,
            "subtree_sizes": subtree_sizes,
            "out_degrees": out_degrees,
            "betweenness": betweenness,
            "laplace_matrix": laplace_matrix,
            "anchor_pe": anchor_pe,
        })
        return results


def build_augmented_math_graph(
    G: nx.DiGraph,
    current_node: str,
    last_seen_map: dict[str, str],
    children_dict: dict[str, list[str]],
    edge_direction: str,
    active_outer_function: str | None = None,
    current_arg_index: int = 0,
) -> None:
    """Recursively augments the mathematical expression graph G using DFS.
    
    Adds "NextUse"/"NextUseBackward" edges for variable tracking and position-aware
    functional nesting edges for nested operations based on the active outer function
    and edge direction configuration.
    
    Arguments:
        G: The NetworkX directed graph to augment with new edges.
        current_node: The current node ID being visited.
        last_seen_map: A dictionary mapping variable names to their last visited Node ID.
        children_dict: A dictionary mapping parent node IDs to list of child node IDs.
        edge_direction: The direction of the edges ("top_down", "bottom_up", or "bidirectional").
        active_outer_function: The Node ID of the closest ancestor function node.
        current_arg_index: The argument index of the current node relative to its active outer function parent.
        
    Returns:
        None
        
    Exceptions:
        None
    """
    node_attrs = G.nodes[current_node]
    
    def add_augmented_edge(u: str, v: str, etype: str) -> None:
        etype_id = encode_edge_type(etype)
        G.add_edge(
            u,
            v,
            child_index=0.0,
            direction=0.0,
            relation_type=float(etype_id),
            etype=etype,
        )

    # 1. Variable NextUse Tracking (Algorithm 1)
    is_variable = (node_attrs.get("type") == "variable")
    if is_variable:
        variable_name = node_attrs.get("label")
        if isinstance(variable_name, str):
            if variable_name in last_seen_map:
                previous_variable_node = last_seen_map[variable_name]
                if edge_direction in ("top_down", "bidirectional"):
                    add_augmented_edge(previous_variable_node, current_node, "NextUse")
                if edge_direction in ("bottom_up", "bidirectional"):
                    add_augmented_edge(current_node, previous_variable_node, "NextUseBackward")
            last_seen_map[variable_name] = current_node

    # 2. TrackFunctionNestingWithSides (Algorithm 2)
    ALLOWED_FUNCTIONS = ["log", "exp", "sin", "cos", "Plus", "Times", "CustomFunc"]
    node_label = node_attrs.get("label")
    is_function_node = False
    if isinstance(node_label, str):
        is_function_node = (node_label.lower() in {f.lower() for f in ALLOWED_FUNCTIONS})

    if is_function_node:
        if active_outer_function is not None:
            edge_type_str = f"OuterToInner_Arg{current_arg_index}"
            backward_edge_type_str = f"InnerToOuter_Arg{current_arg_index}"
            
            if edge_direction in ("top_down", "bidirectional"):
                add_augmented_edge(active_outer_function, current_node, edge_type_str)
            if edge_direction in ("bottom_up", "bidirectional"):
                add_augmented_edge(current_node, active_outer_function, backward_edge_type_str)
        active_outer_function = current_node

    # 3. Recursive DFS traversal down the children (Left to Right)
    children = children_dict.get(current_node, [])
    for idx, child in enumerate(children):
        if is_function_node:
            build_augmented_math_graph(
                G,
                child,
                last_seen_map,
                children_dict,
                edge_direction,
                active_outer_function=active_outer_function,
                current_arg_index=idx,
            )
        else:
            build_augmented_math_graph(
                G,
                child,
                last_seen_map,
                children_dict,
                edge_direction,
                active_outer_function=active_outer_function,
                current_arg_index=current_arg_index,
            )


class ExpressionGraphConverter:
    # node_type codes: 0=global, 1=operator (incl. function/variable/constant), 2=root, 5=supernode.
    # Legacy types (function/variable/constant) all map to 1 for backward compatibility
    # with datasets that still carry those type strings.
    NODE_TYPES = {
        "global": 0,
        "operator": 1,
        "function": 1,
        "variable": 1,
        "constant": 1,
        "root": 2,
        "supernode": SUPERNODE_NODE_TYPE,
    }

    def __init__(self):
        pass

    def _add_ast_edges(
        self,
        G_enriched: nx.DiGraph,
        parent: str,
        child: str,
        child_idx: int,
        etype: str,
        edge_direction: str,
    ) -> None:
        effective = edge_direction
        # Resolve the operand-aware relation type (left/right operand for non-commutative
        # binary operators) so the homogeneous relation_type column matches the
        # heterogeneous metapath keys. For non-child_of edges get_relation_type is a no-op.
        parent_label = G_enriched.nodes[parent].get("label", "") if parent in G_enriched else ""
        rel_forward = get_relation_type(parent_label, etype, float(child_idx))
        rel_reverse = get_relation_type(parent_label, etype + "_reverse", float(child_idx))
        if effective in ("top_down", "bidirectional"):
            G_enriched.add_edge(
                parent,
                child,
                child_index=float(child_idx),
                direction=0.0,
                relation_type=float(self._encode_edge_type(rel_forward)),
                etype=etype,
            )
        if effective in ("bottom_up", "bidirectional"):
            G_enriched.add_edge(
                child,
                parent,
                child_index=float(child_idx),
                direction=1.0 if effective == "bidirectional" else 0.0,
                relation_type=float(self._encode_edge_type(rel_reverse)),
                etype=etype + "_reverse",
            )

    def convert(
        self,
        source: Union[str, Path, dict, nx.DiGraph],
        heterogeneous: bool = False,
        mode: str = "graph",
        edge_direction: str = "top_down",
        add_virtual_supernode: bool = False,
    ) -> Union[Data, HeteroData]:
        edge_direction = validate_edge_direction(edge_direction)
        
        if isinstance(source, nx.DiGraph):
            # Extract pure AST nodes (exclude global and kappa subgraph).
            ast_nodes = [
                n for n in source.nodes
                if n != "global"
                and not str(n).startswith("kappa_")
                and str(n) != SUPERNODE_NODE_ID
            ]
            G_ast = source.subgraph(ast_nodes).copy()
            topo = TopologicalFeatureExtractor.extract_and_annotate(G_ast)
            hist = _compute_subtree_histograms(G_ast)
            ast_node_ids = list(G_ast.nodes)
            ast_id_to_idx = {node_id: idx for idx, node_id in enumerate(ast_node_ids)}

            node_ids = list(source.nodes)
            G_directed = source

            # Enrich/add features to G_enriched
            G_enriched = nx.DiGraph()

            for node in node_ids:
                attrs = source.nodes[node]
                enriched_attrs = attrs.copy()
                ast_idx = ast_id_to_idx.get(node)

                if ast_idx is not None:
                    enriched_attrs["subtree_size"] = float(topo["subtree_sizes"].get(node, 1.0))
                    enriched_attrs["subtree_depth"] = float(topo["heights"].get(node, 0.0))
                    node_hist = hist.get(node, np.zeros(NUM_HISTOGRAM_BINS))
                    for _col, _name in enumerate(HISTOGRAM_FEATURES):
                        enriched_attrs[_name] = float(node_hist[_col])
                    for _col, _name in enumerate(ANCHOR_GROUP_FEATURES):
                        enriched_attrs[_name] = float(topo["anchor_pe"][ast_idx, _col])
                else:
                    enriched_attrs["subtree_size"] = 0.0
                    enriched_attrs["subtree_depth"] = 0.0
                    for _name in HISTOGRAM_FEATURES:
                        enriched_attrs[_name] = 0.0
                    for _name in ANCHOR_GROUP_FEATURES:
                        enriched_attrs[_name] = 0.0

                # Ensure root_color is present (may have been set before conversion).
                if "root_color" not in enriched_attrs:
                    enriched_attrs["root_color"] = float(ROOT_COLOR_VOCAB["none"])

                G_enriched.add_node(node, **enriched_attrs)

            child_counters = {}
            for u, v, attrs in source.edges(data=True):
                if "relation_type" in attrs or "child_index" in attrs or "direction" in attrs:
                    G_enriched.add_edge(u, v, **attrs)
                    continue

                parent = u
                child = v
                etype = attrs.get("type") or attrs.get("etype") or "child_of"

                child_idx = child_counters.get(parent, 0)
                child_counters[parent] = child_idx + 1

                self._add_ast_edges(
                    G_enriched, parent, child, child_idx, etype, edge_direction
                )

            # Include global→root child_of edges so the DFS can traverse from global
            # into each function subtree and correctly track NextUse variable reuse.
            children_dict = {}
            for u, v, attrs in G_directed.edges(data=True):
                etype = attrs.get("etype") or attrs.get("type") or "child_of"
                if etype == "child_of":
                    children_dict.setdefault(u, []).append(v)

            raw = None
        else:
            raw = self._load(source)
            
            # Make a copy of raw to avoid modifying the original dict in-place if passed as object
            raw = dict(raw)
            
            # Check if the raw data is in the new GraphML string container format
            if "graphml_f" in raw:
                nodes_f, edges_f = parse_graphml_to_nodes_and_edges(raw.get("graphml_f", ""), "f")
                
                # If mode is tree, we only compile the function graph f.
                # If mode is tree_derivatives or graph, we compile all three (f, f', f'').
                if mode in ["tree_derivatives", "graph"]:
                    nodes_d1, edges_d1 = parse_graphml_to_nodes_and_edges(raw.get("graphml_derivative1", ""), "d1")
                    nodes_d2, edges_d2 = parse_graphml_to_nodes_and_edges(raw.get("graphml_derivative2", ""), "d2")
                else:
                    nodes_d1, edges_d1 = [], []
                    nodes_d2, edges_d2 = [], []
                
                combined_nodes = nodes_f + nodes_d1 + nodes_d2
                combined_nodes.insert(0, {
                    "id": "global",
                    "label": "GLOBAL",
                    "type": "global",
                    "value": None
                })
                
                roots_f = find_roots(nodes_f, edges_f)
                roots_d1 = find_roots(nodes_d1, edges_d1)
                roots_d2 = find_roots(nodes_d2, edges_d2)

                # Mark actual AST roots with type="root" and root_color.
                _root_specs = [("f", roots_f, nodes_f), ("d1", roots_d1, nodes_d1), ("d2", roots_d2, nodes_d2)]
                for color, roots, nodes_list in _root_specs:
                    root_set = set(roots)
                    for node in nodes_list:
                        if node["id"] in root_set:
                            node["type"] = "root"
                            node["root_color"] = ROOT_COLOR_VOCAB[color]

                combined_edges = edges_f + edges_d1 + edges_d2
                # Direct global → root child_of edges replace the old aggregator chain.
                for color, roots in [("f", roots_f), ("d1", roots_d1), ("d2", roots_d2)]:
                    for root in roots:
                        combined_edges.append({"source": "global", "target": root, "type": "child_of"})

                raw["nodes"] = combined_nodes
                raw["edges"] = combined_edges
            else:
                raw["nodes"] = list(raw.get("nodes", []))
                raw["edges"] = list(raw.get("edges", []))
                _mark_function_roots(raw)

            # Topology and histogram are computed on the pure AST (excludes global + kappa).
            global_id_raw = _find_global_node_id(raw) or "global"
            raw_ast = {
                "nodes": [n for n in raw["nodes"]
                          if n["id"] != global_id_raw and not str(n["id"]).startswith("kappa_")],
                "edges": [e for e in raw["edges"]
                          if e["source"] != global_id_raw and e["target"] != global_id_raw
                          and not str(e["source"]).startswith("kappa_")
                          and not str(e["target"]).startswith("kappa_")],
            }

            # 1. Build AST graph for structural features (excludes global + kappa nodes)
            G_ast = self._build_networkx(raw_ast)
            topo = TopologicalFeatureExtractor.extract_and_annotate(G_ast)
            hist = _compute_subtree_histograms(G_ast)
            ast_node_ids = list(G_ast.nodes)
            ast_id_to_idx = {node_id: idx for idx, node_id in enumerate(ast_node_ids)}

            # 2. Build full directed graph
            G_directed = self._build_networkx(raw)

            # 3. Enrich node attributes
            G_enriched = nx.DiGraph()
            node_ids = list(G_directed.nodes)

            for node in node_ids:
                attrs = G_directed.nodes[node]
                enriched_attrs = attrs.copy()
                ast_idx = ast_id_to_idx.get(node)

                if ast_idx is not None:
                    enriched_attrs["subtree_size"] = float(topo["subtree_sizes"].get(node, 1.0))
                    enriched_attrs["subtree_depth"] = float(topo["heights"].get(node, 0.0))
                    node_hist = hist.get(node, np.zeros(NUM_HISTOGRAM_BINS))
                    for _col, _name in enumerate(HISTOGRAM_FEATURES):
                        enriched_attrs[_name] = float(node_hist[_col])
                    for _col, _name in enumerate(ANCHOR_GROUP_FEATURES):
                        enriched_attrs[_name] = float(topo["anchor_pe"][ast_idx, _col])
                else:
                    enriched_attrs["subtree_size"] = 0.0
                    enriched_attrs["subtree_depth"] = 0.0
                    for _name in HISTOGRAM_FEATURES:
                        enriched_attrs[_name] = 0.0
                    for _name in ANCHOR_GROUP_FEATURES:
                        enriched_attrs[_name] = 0.0

                G_enriched.add_node(node, **enriched_attrs)

            child_counters = {}
            for edge in raw.get("edges", []):
                parent = edge["source"]
                child = edge["target"]
                etype = edge["type"]

                child_idx = child_counters.get(parent, 0)
                child_counters[parent] = child_idx + 1

                self._add_ast_edges(
                    G_enriched,
                    parent,
                    child,
                    child_idx,
                    etype,
                    edge_direction,
                )

            # Include global→root edges so the DFS from global reaches all AST nodes.
            children_dict = {}
            for edge in raw.get("edges", []):
                if edge.get("type") == "child_of":
                    children_dict.setdefault(edge["source"], []).append(edge["target"])

        # Common G_enriched processing (augmented features path).
        # The augmented NextUse / function-nesting edges turn the expression *tree*
        # into a *graph*, so they are only added in graph mode. The tree and
        # tree_derivatives modes keep the pure (multi-)tree structure.
        if mode == "graph":
            last_seen_map = {}
            if "global" in G_enriched:
                build_augmented_math_graph(
                    G_enriched,
                    "global",
                    last_seen_map,
                    children_dict,
                    edge_direction,
                )
            else:
                roots = [n for n, d in G_ast.in_degree() if d == 0]
                for r in roots:
                    build_augmented_math_graph(
                        G_enriched,
                        r,
                        last_seen_map,
                        children_dict,
                        edge_direction,
                    )

        # Optional fully-connected supernode: injected after topology/PE features and the
        # augmented edges so it does not perturb the AST structural features. Independent
        # of mode — it connects to every node regardless of graph/tree/tree_derivatives.
        if add_virtual_supernode:
            inject_virtual_supernode(G_enriched, G_directed, node_ids)

        # Gather node_kappas mapping matching node_ids
        node_kappas = [G_enriched.nodes[nid].get("kappa_value") for nid in node_ids]
        
        # Remove kappa_value from node attributes to prevent PyG from_networkx mismatch error
        for nid in G_enriched.nodes:
            if "kappa_value" in G_enriched.nodes[nid]:
                del G_enriched.nodes[nid]["kappa_value"]

        # Ensure all nodes have exactly the same set of attribute keys
        all_node_keys = set()
        for nid in G_enriched.nodes:
            all_node_keys.update(G_enriched.nodes[nid].keys())
        for nid in G_enriched.nodes:
            for key in all_node_keys:
                if key not in G_enriched.nodes[nid]:
                    G_enriched.nodes[nid][key] = None

        # Ensure all edges have exactly the same set of attribute keys. Seed with the
        # full edge schema so columns that only appear on some edges (e.g. kappa_weight,
        # which is set on kappa edges only) are still present — and default to 0.0 — on
        # graphs that lack those edges. Otherwise from_networkx(group_edge_attrs=...)
        # raises a KeyError on the missing column.
        all_edge_keys = set(EDGE_FEATURE_SCHEMA)
        for u, v in G_enriched.edges:
            all_edge_keys.update(G_enriched.edges[u, v].keys())
        for u, v in G_enriched.edges:
            for key in all_edge_keys:
                if key not in G_enriched.edges[u, v]:
                    if key in ("child_index", "direction", "relation_type", "kappa_weight", "edge_type"):
                        G_enriched.edges[u, v][key] = 0.0
                    else:
                        G_enriched.edges[u, v][key] = None

        if heterogeneous:
            from gnn.shared.utils.heterogeneous_converter import to_hetero
            data = to_hetero(G_enriched, raw or {}, topo)
            data.__class__ = ExpressionHeteroData
            data.node_ids = node_ids
        else:
            from gnn.shared.utils.homogeneous_converter import to_homogeneous
            data = to_homogeneous(G_enriched, raw or {})
            data.__class__ = ExpressionGraphData
            data.node_ids = node_ids

        data.node_kappas = node_kappas

        node_type_tensor = torch.tensor(
            [G_directed.nodes[n]["node_type"] for n in node_ids], dtype=torch.long
        )
        root_color_tensor = torch.tensor(
            [G_directed.nodes[n].get("root_color", ROOT_COLOR_VOCAB["none"]) for n in node_ids],
            dtype=torch.long,
        )
        if not heterogeneous:
            data.node_type = node_type_tensor
            data.root_color = root_color_tensor

        # Add global graph features
        data.tree_depth = topo["tree_depth"]
        data.tree_width = topo["tree_width"]
        data.nodes = G_directed.number_of_nodes()
        data.num_nodes = G_directed.number_of_nodes()
        data.edges = G_directed.number_of_edges()
        data.num_edges = G_directed.number_of_edges()
        data.laplacian = torch.tensor(topo["laplace_matrix"], dtype=torch.float)

        return data

    @staticmethod
    def _load(source: Union[str, Path, dict]) -> dict:
        if isinstance(source, dict):
            return source
        with open(Path(source), encoding="utf-8") as f:
            return json.load(f)

    def _encode_label(self, label: str) -> int:
        return encode_label(label)

    def _encode_edge_type(self, etype: str) -> int:
        return encode_edge_type(etype)

    def _build_networkx(self, raw: dict) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in raw["nodes"]:
            orig_type = node.get("type", "operator")
            node_type_code = self.NODE_TYPES.get(orig_type, 1)
            G.add_node(
                node["id"],
                node_type=node_type_code,
                root_color=float(node.get("root_color", ROOT_COLOR_VOCAB["none"])),
                label_id=self._encode_label(node["label"]),
                type=orig_type,
                label=node["label"],
            )
        for edge in raw["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                edge_type=self._encode_edge_type(edge["type"]),
            )
        return G



class GraphConversionPipeline:
    """Loads all JSON graph files from a directory and converts them to PyG objects."""

    def __init__(
        self,
        experiments_dir: Union[str, Path],
        heterogeneous: bool = False,
        mode: str = "graph",
        edge_direction: str = "top_down",
    ):
        self.experiments_dir = Path(experiments_dir)
        self.heterogeneous = heterogeneous
        self.mode = mode
        self.edge_direction = validate_edge_direction(edge_direction)
        self.converter = ExpressionGraphConverter()
        self.graphs: dict[str, Union[Data, HeteroData]] = {}
        self._convert_all()

    def _discover_json_files(self) -> list[Path]:
        return sorted(self.experiments_dir.glob("**/*.json"))

    def _convert_all(self):
        for json_path in self._discover_json_files():
            raw = self.converter._load(json_path)
            if "nodes" not in raw or "edges" not in raw:
                print(f"Übersprungen: {json_path} — Keys: {list(raw.keys())}")
                continue
            graph_id = raw.get("id", json_path.stem)
            self.graphs[graph_id] = self.converter.convert(
                raw,
                heterogeneous=self.heterogeneous,
                mode=self.mode,
                edge_direction=self.edge_direction,
            )
        print(f"Geladen: {len(self.graphs)} Graphen")

    def get_data(self) -> dict[str, Union[Data, HeteroData]]:
        return self.graphs

    @property
    def label_vocab(self) -> dict[str, int]:
        return dict(CANONICAL_LABEL_VOCAB)

    @property
    def edge_type_vocab(self) -> dict[str, int]:
        return dict(CANONICAL_EDGE_TYPE_VOCAB)

    @property
    def input_dim(self) -> int:
        if not self.graphs:
            return 0

        sample_graph = next(iter(self.graphs.values()))

        if self.heterogeneous:
            return sample_graph["node"].x.shape[1]
        else:
            return sample_graph.x.shape[1]

    def get_feature_schema(self) -> list[str]:
        return list(NODE_FEATURE_SCHEMA)

    def get_edge_feature_schema(self) -> list[str]:
        return list(EDGE_FEATURE_SCHEMA)


def populate_task_virtual_values(
    data,
    *,
    cx_val: float,
    fx_val: float,
    yt_val: float,
    d1x_val: float = 0.0,
    d2x_val: float = 0.0,
    mode: str = "graph",
    set_has_value: bool = False,
    node_id_indices: dict[str, int] | None = None,
) -> None:
    """Write current iterate / function values onto task virtual and aggregator nodes.
    
    Arguments:
        data: The PyG Data or HeteroData object to populate values on.
        cx_val: The current value of x.
        fx_val: The value of function f(x).
        yt_val: The target y value.
        d1x_val: The value of first derivative f'(x).
        d2x_val: The value of second derivative f''(x).
        mode: The graph conversion mode (graph, tree, or tree_derivatives).
        set_has_value: Whether to set has_value flag to 1.0.
        node_id_indices: Optional precomputed node ID to index mapping.
        
    Returns:
        None
        
    Exceptions:
        None
    """
    # Task-value slots (virtual_current_x_val, virtual_delta_target_val, …) were removed
    # from NODE_FEATURE_SCHEMA in the position-aware GNN rewrite. The RL workflow will be
    # redesigned to convey solver state via a separate mechanism. This function is kept as
    # a graceful no-op so call sites don't need to be updated immediately.
    pass


def slice_active_features(x: torch.Tensor, active_features: list[str] | None) -> torch.Tensor:
    if active_features is None:
        return x
    full_schema = NODE_FEATURE_SCHEMA
    indices = []
    for f in active_features:
        if f in full_schema:
            indices.append(full_schema.index(f))
        else:
            raise ValueError(f"Feature '{f}' is not in the schema. Available: {full_schema}")
    return x[:, indices]


def parse_graphml_node_name(name_str: str) -> str:
    name_str = name_str.strip()
    if name_str.startswith('{') and name_str.endswith('}'):
        inner = name_str[1:-1]
        if ',' in inner:
            label = inner.split(',', 1)[0].strip()
            return label
        else:
            return inner.strip()
    return name_str


def _determine_node_type_from_label(label: str) -> str:
    if label in ["Plus", "Times", "Power"]:
        return "operator"
    if label in ["x", "E", "Pi", "I"]:
        return "variable"
    try:
        float(label)
        return "constant"
    except ValueError:
        pass
    if "/" in label:
        try:
            parts = label.split("/")
            if len(parts) == 2:
                float(parts[0])
                float(parts[1])
                return "constant"
        except ValueError:
            pass
    return "function"


def _parse_constant_value(label: str) -> float:
    try:
        if "/" in label:
            parts = label.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(label)
    except Exception:
        return 0.0


def find_roots(nodes_list, edges_list):
    targets = {e["target"] for e in edges_list}
    return [n["id"] for n in nodes_list if n["id"] not in targets]


def parse_graphml_to_nodes_and_edges(graphml_str: str, prefix: str):
    if not graphml_str or not isinstance(graphml_str, str) or graphml_str.strip() == "":
        return [], []
        
    content = graphml_str.replace("attr.type='String'", "attr.type='string'")
    content = content.replace('attr.type="String"', 'attr.type="string"')
    
    G = nx.parse_graphml(content)
    
    nodes = []
    edges = []
    
    for nid, attrs in G.nodes(data=True):
        name_val = attrs.get("Name") or attrs.get("nodeKey1")
        if not name_val:
            label = str(nid)
        else:
            label = parse_graphml_node_name(name_val)
        type_str = _determine_node_type_from_label(label)
        
        val = None
        if type_str == "constant":
            val = _parse_constant_value(label)
            
        nodes.append({
            "id": f"{prefix}_{nid}",
            "label": label,
            "type": type_str,
            "value": val
        })
        
    for u, v in G.edges():
        edges.append({
            "source": f"{prefix}_{u}",
            "target": f"{prefix}_{v}",
            "type": "child_of"
        })
        
    return nodes, edges


def create_virtual_global_node(
    graph_function: Union[str, nx.DiGraph],
    graph_derivative: Union[str, nx.DiGraph],
    graph_secondderivative: Union[str, nx.DiGraph]
) -> nx.DiGraph:
    """
    Creates a combined directed graph by adding a virtual global node 'global'
    that connects to the root nodes (in-degree == 0) of graph_function,
    graph_derivative, and graph_secondderivative.
    """
    def to_digraph(g) -> nx.DiGraph:
        if isinstance(g, str):
            if not g.strip():
                return nx.DiGraph()
            content = g.replace("attr.type='String'", "attr.type='string'")
            content = content.replace('attr.type="String"', 'attr.type="string"')
            return nx.parse_graphml(content)
        elif isinstance(g, nx.DiGraph):
            return g
        else:
            return nx.DiGraph()

    g_f = to_digraph(graph_function)
    g_d1 = to_digraph(graph_derivative)
    g_d2 = to_digraph(graph_secondderivative)

    def normalize_graph(G: nx.DiGraph, color: str):
        G_norm = nx.DiGraph()
        # Identify root nodes (in-degree 0) before building so we can colour them.
        in_degree_zero = {nid for nid, d in G.in_degree() if d == 0}
        for nid, attrs in G.nodes(data=True):
            if "node_type" in attrs and attrs["node_type"] != 1:
                # Already fully encoded (e.g. re-entrant call); just copy.
                G_norm.add_node(nid, **attrs)
                continue
            name_val = attrs.get("Name") or attrs.get("nodeKey1")
            if not name_val:
                label = str(nid)
            else:
                label = parse_graphml_node_name(name_val)
            is_root = nid in in_degree_zero
            type_str = "root" if is_root else _determine_node_type_from_label(label)
            G_norm.add_node(
                nid,
                node_type=ExpressionGraphConverter.NODE_TYPES.get(type_str, 1),
                root_color=float(ROOT_COLOR_VOCAB[color] if is_root else ROOT_COLOR_VOCAB["none"]),
                label_id=encode_label(label),
                label=label,
                type=type_str,
            )
        for u, v, attrs in G.edges(data=True):
            G_norm.add_edge(u, v, **attrs)
        return G_norm

    g_f = normalize_graph(g_f, "f")
    g_d1 = normalize_graph(g_d1, "d1")
    g_d2 = normalize_graph(g_d2, "d2")

    roots_f = [n for n, d in g_f.in_degree() if d == 0]
    roots_d1 = [n for n, d in g_d1.in_degree() if d == 0]
    roots_d2 = [n for n, d in g_d2.in_degree() if d == 0]

    G_combined = nx.DiGraph()
    G_combined.add_node(
        "global",
        node_type=ExpressionGraphConverter.NODE_TYPES["global"],
        root_color=float(ROOT_COLOR_VOCAB["none"]),
        label_id=encode_label("GLOBAL"),
        label="GLOBAL",
        type="global",
    )

    def add_component(g_comp, prefix: str):
        for nid, attrs in g_comp.nodes(data=True):
            G_combined.add_node(f"{prefix}_{nid}", **attrs)
        for u, v, attrs in g_comp.edges(data=True):
            G_combined.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **attrs)

    add_component(g_f, "f")
    add_component(g_d1, "d1")
    add_component(g_d2, "d2")

    # Connect global directly to each function tree root (no aggregator nodes).
    for r in roots_f:
        G_combined.add_edge("global", f"f_{r}", edge_type=encode_edge_type("child_of"))
    for r in roots_d1:
        G_combined.add_edge("global", f"d1_{r}", edge_type=encode_edge_type("child_of"))
    for r in roots_d2:
        G_combined.add_edge("global", f"d2_{r}", edge_type=encode_edge_type("child_of"))

    return G_combined


def compute_normalized_dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> float:
    """
    Computes the normalized Dirichlet energy of node features x:
    E_norm(x) = tr(x^T * L_sym * x) / tr(x^T * x)
    where L_sym = I - D^{-1/2} * A * D^{-1/2}.
    """
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return 0.0

    # Ensure x is float
    x = x.float()

    # Handle multidimensional edge weights (e.g. edge attributes)
    if edge_weight is not None:
        if edge_weight.dim() > 1:
            if edge_weight.size(-1) == 1:
                edge_weight = edge_weight.squeeze(-1)
            else:
                edge_weight = None

    # Calculate degree
    if edge_weight is None:
        from torch_geometric.utils import degree
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=x.dtype)
    else:
        from torch_scatter import scatter
        deg = scatter(edge_weight.float(), edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0.0

    # x_tilde = D^{-1/2} * x
    x_tilde = x * deg_inv_sqrt.unsqueeze(-1)

    # tr(x^T * D^{-1/2} * A * D^{-1/2} * x) = sum_{u, v} A_{uv} * x_tilde_u^T * x_tilde_v
    src, dst = edge_index[0], edge_index[1]
    dot_products = (x_tilde[src] * x_tilde[dst]).sum(dim=-1)

    if edge_weight is not None:
        edge_sum = (edge_weight.float() * dot_products).sum()
    else:
        edge_sum = dot_products.sum()

    tr_x_x = (x * x).sum()
    if tr_x_x == 0:
        return 0.0

    tr_x_L_x = tr_x_x - edge_sum
    normalized_energy = tr_x_L_x / tr_x_x

    return float(normalized_energy.item())


# ==============================================================================
# Augmented Graph Dataloader Enhancement (kappa h-functions integration)
# ==============================================================================


class KappaEdge:
    """Represents a connection between the global node and a kappa root node.

    Attributes:
        source (str): The ID of the source node.
        target (str): The ID of the target node.
        type (str): The type of the edge ("GlobalToKappa" or "KappaToGlobal").
        features (dict[str, float]): A dictionary containing edge features (e.g., "weight").
    """

    def __init__(self, source: str, target: str, type: str):
        """Initializes a new instance of KappaEdge.

        Arguments:
            source: The ID of the source node.
            target: The ID of the target node.
            type: The type of the edge.

        Returns:
            None

        Raises:
            None
        """
        self.source = source
        self.target = target
        self.type = type
        self.features: dict[str, float] = {"weight": 0.0}


class AugmentedFunctionGraph(nx.DiGraph):
    """NetworkX DiGraph wrapper that supports operations required by the LoadAugmentedFunctionGraph algorithm.

    Attributes:
        subgraph_counter (int): Counter to generate unique node IDs when merging subgraphs.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """Initializes the AugmentedFunctionGraph with optional incoming graph data.

        Arguments:
            incoming_graph_data: Graph data to initialize the NetworkX DiGraph.
            attr: Additional graph attributes.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(incoming_graph_data, **attr)
        self.subgraph_counter = 0

    def HasGlobalNode(self) -> bool:
        """Checks if a node of type 'global' exists in the graph.

        Returns:
            bool: True if a global node is present, False otherwise.

        Raises:
            None
        """
        for node, attrs in self.nodes(data=True):
            if attrs.get("type") == "global" or node == "global":
                return True
        return False

    def GetGlobalNode(self) -> str:
        """Retrieves the ID of the global node.

        Returns:
            str: The ID of the global node.

        Raises:
            KeyError: If no global node exists in the graph.
        """
        for node, attrs in self.nodes(data=True):
            if attrs.get("type") == "global" or node == "global":
                return str(node)
        raise KeyError("Global node not found in graph.")

    def CreateVirtualGlobalNode(self, nodeType: str = "GlobalContext") -> str:
        """Creates a virtual global node in the graph and returns its ID.

        Arguments:
            nodeType: The type attribute to assign to the new node.

        Returns:
            str: The ID of the created global node (always 'global').

        Raises:
            None
        """
        global_id = "global"
        self.add_node(
            global_id,
            node_type=ExpressionGraphConverter.NODE_TYPES.get("global", 0),
            root_color=float(ROOT_COLOR_VOCAB["none"]),
            label_id=encode_label("GLOBAL"),
            label="GLOBAL",
            type="global",
            context_type=nodeType,
        )
        return global_id

    def MergeDisjointSubgraph(self, kappa_subgraph: Union[nx.DiGraph, str, dict]) -> str:
        """Merges a disjoint kappa subgraph into the main graph, avoiding ID collisions.

        Arguments:
            kappa_subgraph: The kappa subgraph to merge. Can be an nx.DiGraph, a GraphML string, or a dict.

        Returns:
            str: The shifted node ID of the root node of the merged kappa subgraph.

        Raises:
            TypeError: If the kappa_subgraph has an unsupported type.
            ValueError: If the kappa_subgraph is empty or invalid.
        """
        self.subgraph_counter += 1
        prefix = f"kappa_{self.subgraph_counter}"

        # 1. Parse/normalize the input subgraph into an nx.DiGraph
        if isinstance(kappa_subgraph, str):
            content = kappa_subgraph.replace("attr.type='String'", "attr.type='string'")
            content = content.replace('attr.type="String"', 'attr.type="string"')
            g_kappa = nx.parse_graphml(content)
        elif isinstance(kappa_subgraph, nx.DiGraph):
            g_kappa = kappa_subgraph
        elif isinstance(kappa_subgraph, dict):
            g_kappa = nx.DiGraph()
            for node in kappa_subgraph.get("nodes", []):
                g_kappa.add_node(node["id"], **node)
            for edge in kappa_subgraph.get("edges", []):
                g_kappa.add_edge(edge["source"], edge["target"], **edge)
        else:
            raise TypeError(f"Unsupported subgraph type: {type(kappa_subgraph)}")

        # Normalize nodes in g_kappa so they have all standard attributes
        normalized_g_kappa = nx.DiGraph()
        for nid, attrs in g_kappa.nodes(data=True):
            name_val = attrs.get("Name") or attrs.get("nodeKey1") or attrs.get("label") or str(nid)
            label = parse_graphml_node_name(name_val) if isinstance(name_val, str) else str(name_val)

            type_str = attrs.get("type") or _determine_node_type_from_label(label)
            val = 0.0
            has_val = 0.0
            if type_str == "constant":
                val_attr = attrs.get("value")
                if isinstance(val_attr, (int, float)):
                    val = float(val_attr)
                else:
                    val = _parse_constant_value(label)
                has_val = 1.0

            ntype_code = ExpressionGraphConverter.NODE_TYPES.get(type_str, 1)

            normalized_g_kappa.add_node(
                nid,
                node_type=ntype_code,
                root_color=float(ROOT_COLOR_VOCAB["none"]),
                label_id=encode_label(label),
                label=label,
                type=type_str,
            )

        for u, v, attrs in g_kappa.edges(data=True):
            etype = attrs.get("type") or attrs.get("etype") or "child_of"
            normalized_g_kappa.add_edge(
                u, v,
                edge_type=encode_edge_type(etype),
                etype=etype
            )

        # 2. Identify the root node in the normalized subgraph
        roots = [n for n, d in normalized_g_kappa.in_degree() if d == 0]
        if roots:
            original_root = roots[0]
        else:
            original_root = list(normalized_g_kappa.nodes)[0] if normalized_g_kappa.nodes else None

        if original_root is None:
            raise ValueError("Cannot merge an empty kappa subgraph.")

        # 3. Add nodes and edges to self, shifting the node IDs
        shifted_root_id = f"{prefix}_{original_root}"

        for nid, attrs in normalized_g_kappa.nodes(data=True):
            shifted_id = f"{prefix}_{nid}"
            node_attrs = dict(attrs)
            if nid == original_root:
                node_attrs["node_type"] = ExpressionGraphConverter.NODE_TYPES["root"]
                node_attrs["root_color"] = float(ROOT_COLOR_VOCAB["kappa"])
                node_attrs["type"] = "root"
            self.add_node(shifted_id, **node_attrs)

        for u, v, attrs in normalized_g_kappa.edges(data=True):
            self.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **attrs)

        return shifted_root_id

    def AddEdge(self, edge: KappaEdge) -> None:
        """Adds a KappaEdge connection between the global node and a kappa root node.

        Arguments:
            edge: The KappaEdge to add to the graph.

        Returns:
            None

        Raises:
            None
        """
        weight = edge.features.get("weight", 0.0)
        edge_type_code = encode_edge_type(edge.type)
        direction_val = 0.0 if edge.type == "GlobalToKappa" else 1.0

        self.add_edge(
            edge.source,
            edge.target,
            edge_type=edge_type_code,
            etype=edge.type,
            kappa_weight=weight,
            child_index=0.0,
            direction=direction_val,
            relation_type=float(edge_type_code),
            edge_betweenness_centrality=0.0,
        )


def LoadGraphFromLocalStructure(folder: Union[Path, str], id: str) -> AugmentedFunctionGraph:
    """Loads a mathematical basis graph by ID from local graphs folder and returns it as an AugmentedFunctionGraph.

    Arguments:
        folder: The folder (directory or file) containing the graph data.
        id: The unique ID of the graph to load.

    Returns:
        AugmentedFunctionGraph containing the loaded mathematical graph.

    Raises:
        FileNotFoundError: If the folder does not exist.
        KeyError: If the graph ID is not found.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder or file not found: {folder}")

    raw_data = None
    try:
        if folder_path.is_file():
            with open(folder_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("id") == id:
                        raw_data = item
                        break
            elif isinstance(data, dict):
                if data.get("id") == id:
                    raw_data = data
                elif id in data:
                    raw_data = data[id]
        else:
            direct_files = [
                folder_path / f"{id}.json",
                folder_path / f"{id}_meta.json"
            ]
            for df in direct_files:
                if df.exists() and df.is_file():
                    with open(df, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    break

            if raw_data is None:
                for filepath in folder_path.glob("**/*.json"):
                    if ".pt_cache" in filepath.parts:
                        continue
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and item.get("id") == id:
                                    raw_data = item
                                    break
                        elif isinstance(data, dict):
                            if data.get("id") == id:
                                raw_data = data
                                break
                            elif id in data:
                                raw_data = data[id]
                                break
                    except Exception:
                        continue
                    if raw_data is not None:
                        break
    except Exception as e:
        logger.error(f"Error reading local graph structure for ID {id}: {e}")
        raise

    if raw_data is None:
        raise KeyError(f"Graph ID '{id}' not found in folder '{folder}'")

    if "graphml_f" in raw_data:
        g_f = raw_data.get("graphml_f", "")
        g_d1 = raw_data.get("graphml_derivative1", "")
        g_d2 = raw_data.get("graphml_derivative2", "")
        nx_graph = create_virtual_global_node(g_f, g_d1, g_d2)
    else:
        nx_graph = nx.DiGraph()
        nodes = raw_data.get("nodes", [])
        edges = raw_data.get("edges", [])

        _mark_function_roots(raw_data)

        for node in nodes:
            orig_type = node.get("type", "operator")
            ntype_code = ExpressionGraphConverter.NODE_TYPES.get(orig_type, 1)
            nx_graph.add_node(
                node["id"],
                node_type=ntype_code,
                root_color=float(node.get("root_color", ROOT_COLOR_VOCAB["none"])),
                label_id=encode_label(node["label"]),
                type=orig_type,
                label=node["label"],
            )
        for edge in edges:
            nx_graph.add_edge(
                edge["source"],
                edge["target"],
                edge_type=encode_edge_type(edge["type"]),
                etype=edge["type"]
            )

    return AugmentedFunctionGraph(nx_graph)


def LoadAugmentedFunctionGraph(
    graphId: str, graphsFolder: Union[str, Path], kappasFolder: Union[str, Path]
) -> AugmentedFunctionGraph:
    """Enhances a main function graph by merging matching kappa h-functions.

    Arguments:
        graphId: The unique ID of the main function graph to load.
        graphsFolder: Path to the folder containing mathematical basis graphs.
        kappasFolder: Path to the folder containing kappa h-functions.

    Returns:
        An AugmentedFunctionGraph that is compatible with PyTorch-Geometric/NetworkX.

    Raises:
        FileNotFoundError: If the folders do not exist.
        KeyError: If the graphId is not found.
    """
    mainGraph = LoadGraphFromLocalStructure(folder=graphsFolder, id=graphId)

    if not mainGraph.HasGlobalNode():
        globalNode = mainGraph.CreateVirtualGlobalNode(nodeType="GlobalContext")
    else:
        globalNode = mainGraph.GetGlobalNode()

    kappas_path = Path(kappasFolder)
    if not kappas_path.exists():
        raise FileNotFoundError(f"Kappas folder not found: {kappasFolder}")

    kappa_files = list(kappas_path.glob("**/*.json"))

    for file_path in kappa_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                containers = data
            elif isinstance(data, dict):
                containers = [data]
            else:
                continue

            for kappaContainer in containers:
                if not isinstance(kappaContainer, dict):
                    continue

                if kappaContainer.get("id") == "kappa":
                    kappa_val_raw = kappaContainer.get("value")
                    try:
                        kappaValue = float(kappa_val_raw)
                    except (ValueError, TypeError):
                        kappaValue = 0.0

                    kappaSubgraph = kappaContainer.get("graphStructure") or kappaContainer.get("graphml_h")
                    if not kappaSubgraph:
                        continue

                    kappaRootNodeId = mainGraph.MergeDisjointSubgraph(kappaSubgraph)

                    # Tag all nodes in this merged kappa subgraph with their kappa value
                    prefix_parts = kappaRootNodeId.split("_")
                    if len(prefix_parts) >= 2:
                        prefix_str = f"{prefix_parts[0]}_{prefix_parts[1]}"
                        for node in mainGraph.nodes:
                            if str(node).startswith(prefix_str + "_"):
                                mainGraph.nodes[node]["kappa_value"] = kappaValue

                    newEdge = KappaEdge(
                        source=globalNode,
                        target=kappaRootNodeId,
                        type="GlobalToKappa"
                    )
                    newEdge.features["weight"] = kappaValue
                    mainGraph.AddEdge(newEdge)

                    backwardEdge = KappaEdge(
                        source=kappaRootNodeId,
                        target=globalNode,
                        type="KappaToGlobal"
                    )
                    backwardEdge.features["weight"] = kappaValue
                    mainGraph.AddEdge(backwardEdge)
        except Exception as e:
            logger.warning(f"Error reading or processing kappa file {file_path}: {e}")

    return mainGraph


def filter_active_kappa(
    data: Union[Data, HeteroData], active_kappa: Union[float, int, None]
) -> Union[Data, HeteroData]:
    """Filters the PyG Data object to only keep nodes and edges of the active kappa subgraph.

    All base graph nodes, global nodes, and aggregator nodes are kept. Inactive kappa subgraph nodes
    and their associated edges are removed. If active_kappa is None, 0, or NaN, all kappa subgraphs
    are deactivated and removed.

    Arguments:
        data: The PyG Data or HeteroData object containing node_kappas.
        active_kappa: The kappa value to keep active.

    Returns:
        The filtered PyG Data or HeteroData object.

    Raises:
        None
    """
    import torch
    from torch_geometric.data import Data

    if not isinstance(data, Data) or not hasattr(data, "node_kappas") or data.node_kappas is None:
        return data

    node_kappas = data.node_kappas
    num_nodes = data.x.size(0) if data.x is not None else len(node_kappas)

    # 1. Determine active_kappa validity
    is_active_kappa_valid = False
    if active_kappa is not None:
        try:
            act_k = float(active_kappa)
            if not math.isnan(act_k) and act_k != 0.0:
                is_active_kappa_valid = True
        except (ValueError, TypeError):
            pass

    # 2. Identify nodes to keep
    keep_node_indices = []
    for i in range(num_nodes):
        kappa_val = node_kappas[i]
        if kappa_val is None:
            # Base node / global node / aggregator node
            keep_node_indices.append(i)
        elif is_active_kappa_valid:
            try:
                if abs(float(kappa_val) - float(active_kappa)) < 1e-3:
                    keep_node_indices.append(i)
            except (ValueError, TypeError):
                pass

    # 3. Create node mask
    keep_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    keep_node_mask[keep_node_indices] = True

    # If all nodes are kept, return unmodified
    if keep_node_mask.all():
        return data

    # 4. Filter node-level attributes
    if hasattr(data, "node_ids") and data.node_ids is not None:
        data.node_ids = [data.node_ids[i] for i in keep_node_indices]

    data.node_kappas = [node_kappas[i] for i in keep_node_indices]

    # Map old node indices to new indices
    map_tensor = torch.empty(num_nodes, dtype=torch.long)
    map_tensor[keep_node_mask] = torch.arange(len(keep_node_indices))

    # 5. Filter edge-level attributes
    num_edges = 0
    keep_edge_mask = None
    if hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
        src, dst = data.edge_index[0], data.edge_index[1]
        keep_edge_mask = keep_node_mask[src] & keep_node_mask[dst]
        data.edge_index = map_tensor[data.edge_index[:, keep_edge_mask]]
        num_edges = keep_edge_mask.size(0)

    # 6. Apply masks to all tensors in the data object
    for key, value in list(data.items()):
        if isinstance(value, torch.Tensor):
            if key == "edge_index":
                continue
            # Check known key lists first to prevent size collision
            if key in ("x", "node_type", "label_id", "belongs_to_f", "belongs_to_d1", "belongs_to_d2"):
                data[key] = value[keep_node_mask]
            elif key in ("edge_attr", "edge_type"):
                if keep_edge_mask is not None:
                    data[key] = value[keep_edge_mask]
            elif value.dim() > 0 and value.size(0) == num_nodes:
                data[key] = value[keep_node_mask]
            elif value.dim() > 0 and value.size(0) == num_edges:
                if keep_edge_mask is not None:
                    data[key] = value[keep_edge_mask]

    # Update counts if present
    if hasattr(data, "nodes"):
        data.nodes = len(keep_node_indices)
    if hasattr(data, "num_nodes"):
        data.num_nodes = len(keep_node_indices)
    if hasattr(data, "edges") and keep_edge_mask is not None:
        data.edges = int(keep_edge_mask.sum().item())
    if hasattr(data, "num_edges") and keep_edge_mask is not None:
        data.num_edges = int(keep_edge_mask.sum().item())

    return data


