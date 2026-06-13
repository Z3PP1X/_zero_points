import json
import logging
import math
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
    "f_root",
    "d1_root",
    "d2_root",
)
CANONICAL_LABEL_VOCAB: dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}

FUNCTION_AGGREGATOR_CONFIG: tuple[tuple[str, str], ...] = (
    ("belongs_to_f", "f_root"),
    ("belongs_to_d1", "d1_root"),
    ("belongs_to_d2", "d2_root"),
)
FUNCTION_AGGREGATOR_IDS: frozenset[str] = frozenset(agg for _, agg in FUNCTION_AGGREGATOR_CONFIG)

CANONICAL_EDGE_TYPES: tuple[str, ...] = (
    "<UNK>",
    "child_of",
    "child_of_reverse",
    "belongs_to_f",
    "belongs_to_f_reverse",
    "belongs_to_d1",
    "belongs_to_d1_reverse",
    "belongs_to_d2",
    "belongs_to_d2_reverse",
    "virtual",
    "virtual_reverse",
    "supernode_connection",
    "supernode_connection_reverse",
    "NextUse",
    "NextUseBackward",
    "GlobalToKappa",
    "KappaToGlobal",
) + tuple(f"OuterToInner_Arg{i}" for i in range(10)) + tuple(f"InnerToOuter_Arg{i}" for i in range(10))
CANONICAL_EDGE_TYPE_VOCAB: dict[str, int] = {etype: idx for idx, etype in enumerate(CANONICAL_EDGE_TYPES)}

VIRTUAL_NODE_TYPES = frozenset(FUNCTION_AGGREGATOR_IDS)

# Categorical vocabulary sizes (embedding-table row counts). node_type ids occupy a
# fixed 0..10 code space (11 rows, with gaps); label / edge sizes follow their vocabs.
# Single source of truth — model encoders import these instead of redefining them.
NUM_NODE_TYPES: int = 11
NUM_LABELS: int = len(CANONICAL_LABEL_VOCAB)
NUM_EDGE_TYPES: int = len(CANONICAL_EDGE_TYPE_VOCAB)


def signed_log_value(value: float) -> float:
    """sign(v) * log1p(|v|) — same transform as RL global features."""
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


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


def get_hetero_node_type(raw_type: str) -> str:
    if raw_type in ("operator", "function"):
        return "operator"
    elif raw_type == "variable":
        return "variable"
    elif raw_type == "constant":
        return "constant"
    elif raw_type in ("global", "f_root", "d1_root", "d2_root"):
        return "virtual"
    else:
        return "virtual"


def fourier_frequency_encoding(val: float) -> list[float]:
    # 8-dimensional multi-scale sinusoidal/fourier frequency encoding vector
    frequencies = [2.0 ** i for i in range(4)]
    enc = []
    for freq in frequencies:
        enc.append(math.sin(val * freq))
        enc.append(math.cos(val * freq))
    return enc


def get_relation_type(parent_label: str, etype: str, child_index: float) -> str:
    is_reverse = etype.endswith("_reverse")
    base_etype = etype[:-8] if is_reverse else etype
    
    if base_etype == "child_of":
        if parent_label in ("Plus", "Times", "GLOBAL", "f_root", "d1_root", "d2_root"):
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



def _compute_belongs_to_subtree(
    raw: dict,
    prov_edge_type: str,
    id_prefix: str,
) -> dict[str, float]:
    """Mark nodes in a provenance subtree (aggregator edge + id prefix + child_of BFS)."""
    child_edges = [
        (edge["source"], edge["target"])
        for edge in raw.get("edges", [])
        if edge.get("type") == "child_of"
    ]
    subtree_roots = {
        edge["target"]
        for edge in raw.get("edges", [])
        if edge.get("type") == prov_edge_type
    }

    subtree_nodes: set[str] = set(subtree_roots)
    for node in raw.get("nodes", []):
        node_id = node["id"]
        if str(node_id).startswith(id_prefix):
            subtree_nodes.add(node_id)

    queue = list(subtree_roots)
    while queue:
        parent = queue.pop(0)
        for src, tgt in child_edges:
            if src == parent and tgt not in subtree_nodes:
                subtree_nodes.add(tgt)
                queue.append(tgt)

    return {
        node["id"]: 1.0 if node["id"] in subtree_nodes else 0.0
        for node in raw.get("nodes", [])
    }


def compute_belongs_to_f(raw: dict) -> dict[str, float]:
    return _compute_belongs_to_subtree(raw, "belongs_to_f", "f_")


def compute_belongs_to_d1(raw: dict) -> dict[str, float]:
    return _compute_belongs_to_subtree(raw, "belongs_to_d1", "d1_")


def compute_belongs_to_d2(raw: dict) -> dict[str, float]:
    return _compute_belongs_to_subtree(raw, "belongs_to_d2", "d2_")


NODE_FEATURE_SCHEMA = [
    "node_type",
    "label_id",
    "depth",
    "height",
    "subtree_size",
    "out_degree",
    "betweenness_centrality",
    "value",
    "has_value",
    "lpe_1",
    "lpe_2",
    "lpe_3",
    "lpe_4",
    "rwpe_1",
    "rwpe_2",
    "rwpe_3",
    "rwpe_4",
    "virtual_current_x_val",
    "virtual_delta_target_val",
    "virtual_d1_x_val",
    "virtual_d2_x_val",
    "belongs_to_f",
    "belongs_to_d1",
    "belongs_to_d2",
]

EDGE_FEATURE_SCHEMA = [
    "child_index",
    "direction",
    "relation_type",
    "edge_betweenness_centrality",
]

EDGE_DIRECTIONS: tuple[str, ...] = ("top_down", "bottom_up", "bidirectional")


def validate_edge_direction(edge_direction: str) -> str:
    if edge_direction not in EDGE_DIRECTIONS:
        raise ValueError(
            f"Unsupported edge_direction {edge_direction!r}; "
            f"expected one of {list(EDGE_DIRECTIONS)}"
        )
    return edge_direction


def _find_global_node_id(raw: dict) -> str | None:
    for node in raw.get("nodes", []):
        if node.get("type") == "global":
            return node["id"]
    return None


def _insert_function_aggregators(raw: dict) -> None:
    """Insert provenance-scoped aggregator nodes between global and AST roots."""
    global_id = _find_global_node_id(raw)
    if global_id is None:
        return

    node_ids = {node["id"] for node in raw["nodes"]}
    edges = list(raw.get("edges", []))

    for prov_type, agg_id in FUNCTION_AGGREGATOR_CONFIG:
        roots = [
            edge["target"]
            for edge in edges
            if edge.get("source") == global_id and edge.get("type") == prov_type
        ]
        if not roots:
            continue

        if agg_id not in node_ids:
            raw["nodes"].append({
                "id": agg_id,
                "label": agg_id,
                "type": agg_id,
                "value": None,
            })
            node_ids.add(agg_id)

        edges = [
            edge
            for edge in edges
            if not (edge.get("source") == global_id and edge.get("type") == prov_type)
        ]
        edges.append({"source": global_id, "target": agg_id, "type": prov_type})
        for root in roots:
            edges.append({"source": agg_id, "target": root, "type": "child_of"})

    implicit_specs = (
        ("f_root", "belongs_to_f", lambda node_id: not node_id.startswith(("d1_", "d2_"))),
        ("d1_root", "belongs_to_d1", lambda node_id: node_id.startswith("d1_")),
        ("d2_root", "belongs_to_d2", lambda node_id: node_id.startswith("d2_")),
    )
    for agg_id, prov_type, id_predicate in implicit_specs:
        if agg_id in node_ids:
            continue

        ast_nodes = [
            node["id"]
            for node in raw["nodes"]
            if node["id"] != global_id
            and node.get("type") not in VIRTUAL_NODE_TYPES
            and node.get("type") != "global"
            and id_predicate(str(node["id"]))
        ]
        if not ast_nodes:
            continue

        ast_set = set(ast_nodes)
        internal_child_targets = {
            edge["target"]
            for edge in edges
            if edge.get("type") == "child_of" and edge["source"] in ast_set
        }
        roots = [node_id for node_id in ast_nodes if node_id not in internal_child_targets]
        if not roots:
            roots = ast_nodes

        raw["nodes"].append({
            "id": agg_id,
            "label": agg_id,
            "type": agg_id,
            "value": None,
        })
        node_ids.add(agg_id)
        edges.append({"source": global_id, "target": agg_id, "type": prov_type})
        for root in roots:
            edges.append({"source": agg_id, "target": root, "type": "child_of"})

    raw["edges"] = edges


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

        # Undirected graph representation for Laplacians, LPE, RWPE, and Undirected Centralities
        G_und = G.to_undirected()
        num_nodes = G_und.number_of_nodes()

        # Betweenness Centrality (computed on undirected graph)
        betweenness = nx.betweenness_centrality(G_und)

        # Edge Betweenness Centrality (computed on undirected graph)
        edge_betweenness = nx.edge_betweenness_centrality(G_und)
        eb_lookup = {}
        for (u, v), val in edge_betweenness.items():
            eb_lookup[(u, v)] = val
            eb_lookup[(v, u)] = val

        # Laplace-Matrix (Graph)
        if num_nodes > 0:
            laplace_matrix = nx.laplacian_matrix(G_und).toarray()
        else:
            laplace_matrix = np.zeros((0, 0))

        # Laplacian Positional Encodings (LPE) (dimension 4)
        lpe_features = np.zeros((num_nodes, 4))
        if num_nodes > 1:
            try:
                A = nx.to_numpy_array(G_und)
                d = A.sum(axis=1)
                d_inv_sqrt = np.zeros_like(d)
                d_inv_sqrt[d > 0] = np.power(d[d > 0], -0.5)
                D_inv_sqrt = np.diag(d_inv_sqrt)
                L_norm = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
                
                evals, evecs = np.linalg.eigh(L_norm)
                idx = np.argsort(evals)
                evals = evals[idx]
                evecs = evecs[:, idx]
                
                lpe_list = []
                for i in range(1, 5):
                    if i < num_nodes:
                        # abs() removes arbitrary sign ambiguity from eigh eigenvectors
                        lpe_list.append(np.abs(evecs[:, i]))
                    else:
                        lpe_list.append(np.zeros(num_nodes))
                lpe_features = np.stack(lpe_list, axis=1)
            except Exception:
                lpe_features = np.zeros((num_nodes, 4))

        # Random Walk Positional Encodings (RWPE) (4 steps).
        # NOTE: the AST is a tree, hence bipartite. On a bipartite graph the
        # return probability of a *non-lazy* random walk is exactly 0 for every
        # odd number of steps, which previously made rwpe_1/rwpe_3 dead (all-zero)
        # features. We use a *lazy* random walk P = 1/2 (I + D^-1 A) and record
        # the return probabilities for steps k=2..5, so all four dimensions carry
        # structural information regardless of bipartiteness.
        rwpe_features = np.zeros((num_nodes, 4))
        if num_nodes > 0:
            try:
                A = nx.to_numpy_array(G_und)
                d = A.sum(axis=1)
                d_inv = np.zeros_like(d)
                d_inv[d > 0] = 1.0 / d[d > 0]
                D_inv = np.diag(d_inv)
                P_lazy = 0.5 * (np.eye(num_nodes) + D_inv @ A)

                # Skip k=1 (its diagonal is the constant 0.5 for every node and
                # therefore carries no structural signal); record k=2..5.
                Pk = P_lazy @ P_lazy
                for step in range(4):
                    rwpe_features[:, step] = np.diag(Pk)
                    Pk = Pk @ P_lazy
            except Exception:
                rwpe_features = np.zeros((num_nodes, 4))

        results.update({
            "heights": heights,
            "subtree_sizes": subtree_sizes,
            "out_degrees": out_degrees,
            "betweenness": betweenness,
            "edge_betweenness": eb_lookup,
            "laplace_matrix": laplace_matrix,
            "lpe": lpe_features,
            "rwpe": rwpe_features,
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
            edge_betweenness_centrality=0.0,
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
    NODE_TYPES = {
        "global": 0,
        "operator": 1,
        "constant": 2,
        "variable": 3,
        "function": 4,
        "f_root": 6,
        "d1_root": 9,
        "d2_root": 10,
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
        eb_val: float,
        edge_direction: str,
    ) -> None:
        effective = edge_direction
        if effective in ("top_down", "bidirectional"):
            G_enriched.add_edge(
                parent,
                child,
                child_index=float(child_idx),
                direction=0.0,
                relation_type=float(self._encode_edge_type(etype)),
                edge_betweenness_centrality=eb_val,
                etype=etype,
            )
        if effective in ("bottom_up", "bidirectional"):
            G_enriched.add_edge(
                child,
                parent,
                child_index=float(child_idx),
                direction=1.0 if effective == "bidirectional" else 0.0,
                relation_type=float(self._encode_edge_type(etype + "_reverse")),
                edge_betweenness_centrality=eb_val,
                etype=etype + "_reverse",
            )

    def convert(
        self,
        source: Union[str, Path, dict, nx.DiGraph],
        heterogeneous: bool = False,
        mode: str = "graph",
        edge_direction: str = "top_down",
    ) -> Union[Data, HeteroData]:
        edge_direction = validate_edge_direction(edge_direction)
        
        if isinstance(source, nx.DiGraph):
            # Extract AST nodes and edges to construct G_ast for topological extraction
            ast_nodes = [
                n for n in source.nodes 
                if n not in ("global", "f_root", "d1_root", "d2_root")
                and not str(n).startswith("kappa_")
            ]
            G_ast = source.subgraph(ast_nodes).copy()
            topo = TopologicalFeatureExtractor.extract_and_annotate(G_ast)
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
                    enriched_attrs["depth"] = float(topo["depths"].get(node, 0.0))
                    enriched_attrs["height"] = float(topo["heights"].get(node, 0.0))
                    enriched_attrs["subtree_size"] = float(topo["subtree_sizes"].get(node, 1.0))
                    enriched_attrs["out_degree"] = float(topo["out_degrees"].get(node, 0.0))
                    enriched_attrs["betweenness_centrality"] = float(topo["betweenness"].get(node, 0.0))
                    enriched_attrs["lpe_1"] = float(topo["lpe"][ast_idx, 0])
                    enriched_attrs["lpe_2"] = float(topo["lpe"][ast_idx, 1])
                    enriched_attrs["lpe_3"] = float(topo["lpe"][ast_idx, 2])
                    enriched_attrs["lpe_4"] = float(topo["lpe"][ast_idx, 3])
                    enriched_attrs["rwpe_1"] = float(topo["rwpe"][ast_idx, 0])
                    enriched_attrs["rwpe_2"] = float(topo["rwpe"][ast_idx, 1])
                    enriched_attrs["rwpe_3"] = float(topo["rwpe"][ast_idx, 2])
                    enriched_attrs["rwpe_4"] = float(topo["rwpe"][ast_idx, 3])
                else:
                    enriched_attrs["depth"] = 0.0
                    enriched_attrs["height"] = 0.0
                    enriched_attrs["subtree_size"] = 1.0
                    enriched_attrs["out_degree"] = 0.0
                    enriched_attrs["betweenness_centrality"] = 0.0
                    enriched_attrs["lpe_1"] = 0.0
                    enriched_attrs["lpe_2"] = 0.0
                    enriched_attrs["lpe_3"] = 0.0
                    enriched_attrs["lpe_4"] = 0.0
                    enriched_attrs["rwpe_1"] = 0.0
                    enriched_attrs["rwpe_2"] = 0.0
                    enriched_attrs["rwpe_3"] = 0.0
                    enriched_attrs["rwpe_4"] = 0.0

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

                eb_val = float(topo["edge_betweenness"].get((parent, child), 0.0))
                self._add_ast_edges(
                    G_enriched, parent, child, child_idx, etype, eb_val, edge_direction
                )

            children_dict = {}
            for u, v, attrs in G_ast.edges(data=True):
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

                combined_edges = edges_f + edges_d1 + edges_d2
                if roots_f:
                    combined_nodes.append({
                        "id": "f_root", "label": "f_root", "type": "f_root", "value": None
                    })
                    combined_edges.append({"source": "f_root", "target": "global", "type": "belongs_to_f"})
                    for root in roots_f:
                        combined_edges.append({"source": root, "target": "f_root", "type": "child_of"})
                if roots_d1:
                    combined_nodes.append({
                        "id": "d1_root", "label": "d1_root", "type": "d1_root", "value": None
                    })
                    combined_edges.append({"source": "d1_root", "target": "global", "type": "belongs_to_d1"})
                    for root in roots_d1:
                        combined_edges.append({"source": root, "target": "d1_root", "type": "child_of"})
                if roots_d2:
                    combined_nodes.append({
                        "id": "d2_root", "label": "d2_root", "type": "d2_root", "value": None
                    })
                    combined_edges.append({"source": "d2_root", "target": "global", "type": "belongs_to_d2"})
                    for root in roots_d2:
                        combined_edges.append({"source": root, "target": "d2_root", "type": "child_of"})

                raw["nodes"] = combined_nodes
                raw["edges"] = combined_edges
            else:
                raw["nodes"] = list(raw.get("nodes", []))
                raw["edges"] = list(raw.get("edges", []))
                _insert_function_aggregators(raw)

            # Topology is computed on the pure AST (before virtual nodes are injected).
            raw_ast = {"nodes": list(raw["nodes"]), "edges": list(raw["edges"])}
            belongs_to_f_map = compute_belongs_to_f(raw_ast)
            belongs_to_d1_map = compute_belongs_to_d1(raw_ast)
            belongs_to_d2_map = compute_belongs_to_d2(raw_ast)
            # 1. Build AST graph for structural features (excludes virtual nodes)
            G_ast = self._build_networkx(
                raw_ast,
                belongs_to_f_map=belongs_to_f_map,
                belongs_to_d1_map=belongs_to_d1_map,
                belongs_to_d2_map=belongs_to_d2_map,
            )
            topo = TopologicalFeatureExtractor.extract_and_annotate(G_ast)
            ast_node_ids = list(G_ast.nodes)
            ast_id_to_idx = {node_id: idx for idx, node_id in enumerate(ast_node_ids)}

            # 2. Build full directed graph (includes virtual nodes when mode=graph)
            G_directed = self._build_networkx(
                raw,
                belongs_to_f_map=belongs_to_f_map,
                belongs_to_d1_map=belongs_to_d1_map,
                belongs_to_d2_map=belongs_to_d2_map,
            )

            # 3. Enrich attributes based on mode
            G_enriched = nx.DiGraph()
            node_ids = list(G_directed.nodes)

            for node in node_ids:
                attrs = G_directed.nodes[node]
                enriched_attrs = attrs.copy()
                ast_idx = ast_id_to_idx.get(node)

                if ast_idx is not None:
                    enriched_attrs["depth"] = float(topo["depths"].get(node, 0.0))
                    enriched_attrs["height"] = float(topo["heights"].get(node, 0.0))
                    enriched_attrs["subtree_size"] = float(topo["subtree_sizes"].get(node, 1.0))
                    enriched_attrs["out_degree"] = float(topo["out_degrees"].get(node, 0.0))
                    enriched_attrs["betweenness_centrality"] = float(topo["betweenness"].get(node, 0.0))
                    enriched_attrs["lpe_1"] = float(topo["lpe"][ast_idx, 0])
                    enriched_attrs["lpe_2"] = float(topo["lpe"][ast_idx, 1])
                    enriched_attrs["lpe_3"] = float(topo["lpe"][ast_idx, 2])
                    enriched_attrs["lpe_4"] = float(topo["lpe"][ast_idx, 3])
                    enriched_attrs["rwpe_1"] = float(topo["rwpe"][ast_idx, 0])
                    enriched_attrs["rwpe_2"] = float(topo["rwpe"][ast_idx, 1])
                    enriched_attrs["rwpe_3"] = float(topo["rwpe"][ast_idx, 2])
                    enriched_attrs["rwpe_4"] = float(topo["rwpe"][ast_idx, 3])
                else:
                    enriched_attrs["depth"] = 0.0
                    enriched_attrs["height"] = 0.0
                    enriched_attrs["subtree_size"] = 1.0
                    enriched_attrs["out_degree"] = 0.0
                    enriched_attrs["betweenness_centrality"] = 0.0
                    enriched_attrs["lpe_1"] = 0.0
                    enriched_attrs["lpe_2"] = 0.0
                    enriched_attrs["lpe_3"] = 0.0
                    enriched_attrs["lpe_4"] = 0.0
                    enriched_attrs["rwpe_1"] = 0.0
                    enriched_attrs["rwpe_2"] = 0.0
                    enriched_attrs["rwpe_3"] = 0.0
                    enriched_attrs["rwpe_4"] = 0.0

                G_enriched.add_node(node, **enriched_attrs)

            child_counters = {}
            for edge in raw.get("edges", []):
                parent = edge["source"]
                child = edge["target"]
                etype = edge["type"]

                child_idx = child_counters.get(parent, 0)
                child_counters[parent] = child_idx + 1

                eb_val = float(topo["edge_betweenness"].get((parent, child), 0.0))
                self._add_ast_edges(
                    G_enriched,
                    parent,
                    child,
                    child_idx,
                    etype,
                    eb_val,
                    edge_direction,
                )

            # Build children_dict from all parent-to-child edges in raw_ast
            children_dict = {}
            for edge in raw_ast.get("edges", []):
                parent = edge["source"]
                child = edge["target"]
                children_dict.setdefault(parent, []).append(child)

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

        # Ensure all edges have exactly the same set of attribute keys
        all_edge_keys = set()
        for u, v in G_enriched.edges:
            all_edge_keys.update(G_enriched.edges[u, v].keys())
        for u, v in G_enriched.edges:
            for key in all_edge_keys:
                if key not in G_enriched.edges[u, v]:
                    if key in ("child_index", "direction", "relation_type", "edge_betweenness_centrality", "weight", "edge_type"):
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
        label_id_tensor = torch.tensor(
            [G_directed.nodes[n]["label_id"] for n in node_ids], dtype=torch.long
        )
        belongs_to_f_tensor = torch.tensor(
            [G_directed.nodes[n]["belongs_to_f"] for n in node_ids], dtype=torch.float
        )
        belongs_to_d1_tensor = torch.tensor(
            [G_directed.nodes[n]["belongs_to_d1"] for n in node_ids], dtype=torch.float
        )
        belongs_to_d2_tensor = torch.tensor(
            [G_directed.nodes[n]["belongs_to_d2"] for n in node_ids], dtype=torch.float
        )
        if heterogeneous:
            # Bypassed because true heterogeneous doesn't use a monolithic 'node' type
            pass
        else:
            data.node_type = node_type_tensor
            data.label_id = label_id_tensor
            data.belongs_to_f = belongs_to_f_tensor
            data.belongs_to_d1 = belongs_to_d1_tensor
            data.belongs_to_d2 = belongs_to_d2_tensor

        # Add global graph features
        data.tree_depth = topo["tree_depth"]
        data.tree_width = topo["tree_width"]
        data.treewidth = topo["tree_width"]
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

    def _build_networkx(
        self,
        raw: dict,
        belongs_to_f_map: dict[str, float] | None = None,
        belongs_to_d1_map: dict[str, float] | None = None,
        belongs_to_d2_map: dict[str, float] | None = None,
    ) -> nx.DiGraph:
        if belongs_to_f_map is None:
            belongs_to_f_map = compute_belongs_to_f(raw)
        if belongs_to_d1_map is None:
            belongs_to_d1_map = compute_belongs_to_d1(raw)
        if belongs_to_d2_map is None:
            belongs_to_d2_map = compute_belongs_to_d2(raw)
        G = nx.DiGraph()
        for node in raw["nodes"]:
            val_dict = node.get("value")
            if isinstance(val_dict, dict) and val_dict.get("mantissa") is not None:
                mantissa = val_dict["mantissa"]
                exponent = val_dict.get("exponent", 0)
                actual_value = float(mantissa * (10 ** exponent))
                has_val = 1.0
            else:
                if isinstance(val_dict, (int, float)):
                    actual_value = float(val_dict)
                    has_val = 1.0
                else:
                    actual_value = 0.0
                    has_val = 0.0

            G.add_node(
                node["id"],
                node_type=self.NODE_TYPES[node["type"]],
                label_id=self._encode_label(node["label"]),
                value=float(actual_value),
                has_value=has_val,
                belongs_to_f=float(belongs_to_f_map.get(node["id"], 0.0)),
                belongs_to_d1=float(belongs_to_d1_map.get(node["id"], 0.0)),
                belongs_to_d2=float(belongs_to_d2_map.get(node["id"], 0.0)),
                virtual_current_x_val=0.0,
                virtual_delta_target_val=0.0,
                virtual_d1_x_val=0.0,
                virtual_d2_x_val=0.0,
                type=node["type"],
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
    if isinstance(data, HeteroData):
        if 'virtual' not in data.node_types or not hasattr(data['virtual'], 'node_ids') or data['virtual'].node_ids is None:
            return
        if not hasattr(data['virtual'], 'x') or data['virtual'].x is None:
            return
            
        virtual_node_ids = data['virtual'].node_ids
        
        def write_hetero(node_id: str, col_idx: int, value: float) -> None:
            if node_id_indices is not None:
                if node_id in node_id_indices:
                    idx = node_id_indices[node_id]
                    data['virtual'].x[idx, col_idx] = float(value)
            elif node_id in virtual_node_ids:
                idx = virtual_node_ids.index(node_id)
                data['virtual'].x[idx, col_idx] = float(value)
                
        try:
            delta_val = yt_val - fx_val
            task_target = "f_root" if "f_root" in virtual_node_ids else "global"
            write_hetero(task_target, 0, cx_val)
            write_hetero(task_target, 1, delta_val)
            write_hetero("d1_root", 2, d1x_val)
            write_hetero("d2_root", 3, d2x_val)
        except ValueError:
            pass
        return

    if not hasattr(data, "node_ids") or data.node_ids is None or data.x is None:
        return
    if len(data.x.shape) != 2:
        return

    schema = NODE_FEATURE_SCHEMA
    expected_count = len(schema)
    if data.x.shape[1] != expected_count:
        return

    cx_idx = schema.index("virtual_current_x_val")
    dt_idx = schema.index("virtual_delta_target_val")
    d1_idx = schema.index("virtual_d1_x_val")
    d2_idx = schema.index("virtual_d2_x_val")
    has_idx = schema.index("has_value") if set_has_value else None

    def write(node_id: str, col_idx: int, value: float) -> None:
        if node_id_indices is not None:
            if node_id not in node_id_indices:
                raise ValueError
            idx = node_id_indices[node_id]
        else:
            idx = data.node_ids.index(node_id)
        data.x[idx, col_idx] = float(value)
        if has_idx is not None:
            data.x[idx, has_idx] = 1.0

    try:
        delta_val = yt_val - fx_val
        task_target = "f_root" if "f_root" in data.node_ids else "global"
        write(task_target, cx_idx, cx_val)
        write(task_target, dt_idx, delta_val)
        if "d1_root" in data.node_ids:
            write("d1_root", d1_idx, d1x_val)
        if "d2_root" in data.node_ids:
            write("d2_root", d2_idx, d2x_val)
    except ValueError:
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

    def normalize_graph(G: nx.DiGraph):
        G_norm = nx.DiGraph()
        for nid, attrs in G.nodes(data=True):
            if "node_type" in attrs:
                G_norm.add_node(nid, **attrs)
                continue
            name_val = attrs.get("Name") or attrs.get("nodeKey1")
            if not name_val:
                label = str(nid)
            else:
                label = parse_graphml_node_name(name_val)
            type_str = _determine_node_type_from_label(label)
            val = 0.0
            has_val = 0.0
            if type_str == "constant":
                val = _parse_constant_value(label)
                has_val = 1.0
            G_norm.add_node(
                nid,
                node_type=ExpressionGraphConverter.NODE_TYPES[type_str],
                label_id=encode_label(label),
                label=label,
                type=type_str,
                value=float(val),
                has_value=has_val,
                belongs_to_f=0.0,
                belongs_to_d1=0.0,
                belongs_to_d2=0.0,
                virtual_current_x_val=0.0,
                virtual_delta_target_val=0.0,
                virtual_d1_x_val=0.0,
                virtual_d2_x_val=0.0
            )
        for u, v, attrs in G.edges(data=True):
            G_norm.add_edge(u, v, **attrs)
        return G_norm

    g_f = normalize_graph(g_f)
    g_d1 = normalize_graph(g_d1)
    g_d2 = normalize_graph(g_d2)

    roots_f = [n for n, d in g_f.in_degree() if d == 0]
    roots_d1 = [n for n, d in g_d1.in_degree() if d == 0]
    roots_d2 = [n for n, d in g_d2.in_degree() if d == 0]

    G_combined = nx.DiGraph()
    G_combined.add_node(
        "global",
        node_type=ExpressionGraphConverter.NODE_TYPES["global"],
        label_id=encode_label("GLOBAL"),
        label="GLOBAL",
        type="global",
        value=0.0,
        has_value=0.0,
        belongs_to_f=0.0,
        belongs_to_d1=0.0,
        belongs_to_d2=0.0,
        virtual_current_x_val=0.0,
        virtual_delta_target_val=0.0,
        virtual_d1_x_val=0.0,
        virtual_d2_x_val=0.0
    )

    def add_component(g_comp, prefix: str):
        for nid, attrs in g_comp.nodes(data=True):
            G_combined.add_node(f"{prefix}_{nid}", **attrs)
        for u, v, attrs in g_comp.edges(data=True):
            G_combined.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **attrs)

    add_component(g_f, "f")
    add_component(g_d1, "d1")
    add_component(g_d2, "d2")

    prov_edge_by_agg = {
        "f_root": "belongs_to_f",
        "d1_root": "belongs_to_d1",
        "d2_root": "belongs_to_d2",
    }

    def add_aggregator(agg_id: str, agg_type: str, roots: list, prefix: str):
        if not roots:
            return
        G_combined.add_node(
            agg_id,
            node_type=ExpressionGraphConverter.NODE_TYPES[agg_type],
            label_id=encode_label(agg_id),
            label=agg_id,
            type=agg_type,
            value=0.0,
            has_value=0.0,
            belongs_to_f=1.0 if agg_type == "f_root" else 0.0,
            belongs_to_d1=1.0 if agg_type == "d1_root" else 0.0,
            belongs_to_d2=1.0 if agg_type == "d2_root" else 0.0,
            virtual_current_x_val=0.0,
            virtual_delta_target_val=0.0,
            virtual_d1_x_val=0.0,
            virtual_d2_x_val=0.0,
        )
        G_combined.add_edge(
            "global", agg_id, edge_type=encode_edge_type(prov_edge_by_agg[agg_type])
        )
        for r in roots:
            G_combined.add_edge(agg_id, f"{prefix}_{r}", edge_type=encode_edge_type("child_of"))

    add_aggregator("f_root", "f_root", roots_f, "f")
    add_aggregator("d1_root", "d1_root", roots_d1, "d1")
    add_aggregator("d2_root", "d2_root", roots_d2, "d2")

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
            label_id=encode_label("GLOBAL"),
            label="GLOBAL",
            type="global",
            value=0.0,
            has_value=0.0,
            belongs_to_f=0.0,
            belongs_to_d1=0.0,
            belongs_to_d2=0.0,
            virtual_current_x_val=0.0,
            virtual_delta_target_val=0.0,
            virtual_d1_x_val=0.0,
            virtual_d2_x_val=0.0,
            context_type=nodeType
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

            ntype_code = ExpressionGraphConverter.NODE_TYPES.get(type_str, 4)

            normalized_g_kappa.add_node(
                nid,
                node_type=ntype_code,
                label_id=encode_label(label),
                label=label,
                type=type_str,
                value=float(val),
                has_value=has_val,
                belongs_to_f=0.0,
                belongs_to_d1=0.0,
                belongs_to_d2=0.0,
                virtual_current_x_val=0.0,
                virtual_delta_target_val=0.0,
                virtual_d1_x_val=0.0,
                virtual_d2_x_val=0.0
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
            self.add_node(shifted_id, **attrs)

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
            weight=weight,
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

        belongs_to_f_map = compute_belongs_to_f(raw_data)
        belongs_to_d1_map = compute_belongs_to_d1(raw_data)
        belongs_to_d2_map = compute_belongs_to_d2(raw_data)

        for node in nodes:
            val_dict = node.get("value")
            if isinstance(val_dict, dict) and val_dict.get("mantissa") is not None:
                mantissa = val_dict["mantissa"]
                exponent = val_dict.get("exponent", 0)
                actual_value = float(mantissa * (10 ** exponent))
                has_val = 1.0
            else:
                if isinstance(val_dict, (int, float)):
                    actual_value = float(val_dict)
                    has_val = 1.0
                else:
                    actual_value = 0.0
                    has_val = 0.0

            nx_graph.add_node(
                node["id"],
                node_type=ExpressionGraphConverter.NODE_TYPES[node["type"]],
                label_id=encode_label(node["label"]),
                value=float(actual_value),
                has_value=has_val,
                belongs_to_f=float(belongs_to_f_map.get(node["id"], 0.0)),
                belongs_to_d1=float(belongs_to_d1_map.get(node["id"], 0.0)),
                belongs_to_d2=float(belongs_to_d2_map.get(node["id"], 0.0)),
                virtual_current_x_val=0.0,
                virtual_delta_target_val=0.0,
                virtual_d1_x_val=0.0,
                virtual_d2_x_val=0.0,
                type=node["type"],
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
    import math
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


