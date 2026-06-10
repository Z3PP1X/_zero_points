import json
import math
import torch
import networkx as nx
from pathlib import Path
from typing import Union, Any, Dict
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
import numpy as np


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
    "virtual_current_x",
    "f_root",
    "d1_root",
    "d2_root",
    "virtual_y_target",
    "virtual_supernode",
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
)
CANONICAL_EDGE_TYPE_VOCAB: dict[str, int] = {etype: idx for idx, etype in enumerate(CANONICAL_EDGE_TYPES)}

VIRTUAL_NODE_TYPES = frozenset(
    {"virtual_current_x", "virtual_y_target", "virtual_supernode"} | FUNCTION_AGGREGATOR_IDS
)


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
    elif raw_type in ("global", "virtual_current_x", "virtual_y_target", "virtual_supernode", "f_root", "d1_root", "d2_root"):
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
        if parent_label in ("Plus", "Times", "GLOBAL", "f_root", "d1_root", "d2_root") or parent_label.startswith("virtual"):
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


ENRICHED_NODE_FEATURE_SCHEMA = [
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

ENRICHED_EDGE_FEATURE_SCHEMA = [
    "child_index",
    "direction",
    "relation_type",
    "edge_betweenness_centrality",
]

BASIC_EDGE_FEATURE_SCHEMA = [
    "edge_type",
]

BASIC_NODE_FEATURE_SCHEMA = [
    "node_type",
    "label_id",
    "value",
    "has_value",
    "degree_centrality",
    "virtual_current_x_val",
    "virtual_delta_target_val",
    "virtual_d1_x_val",
    "virtual_d2_x_val",
    "belongs_to_f",
    "belongs_to_d1",
    "belongs_to_d2",
]


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
    def extract_and_annotate(G: nx.DiGraph, enrich: bool = True) -> dict:
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

        # Return early if enrichment is not requested (supervised learning legacy mode)
        if not enrich:
            return results

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


class ExpressionGraphConverter:
    NODE_TYPES = {
        "global": 0,
        "operator": 1,
        "constant": 2,
        "variable": 3,
        "function": 4,
        "virtual_current_x": 5,
        "f_root": 6,
        "virtual_y_target": 7,
        "virtual_supernode": 8,
        "d1_root": 9,
        "d2_root": 10,
    }

    def __init__(self):
        pass

    def convert(
        self, source: Union[str, Path, dict], heterogeneous: bool = False, enrich: bool = True, mode: str = "graph"
    ) -> Union[Data, HeteroData]:
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
        
        if mode == "graph":
            # Find variable nodes and global node
            variable_node_ids = []
            global_node_id = None
            for node in raw["nodes"]:
                if node.get("type") == "variable":
                    variable_node_ids.append(node["id"])
                elif node.get("type") == "global":
                    global_node_id = node["id"]
            
            # Add task virtual nodes (function values live on f_root/d1_root/d2_root aggregators)
            raw["nodes"].append({
                "id": "virtual_current_x",
                "label": "virtual_current_x",
                "type": "virtual_current_x",
                "value": None
            })
            raw["nodes"].append({
                "id": "virtual_y_target",
                "label": "virtual_y_target",
                "type": "virtual_y_target",
                "value": None
            })
            raw["nodes"].append({
                "id": "virtual_supernode",
                "label": "virtual_supernode",
                "type": "virtual_supernode",
                "value": None
            })

            existing_aggregators = [
                agg_id for agg_id in FUNCTION_AGGREGATOR_IDS
                if any(node["id"] == agg_id for node in raw["nodes"])
            ]

            # virtual_current_x -> all variables
            for var_id in variable_node_ids:
                raw["edges"].append({
                    "source": "virtual_current_x",
                    "target": var_id,
                    "type": "virtual"
                })
            # virtual_current_x <-> per-function aggregators (couple x to f, f', f'')
            for agg_id in existing_aggregators:
                raw["edges"].append({
                    "source": "virtual_current_x",
                    "target": agg_id,
                    "type": "virtual"
                })
                raw["edges"].append({
                    "source": agg_id,
                    "target": "virtual_current_x",
                    "type": "virtual"
                })
            # f_root <-> virtual_y_target (root-finding target applies to f only)
            if "f_root" in existing_aggregators:
                raw["edges"].append({
                    "source": "f_root",
                    "target": "virtual_y_target",
                    "type": "virtual"
                })
                raw["edges"].append({
                    "source": "virtual_y_target",
                    "target": "f_root",
                    "type": "virtual"
                })
            # Newton/Halley coupling between derivative aggregators
            for src, tgt in (("f_root", "d1_root"), ("d1_root", "d2_root"), ("f_root", "d2_root")):
                if src in existing_aggregators and tgt in existing_aggregators:
                    raw["edges"].append({"source": src, "target": tgt, "type": "virtual"})
                    raw["edges"].append({"source": tgt, "target": src, "type": "virtual"})
            # global node -> virtual_y_target
            if global_node_id is not None:
                raw["edges"].append({
                    "source": global_node_id,
                    "target": "virtual_y_target",
                    "type": "virtual"
                })
        
        # 1. Build AST graph for structural features (excludes virtual nodes)
        G_ast = self._build_networkx(
            raw_ast,
            belongs_to_f_map=belongs_to_f_map,
            belongs_to_d1_map=belongs_to_d1_map,
            belongs_to_d2_map=belongs_to_d2_map,
        )
        topo = TopologicalFeatureExtractor.extract_and_annotate(G_ast, enrich=enrich)
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
        
        if enrich:
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

            # Build bidirectional edges for rich representation
            child_counters = {}
            for edge in raw.get("edges", []):
                parent = edge["source"]
                child = edge["target"]
                etype = edge["type"]

                child_idx = child_counters.get(parent, 0)
                child_counters[parent] = child_idx + 1

                # Fetch edge betweenness centrality
                eb_val = float(topo["edge_betweenness"].get((parent, child), 0.0))

                # Forward Edge (Child -> Parent)
                G_enriched.add_edge(
                    child,
                    parent,
                    child_index=float(child_idx),
                    direction=0.0,
                    relation_type=float(self._encode_edge_type(etype)),
                    edge_betweenness_centrality=eb_val,
                    etype=etype,
                )

                # Backward Edge (Parent -> Child)
                G_enriched.add_edge(
                    parent,
                    child,
                    child_index=float(child_idx),
                    direction=1.0,
                    relation_type=float(self._encode_edge_type(etype + "_reverse")),
                    edge_betweenness_centrality=eb_val,
                    etype=etype + "_reverse",
                )

            # Add virtual supernode edges here, after all other edges are constructed
            if "virtual_supernode" in node_ids:
                supernode_etype = self._encode_edge_type("virtual")
                for node in node_ids:
                    if node != "virtual_supernode":
                        # Forward Edge (virtual_supernode -> node)
                        G_enriched.add_edge(
                            "virtual_supernode",
                            node,
                            child_index=0.0,
                            direction=0.0,
                            relation_type=float(supernode_etype),
                            edge_betweenness_centrality=0.0,
                            etype="supernode_connection",
                        )
                        # Backward Edge (node -> virtual_supernode)
                        G_enriched.add_edge(
                            node,
                            "virtual_supernode",
                            child_index=0.0,
                            direction=1.0,
                            relation_type=float(self._encode_edge_type("virtual_reverse")),
                            edge_betweenness_centrality=0.0,
                            etype="supernode_connection_reverse",
                        )
        else:
            ast_deg_cent = nx.degree_centrality(G_ast) if G_ast.number_of_nodes() > 0 else {}
            for node in node_ids:
                attrs = G_directed.nodes[node]
                enriched_attrs = attrs.copy()
                enriched_attrs["degree_centrality"] = float(ast_deg_cent.get(node, 0.0))
                G_enriched.add_node(node, **enriched_attrs)

            # Build edges for basic representation
            child_counters = {}
            for edge in raw.get("edges", []):
                parent = edge["source"]
                child = edge["target"]
                etype = edge["type"]

                child_idx = child_counters.get(parent, 0)
                child_counters[parent] = child_idx + 1

                G_enriched.add_edge(
                    parent,
                    child,
                    edge_type=self._encode_edge_type(etype),
                    child_index=float(child_idx),
                    etype=etype,
                )

            # Add supernode edges when enrich=False
            if "virtual_supernode" in node_ids:
                supernode_etype = self._encode_edge_type("virtual")
                for node in node_ids:
                    if node != "virtual_supernode":
                        G_enriched.add_edge(
                            "virtual_supernode",
                            node,
                            edge_type=supernode_etype,
                            child_index=0.0,
                            etype="supernode_connection",
                        )
                        G_enriched.add_edge(
                            node,
                            "virtual_supernode",
                            edge_type=self._encode_edge_type("virtual_reverse"),
                            child_index=0.0,
                            etype="supernode_connection_reverse",
                        )

        if heterogeneous:
            data = self._to_hetero(G_enriched, raw, topo, enrich)
            data.__class__ = ExpressionHeteroData
            data.node_ids = node_ids
        else:
            data = self._to_homogeneous(G_enriched, raw, enrich)
            data.__class__ = ExpressionGraphData
            data.node_ids = node_ids

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
        if enrich:
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
                value=signed_log_value(actual_value),
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

    def _to_homogeneous(self, G: nx.DiGraph, raw: dict, enrich: bool) -> Data:
        if enrich:
            group_node_attrs = list(ENRICHED_NODE_FEATURE_SCHEMA)
            if G.number_of_edges() == 0:
                data = from_networkx(
                    G,
                    group_node_attrs=group_node_attrs,
                )
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
                data.edge_attr = torch.empty((0, 4), dtype=torch.float)
                return data

            return from_networkx(
                G,
                group_node_attrs=group_node_attrs,
                group_edge_attrs=["child_index", "direction", "relation_type", "edge_betweenness_centrality"],
            )
        else:
            group_node_attrs = list(BASIC_NODE_FEATURE_SCHEMA)
            if G.number_of_edges() == 0:
                data = from_networkx(
                    G,
                    group_node_attrs=group_node_attrs,
                )
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
                data.edge_attr = torch.empty((0, 1), dtype=torch.float)
                return data

            return from_networkx(
                G,
                group_node_attrs=group_node_attrs,
                group_edge_attrs=["edge_type"],
            )

    def _to_hetero(self, G: nx.DiGraph, raw: dict, topo: dict, enrich: bool) -> HeteroData:
        node_ids = list(G.nodes)
        
        # 1. Separate node IDs by their heterogeneous type
        operators = []
        variables = []
        constants = []
        virtuals = []

        for nid in node_ids:
            attrs = G.nodes[nid]
            raw_type = attrs.get("type", "")
            h_type = get_hetero_node_type(raw_type)
            if h_type == "operator":
                operators.append(nid)
            elif h_type == "variable":
                variables.append(nid)
            elif h_type == "constant":
                constants.append(nid)
            else:
                virtuals.append(nid)

        # 2. Local re-indexing mappings
        op_idx = {nid: i for i, nid in enumerate(operators)}
        var_idx = {nid: i for i, nid in enumerate(variables)}
        const_idx = {nid: i for i, nid in enumerate(constants)}
        virt_idx = {nid: i for i, nid in enumerate(virtuals)}

        type_to_local_idx = {
            "operator": op_idx,
            "variable": var_idx,
            "constant": const_idx,
            "virtual": virt_idx,
        }

        # Helper to construct LPE and RWPE
        def get_lpe_rwpe(nid: str, ast_idx: int | None):
            if ast_idx is not None:
                if "lpe" in topo and topo["lpe"] is not None:
                    lpe_vals = [float(topo["lpe"][ast_idx, j]) for j in range(4)]
                else:
                    lpe_vals = [0.0] * 4
                if "rwpe" in topo and topo["rwpe"] is not None:
                    rwpe_vals = [float(topo["rwpe"][ast_idx, j]) for j in range(4)]
                else:
                    rwpe_vals = [0.0] * 4
            else:
                lpe_vals = [0.0] * 4
                rwpe_vals = [0.0] * 4
            return lpe_vals, rwpe_vals

        # Helper to construct topology features
        def get_topology(nid: str):
            if nid in topo.get("depths", {}):
                depth = float(topo["depths"].get(nid, 0.0))
                height = float(topo["heights"].get(nid, 0.0)) if "heights" in topo else 0.0
                subtree_size = float(topo["subtree_sizes"].get(nid, 1.0)) if "subtree_sizes" in topo else 1.0
                out_degree = float(topo["out_degrees"].get(nid, 0.0)) if "out_degrees" in topo else 0.0
                betweenness = float(topo["betweenness"].get(nid, 0.0)) if "betweenness" in topo else 0.0
            else:
                depth = 0.0
                height = 0.0
                subtree_size = 1.0
                out_degree = 0.0
                betweenness = 0.0
            return [depth, height, subtree_size, out_degree, betweenness]

        # 3. Build features for 'operator'
        x_ops_list = []
        ast_node_ids = [nid for nid in node_ids if nid not in ("virtual_current_x", "virtual_y_target", "virtual_supernode")]
        ast_id_to_idx = {node_id: idx for idx, node_id in enumerate(ast_node_ids)}

        for nid in operators:
            attrs = G.nodes[nid]
            label_id = attrs["label_id"]
            # One-hot label encoding
            label_oh = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float)
            label_oh[label_id] = 1.0
            
            # Topology metrics
            topo_feats = torch.tensor(get_topology(nid), dtype=torch.float)
            
            # LPE / RWPE
            ast_idx = ast_id_to_idx.get(nid)
            lpe_vals, rwpe_vals = get_lpe_rwpe(nid, ast_idx)
            struct_feats = torch.tensor(lpe_vals + rwpe_vals, dtype=torch.float)
            
            # Combine
            x_ops_list.append(torch.cat([label_oh, topo_feats, struct_feats]))
            
        if x_ops_list:
            x_ops = torch.stack(x_ops_list, dim=0)
        else:
            x_ops = torch.empty((0, 59), dtype=torch.float)

        # 4. Build features for 'variable'
        x_vars_list = []
        for nid in variables:
            attrs = G.nodes[nid]
            label_id = attrs["label_id"]
            label_oh = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float)
            label_oh[label_id] = 1.0
            
            topo_feats = torch.tensor(get_topology(nid), dtype=torch.float)
            
            ast_idx = ast_id_to_idx.get(nid)
            lpe_vals, rwpe_vals = get_lpe_rwpe(nid, ast_idx)
            struct_feats = torch.tensor(lpe_vals + rwpe_vals, dtype=torch.float)
            
            x_vars_list.append(torch.cat([label_oh, topo_feats, struct_feats]))
            
        if x_vars_list:
            x_vars = torch.stack(x_vars_list, dim=0)
        else:
            x_vars = torch.empty((0, 59), dtype=torch.float)

        # 5. Build features for 'constant'
        x_consts_list = []
        for nid in constants:
            attrs = G.nodes[nid]
            val = attrs["value"]
            
            fourier = torch.tensor(fourier_frequency_encoding(val), dtype=torch.float)
            val_tensor = torch.tensor([val], dtype=torch.float)
            x_consts_list.append(torch.cat([val_tensor, fourier]))
            
        if x_consts_list:
            x_consts = torch.stack(x_consts_list, dim=0)
        else:
            x_consts = torch.empty((0, 9), dtype=torch.float)

        # 6. Build features for 'virtual'
        x_virts_list = []
        for nid in virtuals:
            attrs = G.nodes[nid]
            v_cx = attrs["virtual_current_x_val"]
            v_dt = attrs["virtual_delta_target_val"]
            v_d1x = attrs["virtual_d1_x_val"]
            v_d2x = attrs["virtual_d2_x_val"]
            
            belongs_f = attrs["belongs_to_f"]
            belongs_d1 = attrs["belongs_to_d1"]
            belongs_d2 = attrs["belongs_to_d2"]
            
            x_virts_list.append(torch.tensor([
                v_cx, v_dt, v_d1x, v_d2x,
                belongs_f, belongs_d1, belongs_d2
            ], dtype=torch.float))
            
        if x_virts_list:
            x_virt = torch.stack(x_virts_list, dim=0)
        else:
            x_virt = torch.empty((0, 7), dtype=torch.float)

        # 7. Map edges to metapaths
        edge_buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        for u, v, attrs in G.edges(data=True):
            src_type = get_hetero_node_type(G.nodes[u].get("type", ""))
            dst_type = get_hetero_node_type(G.nodes[v].get("type", ""))
            
            is_reverse = (attrs.get("direction", 0.0) == 1.0)
            parent = u if is_reverse else v
            parent_label = G.nodes[parent].get("label", "")
            
            etype = attrs.get("etype", "")
            if not etype:
                etype = "child_of"
                
            child_idx = attrs.get("child_index", 0.0)
            relation_type = get_relation_type(parent_label, etype, child_idx)
            
            src_local = type_to_local_idx[src_type][u]
            dst_local = type_to_local_idx[dst_type][v]
            
            triplet = (src_type, relation_type, dst_type)
            edge_buckets.setdefault(triplet, []).append((src_local, dst_local))

        # 8. Construct HeteroData
        hetero = HeteroData()
        hetero["operator"].x = x_ops
        hetero["variable"].x = x_vars
        hetero["constant"].x = x_consts
        hetero["virtual"].x = x_virt
        
        hetero["operator"].node_ids = operators
        hetero["variable"].node_ids = variables
        hetero["constant"].node_ids = constants
        hetero["virtual"].node_ids = virtuals

        for triplet, pairs in edge_buckets.items():
            src_ids, dst_ids = zip(*pairs)
            hetero[triplet].edge_index = torch.tensor(
                [list(src_ids), list(dst_ids)], dtype=torch.long
            )

        return hetero


class GraphConversionPipeline:
    """Loads all JSON graph files from a directory and converts them to PyG objects."""

    def __init__(self, experiments_dir: Union[str, Path], heterogeneous: bool = False, enrich: bool = True, mode: str = "graph"):
        self.experiments_dir = Path(experiments_dir)
        self.heterogeneous = heterogeneous
        self.enrich = enrich
        self.mode = mode
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
                raw, heterogeneous=self.heterogeneous, enrich=self.enrich, mode=self.mode
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
        if self.enrich:
            return list(ENRICHED_NODE_FEATURE_SCHEMA)
        return list(BASIC_NODE_FEATURE_SCHEMA)

    def get_edge_feature_schema(self) -> list[str]:
        if self.enrich:
            return list(ENRICHED_EDGE_FEATURE_SCHEMA)
        return list(BASIC_EDGE_FEATURE_SCHEMA)


def populate_task_virtual_values(
    data,
    *,
    cx_val: float,
    fx_val: float,
    yt_val: float,
    d1x_val: float = 0.0,
    d2x_val: float = 0.0,
    mode: str = "graph",
    enrich: bool = True,
    set_has_value: bool = False,
) -> None:
    """Write current iterate / function values onto task virtual and aggregator nodes."""
    if isinstance(data, HeteroData):
        if 'virtual' not in data.node_types or not hasattr(data['virtual'], 'node_ids') or data['virtual'].node_ids is None:
            return
        if not hasattr(data['virtual'], 'x') or data['virtual'].x is None:
            return
            
        virtual_node_ids = data['virtual'].node_ids
        
        def write_hetero(node_id: str, col_idx: int, value: float) -> None:
            if node_id in virtual_node_ids:
                idx = virtual_node_ids.index(node_id)
                data['virtual'].x[idx, col_idx] = float(signed_log_value(value))
                
        try:
            delta_val = yt_val - fx_val
            if mode == "graph":
                write_hetero("virtual_current_x", 0, cx_val)
                write_hetero("virtual_y_target", 1, delta_val)
                write_hetero("f_root", 1, delta_val)
                write_hetero("d1_root", 2, d1x_val)
                write_hetero("d2_root", 3, d2x_val)
            elif mode in ("tree", "tree_derivatives"):
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

    schema = ENRICHED_NODE_FEATURE_SCHEMA if enrich else BASIC_NODE_FEATURE_SCHEMA
    expected_count = len(schema)
    if data.x.shape[1] != expected_count:
        return

    cx_idx = schema.index("virtual_current_x_val")
    dt_idx = schema.index("virtual_delta_target_val")
    d1_idx = schema.index("virtual_d1_x_val")
    d2_idx = schema.index("virtual_d2_x_val")
    has_idx = schema.index("has_value") if set_has_value else None

    def write(node_id: str, col_idx: int, value: float) -> None:
        idx = data.node_ids.index(node_id)
        data.x[idx, col_idx] = float(signed_log_value(value))
        if has_idx is not None:
            data.x[idx, has_idx] = 1.0

    try:
        delta_val = yt_val - fx_val
        if mode == "graph":
            write("virtual_current_x", cx_idx, cx_val)
            write("virtual_y_target", dt_idx, delta_val)
            if "f_root" in data.node_ids:
                write("f_root", dt_idx, delta_val)
            if "d1_root" in data.node_ids:
                write("d1_root", d1_idx, d1x_val)
            if "d2_root" in data.node_ids:
                write("d2_root", d2_idx, d2x_val)
        elif mode in ("tree", "tree_derivatives"):
            task_target = "f_root" if "f_root" in data.node_ids else "global"
            write(task_target, cx_idx, cx_val)
            write(task_target, dt_idx, delta_val)
            if "d1_root" in data.node_ids:
                write("d1_root", d1_idx, d1x_val)
            if "d2_root" in data.node_ids:
                write("d2_root", d2_idx, d2x_val)
    except ValueError:
        pass


def slice_active_features(x: torch.Tensor, active_features: list[str] | None, enrich: bool) -> torch.Tensor:
    if active_features is None:
        return x
    full_schema = ENRICHED_NODE_FEATURE_SCHEMA if enrich else BASIC_NODE_FEATURE_SCHEMA
    indices = []
    for f in active_features:
        if f in full_schema:
            indices.append(full_schema.index(f))
        else:
            raise ValueError(f"Feature '{f}' is not in the schema (enrich={enrich}). Available: {full_schema}")
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
                value=signed_log_value(val),
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
