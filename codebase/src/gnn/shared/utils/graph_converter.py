import json
import logging
from typing import Union, Any, Dict
import torch
import networkx as nx
import numpy as np
from pathlib import Path
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
from gnn.shared.utils.graph_vocab import (
    CANONICAL_LABEL_VOCAB, ROOT_COLOR_VOCAB,
    NUM_HISTOGRAM_BINS, HISTOGRAM_FEATURES, ANCHOR_GROUP_FEATURES,
    SUPERNODE_NODE_ID, SUPERNODE_NODE_TYPE,
    NODE_FEATURE_SCHEMA, EDGE_FEATURE_SCHEMA,
    encode_label, encode_edge_type, validate_edge_direction,
)
from gnn.shared.utils.feature_extraction import (
    TopologicalFeatureExtractor, _compute_subtree_histograms,
    inject_virtual_supernode,
)

logger = logging.getLogger(__name__)


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
    pass


class ExpressionHeteroData(HeteroData):
    pass


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

    @staticmethod
    def _enrich_nodes(
        node_ids: list,
        G_source: nx.DiGraph,
        ast_id_to_idx: dict,
        topo: dict,
        hist: dict,
        G_enriched: nx.DiGraph,
    ) -> None:
        """Populate G_enriched with topology- and histogram-enriched node attributes."""
        for node in node_ids:
            enriched_attrs = dict(G_source.nodes[node])
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

            enriched_attrs.setdefault("root_color", float(ROOT_COLOR_VOCAB["none"]))
            G_enriched.add_node(node, **enriched_attrs)

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

            G_enriched = nx.DiGraph()
            self._enrich_nodes(node_ids, source, ast_id_to_idx, topo, hist, G_enriched)

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
            self._enrich_nodes(node_ids, G_directed, ast_id_to_idx, topo, hist, G_enriched)

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

        # Ensure all edges have exactly the same set of attribute keys so
        # from_networkx(group_edge_attrs=EDGE_FEATURE_SCHEMA) doesn't raise KeyError on
        # edges that are missing a schema column.
        all_edge_keys = set(EDGE_FEATURE_SCHEMA)
        for u, v in G_enriched.edges:
            all_edge_keys.update(G_enriched.edges[u, v].keys())
        for u, v in G_enriched.edges:
            for key in all_edge_keys:
                if key not in G_enriched.edges[u, v]:
                    if key in ("child_index", "direction", "relation_type", "edge_type"):
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
