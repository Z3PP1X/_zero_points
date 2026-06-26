import logging
import math
from collections import deque
from typing import Any
import torch
import networkx as nx
import numpy as np
from gnn.shared.utils.graph_vocab import (
    NUM_ANCHOR_GROUPS, ANCHOR_EXCLUDED_NODE_IDS, ANCHOR_GROUP_FEATURES,
    HISTOGRAM_GROUP_BY_LABEL, HISTOGRAM_VARIABLE_BIN, HISTOGRAM_CONSTANT_BIN,
    NUM_HISTOGRAM_BINS, HISTOGRAM_FEATURES,
    NODE_FEATURE_SCHEMA,
    SUPERNODE_NODE_ID, SUPERNODE_NODE_TYPE, ROOT_COLOR_VOCAB,
    LABEL_ONEHOT_NAMES,
    anchor_group_for_node,
)

logger = logging.getLogger(__name__)


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


def _histogram_bin_for_node(label: str, orig_type: str) -> int:
    """Return the 0-based histogram bin for a single node (based on its own label/type)."""
    if label in HISTOGRAM_GROUP_BY_LABEL:
        return HISTOGRAM_GROUP_BY_LABEL[label]
    if orig_type == "variable":
        return HISTOGRAM_VARIABLE_BIN
    if orig_type == "constant":
        return HISTOGRAM_CONSTANT_BIN
    return HISTOGRAM_CONSTANT_BIN


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

    for nid, attrs in G.nodes(data=True):
        b = _histogram_bin_for_node(attrs.get("label", ""), attrs.get("type", "operator"))
        hists[nid][b] = 1.0

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return hists

    for nid in reversed(topo_order):
        for child in G.successors(nid):
            hists[nid] += hists[child]

    return hists


class TopologicalFeatureExtractor:

    @staticmethod
    def extract_and_annotate(G: nx.DiGraph) -> dict:
        roots = [n for n, d in G.in_degree() if d == 0]

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

        tree_depth = max(levels.values()) if levels else 0

        level_counts = {}
        for lvl in levels.values():
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        tree_width = max(level_counts.values()) if level_counts else 0

        results = {
            "tree_depth": tree_depth,
            "tree_width": tree_width,
            "depths": levels,
        }

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

        subtree_sizes = {}
        for node in G.nodes:
            subtree_sizes[node] = len(nx.descendants(G, node)) + 1

        anchor_pe = _compute_anchor_positional_encoding(G)

        results.update({
            "heights": heights,
            "subtree_sizes": subtree_sizes,
            "anchor_pe": anchor_pe,
        })
        return results


def inject_virtual_supernode(
    G_enriched: nx.DiGraph,
    G_directed: nx.DiGraph,
    node_ids: list,
) -> None:
    """Inject one fully-connected virtual supernode into an already-built graph.

    The supernode is wired with bidirectional edges to every pre-existing node so a
    message can travel between any two nodes in at most two hops, shrinking the
    effective graph diameter and boosting long-range message passing.

    It is injected *after* the AST topology / positional features and any kappa
    augmentation have been computed, so adding it does not perturb
    those structural node features. The supernode carries its own node_type code
    (``SUPERNODE_NODE_TYPE``); the model treats it as an ordinary message-passing node.

    Mutates ``G_enriched`` (feature graph), ``G_directed`` (source of the node-type /
    label / belongs tensors and the node/edge counts) and appends the supernode id to
    ``node_ids`` in place. Idempotent: a second call is a no-op.
    """
    if SUPERNODE_NODE_ID in G_enriched:
        return

    existing_nodes = [nid for nid in node_ids if nid != SUPERNODE_NODE_ID]

    supernode_attrs: dict[str, Any] = {
        "node_type_global": 1.0, "node_type_operator": 0.0, "node_type_function": 0.0,
        # root_color one-hot: none (0)
        "root_color_none": 1.0, "root_color_f": 0.0,
        "root_color_d1": 0.0, "root_color_d2": 0.0, "root_color_kappa": 0.0,
        # label one-hot: all zero (supernode carries no AST label)
        **{name: 0.0 for name in LABEL_ONEHOT_NAMES},
        # topology
        "subtree_size": 0.0, "subtree_depth": 0.0,
        # histograms
        **{name: 0.0 for name in HISTOGRAM_FEATURES},
        # anchor PE
        **{name: 0.0 for name in ANCHOR_GROUP_FEATURES},
        # non-schema attrs for G_directed compat
        "node_type": SUPERNODE_NODE_TYPE,
        "root_color": 0.0,
        "label_id": 2,
        "type": "supernode",
        "label": "GLOBAL",
    }

    G_enriched.add_node(SUPERNODE_NODE_ID, **supernode_attrs)
    G_directed.add_node(SUPERNODE_NODE_ID, **supernode_attrs)

    for nid in existing_nodes:
        # Both directions are always added so the virtual supernode shortcut stays
        # bidirectional (the AST itself is top-down).
        G_enriched.add_edge(
            SUPERNODE_NODE_ID,
            nid,
            child_index=0.0,
            direction=0.0,
            etype="supernode_connection",
        )
        G_enriched.add_edge(
            nid,
            SUPERNODE_NODE_ID,
            child_index=0.0,
            direction=1.0,
            etype="supernode_connection_reverse",
        )
        # Mirror the structural edges into G_directed so num_edges/num_nodes stay accurate.
        G_directed.add_edge(SUPERNODE_NODE_ID, nid)
        G_directed.add_edge(nid, SUPERNODE_NODE_ID)

    node_ids.append(SUPERNODE_NODE_ID)


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


def compute_normalized_dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> float:
    """
    Computes the normalized Dirichlet energy of node features x:
    E_norm(x) = tr(x^T * L_sym * x) / tr(x^T * x)
    where L_sym = I - D^{-1/2} * A * D^{-1/2}.
    """
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return 0.0

    x = x.float()

    if edge_weight is not None:
        if edge_weight.dim() > 1:
            if edge_weight.size(-1) == 1:
                edge_weight = edge_weight.squeeze(-1)
            else:
                edge_weight = None

    if edge_weight is None:
        from torch_geometric.utils import degree
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=x.dtype)
    else:
        from torch_scatter import scatter
        deg = scatter(edge_weight.float(), edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0.0

    x_tilde = x * deg_inv_sqrt.unsqueeze(-1)

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
