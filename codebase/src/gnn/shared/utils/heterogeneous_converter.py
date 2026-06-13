import networkx as nx
import torch
from torch_geometric.data import HeteroData

from gnn.shared.utils.graph_utils import (
    NODE_FEATURE_SCHEMA,
    get_hetero_node_type,
    get_relation_type,
)


def to_hetero(
    G: nx.DiGraph, raw: dict, topo: dict
) -> HeteroData:
    """Converts a NetworkX DiGraph to a heterogeneous PyG HeteroData object.

    Node types are: global, operator, root.
    All share the same feature vector layout (NODE_FEATURE_SCHEMA), populated
    from node attributes written during conversion.

    Arguments:
        G: The NetworkX DiGraph representing the expression graph.
        raw: The raw input dictionary of the graph.
        topo: Pre-computed topological features dictionary of the graph.

    Returns:
        The heterogeneous PyG HeteroData object.
    """
    node_ids = list(G.nodes)

    # 1. Partition nodes by heterogeneous type
    globals_ = []
    operators = []
    roots = []

    for nid in node_ids:
        attrs = G.nodes[nid]
        raw_type = attrs.get("type", "operator")
        h_type = get_hetero_node_type(raw_type)
        if h_type == "global":
            globals_.append(nid)
        elif h_type == "root":
            roots.append(nid)
        else:
            operators.append(nid)

    # 2. Local re-indexing mappings
    global_idx = {nid: i for i, nid in enumerate(globals_)}
    op_idx = {nid: i for i, nid in enumerate(operators)}
    root_idx = {nid: i for i, nid in enumerate(roots)}

    type_to_local_idx = {
        "global": global_idx,
        "operator": op_idx,
        "root": root_idx,
    }

    n_features = len(NODE_FEATURE_SCHEMA)

    def _build_feature_vec(nid: str) -> list[float]:
        attrs = G.nodes[nid]
        return [float(attrs.get(feat, 0.0)) for feat in NODE_FEATURE_SCHEMA]

    # 3. Build feature tensors per type
    def _build_features(nid_list: list) -> torch.Tensor:
        if not nid_list:
            return torch.empty((0, n_features), dtype=torch.float)
        rows = [_build_feature_vec(nid) for nid in nid_list]
        return torch.tensor(rows, dtype=torch.float)

    x_globals = _build_features(globals_)
    x_ops = _build_features(operators)
    x_roots = _build_features(roots)

    def _is_augmented_edge(etype: str) -> bool:
        return etype in ("NextUse", "NextUseBackward") or etype.startswith(
            ("OuterToInner_Arg", "InnerToOuter_Arg")
        )

    def _augmented_edge_feature(etype: str) -> list[float]:
        direction = 1.0 if ("Backward" in etype or etype.startswith("InnerToOuter")) else 0.0
        arg_idx = 0.0
        if "Arg" in etype:
            try:
                arg_idx = float(etype.split("Arg")[-1])
            except ValueError:
                pass
        return [direction, arg_idx]

    # 4. Map edges to metapaths
    edge_buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    edge_feature_buckets: dict[tuple[str, str, str], list[list[float]]] = {}

    for u, v, attrs in G.edges(data=True):
        src_h = get_hetero_node_type(G.nodes[u].get("type", "operator"))
        dst_h = get_hetero_node_type(G.nodes[v].get("type", "operator"))

        src_local_map = type_to_local_idx.get(src_h)
        dst_local_map = type_to_local_idx.get(dst_h)
        if src_local_map is None or dst_local_map is None:
            continue
        if u not in src_local_map or v not in dst_local_map:
            continue

        is_reverse = attrs.get("direction", 0.0) == 1.0
        parent = u if is_reverse else v
        parent_label = G.nodes[parent].get("label", "")

        etype = attrs.get("etype", "") or "child_of"
        child_idx = attrs.get("child_index", 0.0)
        relation_type = get_relation_type(parent_label, etype, child_idx)

        src_local = src_local_map[u]
        dst_local = dst_local_map[v]

        triplet = (src_h, relation_type, dst_h)
        edge_buckets.setdefault(triplet, []).append((src_local, dst_local))

        if _is_augmented_edge(etype):
            edge_feature_buckets.setdefault(triplet, []).append(
                _augmented_edge_feature(etype)
            )

    # 5. Construct HeteroData
    hetero = HeteroData()
    hetero["global"].x = x_globals
    hetero["operator"].x = x_ops
    hetero["root"].x = x_roots

    hetero["global"].node_ids = globals_
    hetero["operator"].node_ids = operators
    hetero["root"].node_ids = roots

    for triplet, pairs in edge_buckets.items():
        src_ids, dst_ids = zip(*pairs)
        hetero[triplet].edge_index = torch.tensor(
            [list(src_ids), list(dst_ids)], dtype=torch.long
        )
        if triplet in edge_feature_buckets:
            hetero[triplet].edge_attr = torch.tensor(
                edge_feature_buckets[triplet], dtype=torch.float
            )

    return hetero
