import networkx as nx
import torch
from torch_geometric.data import HeteroData

from gnn.shared.utils.graph_utils import (
    ANCHOR_GROUP_FEATURES,
    get_hetero_node_type,
    get_relation_type,
)


def to_hetero(
    G: nx.DiGraph, raw: dict, topo: dict
) -> HeteroData:
    """Converts a NetworkX DiGraph to a heterogeneous PyG HeteroData object.

    Arguments:
        G: The NetworkX DiGraph representing the expression graph.
        raw: The raw input dictionary of the graph.
        topo: Pre-computed topological features dictionary of the graph.

    Returns:
        The heterogeneous PyG HeteroData object.

    Raises:
        None
    """
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

    # Helper to construct the anchor positional encoding (5 columns).
    def get_anchor_pe(nid: str, ast_idx: int | None):
        n_groups = len(ANCHOR_GROUP_FEATURES)
        if ast_idx is not None and "anchor_pe" in topo and topo["anchor_pe"] is not None:
            return [float(topo["anchor_pe"][ast_idx, j]) for j in range(n_groups)]
        return [0.0] * n_groups

    # Helper to construct topology features
    def get_topology(nid: str):
        if nid in topo.get("depths", {}):
            depth = float(topo["depths"].get(nid, 0.0))
            height = (
                float(topo["heights"].get(nid, 0.0))
                if "heights" in topo
                else 0.0
            )
            subtree_size = (
                float(topo["subtree_sizes"].get(nid, 1.0))
                if "subtree_sizes" in topo
                else 1.0
            )
            out_degree = (
                float(topo["out_degrees"].get(nid, 0.0))
                if "out_degrees" in topo
                else 0.0
            )
            betweenness = (
                float(topo["betweenness"].get(nid, 0.0))
                if "betweenness" in topo
                else 0.0
            )
        else:
            depth = 0.0
            height = 0.0
            subtree_size = 1.0
            out_degree = 0.0
            betweenness = 0.0
        return [depth, height, subtree_size, out_degree, betweenness]

    # 3. Build features for 'operator'
    x_ops_list = []
    ast_node_ids = [
        nid
        for nid in node_ids
        if nid not in ("global", "f_root", "d1_root", "d2_root")
    ]
    ast_id_to_idx = {
        node_id: idx for idx, node_id in enumerate(ast_node_ids)
    }

    # Operator / variable features: integer label_id (col 0, for a learnable
    # embedding on the model side) followed by raw continuous topology + PE columns.
    # One-hot label encoding is intentionally removed; the model embeds label_id.
    OP_VAR_FEATURE_DIM = 1 + 5 + len(ANCHOR_GROUP_FEATURES)  # label_id + topology(5) + anchor PE(5) = 11

    for nid in operators:
        attrs = G.nodes[nid]
        label_feat = torch.tensor([float(attrs["label_id"])], dtype=torch.float)

        # Topology metrics
        topo_feats = torch.tensor(get_topology(nid), dtype=torch.float)

        # Anchor positional encoding
        ast_idx = ast_id_to_idx.get(nid)
        struct_feats = torch.tensor(
            get_anchor_pe(nid, ast_idx), dtype=torch.float
        )

        # Combine
        x_ops_list.append(torch.cat([label_feat, topo_feats, struct_feats]))

    if x_ops_list:
        x_ops = torch.stack(x_ops_list, dim=0)
    else:
        x_ops = torch.empty((0, OP_VAR_FEATURE_DIM), dtype=torch.float)

    # 4. Build features for 'variable'
    x_vars_list = []
    for nid in variables:
        attrs = G.nodes[nid]
        label_feat = torch.tensor([float(attrs["label_id"])], dtype=torch.float)

        topo_feats = torch.tensor(get_topology(nid), dtype=torch.float)

        ast_idx = ast_id_to_idx.get(nid)
        struct_feats = torch.tensor(
            get_anchor_pe(nid, ast_idx), dtype=torch.float
        )

        x_vars_list.append(torch.cat([label_feat, topo_feats, struct_feats]))

    if x_vars_list:
        x_vars = torch.stack(x_vars_list, dim=0)
    else:
        x_vars = torch.empty((0, OP_VAR_FEATURE_DIM), dtype=torch.float)

    # 5. Build features for 'constant': raw value only (fourier encoding removed).
    x_consts_list = []
    for nid in constants:
        attrs = G.nodes[nid]
        val = attrs["value"]
        x_consts_list.append(torch.tensor([float(val)], dtype=torch.float))

    if x_consts_list:
        x_consts = torch.stack(x_consts_list, dim=0)
    else:
        x_consts = torch.empty((0, 1), dtype=torch.float)

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

        x_virts_list.append(
            torch.tensor(
                [
                    v_cx,
                    v_dt,
                    v_d1x,
                    v_d2x,
                    belongs_f,
                    belongs_d1,
                    belongs_d2,
                ],
                dtype=torch.float,
            )
        )

    if x_virts_list:
        x_virt = torch.stack(x_virts_list, dim=0)
    else:
        x_virt = torch.empty((0, 7), dtype=torch.float)

    # 7. Map edges to metapaths
    edge_buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for u, v, attrs in G.edges(data=True):
        src_type = get_hetero_node_type(G.nodes[u].get("type", ""))
        dst_type = get_hetero_node_type(G.nodes[v].get("type", ""))

        is_reverse = attrs.get("direction", 0.0) == 1.0
        parent = u if is_reverse else v
        parent_label = G.nodes[parent].get("label", "")

        etype = attrs.get("etype", "")
        if not etype:
            etype = "child_of"

        child_idx = attrs.get("child_index", 0.0)
        relation_type = get_relation_type(
            parent_label, etype, child_idx
        )

        src_local = type_to_local_idx[src_type][u]
        dst_local = type_to_local_idx[dst_type][v]

        triplet = (src_type, relation_type, dst_type)
        edge_buckets.setdefault(triplet, []).append(
            (src_local, dst_local)
        )

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
