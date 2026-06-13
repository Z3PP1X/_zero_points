"""Integration: heterogeneous routing in the GraphGym loader/network.

Confirms heterogeneous:true keeps HeteroData (pad + collate, no homogenization) and that
ExpressionClassifierNetwork routes to the to_hetero model and returns (logits, y) for the
GraphGym training step. Package-style imports (the repo's dual import hazard).
"""

import torch
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.loader import DataLoader

from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
    EDGE_FEATURE_SCHEMA,
)
from gnn.supervised_learning.loader_graphgym import (
    ExpressionClassifierNetwork,
    ExpressionGraphDataset,
    prepare_hetero_data_list,
)


def _raw(node_specs, edge_specs, gid):
    return {
        "id": gid,
        "nodes": [
            {"id": n, "label": lbl, "type": t, "value": None}
            for n, lbl, t in node_specs
        ],
        "edges": [{"source": s, "target": d, "type": e} for s, d, e in edge_specs],
    }


def _hetero(gid, node_specs, edge_specs, label):
    raw = _raw(node_specs, edge_specs, gid)
    data = ExpressionGraphConverter().convert(raw, heterogeneous=True, mode="graph")
    if hasattr(data, "laplacian"):
        del data.laplacian
    data.y = torch.tensor([label], dtype=torch.long)
    return data


def _dataset():
    g1 = _hetero(
        "g1",
        [("global", "GLOBAL", "global"), ("f1", "Plus", "operator"), ("f2", "x", "variable")],
        [("global", "f1", "belongs_to_f"), ("f1", "f2", "child_of")],
        0,
    )
    g2 = _hetero(
        "g2",
        [
            ("global", "GLOBAL", "global"),
            ("f1", "Times", "operator"),
            ("f2", "Sin", "function"),
            ("f3", "x", "variable"),
        ],
        [("global", "f1", "belongs_to_f"), ("f1", "f2", "child_of"), ("f2", "f3", "child_of")],
        1,
    )
    return [g1, g2]


def _setup_hetero_cfg(edge_types):
    set_cfg(cfg)  # triggers the registered set_custom_cfg
    cfg.expression_graph.heterogeneous = True
    cfg.expression_graph.active_feature_names = []
    cfg.expression_graph.hetero_edge_types = [list(et) for et in edge_types]
    cfg.gnn.dim_inner = 16
    cfg.gnn.layers_mp = 2
    cfg.dataset.edge_dim = len(EDGE_FEATURE_SCHEMA)


def test_prepare_hetero_data_list_pads_and_collects_metadata():
    data_list = _dataset()
    padded, edge_types = prepare_hetero_data_list(data_list)

    assert len(padded) == 2
    # Every padded graph exposes the full edge-type union -> uniform layout.
    for graph in padded:
        assert set(graph.edge_types) == set(edge_types)


def test_dataset_collates_heterodata_without_homogenizing():
    padded, _ = prepare_hetero_data_list(_dataset())
    ds = ExpressionGraphDataset(padded, train_idx=[0], val_idx=[1], test_idx=[])

    assert len(ds) == 2
    item = ds[0]
    # Still heterogeneous (NOT flattened to a homogeneous Data).
    assert hasattr(item, "x_dict")
    assert "global" in item.node_types


def test_network_routes_to_hetero_and_returns_logits_and_y():
    data_list = _dataset()
    padded, edge_types = prepare_hetero_data_list(data_list)
    _setup_hetero_cfg(edge_types)

    net = ExpressionClassifierNetwork(dim_in=len(NODE_FEATURE_SCHEMA), dim_out=1)
    assert net._hetero is True

    batch = next(iter(DataLoader(padded, batch_size=2)))
    net.eval()
    with torch.no_grad():
        logits, y = net(batch)

    assert logits.shape == (2,)  # single-logit BCE path, squeezed
    assert torch.equal(y.view(-1), torch.tensor([0, 1]))
    assert torch.isfinite(logits).all()
    assert float(net._last_aux_loss) == 0.0  # no DiffPool aux loss on the hetero path
