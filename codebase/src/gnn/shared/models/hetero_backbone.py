"""Heterogeneous expression-graph classifier (PyG ``to_hetero`` path).

A true heterogeneous training path for ``heterogeneous: true``: consume ``HeteroData``
(``x_dict`` / ``edge_index_dict``) with per-edge-type message passing, rather than
flattening to a homogeneous graph. Built on ``torch_geometric.nn.to_hetero`` — the library
transform that lifts a homogeneous GNN to a heterogeneous one over a fixed metadata.

STATUS: stubs only — the tests in ``test_hetero_backbone.py`` specify the behaviour; the
implementations land next.

Node types are the three from ``get_hetero_node_type`` (global / operator / root). Edge
types are relation-typed triplets ``(src, relation, dst)``; the set varies per graph, so a
model built for fixed metadata requires every batched graph to carry every edge type (empty
``edge_index`` for the absent ones) — see :func:`pad_edge_types`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, global_mean_pool, to_hetero

# The fixed heterogeneous node-type vocabulary (mirrors graph_utils.get_hetero_node_type).
HETERO_NODE_TYPES: tuple[str, ...] = ("global", "operator", "root")

EdgeType = tuple[str, str, str]
Metadata = tuple[list[str], list[EdgeType]]


def collect_edge_types(data_list: list[HeteroData]) -> list[EdgeType]:
    """Sorted union of edge-type triplets present across a list of HeteroData graphs."""
    seen: set[EdgeType] = set()
    for data in data_list:
        for edge_type in data.edge_types:
            seen.add(tuple(edge_type))
    return sorted(seen)


def build_hetero_metadata(data_list: list[HeteroData]) -> Metadata:
    """PyG ``metadata = (node_types, edge_types)`` for ``to_hetero``, from a dataset.

    ``node_types`` is the full :data:`HETERO_NODE_TYPES` vocabulary (stores always exist,
    possibly empty); ``edge_types`` is :func:`collect_edge_types`.
    """
    return list(HETERO_NODE_TYPES), collect_edge_types(data_list)


def pad_edge_types(data: HeteroData, edge_types: list[EdgeType]) -> HeteroData:
    """Ensure ``data`` carries every edge type in ``edge_types``.

    Missing edge types are added with an empty ``edge_index`` of shape ``[2, 0]`` so a list
    of graphs has a uniform store layout and collates (the root cause of the original
    ``InMemoryDataset`` IndexError). Existing edge types and all node stores are untouched.
    """
    present = set(data.edge_types)
    for edge_type in edge_types:
        edge_type = tuple(edge_type)
        if edge_type not in present:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    return data


class _ConvStack(nn.Module):
    """Homogeneous message-passing stack ``to_hetero`` lifts to per-edge-type convs."""

    def __init__(self, hidden_dim: int, num_layers: int, sage_aggr: str):
        super().__init__()
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x


class HeteroExpressionClassifier(nn.Module):
    """A ``to_hetero``-wrapped GNN + per-node-type readout → graph-level logits.

    Args:
        metadata: ``(node_types, edge_types)`` the model is specialised to.
        in_dim: node feature width (the shared ``NODE_FEATURE_SCHEMA`` length).
        hidden_dim / out_dim / num_layers / aggr: backbone hyperparameters.

    A shared input projection runs over *every* node type before message passing. The model
    also injects a self-loop edge type ``(nt, 'self_loop', nt)`` for each node type, so types
    that are only ever an edge source (``global``) are still a destination and get updated —
    ``to_hetero`` otherwise refuses to build a model where a node type is never updated. The
    real edge metadata stays untouched (self-loops are an internal model detail, not part of
    ``build_hetero_metadata``). ``aggr`` is the cross-relation aggregation ``to_hetero``
    applies when a node type is the destination of several edge types. ``forward(data)`` takes
    a (batched) ``HeteroData`` and returns logits of shape ``[num_graphs, out_dim]``.
    """

    def __init__(
        self,
        metadata: Metadata,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 2,
        num_layers: int = 2,
        aggr: str = "sum",
    ):
        super().__init__()
        self.node_types = list(metadata[0])
        self.hidden_dim = hidden_dim
        self._loop_edge_types = [(nt, "self_loop", nt) for nt in self.node_types]
        hetero_metadata = (
            self.node_types,
            list(metadata[1]) + self._loop_edge_types,
        )
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.gnn = to_hetero(
            _ConvStack(hidden_dim, num_layers, sage_aggr="mean"),
            hetero_metadata,
            aggr=aggr,
        )
        self.head = nn.Sequential(
            nn.Linear(len(self.node_types) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def _with_self_loops(self, data: HeteroData, device) -> dict:
        """edge_index_dict augmented with a self-loop edge_index per node type."""
        edge_index_dict = dict(data.edge_index_dict)
        for nt, loop_type in zip(self.node_types, self._loop_edge_types):
            n = data[nt].x.size(0)
            idx = torch.arange(n, device=device, dtype=torch.long)
            edge_index_dict[loop_type] = idx.unsqueeze(0).repeat(2, 1)
        return edge_index_dict

    def forward(self, data: HeteroData):
        num_graphs = data.num_graphs
        device = self.lin_in.weight.device

        # Project every node type up front, so even isolated types carry a representation.
        h0 = {nt: F.relu(self.lin_in(x)) for nt, x in data.x_dict.items()}
        h = self.gnn(h0, self._with_self_loops(data, device))

        pooled = []
        for nt in self.node_types:
            x = h.get(nt)
            if x is None:  # conv stack did not touch this node type
                x = h0.get(nt)
            if x is not None and x.size(0) > 0:
                pooled.append(global_mean_pool(x, data[nt].batch, size=num_graphs))
            else:
                pooled.append(torch.zeros(num_graphs, self.hidden_dim, device=device))

        return self.head(torch.cat(pooled, dim=-1))
