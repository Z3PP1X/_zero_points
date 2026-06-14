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
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

from gnn.shared.models.feature_encoders import TwoWayFeatureEncoder
from gnn.shared.models.gnn_backbones import (
    functional_activation,
    make_activation,
    resolve_global_pool,
    resolve_node_feature_names,
)
from gnn.shared.utils.feature_config import NODE_CATEGORICAL_REGISTRY

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


def collect_edge_attr_dims(data_list: list[HeteroData]) -> dict[EdgeType, int]:
    """For each edge type that has ``edge_attr`` anywhere in the dataset, return its dim.

    PyG collation requires every graph that participates in a given edge-type store to
    carry the same set of attributes. We scan the whole list so that ``pad_edge_types``
    knows which padded stores need an empty ``edge_attr`` and at what width.
    """
    dims: dict[EdgeType, int] = {}
    for data in data_list:
        for edge_type in data.edge_types:
            et = tuple(edge_type)
            store = data[et]
            if hasattr(store, "edge_attr") and store.edge_attr is not None and et not in dims:
                dims[et] = store.edge_attr.size(1) if store.edge_attr.dim() > 1 else 1
    return dims


def build_hetero_metadata(data_list: list[HeteroData]) -> Metadata:
    """PyG ``metadata = (node_types, edge_types)`` for ``to_hetero``, from a dataset.

    ``node_types`` is the full :data:`HETERO_NODE_TYPES` vocabulary (stores always exist,
    possibly empty); ``edge_types`` is :func:`collect_edge_types`.
    """
    return list(HETERO_NODE_TYPES), collect_edge_types(data_list)


def pad_edge_types(
    data: HeteroData,
    edge_types: list[EdgeType],
    edge_attr_dims: dict[EdgeType, int] | None = None,
) -> HeteroData:
    """Ensure ``data`` carries every edge type in ``edge_types``.

    Missing edge types are added with an empty ``edge_index`` of shape ``[2, 0]``.
    When ``edge_attr_dims`` maps an edge type to a feature width, an empty
    ``edge_attr`` of shape ``[0, dim]`` is also added so PyG's collator finds a
    uniform store layout (fixes ``KeyError: 'edge_attr'`` during collation).
    Existing edge types and all node stores are untouched.
    """
    present = set(data.edge_types)
    attr_dims = edge_attr_dims or {}
    for edge_type in edge_types:
        edge_type = tuple(edge_type)
        if edge_type not in present:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            if edge_type in attr_dims:
                data[edge_type].edge_attr = torch.empty((0, attr_dims[edge_type]), dtype=torch.float)
    return data


class _ConvStack(nn.Module):
    """Homogeneous message-passing stack ``to_hetero`` lifts to per-edge-type convs."""

    def __init__(self, hidden_dim: int, num_layers: int, sage_aggr: str, activation: str = "relu"):
        super().__init__()
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr) for _ in range(num_layers)]
        )
        # cfg.gnn.act drives the activation, but it MUST be functional here: to_hetero traces
        # this stack and would turn an nn.Module activation into a per-node-type call_module
        # whose generated name clashes with the 'global' node type (a Python keyword). See
        # functional_activation() (PReLU is realised as its default-init LeakyReLU).
        self.act_fn = functional_activation(activation)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.act_fn(conv(x, edge_index))
        return x


class HeteroExpressionClassifier(nn.Module):
    """A ``to_hetero``-wrapped GNN + per-node-type readout → graph-level logits.

    Args:
        metadata: ``(node_types, edge_types)`` the model is specialised to.
        in_dim: node feature width (the shared ``NODE_FEATURE_SCHEMA`` length).
        hidden_dim / out_dim / num_layers / aggr: backbone hyperparameters.

    A shared ``TwoWayFeatureEncoder`` runs over *every* node type before message passing —
    the same encoder the homogeneous backbone uses, so categorical columns (``node_type`` /
    ``label_id``) are embedded by name rather than fed as raw ordinal codes, and continuous
    columns are LayerNorm'd + linearly projected. All node types share the encoder because
    they share the ``NODE_FEATURE_SCHEMA`` layout. The model also injects a self-loop edge
    type ``(nt, 'self_loop', nt)`` for each node type, so types that are only ever an edge
    source (``global``) are still a destination and get updated — ``to_hetero`` otherwise
    refuses to build a model where a node type is never updated. The real edge metadata stays
    untouched (self-loops are an internal model detail, not part of ``build_hetero_metadata``).
    ``aggr`` is the cross-relation aggregation ``to_hetero`` applies when a node type is the
    destination of several edge types. ``forward(data)`` takes a (batched) ``HeteroData`` and
    returns logits of shape ``[num_graphs, out_dim]``.
    """

    def __init__(
        self,
        metadata: Metadata,
        active_features: list[str] | None = None,
        hidden_dim: int = 128,
        out_dim: int = 2,
        num_layers: int = 2,
        aggr: str = "sum",
        activation: str = "prelu",
        dropout: float = 0.2,
        graph_pooling: str = "mean",
    ):
        super().__init__()
        self.node_types = list(metadata[0])
        self.hidden_dim = hidden_dim
        # Graph-level readout, selectable via cfg.model.graph_pooling (mean|add|max).
        self.pool_fn = resolve_global_pool(graph_pooling)
        self._loop_edge_types = [(nt, "self_loop", nt) for nt in self.node_types]
        hetero_metadata = (
            self.node_types,
            list(metadata[1]) + self._loop_edge_types,
        )
        # Embed categoricals by name + encode continuous columns, exactly like the
        # homogeneous TestGraphNetwork. Shared across node types (uniform feature layout).
        self.node_feature_names = resolve_node_feature_names(active_features)
        self.node_encoder = TwoWayFeatureEncoder(
            self.node_feature_names,
            hidden_dim,
            NODE_CATEGORICAL_REGISTRY,
            activation=make_activation(activation),
        )
        self.gnn = to_hetero(
            _ConvStack(hidden_dim, num_layers, sage_aggr="mean", activation=activation),
            hetero_metadata,
            aggr=aggr,
        )
        self.head = nn.Sequential(
            nn.Linear(len(self.node_types) * hidden_dim, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
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
        device = self.head[0].weight.device

        # Encode every node type up front (categorical embeddings + continuous linear), so
        # even isolated types carry a learned representation into message passing.
        h0 = {nt: self.node_encoder(x)[0] for nt, x in data.x_dict.items()}
        h = self.gnn(h0, self._with_self_loops(data, device))

        pooled = []
        for nt in self.node_types:
            x = h.get(nt)
            if x is None:  # conv stack did not touch this node type
                x = h0.get(nt)
            if x is not None and x.size(0) > 0:
                pooled.append(self.pool_fn(x, data[nt].batch, size=num_graphs))
            else:
                pooled.append(torch.zeros(num_graphs, self.hidden_dim, device=device))

        return self.head(torch.cat(pooled, dim=-1))
