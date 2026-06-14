"""Heterogeneous expression-graph classifier (PyG ``to_hetero`` path).

A true heterogeneous training path for ``heterogeneous: true``: consume ``HeteroData``
(``x_dict`` / ``edge_index_dict``) with per-edge-type message passing, rather than
flattening to a homogeneous graph. Built on ``torch_geometric.nn.to_hetero`` — the library
transform that lifts a homogeneous GNN to a heterogeneous one over a fixed metadata.

Node types are the three from ``get_hetero_node_type`` (global / operator / root). Edge
types are relation-typed triplets ``(src, relation, dst)``; the set varies per graph, so a
model built for fixed metadata requires every batched graph to carry every edge type (empty
``edge_index`` for the absent ones) — see :func:`pad_edge_types`.

Pooling variants (``variant`` arg):
  ``legacy`` — flat graph-level readout (mean/add/max) independently per node type.
  ``pooling`` — one DiffPool step per node type; intra-type edges form the adjacency.

For the ``pooling`` variant, :data:`HETERO_DIFFPOOL_CLUSTERS` clusters are used per
node type regardless of how many nodes that type has in a given graph.  Small graphs
(e.g. ``global`` with 1 node) degenerate to a learned soft-assignment but remain
numerically valid.  The per-type link+entropy losses are summed into ``_last_aux_loss``
and picked up by the ``_shared_step_with_aux`` monkeypatch in ``loader_graphgym``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINConv, to_hetero
from torch_geometric.utils import to_dense_adj, to_dense_batch

from gnn.shared.models.feature_encoders import TwoWayFeatureEncoder
from gnn.shared.models.gnn_backbones import (
    DiffPoolBlock,
    _gin_mlp,
    functional_activation,
    make_activation,
    resolve_global_pool,
    resolve_node_feature_names,
)
from gnn.shared.utils.feature_config import NODE_CATEGORICAL_REGISTRY

# The fixed heterogeneous node-type vocabulary (mirrors graph_utils.get_hetero_node_type).
HETERO_NODE_TYPES: tuple[str, ...] = ("global", "operator", "root")

# Single DiffPool step: collapse each node type to this many clusters.
# Expression trees are small, so a compact cluster count is appropriate.
HETERO_DIFFPOOL_CLUSTERS: int = 4

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

    def __init__(self, hidden_dim: int, num_layers: int, activation: str = "relu"):
        super().__init__()
        self.convs = nn.ModuleList(
            [GINConv(_gin_mlp(hidden_dim, hidden_dim, activation)) for _ in range(num_layers)]
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
        hidden_dim / out_dim / num_layers / aggr: backbone hyperparameters.
        variant: ``"legacy"`` for flat graph-level readout, ``"pooling"`` for DiffPool.
        pool_type: ``"diffpool"`` (the only hierarchical pooling supported for hetero).

    A shared ``TwoWayFeatureEncoder`` runs over *every* node type before message passing.
    The model injects a self-loop edge type ``(nt, 'self_loop', nt)`` for each node type so
    ``to_hetero`` can update all node types. The real edge metadata stays untouched.

    When ``variant == "pooling"`` and ``pool_type == "diffpool"``, one
    :class:`~gnn.shared.models.gnn_backbones.DiffPoolBlock` is built per node type.
    Each block uses only the intra-type edges (``src_type == dst_type``) as its adjacency;
    missing intra-type edges fall back to self-loops. The sum of per-type
    link+entropy losses is accumulated in ``_last_aux_loss`` (scalar tensor) and
    picked up by the ``_shared_step_with_aux`` monkeypatch.
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
        variant: str = "legacy",
        pool_type: str = "diffpool",
    ):
        super().__init__()
        self.node_types = list(metadata[0])
        self.hidden_dim = hidden_dim
        self.variant = variant
        self.pool_type = pool_type
        self._last_aux_loss = torch.zeros(())
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
            _ConvStack(hidden_dim, num_layers, activation=activation),
            hetero_metadata,
            aggr=aggr,
        )
        if variant == "pooling" and pool_type == "diffpool":
            self.hetero_diffpool_blocks = nn.ModuleList([
                DiffPoolBlock(hidden_dim, hidden_dim, HETERO_DIFFPOOL_CLUSTERS, activation)
                for _ in self.node_types
            ])
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

    def _intra_type_edges(self, data: HeteroData, node_type: str, device) -> torch.Tensor:
        """Return edge_index of all intra-type edges (src == dst == node_type).

        Falls back to identity self-loops when no intra-type edges exist, so
        DenseSAGEConv inside DiffPoolBlock always has a valid (non-empty) adjacency.
        """
        parts = []
        for et in data.edge_types:
            src_t, _, dst_t = et
            if src_t == node_type and dst_t == node_type:
                ei = data[et].edge_index
                if ei.numel() > 0 and ei.size(1) > 0:
                    parts.append(ei)
        if parts:
            return torch.cat(parts, dim=1)
        n = data[node_type].x.size(0)
        idx = torch.arange(n, device=device, dtype=torch.long)
        return idx.unsqueeze(0).repeat(2, 1)

    def _flat_readout(self, h, h0, data, num_graphs, device):
        """Flat graph-level readout (pool_fn) per node type."""
        pooled = []
        for nt in self.node_types:
            x = h.get(nt)
            if x is None:
                x = h0.get(nt)
            if x is not None and x.size(0) > 0:
                pooled.append(self.pool_fn(x, data[nt].batch, size=num_graphs))
            else:
                pooled.append(torch.zeros(num_graphs, self.hidden_dim, device=device))
        return pooled

    def _diffpool_readout(self, h, h0, data, num_graphs, device):
        """Per-node-type DiffPool readout.

        Returns (list_of_pooled_tensors, aux_loss) where each tensor is
        ``[num_graphs, hidden_dim]`` and aux_loss is the summed link+entropy loss.
        """
        aux = torch.zeros((), device=device)
        pooled = []
        for nt_idx, nt in enumerate(self.node_types):
            x = h.get(nt)
            if x is None:
                x = h0.get(nt)
            if x is None or x.size(0) == 0:
                pooled.append(torch.zeros(num_graphs, self.hidden_dim, device=device))
                continue
            batch_nt = data[nt].batch
            intra_ei = self._intra_type_edges(data, nt, device)
            max_nodes = max(1, int(batch_nt.bincount().max().item()))
            x_dense, mask = to_dense_batch(x, batch_nt, max_num_nodes=max_nodes)
            adj = to_dense_adj(intra_ei, batch_nt, max_num_nodes=max_nodes)
            x_dense, _, loss = self.hetero_diffpool_blocks[nt_idx](x_dense, adj, mask)
            aux = aux + loss
            pooled.append(x_dense.mean(dim=1))
        return pooled, aux

    def forward(self, data: HeteroData):
        num_graphs = data.num_graphs
        device = self.head[0].weight.device

        # Encode every node type up front (categorical embeddings + continuous linear), so
        # even isolated types carry a learned representation into message passing.
        h0 = {nt: self.node_encoder(x)[0] for nt, x in data.x_dict.items()}
        h = self.gnn(h0, self._with_self_loops(data, device))

        if self.variant == "pooling" and self.pool_type == "diffpool":
            pooled, aux = self._diffpool_readout(h, h0, data, num_graphs, device)
            self._last_aux_loss = aux
        else:
            pooled = self._flat_readout(h, h0, data, num_graphs, device)
            self._last_aux_loss = torch.zeros((), device=device)

        return self.head(torch.cat(pooled, dim=-1))
