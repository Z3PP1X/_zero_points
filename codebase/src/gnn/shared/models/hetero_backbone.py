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

import torch.nn as nn
from torch_geometric.data import HeteroData

# The fixed heterogeneous node-type vocabulary (mirrors graph_utils.get_hetero_node_type).
HETERO_NODE_TYPES: tuple[str, ...] = ("global", "operator", "root")

EdgeType = tuple[str, str, str]
Metadata = tuple[list[str], list[EdgeType]]


def collect_edge_types(data_list: list[HeteroData]) -> list[EdgeType]:
    """Sorted union of edge-type triplets present across a list of HeteroData graphs."""
    raise NotImplementedError("pending: collect_edge_types")


def build_hetero_metadata(data_list: list[HeteroData]) -> Metadata:
    """PyG ``metadata = (node_types, edge_types)`` for ``to_hetero``, from a dataset.

    ``node_types`` is the full :data:`HETERO_NODE_TYPES` vocabulary (stores always exist,
    possibly empty); ``edge_types`` is :func:`collect_edge_types`.
    """
    raise NotImplementedError("pending: build_hetero_metadata")


def pad_edge_types(data: HeteroData, edge_types: list[EdgeType]) -> HeteroData:
    """Ensure ``data`` carries every edge type in ``edge_types``.

    Missing edge types are added with an empty ``edge_index`` of shape ``[2, 0]`` so a list
    of graphs has a uniform store layout and collates (the root cause of the original
    ``InMemoryDataset`` IndexError). Existing edge types and all node stores are untouched.
    """
    raise NotImplementedError("pending: pad_edge_types")


class HeteroExpressionClassifier(nn.Module):
    """A ``to_hetero``-wrapped GNN + per-node-type readout → graph-level logits.

    Args:
        metadata: ``(node_types, edge_types)`` the model is specialised to.
        in_dim: node feature width (the shared ``NODE_FEATURE_SCHEMA`` length).
        hidden_dim / out_dim / num_layers / aggr: backbone hyperparameters.

    ``forward(data)`` takes a (batched) ``HeteroData`` and returns logits of shape
    ``[num_graphs, out_dim]``.
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
        raise NotImplementedError("pending: HeteroExpressionClassifier.__init__")

    def forward(self, data: HeteroData):
        raise NotImplementedError("pending: HeteroExpressionClassifier.forward")
