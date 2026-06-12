import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from gnn.shared.utils.graph_utils import NODE_FEATURE_SCHEMA


def to_homogeneous(G: nx.DiGraph, raw: dict) -> Data:
    """Converts a NetworkX DiGraph to a homogeneous PyG Data object.

    Arguments:
        G: The NetworkX DiGraph representing the expression graph.
        raw: The raw input dictionary of the graph.

    Returns:
        The homogeneous PyG Data object.

    Raises:
        None
    """
    group_node_attrs = list(NODE_FEATURE_SCHEMA)
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
        group_edge_attrs=[
            "child_index",
            "direction",
            "relation_type",
            "edge_betweenness_centrality",
        ],
    )
