from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv, global_mean_pool

from feature_layout import (
    FeatureLayout,
    GNN_ARCHITECTURE_CHOICES,
    GNN_LAYER_COUNT_CHOICES,
)


def _graph_mlp_tail(
    hidden_dim: int,
    global_input_dim: int,
    dropout: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim + global_input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        LeakyReLU(),
        nn.Dropout(dropout),
    )


def _gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        LeakyReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GraphPolicyBackbone(nn.Module):
    def __init__(
        self,
        layout: FeatureLayout,
        architecture: str,
        hidden_dim: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if architecture not in GNN_ARCHITECTURE_CHOICES:
            raise ValueError(
                f"Unknown architecture {architecture!r}; "
                f"expected one of {GNN_ARCHITECTURE_CHOICES}"
            )
        if num_layers not in GNN_LAYER_COUNT_CHOICES:
            raise ValueError(
                f"num_layers must be one of {GNN_LAYER_COUNT_CHOICES}, got {num_layers}"
            )

        self.layout = layout
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.output_dim = hidden_dim

        self.node_encoder = nn.Linear(
            layout.padded_node_feature_count,
            layout.node_input_dim,
        )
        self.global_encoder = nn.Linear(
            layout.padded_global_feature_count,
            layout.global_input_dim,
        )
        self.convs = nn.ModuleList(self._build_convs(architecture, layout, hidden_dim, heads))
        self.shared = _graph_mlp_tail(hidden_dim, layout.global_input_dim, dropout)
        self.activation = LeakyReLU()

    def _build_convs(
        self,
        architecture: str,
        layout: FeatureLayout,
        hidden_dim: int,
        heads: int,
    ) -> List[nn.Module]:
        builders: dict[str, Callable[[], List[nn.Module]]] = {
            "gatv2_stack": lambda: self._gatv2_layers(layout, hidden_dim, heads),
            "gcn_stack": lambda: self._gcn_layers(layout, hidden_dim),
            "sage_stack": lambda: self._sage_layers(layout, hidden_dim),
            "gin_stack": lambda: self._gin_layers(layout, hidden_dim),
        }
        return builders[architecture]()

    def _gatv2_layers(
        self,
        layout: FeatureLayout,
        hidden_dim: int,
        heads: int,
    ) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for layer_index in range(self.num_layers):
            is_last = layer_index == self.num_layers - 1
            out_heads = 1 if is_last else heads
            concat = not is_last
            out_dim = hidden_dim if is_last else hidden_dim
            layers.append(
                GATv2Conv(
                    in_dim,
                    out_dim,
                    heads=out_heads,
                    concat=concat,
                )
            )
            in_dim = hidden_dim * out_heads if concat else hidden_dim
        return layers

    def _gcn_layers(self, layout: FeatureLayout, hidden_dim: int) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        return layers

    def _sage_layers(self, layout: FeatureLayout, hidden_dim: int) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            in_dim = hidden_dim
        return layers

    def _gin_layers(self, layout: FeatureLayout, hidden_dim: int) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(GINConv(_gin_mlp(in_dim, hidden_dim)))
            in_dim = hidden_dim
        return layers

    def forward(self, x, edge_index, batch_index, global_features=None):
        x = self.activation(self.node_encoder(x))
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            global_features = self.activation(self.global_encoder(global_features))
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


def build_graph_policy_backbone(
    layout: FeatureLayout,
    architecture: str,
    hidden_dim: int,
    num_layers: int,
    heads: int = 4,
) -> GraphPolicyBackbone:
    return GraphPolicyBackbone(
        layout=layout,
        architecture=architecture,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
    )
