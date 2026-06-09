import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, GINEConv, global_mean_pool

from gnn.shared.models.gnn_backbones import (
    EDGE_AWARE_ARCHITECTURE_NAMES,
    _gin_mlp,
    apply_edge_conv,
    coalesce_edge_attr,
)


class SupervisedGraphClassifier(nn.Module):
    """
    A generic supervised GNN graph classifier.
    Wraps any feature-extracting GNN backbone and appends a classification head.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, global_dim: int, output_dim: int = 2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        features = self.backbone(x, edge_index, batch, global_features, edge_attr=edge_attr)
        if global_features is not None:
            global_features = global_features.view(features.size(0), -1)
            features = torch.cat([features, global_features], dim=-1)
        return self.classifier(features)


class TestGraphNetwork(nn.Module):
    """
    Supervised graph classifier with split real/virtual pooling and native edge features.
    Supports GATv2 (attention over edges) and GINE (edge attrs in update MLP).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        global_dim=2,
        output_dim=2,
        heads=4,
        architecture="gatv2_stack",
        edge_dim=4,
    ):
        super().__init__()
        if architecture not in EDGE_AWARE_ARCHITECTURE_NAMES:
            raise ValueError(
                f"Unsupported architecture {architecture!r}; "
                f"expected one of {EDGE_AWARE_ARCHITECTURE_NAMES}"
            )

        self.architecture = architecture
        self.edge_dim = edge_dim
        self.activation = LeakyReLU()

        if architecture == "gine_stack":
            self.conv1 = GINEConv(_gin_mlp(input_dim, hidden_dim, self.activation), edge_dim=edge_dim)
            self.conv2 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, self.activation), edge_dim=edge_dim)
            self.conv3 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, self.activation), edge_dim=edge_dim)
        else:
            self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
            self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_dim, hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs):
        input_dim = pipeline.input_dim
        global_dim = getattr(pipeline, "global_dim", 0)
        edge_dim = getattr(pipeline, "edge_dim", 4)
        architecture = getattr(pipeline, "architecture", "gatv2_stack")
        return cls(
            input_dim=input_dim,
            global_dim=global_dim,
            edge_dim=edge_dim,
            architecture=architecture,
            **kwargs,
        )

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        node_types = x[:, 0].round().long()
        is_virtual = (node_types >= 5) & (node_types <= 10)
        is_real = ~is_virtual

        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
        x = self.activation(apply_edge_conv(self.conv1, x, edge_index, edge_attr))
        x = self.activation(apply_edge_conv(self.conv2, x, edge_index, edge_attr))
        x = self.activation(apply_edge_conv(self.conv3, x, edge_index, edge_attr))

        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0

        if is_real.any():
            x_real_pooled = global_mean_pool(x[is_real], batch[is_real], size=num_graphs)
        else:
            x_real_pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)

        if is_virtual.any():
            x_virt_pooled = global_mean_pool(x[is_virtual], batch[is_virtual], size=num_graphs)
        else:
            x_virt_pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)

        x_pooled = torch.cat([x_real_pooled, x_virt_pooled], dim=-1)

        if global_features is not None:
            global_features = global_features.view(x_pooled.size(0), -1)
            x_pooled = torch.cat([x_pooled, global_features], dim=-1)

        return self.classifier(x_pooled)
