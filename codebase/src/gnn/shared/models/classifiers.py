import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, global_mean_pool


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

    def forward(self, x, edge_index, batch, global_features=None):
        features = self.backbone(x, edge_index, batch, global_features)
        # Backbone is expected to output a pooled representation of shape (batch_size, hidden_dim)
        if global_features is not None:
            global_features = global_features.view(features.size(0), -1)
            features = torch.cat([features, global_features], dim=-1)
        return self.classifier(features)


class TestGraphNetwork(nn.Module):
    """
    Identical replica of the legacy TestGraphNetwork used in supervised learning.
    Maintained for full backward compatibility and exact experiment reproducibility.
    """
    def __init__(self, input_dim, hidden_dim=128, global_dim=2, output_dim=2, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        self.activation = LeakyReLU()

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs):
        input_dim = pipeline.input_dim
        global_dim = getattr(pipeline, "global_dim", 0)
        return cls(input_dim=input_dim, global_dim=global_dim, **kwargs)

    def forward(self, x, edge_index, batch, global_features=None):
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)

        return self.classifier(x)
