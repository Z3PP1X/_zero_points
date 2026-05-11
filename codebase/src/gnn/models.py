import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, global_mean_pool


class TestGraphNetwork(nn.Module):
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
