import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, global_mean_pool


class TestGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, global_dim=9, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)

        # Gemeinsame Repräsentation
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
        )
        
        # Head 1: Solver (Binäre Entscheidung 0 oder 1)
        self.solver_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Tolerance Skalierungsfaktor (0 bis 1)
        self.tolerance_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.activation = LeakyReLU()

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs):
        input_dim = pipeline.input_dim
        global_dim = getattr(pipeline, "global_dim", 9) # 9 extrahierte Keys im Preprocessor
        return cls(input_dim=input_dim, global_dim=global_dim, **kwargs)

    def forward(self, x, edge_index, batch_index, global_features=None):
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))

        # Graph-Level Pooling
        x = global_mean_pool(x, batch_index)

        # Globale Features nach dem Pooling verketten
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)

        # Durch das gemeinsame MLP
        shared_rep = self.shared(x)
        
        # Beide Heads berechnen
        solver_prob = self.solver_head(shared_rep)
        tolerance_scale = self.tolerance_head(shared_rep)

        return solver_prob, tolerance_scale
