"""
Graph neural network stacks for the Mathematica RL pipeline.

All modules share the same forward contract as the legacy ``TestGraphNetwork``:
``forward(x, edge_index, batch_index, global_features) -> (batch, hidden_dim)``.
"""
from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv, global_mean_pool

# Optuna / CLI categorical choices (exactly four architectures)
ARCHITECTURE_NAMES: List[str] = [
    "gatv2_stack",
    "gcn_stack",
    "sage_stack",
    "gin_stack",
]


def _graph_mlp_tail(hidden_dim: int, global_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim + global_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        LeakyReLU(),
        nn.Dropout(dropout),
    )


class GATv2StackNetwork(nn.Module):
    """Three GATv2 layers + graph mean pool + MLP (original stack)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None):
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


class GCNStackNetwork(nn.Module):
    """Three GCN layers + pool + MLP. ``heads`` is accepted for API parity but ignored."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        _ = heads
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None):
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


class SAGEStackNetwork(nn.Module):
    """Three GraphSAGE (mean) layers + pool + MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        _ = heads
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None):
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


def _gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        LeakyReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GINStackNetwork(nn.Module):
    """Three GIN layers + pool + MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        _ = heads
        self.conv1 = GINConv(_gin_mlp(input_dim, hidden_dim))
        self.conv2 = GINConv(_gin_mlp(hidden_dim, hidden_dim))
        self.conv3 = GINConv(_gin_mlp(hidden_dim, hidden_dim))
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None):
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


def build_gnn(
    architecture: str,
    input_dim: int = 5,
    hidden_dim: int = 128,
    global_dim: int = 9,
    heads: int = 4,
) -> nn.Module:
    """Instantiate one of the registered graph stacks."""
    builders = {
        "gatv2_stack": GATv2StackNetwork,
        "gcn_stack": GCNStackNetwork,
        "sage_stack": SAGEStackNetwork,
        "gin_stack": GINStackNetwork,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURE_NAMES}")
    return builders[architecture](input_dim, hidden_dim, global_dim, heads)


def maybe_torch_compile(module: nn.Module, enabled: bool) -> nn.Module:
    """
    Wrap ``module`` with ``torch.compile`` when supported (including CPU).

    PyG + changing graph sizes often needs ``dynamic=True``. On failure, returns
    the original module and emits a warning.
    """
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile unavailable in this PyTorch build; using eager GNN.")
        return module

    try_kw = {"mode": "default", "dynamic": True}
    try:
        return torch.compile(module, **try_kw)
    except TypeError:
        try:
            return torch.compile(module, mode="default")
        except Exception as exc:
            warnings.warn(f"torch.compile failed ({exc}); using eager GNN.")
            return module
    except Exception as exc:
        warnings.warn(f"torch.compile failed ({exc}); using eager GNN.")
        return module
