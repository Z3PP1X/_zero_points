from __future__ import annotations
import warnings
from typing import List, Callable
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

from gnn.shared.utils.graph_utils import NODE_FEATURE_SCHEMA  # noqa: F401

GLOBAL_POOL_FUNCTIONS: dict[str, Callable] = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "sum": global_add_pool,
    "max": global_max_pool,
}


def resolve_global_pool(pooling_name: str) -> Callable:
    key = (pooling_name or "mean").lower()
    if key not in GLOBAL_POOL_FUNCTIONS:
        raise ValueError(
            f"Unknown graph_pooling {pooling_name!r}; expected one of {sorted(GLOBAL_POOL_FUNCTIONS)}"
        )
    return GLOBAL_POOL_FUNCTIONS[key]


def get_activation_module(activation_name: str) -> nn.Module:
    name_lower = activation_name.lower().replace("_", "")
    if name_lower == "leakyrelu":
        return nn.LeakyReLU()
    elif name_lower == "prelu":
        return nn.PReLU()
    elif name_lower == "relu":
        return nn.ReLU()
    elif name_lower == "elu":
        return nn.ELU()
    elif name_lower == "tanh":
        return nn.Tanh()
    elif name_lower == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


def make_activation(activation_name: str = "prelu") -> nn.Module:
    return get_activation_module(activation_name)


def functional_activation(activation_name: str = "relu") -> Callable[[torch.Tensor], torch.Tensor]:
    import torch.nn.functional as F

    name = activation_name.lower().replace("_", "")
    table: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "relu": F.relu,
        "gelu": F.gelu,
        "elu": F.elu,
        "tanh": torch.tanh,
        "leakyrelu": F.leaky_relu,
        "prelu": lambda x: F.leaky_relu(x, 0.25),
    }
    if name not in table:
        raise ValueError(f"Unknown activation function: {activation_name}")
    return table[name]


def _gin_mlp(in_dim: int, out_dim: int, activation_name: str = "prelu") -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        make_activation(activation_name),
        nn.Linear(out_dim, out_dim),
    )


class _MPCapture:
    """Lightweight carrier exposing message-passing output for Dirichlet-energy hooks.

    Forward hooks in the logging pipeline access .x / .edge_index / .edge_attr on the
    hooked module's output. This wraps the captured tensors without the cost of a full
    PyG Data object.
    """

    __slots__ = ("x", "edge_index", "edge_attr")

    def __init__(self, x, edge_index, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


class DirichletProbe(nn.Module):
    """Identity probe at the message-passing output for over-smoothing diagnostics.

    A forward hook on this module captures node embeddings before pooling collapses
    the node set. It adds no parameters and does not alter any numerics.
    """

    def forward(self, x, edge_index, edge_attr=None):
        return _MPCapture(x, edge_index, edge_attr)


class ExpressionGNN(nn.Module):
    """GNN backbone for supervised learning (classify=True) and RL (classify=False).

    Architecture:
        node_encoder: LayerNorm → Linear(input_dim, hidden_dim) → activation
        convs: num_layers × GINConv (2-layer MLP inside each conv, no edge features)
        tail: Linear(hidden_dim, hidden_dim) → LayerNorm → act → Dropout
        head (classify=True only): Linear(hidden_dim, output_dim)

    All information must be encoded in node features before the forward pass.
    In RL, solver-state scalars are concatenated to each node's feature vector
    by the preprocessor so they participate in message passing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        activation: str = "prelu",
        graph_pooling: str = "mean",
        classify: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_dim = hidden_dim
        self.pool_fn = resolve_global_pool(graph_pooling)

        self.node_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            make_activation(activation),
        )

        convs: List[nn.Module] = []
        for _ in range(num_layers):
            convs.append(GINConv(_gin_mlp(hidden_dim, hidden_dim, activation)))
        self.convs = nn.ModuleList(convs)
        self.dirichlet_probe = DirichletProbe()
        self.layer_activations = nn.ModuleList(
            [make_activation(activation) for _ in range(num_layers)]
        )

        self.tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(hidden_dim, output_dim) if classify else None

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.node_encoder(x)
        for i, conv in enumerate(self.convs):
            x = self.layer_activations[i](conv(x, edge_index))
        self.dirichlet_probe(x, edge_index)
        h = self.pool_fn(x, batch)
        out = self.tail(h)
        if self.head is not None:
            out = self.head(out)
        return out


def maybe_torch_compile(module: nn.Module, enabled: bool) -> nn.Module:
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
