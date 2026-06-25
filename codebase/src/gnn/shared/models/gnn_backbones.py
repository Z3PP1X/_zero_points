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
        feature dropout: per-graph column-wise dropout on the raw input features
            (training only) — driven by the `dropout` arg; see _drop_feature_columns
        node_encoder: LayerNorm → Linear(input_dim, hidden_dim) → activation
        convs: num_layers × GINConv (2-layer MLP inside each conv, no edge features)
        global encoder: LayerNorm(global_dim) → Linear(global_dim, global_hidden_dim) → act
        tail: Linear(hidden_dim + global_hidden_dim, hidden_dim) → LayerNorm → act
        head (classify=True only): Linear(hidden_dim, output_dim)

    Note: `dropout` controls input feature-column dropout (whole features zeroed for a
    graph), NOT the classic post-pooling tail dropout. This forces the model to make
    predictions even when arbitrary subsets of features are missing.

    Separation of concerns: structural information flows through the message-passing
    stack into the pooled graph embedding, while per-problem scalar values (e.g.
    x0, f(x0), f'(x0), f''(x0), y_target) are encoded by a small global MLP and fused
    with the graph embedding *after* pooling.

    When global_dim=0 no global encoder is built and the tail input width is hidden_dim
    (behaviour identical to a pure structural GNN). When global_features=None the global
    contribution is zero-padded, so a global-enabled model still runs without scalars.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        global_dim: int = 0,
        global_hidden_dim: int = 8,
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
        self.global_dim = global_dim
        self.pool_fn = resolve_global_pool(graph_pooling)

        # `dropout` drives per-graph column-wise dropout on the input features
        # (see _drop_feature_columns), applied at the top of forward in training mode.
        self.feature_dropout_p = float(dropout)

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

        # Optional global feature encoder — only built when global_dim > 0.
        _g_enc_dim = global_hidden_dim if global_dim > 0 else 0
        self._g_enc_dim = _g_enc_dim
        if global_dim > 0:
            self.global_norm = nn.LayerNorm(global_dim)
            self.global_encoder = nn.Linear(global_dim, global_hidden_dim)
            self.global_activation = make_activation(activation)

        self.tail = nn.Sequential(
            nn.Linear(hidden_dim + _g_enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            make_activation(activation),
        )

        self.head = nn.Linear(hidden_dim, output_dim) if classify else None

    def _drop_feature_columns(self, x, batch):
        """Per-graph column-wise dropout on raw input features (training only).

        For every graph in the batch an independent Bernoulli mask over the feature
        columns is drawn, so a dropped feature is zeroed for *all* nodes of that graph
        (the graph loses that feature entirely). The mask is resampled on every forward
        pass, so different batches/graphs see different missing features. Inverted-dropout
        scaling (1/(1-p)) preserves the expected activation, so eval needs no rescaling.
        """
        p = self.feature_dropout_p
        if not self.training or p <= 0.0:
            return x
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        keep = x.new_empty(num_graphs, x.size(1)).bernoulli_(1.0 - p) / (1.0 - p)
        return x * keep[batch]

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        x = self._drop_feature_columns(x, batch)
        x = self.node_encoder(x)
        for i, conv in enumerate(self.convs):
            x = self.layer_activations[i](conv(x, edge_index))
        self.dirichlet_probe(x, edge_index)
        x_pooled = self.pool_fn(x, batch)

        if self._g_enc_dim > 0:
            if global_features is not None:
                gf = global_features.view(x_pooled.size(0), -1)
                g = self.global_activation(self.global_encoder(self.global_norm(gf)))
            else:
                g = x_pooled.new_zeros(x_pooled.size(0), self._g_enc_dim)
            h = torch.cat([x_pooled, g], dim=-1)
        else:
            h = x_pooled

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
