from __future__ import annotations
import warnings
from typing import List, Callable
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv, global_mean_pool

# Architecture choices
ARCHITECTURE_NAMES: List[str] = [
    "gatv2_stack",
    "gcn_stack",
    "sage_stack",
    "gin_stack",
]


def get_activation_module(activation_name: str) -> nn.Module:
    name_lower = activation_name.lower().replace("_", "")
    if name_lower == "leakyrelu":
        return nn.LeakyReLU()
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


def _graph_mlp_tail(
    hidden_dim: int,
    global_dim: int,
    activation: nn.Module = LeakyReLU(),
    dropout: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim + global_dim, hidden_dim),
        nn.LayerNorm(hidden_dim) if isinstance(activation, LeakyReLU) else nn.Identity(), # matching gnn_policy_backbone vs architectures
        activation,
        nn.Dropout(dropout),
    )


def _gin_mlp(in_dim: int, out_dim: int, activation: nn.Module = LeakyReLU()) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation,
        nn.Linear(out_dim, out_dim),
    )


# ------------------------------------------------------------------ #
# Fixed-layer stacks (Legacy & SAC compatible)
# ------------------------------------------------------------------ #

class GATv2StackNetwork(nn.Module):
    """Three GATv2 layers + graph mean pool + MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
        )
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
    """Three GCN layers + pool + MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        _ = heads
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
        )
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
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
        )
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


class GINStackNetwork(nn.Module):
    """Three GIN layers + pool + MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4):
        super().__init__()
        _ = heads
        self.conv1 = GINConv(_gin_mlp(input_dim, hidden_dim))
        self.conv2 = GINConv(_gin_mlp(hidden_dim, hidden_dim))
        self.conv3 = GINConv(_gin_mlp(hidden_dim, hidden_dim))
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
        )
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
    """Instantiate one of the registered fixed GNN graph stacks."""
    builders = {
        "gatv2_stack": GATv2StackNetwork,
        "gcn_stack": GCNStackNetwork,
        "sage_stack": SAGEStackNetwork,
        "gin_stack": GINStackNetwork,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURE_NAMES}")
    return builders[architecture](input_dim, hidden_dim, global_dim, heads)


# ------------------------------------------------------------------ #
# Flexible Backbone (PPO / Optuna Search Search space compatible)
# ------------------------------------------------------------------ #

class GraphPolicyBackbone(nn.Module):
    def __init__(
        self,
        layout: Any,  # FeatureLayout
        architecture: str,
        activation: str = "leaky_relu",
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
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
        self.activation = get_activation_module(activation)
        self.convs = nn.ModuleList(self._build_convs(architecture, layout, hidden_dim, heads, self.activation))
        
        # Build shared projection head
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + layout.global_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
        )

    def _build_convs(
        self,
        architecture: str,
        layout: Any,
        hidden_dim: int,
        heads: int,
        activation: nn.Module,
    ) -> List[nn.Module]:
        builders: dict[str, Callable[[], List[nn.Module]]] = {
            "gatv2_stack": lambda: self._gatv2_layers(layout, hidden_dim, heads),
            "gcn_stack": lambda: self._gcn_layers(layout, hidden_dim),
            "sage_stack": lambda: self._sage_layers(layout, hidden_dim),
            "gin_stack": lambda: self._gin_layers(layout, hidden_dim, activation),
        }
        arch_key = architecture if architecture in builders else "gatv2_stack"
        if architecture not in builders:
            # Fallback to gin_stack or gatv2 if needed
            arch_key = "gin_stack" if "gin" in architecture else "gatv2_stack"
        return builders[arch_key]()

    def _gatv2_layers(
        self,
        layout: Any,
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

    def _gcn_layers(self, layout: Any, hidden_dim: int) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        return layers

    def _sage_layers(self, layout: Any, hidden_dim: int) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            in_dim = hidden_dim
        return layers

    def _gin_layers(self, layout: Any, hidden_dim: int, activation: nn.Module) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(GINConv(_gin_mlp(in_dim, hidden_dim, activation)))
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
        else:
            dummy_global = torch.zeros(x.size(0), self.layout.global_input_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, dummy_global], dim=-1)
        return self.shared(x)


def build_graph_policy_backbone(
    layout: Any,
    architecture: str,
    activation: str = "leaky_relu",
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
) -> GraphPolicyBackbone:
    return GraphPolicyBackbone(
        layout=layout,
        architecture=architecture,
        activation=activation,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
    )


# ------------------------------------------------------------------ #
# Torch Compile Helper
# ------------------------------------------------------------------ #

def maybe_torch_compile(module: nn.Module, enabled: bool) -> nn.Module:
    """Wrap module with torch.compile when supported."""
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
