from __future__ import annotations
import warnings
from typing import Any, List, Callable
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GINEConv, SAGEConv, global_mean_pool

from gnn.shared.utils.graph_utils import (
    CANONICAL_EDGE_TYPE_VOCAB,
    CANONICAL_LABEL_VOCAB,
    ENRICHED_NODE_FEATURE_SCHEMA,
)

NUM_NODE_TYPES = 11
NUM_LABELS = len(CANONICAL_LABEL_VOCAB)
NUM_EDGE_TYPES = len(CANONICAL_EDGE_TYPE_VOCAB)
NODE_TYPE_COL = ENRICHED_NODE_FEATURE_SCHEMA.index("node_type")
LABEL_ID_COL = ENRICHED_NODE_FEATURE_SCHEMA.index("label_id")
BELONGS_TO_F_COL = ENRICHED_NODE_FEATURE_SCHEMA.index("belongs_to_f")
BELONGS_TO_D1_COL = ENRICHED_NODE_FEATURE_SCHEMA.index("belongs_to_d1")
BELONGS_TO_D2_COL = ENRICHED_NODE_FEATURE_SCHEMA.index("belongs_to_d2")
EDGE_RELATION_TYPE_COL = 2


EDGE_RELATION_TYPE_COL = 2


class NodeFeatureEncoder(nn.Module):
    """Embed node_type + label_id; encode remaining continuous columns."""

    def __init__(
        self,
        padded_node_feature_count: int,
        output_dim: int,
        node_type_emb_dim: int = 8,
        label_emb_dim: int = 16,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.node_type_col = NODE_TYPE_COL
        self.label_id_col = LABEL_ID_COL
        continuous_dim = padded_node_feature_count - 2
        cont_hidden = max(output_dim - node_type_emb_dim - label_emb_dim, 8)
        self.continuous_encoder = nn.Linear(continuous_dim, cont_hidden)
        self.node_type_emb = nn.Embedding(NUM_NODE_TYPES, node_type_emb_dim)
        self.label_emb = nn.Embedding(NUM_LABELS, label_emb_dim)
        self.fusion = nn.Linear(cont_hidden + node_type_emb_dim + label_emb_dim, output_dim)
        self.activation = activation if activation is not None else LeakyReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_types = x[:, self.node_type_col].round().long().clamp(0, NUM_NODE_TYPES - 1)
        label_ids = x[:, self.label_id_col].round().long().clamp(0, NUM_LABELS - 1)
        continuous_cols = [idx for idx in range(x.size(1)) if idx not in (self.node_type_col, self.label_id_col)]
        x_cont = x[:, continuous_cols]
        fused = torch.cat(
            [
                self.continuous_encoder(x_cont),
                self.node_type_emb(node_types),
                self.label_emb(label_ids),
            ],
            dim=-1,
        )
        return self.activation(self.fusion(fused)), node_types


class EdgeFeatureEncoder(nn.Module):
    """Embed relation_type; encode remaining continuous edge columns."""

    def __init__(
        self,
        padded_edge_feature_count: int,
        output_dim: int,
        relation_emb_dim: int = 8,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        continuous_dim = padded_edge_feature_count - 1
        cont_hidden = max(output_dim - relation_emb_dim, 4)
        self.continuous_encoder = nn.Linear(continuous_dim, cont_hidden)
        self.relation_type_emb = nn.Embedding(NUM_EDGE_TYPES, relation_emb_dim)
        self.fusion = nn.Linear(cont_hidden + relation_emb_dim, output_dim)
        self.activation = activation if activation is not None else LeakyReLU()

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        relation_ids = edge_attr[:, EDGE_RELATION_TYPE_COL].round().long().clamp(0, NUM_EDGE_TYPES - 1)
        continuous_cols = [idx for idx in range(edge_attr.size(1)) if idx != EDGE_RELATION_TYPE_COL]
        edge_cont = edge_attr[:, continuous_cols]
        fused = torch.cat(
            [self.continuous_encoder(edge_cont), self.relation_type_emb(relation_ids)],
            dim=-1,
        )
        return self.activation(self.fusion(fused))


def split_global_mean_pool(x: torch.Tensor, batch_index: torch.Tensor, is_virtual: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
    is_real = ~is_virtual

    if is_real.any():
        x_real_pooled = global_mean_pool(x[is_real], batch_index[is_real], size=num_graphs)
    else:
        x_real_pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)

    if is_virtual.any():
        x_virt_pooled = global_mean_pool(x[is_virtual], batch_index[is_virtual], size=num_graphs)
    else:
        x_virt_pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)

    return torch.cat([x_real_pooled, x_virt_pooled], dim=-1)


def pool_split_embeddings(
    h_real: torch.Tensor,
    batch_real: torch.Tensor,
    h_virt: torch.Tensor | None,
    batch_virt: torch.Tensor | None,
    num_graphs: int,
    readout_dim: int,
) -> torch.Tensor:
    """Pool real and virtual embedding tensors that may differ in feature dim until the final layer."""
    ref = h_real if h_real.numel() > 0 else h_virt
    assert ref is not None
    device, dtype = ref.device, ref.dtype

    if h_real.numel() > 0:
        x_real_pooled = global_mean_pool(h_real, batch_real, size=num_graphs)
    else:
        x_real_pooled = torch.zeros(num_graphs, readout_dim, device=device, dtype=dtype)

    if h_virt is not None and h_virt.numel() > 0:
        x_virt_pooled = global_mean_pool(h_virt, batch_virt, size=num_graphs)
    else:
        x_virt_pooled = torch.zeros(num_graphs, readout_dim, device=device, dtype=dtype)

    return torch.cat([x_real_pooled, x_virt_pooled], dim=-1)


def coalesce_edge_attr(
    edge_attr: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if edge_attr is not None and edge_attr.numel() > 0:
        return edge_attr.to(device=device, dtype=dtype)
    num_edges = edge_index.size(1)
    return torch.zeros(num_edges, edge_dim, device=device, dtype=dtype)


def apply_edge_conv(
    conv: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    if isinstance(conv, GINEConv):
        return conv(x, edge_index, edge_attr)
    return conv(x, edge_index, edge_attr=edge_attr)


def filter_real_subgraph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    is_real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only real->real edges and remap endpoints to a dense 0..N_real-1 index space."""
    src, dst = edge_index[0], edge_index[1]
    edge_mask = is_real[src] & is_real[dst]
    filtered_edges = edge_index[:, edge_mask]
    filtered_attr = edge_attr[edge_mask]

    real_idx = is_real.nonzero(as_tuple=False).view(-1)
    remap = torch.full((is_real.size(0),), -1, dtype=torch.long, device=is_real.device)
    remap[real_idx] = torch.arange(real_idx.numel(), device=is_real.device, dtype=torch.long)
    remapped_edges = torch.stack([remap[filtered_edges[0]], remap[filtered_edges[1]]], dim=0)
    return remapped_edges, filtered_attr, real_idx


EDGE_AWARE_ARCHITECTURE_NAMES: List[str] = [
    "gatv2_stack",
    "gine_stack",
]

LEGACY_ARCHITECTURE_NAMES: List[str] = [
    "gcn_stack",
    "sage_stack",
    "gin_stack",
]

ARCHITECTURE_NAMES: List[str] = EDGE_AWARE_ARCHITECTURE_NAMES + LEGACY_ARCHITECTURE_NAMES


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
        nn.LayerNorm(hidden_dim) if isinstance(activation, LeakyReLU) else nn.Identity(),
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
# Edge-aware stacks
# ------------------------------------------------------------------ #

class GATv2StackNetwork(nn.Module):
    """Three GATv2 layers with edge features + graph mean pool + MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        global_dim: int = 9,
        heads: int = 4,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
        x = self.activation(apply_edge_conv(self.conv1, x, edge_index, edge_attr))
        x = self.activation(apply_edge_conv(self.conv2, x, edge_index, edge_attr))
        x = self.activation(apply_edge_conv(self.conv3, x, edge_index, edge_attr))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


class GINEStackNetwork(nn.Module):
    """Three GINE layers with edge features + graph mean pool + MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        global_dim: int = 9,
        heads: int = 4,
        edge_dim: int = 4,
        activation: nn.Module = LeakyReLU(),
    ):
        super().__init__()
        _ = heads
        self.edge_dim = edge_dim
        self.conv1 = GINEConv(_gin_mlp(input_dim, hidden_dim, activation), edge_dim=edge_dim)
        self.conv2 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, activation), edge_dim=edge_dim)
        self.conv3 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, activation), edge_dim=edge_dim)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim, activation)
        self.activation = activation
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
        x = self.activation(apply_edge_conv(self.conv1, x, edge_index, edge_attr))
        x = self.activation(apply_edge_conv(self.conv2, x, edge_index, edge_attr))
        x = self.activation(apply_edge_conv(self.conv3, x, edge_index, edge_attr))
        x = global_mean_pool(x, batch_index)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


# ------------------------------------------------------------------ #
# Legacy stacks (edge-blind)
# ------------------------------------------------------------------ #

class GCNStackNetwork(nn.Module):
    """Three GCN layers + pool + MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4, edge_dim: int = 4):
        super().__init__()
        _ = heads, edge_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        _ = edge_attr
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

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4, edge_dim: int = 4):
        super().__init__()
        _ = heads, edge_dim
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        _ = edge_attr
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

    def __init__(self, input_dim: int, hidden_dim: int = 128, global_dim: int = 9, heads: int = 4, edge_dim: int = 4):
        super().__init__()
        _ = heads, edge_dim
        self.conv1 = GINConv(_gin_mlp(input_dim, hidden_dim))
        self.conv2 = GINConv(_gin_mlp(hidden_dim, hidden_dim))
        self.conv3 = GINConv(_gin_mlp(hidden_dim, hidden_dim))
        self.shared = _graph_mlp_tail(hidden_dim, global_dim)
        self.activation = LeakyReLU()
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        _ = edge_attr
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
    edge_dim: int = 4,
) -> nn.Module:
    """Instantiate one of the registered GNN graph stacks."""
    builders = {
        "gatv2_stack": GATv2StackNetwork,
        "gine_stack": GINEStackNetwork,
        "gcn_stack": GCNStackNetwork,
        "sage_stack": SAGEStackNetwork,
        "gin_stack": GINStackNetwork,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURE_NAMES}")
    return builders[architecture](input_dim, hidden_dim, global_dim, heads, edge_dim)


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
        self.edge_dim = layout.edge_input_dim
        self.activation = get_activation_module(activation)

        self.node_encoder = NodeFeatureEncoder(
            layout.padded_node_feature_count,
            layout.node_input_dim,
            activation=self.activation,
        )
        self.edge_encoder = EdgeFeatureEncoder(
            layout.padded_edge_feature_count,
            layout.edge_input_dim,
            activation=self.activation,
        )
        self.global_encoder = nn.Linear(
            layout.padded_global_feature_count,
            layout.global_input_dim,
        )
        self.convs = nn.ModuleList(
            self._build_convs(architecture, layout, hidden_dim, heads, self.activation)
        )

        self.current_x_proj = nn.Linear(1, layout.node_input_dim)
        self.y_target_proj = nn.Linear(1, layout.node_input_dim)

        self.virtual_update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._virtual_mlp_input_dim(layer_idx), self._layer_output_dim(layer_idx)),
                    nn.BatchNorm1d(self._layer_output_dim(layer_idx)),
                    nn.ReLU(),
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.shared = nn.Sequential(
            nn.Linear(2 * hidden_dim + layout.global_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
        )

    def _layer_output_dim(self, layer_idx: int) -> int:
        is_last = layer_idx == self.num_layers - 1
        if self.architecture == "gatv2_stack":
            if is_last:
                return self.hidden_dim
            return self.hidden_dim * self.heads
        return self.hidden_dim

    def _virtual_mlp_input_dim(self, layer_idx: int) -> int:
        if layer_idx == 0:
            return self.layout.node_input_dim
        return self._layer_output_dim(layer_idx - 1)

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
            "gine_stack": lambda: self._gine_layers(layout, hidden_dim, activation),
            "gcn_stack": lambda: self._gcn_layers(layout, hidden_dim),
            "sage_stack": lambda: self._sage_layers(layout, hidden_dim),
            "gin_stack": lambda: self._gin_layers(layout, hidden_dim, activation),
        }
        arch_key = architecture if architecture in builders else "gatv2_stack"
        if architecture not in builders:
            arch_key = "gine_stack" if "gine" in architecture else "gatv2_stack"
        return builders[arch_key]()

    def _gatv2_layers(
        self,
        layout: Any,
        hidden_dim: int,
        heads: int,
    ) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        edge_dim = layout.edge_input_dim
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
                    edge_dim=edge_dim,
                )
            )
            in_dim = hidden_dim * out_heads if concat else hidden_dim
        return layers

    def _gine_layers(self, layout: Any, hidden_dim: int, activation: nn.Module) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        edge_dim = layout.edge_input_dim
        for _ in range(self.num_layers):
            layers.append(GINEConv(_gin_mlp(in_dim, hidden_dim, activation), edge_dim=edge_dim))
            in_dim = hidden_dim
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

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        x_enc, node_types = self.node_encoder(x)
        is_cx = (node_types == 5)
        is_f_root = (node_types == 6)
        is_yt = (node_types == 7)
        is_super = (node_types == 8)
        is_d1_root = (node_types == 9)
        is_d2_root = (node_types == 10)
        is_virtual = (node_types >= 5) & (node_types <= 10)
        is_real = ~is_virtual
        is_func_op = (node_types == 1) | (node_types == 4)
        belongs_to_f = (
            x[:, BELONGS_TO_F_COL] > 0.5
            if x.size(1) > BELONGS_TO_F_COL
            else torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        )
        belongs_to_d1 = (
            x[:, BELONGS_TO_D1_COL] > 0.5
            if x.size(1) > BELONGS_TO_D1_COL
            else torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        )
        belongs_to_d2 = (
            x[:, BELONGS_TO_D2_COL] > 0.5
            if x.size(1) > BELONGS_TO_D2_COL
            else torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        )

        edge_emb = self.edge_encoder(
            coalesce_edge_attr(edge_attr, edge_index, self.layout.padded_edge_feature_count, x.device, x.dtype)
        )

        if global_features is not None:
            current_x = global_features[:, 0:1]
            y_target = global_features[:, 1:2]

            cx_proj = self.current_x_proj(current_x)
            yt_proj = self.y_target_proj(y_target)

            if is_cx.any():
                x_enc[is_cx] = x_enc[is_cx] + cx_proj[batch_index[is_cx]]
            if is_yt.any():
                x_enc[is_yt] = x_enc[is_yt] + yt_proj[batch_index[is_yt]]

        num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0

        aggregator_specs = (
            (is_f_root, belongs_to_f),
            (is_d1_root, belongs_to_d1),
            (is_d2_root, belongs_to_d2),
        )
        for is_agg, belongs_mask in aggregator_specs:
            if is_func_op.any() and is_agg.any():
                is_func_op_subtree = is_func_op & belongs_mask
                if is_func_op_subtree.any():
                    fo_mean = global_mean_pool(
                        x_enc[is_func_op_subtree],
                        batch_index[is_func_op_subtree],
                        size=num_graphs,
                    )
                    x_enc[is_agg] = x_enc[is_agg] + fo_mean[batch_index[is_agg]]

        if is_super.any() and is_real.any():
            real_mean = global_mean_pool(x_enc[is_real], batch_index[is_real], size=num_graphs)
            x_enc[is_super] = x_enc[is_super] + real_mean[batch_index[is_super]]

        real_edge_index, real_edge_emb, _ = filter_real_subgraph(edge_index, edge_emb, is_real)

        h_real = x_enc[is_real]
        h_virt = x_enc[is_virtual] if is_virtual.any() else None
        batch_real = batch_index[is_real]
        batch_virt = batch_index[is_virtual] if is_virtual.any() else None

        for layer_idx, conv in enumerate(self.convs):
            h_real = self.activation(apply_edge_conv(conv, h_real, real_edge_index, real_edge_emb))

            if is_virtual.any():
                h_virt = self.virtual_update_mlps[layer_idx](h_virt)

            if is_super.any() and is_real.any() and is_virtual.any():
                super_emb = h_virt[is_super[is_virtual]]
                h_real = h_real + super_emb[batch_index[is_real]]

        h_pooled = pool_split_embeddings(
            h_real,
            batch_real,
            h_virt,
            batch_virt,
            num_graphs,
            self.hidden_dim,
        )

        if global_features is not None:
            global_features = global_features.view(h_pooled.size(0), -1)
            global_features = self.activation(self.global_encoder(global_features))
            h_pooled = torch.cat([h_pooled, global_features], dim=-1)
        else:
            dummy_global = torch.zeros(
                h_pooled.size(0), self.layout.global_input_dim, device=h_pooled.device, dtype=h_pooled.dtype
            )
            h_pooled = torch.cat([h_pooled, dummy_global], dim=-1)

        return self.shared(h_pooled)


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
