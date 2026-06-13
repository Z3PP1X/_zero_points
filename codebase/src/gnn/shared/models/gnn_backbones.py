from __future__ import annotations
import warnings
from typing import Any, List, Callable
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    SAGEConv,
    TopKPooling,
    DenseSAGEConv,
    dense_diff_pool,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.utils import to_dense_batch, to_dense_adj

from gnn.shared.utils.graph_utils import (
    EDGE_FEATURE_SCHEMA,
    NODE_FEATURE_SCHEMA,
)
from gnn.shared.utils.feature_config import (
    EDGE_CATEGORICAL_REGISTRY,
    NODE_CATEGORICAL_REGISTRY,
)
from gnn.shared.models.feature_encoders import TwoWayFeatureEncoder

# Column index of node_type in the FULL node schema; imported by tests.
NODE_TYPE_COL = NODE_FEATURE_SCHEMA.index("node_type")


def resolve_node_feature_names(active_feature_names) -> list[str]:
    """Ordered node-feature names present in x; falls back to the full schema."""
    if active_feature_names:
        return list(active_feature_names)
    return list(NODE_FEATURE_SCHEMA)


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
    # Edge-blind convolutions ignore edge attributes entirely; calling them with an
    # ``edge_attr`` kwarg would raise (GCNConv expects ``edge_weight``, etc.).
    if isinstance(conv, (GCNConv, SAGEConv, GINConv)):
        return conv(x, edge_index)
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

# Structural variants, orthogonal to the layer-type ``architecture`` string above.
#   legacy / standard -> the original conv stack + real/virtual split pooling.
#   pooling           -> conv stack with hierarchical pooling between blocks.
#   pooling_skip      -> pooling + JK-style skip aggregation of per-block readouts.
VARIANT_NAMES: List[str] = ["legacy", "standard", "pooling", "pooling_skip"]
LEGACY_VARIANTS = frozenset({"legacy", "standard"})
POOL_TYPE_NAMES: List[str] = ["topk", "diffpool"]

# Fixed cluster sizes for the DiffPool path (size-robust, memory-bounded). DiffPool
# requires a fixed assignment width independent of the input node count.
DIFFPOOL_CLUSTERS: tuple[int, ...] = (16, 4)


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
    """Return a FRESH activation module.

    PReLU carries a learnable slope, so every usage site must own its own instance
    -- never share a single module across layers/encoders.
    """
    return get_activation_module(activation_name)


def _graph_mlp_tail(
    hidden_dim: int,
    global_dim: int,
    activation_name: str = "prelu",
    dropout: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim + global_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        make_activation(activation_name),
        nn.Dropout(dropout),
    )


def _gin_mlp(in_dim: int, out_dim: int, activation_name: str = "prelu") -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        make_activation(activation_name),
        nn.Linear(out_dim, out_dim),
    )


def build_uniform_conv(
    architecture: str,
    in_dim: int,
    hidden_dim: int,
    heads: int,
    edge_dim: int,
    activation_name: str,
) -> nn.Module:
    """One message-passing layer that always outputs ``hidden_dim``.

    Used by the uniform-pool variants. GATv2 is forced to ``concat=False`` so the
    width stays ``hidden_dim`` across pooling boundaries (multi-head attention is
    kept, head-concat width is not).
    """
    if architecture == "gine_stack":
        return GINEConv(_gin_mlp(in_dim, hidden_dim, activation_name), edge_dim=edge_dim)
    if architecture == "gcn_stack":
        return GCNConv(in_dim, hidden_dim)
    if architecture == "sage_stack":
        return SAGEConv(in_dim, hidden_dim, aggr="mean")
    if architecture == "gin_stack":
        return GINConv(_gin_mlp(in_dim, hidden_dim, activation_name))
    # default / gatv2_stack
    return GATv2Conv(in_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)


# ------------------------------------------------------------------ #
# Hierarchical-pooling building blocks
# ------------------------------------------------------------------ #

class DiffPoolBlock(nn.Module):
    """One DiffPool step: dense embed-GNN + assign-GNN + ``dense_diff_pool``.

    Always uses ``DenseSAGEConv`` (the canonical PyG DiffPool reference); it does
    not consume edge features and is independent of the selected layer type.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_clusters: int, activation_name: str = "prelu"):
        super().__init__()
        self.embed = DenseSAGEConv(in_dim, hidden_dim)
        self.assign = DenseSAGEConv(in_dim, num_clusters)
        self.act = make_activation(activation_name)
        self.out_dim = hidden_dim

    def forward(self, x, adj, mask=None):
        s = self.assign(x, adj, mask)
        z = self.act(self.embed(x, adj, mask))
        x_new, adj_new, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        return x_new, adj_new, link_loss + ent_loss


class UniformPoolMixin:
    """Forward logic for the ``pooling`` / ``pooling_skip`` variants.

    The variants pool the WHOLE graph uniformly -- no real/virtual node split. The
    host class calls :meth:`_init_uniform_pool` in ``__init__`` (after the legacy
    modules are built) and routes its ``forward`` to :meth:`_uniform_pool_forward`
    when ``self.variant`` is not legacy. The pooled readout is projected to
    ``tail_in_dim`` so the host's existing MLP tail is reused unchanged.
    """

    def _init_uniform_pool(
        self,
        *,
        first_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        edge_dim: int,
        architecture: str,
        activation_name: str,
        variant: str,
        pool_type: str,
        tail_in_dim: int,
        pool_ratio: float = 0.5,
        diffpool_clusters: tuple[int, ...] = DIFFPOOL_CLUSTERS,
    ) -> None:
        self.variant = variant
        self.pool_type = pool_type
        self._uniform_edge_dim = edge_dim
        self._last_aux_loss = torch.zeros(())

        if pool_type == "topk":
            convs: List[nn.Module] = []
            in_dim = first_in_dim
            for _ in range(num_layers):
                convs.append(
                    build_uniform_conv(architecture, in_dim, hidden_dim, heads, edge_dim, activation_name)
                )
                in_dim = hidden_dim
            self.uniform_convs = nn.ModuleList(convs)
            self.uniform_acts = nn.ModuleList([make_activation(activation_name) for _ in range(num_layers)])
            self.topk_pools = nn.ModuleList(
                [TopKPooling(hidden_dim, ratio=pool_ratio) for _ in range(num_layers)]
            )
            block_readout = 2 * hidden_dim  # mean || max
            jk_in = block_readout * num_layers if variant == "pooling_skip" else block_readout
            self.jk_proj = nn.Linear(jk_in, tail_in_dim)
        elif pool_type == "diffpool":
            blocks: List[nn.Module] = []
            in_dim = first_in_dim
            for num_clusters in diffpool_clusters:
                blocks.append(DiffPoolBlock(in_dim, hidden_dim, num_clusters, activation_name))
                in_dim = hidden_dim
            self.diffpool_blocks = nn.ModuleList(blocks)
            jk_in = hidden_dim * len(diffpool_clusters) if variant == "pooling_skip" else hidden_dim
            self.jk_proj = nn.Linear(jk_in, tail_in_dim)
        else:
            raise ValueError(f"Unknown pool_type {pool_type!r}; expected one of {POOL_TYPE_NAMES}")

    def _uniform_pool_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.pool_type == "topk":
            readout = self._topk_pool_forward(x, edge_index, edge_attr, batch_index, num_graphs)
        else:
            readout = self._diffpool_forward(x, edge_index, batch_index, num_graphs)
        return self.jk_proj(readout)

    def _topk_pool_forward(self, x, edge_index, edge_attr, batch, num_graphs):
        readouts: List[torch.Tensor] = []
        for layer_idx in range(len(self.uniform_convs)):
            x = apply_edge_conv(self.uniform_convs[layer_idx], x, edge_index, edge_attr)
            x = self.uniform_acts[layer_idx](x)
            x, edge_index, edge_attr, batch, _, _ = self.topk_pools[layer_idx](
                x, edge_index, edge_attr=edge_attr, batch=batch
            )
            block_readout = torch.cat(
                [
                    global_mean_pool(x, batch, size=num_graphs),
                    global_max_pool(x, batch, size=num_graphs),
                ],
                dim=-1,
            )
            readouts.append(block_readout)
        self._last_aux_loss = x.new_zeros(())
        if self.variant == "pooling_skip":
            return torch.cat(readouts, dim=-1)
        return readouts[-1]

    def _diffpool_forward(self, x, edge_index, batch, num_graphs):
        max_nodes = int(batch.bincount().max().item()) if batch.numel() > 0 else 1
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=max_nodes)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes)

        aux = x.new_zeros(())
        readouts: List[torch.Tensor] = []
        cur_mask = mask
        for block in self.diffpool_blocks:
            x_dense, adj, loss = block(x_dense, adj, cur_mask)
            aux = aux + loss
            cur_mask = None  # after the first pool every cluster is populated
            readouts.append(x_dense.mean(dim=1))
        self._last_aux_loss = aux
        if self.variant == "pooling_skip":
            return torch.cat(readouts, dim=-1)
        return readouts[-1]


# ------------------------------------------------------------------ #
# Edge-aware stacks
# ------------------------------------------------------------------ #

class GATv2StackNetwork(UniformPoolMixin, nn.Module):
    """Three GATv2 layers with edge features + graph pooling + MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        global_dim: int = 8,
        heads: int = 4,
        edge_dim: int = 4,
        activation: str = "prelu",
        variant: str = "legacy",
        pool_type: str = "topk",
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.architecture = "gatv2_stack"
        self.activation_name = activation
        self.variant = variant
        self.pool_type = pool_type
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim, activation)
        self.layer_activations = nn.ModuleList([make_activation(activation) for _ in range(3)])
        self.output_dim = hidden_dim
        self._last_aux_loss = torch.zeros(())
        if variant not in LEGACY_VARIANTS:
            self._init_uniform_pool(
                first_in_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=3,
                heads=heads,
                edge_dim=edge_dim,
                architecture=self.architecture,
                activation_name=activation,
                variant=variant,
                pool_type=pool_type,
                tail_in_dim=hidden_dim,
            )

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
        if self.variant in LEGACY_VARIANTS:
            x = self.layer_activations[0](apply_edge_conv(self.conv1, x, edge_index, edge_attr))
            x = self.layer_activations[1](apply_edge_conv(self.conv2, x, edge_index, edge_attr))
            x = self.layer_activations[2](apply_edge_conv(self.conv3, x, edge_index, edge_attr))
            x = global_mean_pool(x, batch_index)
        else:
            num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
            x = self._uniform_pool_forward(x, edge_index, edge_attr, batch_index, num_graphs)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


class GINEStackNetwork(UniformPoolMixin, nn.Module):
    """Three GINE layers with edge features + graph pooling + MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        global_dim: int = 8,
        heads: int = 4,
        edge_dim: int = 4,
        activation: str = "prelu",
        variant: str = "legacy",
        pool_type: str = "topk",
    ):
        super().__init__()
        _ = heads
        self.edge_dim = edge_dim
        self.architecture = "gine_stack"
        self.activation_name = activation
        self.variant = variant
        self.pool_type = pool_type
        self.conv1 = GINEConv(_gin_mlp(input_dim, hidden_dim, activation), edge_dim=edge_dim)
        self.conv2 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, activation), edge_dim=edge_dim)
        self.conv3 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, activation), edge_dim=edge_dim)
        self.shared = _graph_mlp_tail(hidden_dim, global_dim, activation)
        self.layer_activations = nn.ModuleList([make_activation(activation) for _ in range(3)])
        self.output_dim = hidden_dim
        self._last_aux_loss = torch.zeros(())
        if variant not in LEGACY_VARIANTS:
            self._init_uniform_pool(
                first_in_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=3,
                heads=heads,
                edge_dim=edge_dim,
                architecture=self.architecture,
                activation_name=activation,
                variant=variant,
                pool_type=pool_type,
                tail_in_dim=hidden_dim,
            )

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
        if self.variant in LEGACY_VARIANTS:
            x = self.layer_activations[0](apply_edge_conv(self.conv1, x, edge_index, edge_attr))
            x = self.layer_activations[1](apply_edge_conv(self.conv2, x, edge_index, edge_attr))
            x = self.layer_activations[2](apply_edge_conv(self.conv3, x, edge_index, edge_attr))
            x = global_mean_pool(x, batch_index)
        else:
            num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
            x = self._uniform_pool_forward(x, edge_index, edge_attr, batch_index, num_graphs)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


# ------------------------------------------------------------------ #
# Legacy stacks (edge-blind)
# ------------------------------------------------------------------ #

class _EdgeBlindStack(UniformPoolMixin, nn.Module):
    """Shared body for the three edge-blind simple stacks (GCN/SAGE/GIN)."""

    architecture = "gcn_stack"

    def _build_convs(self, input_dim: int, hidden_dim: int, activation: str) -> List[nn.Module]:
        raise NotImplementedError

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        global_dim: int = 8,
        heads: int = 4,
        edge_dim: int = 4,
        activation: str = "prelu",
        variant: str = "legacy",
        pool_type: str = "topk",
    ):
        super().__init__()
        _ = heads
        self.edge_dim = edge_dim
        self.activation_name = activation
        self.variant = variant
        self.pool_type = pool_type
        convs = self._build_convs(input_dim, hidden_dim, activation)
        self.conv1, self.conv2, self.conv3 = convs
        self.shared = _graph_mlp_tail(hidden_dim, global_dim, activation)
        self.layer_activations = nn.ModuleList([make_activation(activation) for _ in range(3)])
        self.output_dim = hidden_dim
        self._last_aux_loss = torch.zeros(())
        if variant not in LEGACY_VARIANTS:
            self._init_uniform_pool(
                first_in_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=3,
                heads=heads,
                edge_dim=edge_dim,
                architecture=self.architecture,
                activation_name=activation,
                variant=variant,
                pool_type=pool_type,
                tail_in_dim=hidden_dim,
            )

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        if self.variant in LEGACY_VARIANTS:
            x = self.layer_activations[0](self.conv1(x, edge_index))
            x = self.layer_activations[1](self.conv2(x, edge_index))
            x = self.layer_activations[2](self.conv3(x, edge_index))
            x = global_mean_pool(x, batch_index)
        else:
            num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
            # Edge-blind convs ignore edge_attr; the pooling layer still carries it
            # harmlessly. Coalesce so TopKPooling has a real tensor to filter.
            edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
            x = self._uniform_pool_forward(x, edge_index, edge_attr, batch_index, num_graphs)
        if global_features is not None:
            global_features = global_features.view(x.size(0), -1)
            x = torch.cat([x, global_features], dim=-1)
        return self.shared(x)


class GCNStackNetwork(_EdgeBlindStack):
    """Three GCN layers + pool + MLP."""

    architecture = "gcn_stack"

    def _build_convs(self, input_dim, hidden_dim, activation):
        return [GCNConv(input_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)]


class SAGEStackNetwork(_EdgeBlindStack):
    """Three GraphSAGE (mean) layers + pool + MLP."""

    architecture = "sage_stack"

    def _build_convs(self, input_dim, hidden_dim, activation):
        return [
            SAGEConv(input_dim, hidden_dim, aggr="mean"),
            SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
        ]


class GINStackNetwork(_EdgeBlindStack):
    """Three GIN layers + pool + MLP."""

    architecture = "gin_stack"

    def _build_convs(self, input_dim, hidden_dim, activation):
        return [
            GINConv(_gin_mlp(input_dim, hidden_dim, activation)),
            GINConv(_gin_mlp(hidden_dim, hidden_dim, activation)),
            GINConv(_gin_mlp(hidden_dim, hidden_dim, activation)),
        ]


def build_gnn(
    architecture: str,
    input_dim: int = 5,
    hidden_dim: int = 128,
    global_dim: int = 8,
    heads: int = 4,
    edge_dim: int = 4,
    activation: str = "prelu",
    variant: str = "legacy",
    pool_type: str = "topk",
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
    return builders[architecture](
        input_dim,
        hidden_dim,
        global_dim,
        heads,
        edge_dim,
        activation=activation,
        variant=variant,
        pool_type=pool_type,
    )


# ------------------------------------------------------------------ #
# Flexible Backbone (PPO / Optuna Search Search space compatible)
# ------------------------------------------------------------------ #

class GraphPolicyBackbone(UniformPoolMixin, nn.Module):
    def __init__(
        self,
        layout: Any,  # FeatureLayout
        architecture: str,
        activation: str = "prelu",
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        variant: str = "legacy",
        pool_type: str = "topk",
    ):
        super().__init__()

        self.layout = layout
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.output_dim = hidden_dim
        self.edge_dim = layout.edge_input_dim
        self.activation_name = activation
        self.variant = variant
        self.pool_type = pool_type
        self._last_aux_loss = torch.zeros(())

        self.node_feature_names = resolve_node_feature_names(
            getattr(layout, "active_feature_names", None)
        )
        # name -> column map for the active node-feature ordering (routing index reads).
        self._node_col = {name: idx for idx, name in enumerate(self.node_feature_names)}
        self.node_encoder = TwoWayFeatureEncoder(
            self.node_feature_names,
            layout.node_input_dim,
            NODE_CATEGORICAL_REGISTRY,
            activation=make_activation(activation),
        )
        self.edge_encoder = TwoWayFeatureEncoder(
            list(EDGE_FEATURE_SCHEMA),
            layout.edge_input_dim,
            EDGE_CATEGORICAL_REGISTRY,
            activation=make_activation(activation),
        )
        # Globals lost their hand-crafted sign-log; a learnable LayerNorm tames scale.
        self.global_norm = nn.LayerNorm(layout.padded_global_feature_count)
        self.global_encoder = nn.Linear(
            layout.padded_global_feature_count,
            layout.global_input_dim,
        )
        self.global_activation = make_activation(activation)
        self.convs = nn.ModuleList(
            self._build_convs(architecture, layout, hidden_dim, heads, activation)
        )
        # One activation per conv layer so PReLU slopes are independent per layer.
        self.layer_activations = nn.ModuleList(
            [make_activation(activation) for _ in range(num_layers)]
        )

        self.current_x_proj = nn.Linear(1, layout.node_input_dim)
        self.y_target_proj = nn.Linear(1, layout.node_input_dim)

        self.virtual_update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._virtual_mlp_input_dim(layer_idx), self._layer_output_dim(layer_idx)),
                    nn.BatchNorm1d(self._layer_output_dim(layer_idx)),
                    make_activation(activation),
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.shared = nn.Sequential(
            nn.Linear(2 * hidden_dim + layout.global_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
        )

        # Uniform-pool variants reuse the same self.shared tail (2*hidden_dim + global).
        if variant not in LEGACY_VARIANTS:
            self._init_uniform_pool(
                first_in_dim=layout.node_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                edge_dim=layout.edge_input_dim,
                architecture=architecture,
                activation_name=activation,
                variant=variant,
                pool_type=pool_type,
                tail_in_dim=2 * hidden_dim,
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
        activation: str,
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

    def _gine_layers(self, layout: Any, hidden_dim: int, activation: str) -> List[nn.Module]:
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

    def _gin_layers(self, layout: Any, hidden_dim: int, activation: str) -> List[nn.Module]:
        layers: List[nn.Module] = []
        in_dim = layout.node_input_dim
        for _ in range(self.num_layers):
            layers.append(GINConv(_gin_mlp(in_dim, hidden_dim, activation)))
            in_dim = hidden_dim
        return layers

    def forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        if self.variant in LEGACY_VARIANTS:
            return self._legacy_forward(x, edge_index, batch_index, global_features, edge_attr)
        return self._uniform_forward(x, edge_index, batch_index, global_features, edge_attr)

    def _legacy_forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        x_enc, node_types = self.node_encoder(x)
        is_f_root = (node_types == 6)
        is_d1_root = (node_types == 9)
        is_d2_root = (node_types == 10)
        is_virtual = is_f_root | is_d1_root | is_d2_root
        is_real = ~is_virtual

        edge_emb, _ = self.edge_encoder(
            coalesce_edge_attr(edge_attr, edge_index, self.layout.padded_edge_feature_count, x.device, x.dtype)
        )

        num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0

        real_edge_index, real_edge_emb, _ = filter_real_subgraph(edge_index, edge_emb, is_real)

        h_real = x_enc[is_real]
        h_virt = x_enc[is_virtual] if is_virtual.any() else None
        batch_real = batch_index[is_real]
        batch_virt = batch_index[is_virtual] if is_virtual.any() else None

        for layer_idx, conv in enumerate(self.convs):
            h_real = self.layer_activations[layer_idx](apply_edge_conv(conv, h_real, real_edge_index, real_edge_emb))

            if is_virtual.any():
                h_virt = self.virtual_update_mlps[layer_idx](h_virt)

        h_pooled = pool_split_embeddings(
            h_real,
            batch_real,
            h_virt,
            batch_virt,
            num_graphs,
            self.hidden_dim,
        )

        return self._apply_tail(h_pooled, global_features)

    def _uniform_forward(self, x, edge_index, batch_index, global_features=None, edge_attr=None):
        x_enc, _ = self.node_encoder(x)
        edge_emb, _ = self.edge_encoder(
            coalesce_edge_attr(edge_attr, edge_index, self.layout.padded_edge_feature_count, x.device, x.dtype)
        )
        num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
        h_pooled = self._uniform_pool_forward(x_enc, edge_index, edge_emb, batch_index, num_graphs)
        return self._apply_tail(h_pooled, global_features)

    def _apply_tail(self, h_pooled, global_features):
        if global_features is not None:
            global_features = global_features.view(h_pooled.size(0), -1)
            global_features = self.global_activation(
                self.global_encoder(self.global_norm(global_features))
            )
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
    activation: str = "prelu",
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    variant: str = "legacy",
    pool_type: str = "topk",
) -> GraphPolicyBackbone:
    return GraphPolicyBackbone(
        layout=layout,
        architecture=architecture,
        activation=activation,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        variant=variant,
        pool_type=pool_type,
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
