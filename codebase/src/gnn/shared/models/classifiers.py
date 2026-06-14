import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINEConv

from gnn.shared.models.gnn_backbones import (
    DirichletProbe,
    EDGE_AWARE_ARCHITECTURE_NAMES,
    LEGACY_VARIANTS,
    UniformPoolMixin,
    _gin_mlp,
    apply_edge_conv,
    coalesce_edge_attr,
    filter_real_subgraph,
    make_activation,
    pool_split_embeddings,
    resolve_global_pool,
    resolve_node_feature_names,
)
from gnn.shared.models.feature_encoders import TwoWayFeatureEncoder
from gnn.shared.utils.feature_config import (
    EDGE_CATEGORICAL_REGISTRY,
    NODE_CATEGORICAL_REGISTRY,
)
from gnn.shared.utils.graph_utils import EDGE_FEATURE_SCHEMA

# Edge feature dim at which the relation-type column (index 2) is present and the
# edge encoder can embed it. Derived from the schema so it tracks edge-column changes.
ENRICHED_EDGE_DIM = len(EDGE_FEATURE_SCHEMA)


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
            make_activation("prelu"),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        features = self.backbone(x, edge_index, batch, global_features, edge_attr=edge_attr)
        if global_features is not None:
            global_features = global_features.view(features.size(0), -1)
            features = torch.cat([features, global_features], dim=-1)
        return self.classifier(features)


class TestGraphNetwork(UniformPoolMixin, nn.Module):
    """
    Supervised graph classifier. The legacy variant uses split real/virtual pooling
    and native edge features (GATv2 attention over edges, or GINE edge attrs in the
    update MLP). The ``pooling`` / ``pooling_skip`` variants pool the whole graph
    uniformly via TopK or DiffPool. PReLU activations throughout.
    """

    __test__ = False  # prevent pytest from collecting this nn.Module as a test case

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        global_dim=2,
        output_dim=2,
        heads=4,
        architecture="gatv2_stack",
        edge_dim=4,
        active_features=None,
        activation="prelu",
        variant="legacy",
        pool_type="topk",
        num_layers=3,
        dropout=0.2,
        graph_pooling="mean",
    ):
        super().__init__()
        if architecture not in EDGE_AWARE_ARCHITECTURE_NAMES:
            raise ValueError(
                f"Unsupported architecture {architecture!r}; "
                f"expected one of {EDGE_AWARE_ARCHITECTURE_NAMES}"
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_dim = input_dim
        self.architecture = architecture
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.activation_name = activation
        self.variant = variant
        self.pool_type = pool_type
        self.dropout = dropout
        # Graph-level readout for the legacy real/virtual split path. Selectable via
        # cfg.model.graph_pooling (mean|add|max); 'add' keeps node-count/multiplicity signal.
        self.graph_pooling = graph_pooling
        self.pool_fn = resolve_global_pool(graph_pooling)
        self._last_aux_loss = torch.zeros(())
        self.num_layers = num_layers

        # Categorical features (node_type; edge relation_type) are integer codes.
        # Feeding them as raw floats imposes a spurious ordinal scale (e.g. Log=17 ≈
        # 17 × Plus=3), discarding the most discriminative signal — operator identity.
        # TwoWayFeatureEncoder embeds them BY NAME so it works under any active-feature
        # subset/reorder and projects the LayerNorm'd continuous columns linearly.
        self.node_feature_names = resolve_node_feature_names(active_features)
        self._node_col = {name: idx for idx, name in enumerate(self.node_feature_names)}
        self.node_encoder = TwoWayFeatureEncoder(
            self.node_feature_names,
            hidden_dim,
            NODE_CATEGORICAL_REGISTRY,
            activation=make_activation(activation),
        )
        conv_in_dim = hidden_dim
        self.conv_in_dim = conv_in_dim

        # Edge encoder maps to the same edge_dim so the conv layers are constructed
        # identically; it only swaps the raw relation code for a learned embedding
        # fused with the LayerNorm'd continuous edge columns.
        self.use_edge_encoder = edge_dim == ENRICHED_EDGE_DIM
        if self.use_edge_encoder:
            self.edge_encoder = TwoWayFeatureEncoder(
                list(EDGE_FEATURE_SCHEMA),
                edge_dim,
                EDGE_CATEGORICAL_REGISTRY,
                activation=make_activation(activation),
            )
        else:
            self.edge_encoder = None

        # Depth is config-driven (cfg.gnn.layers_mp): build exactly num_layers convs with
        # the correct per-layer widths instead of a fixed conv1/conv2/conv3 triple. For GATv2
        # the last layer collapses the heads (heads=1, concat=False) so the readout width
        # stays hidden_dim regardless of att_heads / depth; intermediate layers concat heads.
        convs: list[nn.Module] = []
        in_dim = conv_in_dim
        for layer_idx in range(self.num_layers):
            is_last = layer_idx == self.num_layers - 1
            if architecture == "gine_stack":
                convs.append(GINEConv(_gin_mlp(in_dim, hidden_dim, activation), edge_dim=edge_dim))
                in_dim = hidden_dim
            else:
                out_heads = 1 if is_last else heads
                concat = not is_last
                convs.append(
                    GATv2Conv(in_dim, hidden_dim, heads=out_heads, concat=concat, edge_dim=edge_dim)
                )
                in_dim = hidden_dim * out_heads if concat else hidden_dim
        self.convs = nn.ModuleList(convs)
        # Forward-hook target for Dirichlet-energy / over-smoothing diagnostics. Parameter-
        # free, so it leaves the checkpoint and numerics untouched; main_graphgym discovers
        # it as the message-passing probe in place of PyG GNN's absent .mp stage.
        self.dirichlet_probe = DirichletProbe()
        # One activation per conv layer so PReLU slopes are independent per layer.
        self.layer_activations = nn.ModuleList(
            [make_activation(activation) for _ in range(self.num_layers)]
        )
        self.virtual_update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._virtual_mlp_input_dim(layer_idx), self._layer_output_dim(layer_idx)),
                    nn.BatchNorm1d(self._layer_output_dim(layer_idx)),
                    make_activation(activation),
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_dim, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Uniform-pool variants reuse the same classifier tail (2*hidden_dim + global).
        if variant not in LEGACY_VARIANTS:
            self._init_uniform_pool(
                first_in_dim=conv_in_dim,
                hidden_dim=hidden_dim,
                num_layers=self.num_layers,
                heads=heads,
                edge_dim=edge_dim,
                architecture=architecture,
                activation_name=activation,
                variant=variant,
                pool_type=pool_type,
                tail_in_dim=2 * hidden_dim,
            )

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs):
        input_dim = pipeline.input_dim
        global_dim = getattr(pipeline, "global_dim", 0)
        edge_dim = getattr(pipeline, "edge_dim", 4)
        architecture = getattr(pipeline, "architecture", "gatv2_stack")
        # The encoder locates categorical columns BY NAME, so an active-feature
        # subset/reorder is handled correctly — thread the names through.
        kwargs.setdefault("active_features", getattr(pipeline, "active_features", None))
        return cls(
            input_dim=input_dim,
            global_dim=global_dim,
            edge_dim=edge_dim,
            architecture=architecture,
            **kwargs,
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
            return self.conv_in_dim
        return self._layer_output_dim(layer_idx - 1)

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        if self.variant in LEGACY_VARIANTS:
            x_pooled = self._legacy_forward(x, edge_index, batch, edge_attr)
        else:
            x_pooled = self._uniform_forward(x, edge_index, batch, edge_attr)

        if global_features is not None:
            global_features = global_features.view(x_pooled.size(0), -1)
            x_pooled = torch.cat([x_pooled, global_features], dim=-1)

        return self.classifier(x_pooled)

    def _legacy_forward(self, x, edge_index, batch, edge_attr=None):
        # Derive virtual/real partition from the raw node_type column before any
        # encoding (the encoder consumes that column). Resolve the column BY NAME so
        # it survives active-feature subset/reorder.
        node_type_col = self._node_col.get("node_type", 0)
        node_types = x[:, node_type_col].round().long()
        is_virtual = (node_types == 6) | (node_types == 9) | (node_types == 10)
        is_real = ~is_virtual

        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)

        x, _ = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr, _ = self.edge_encoder(edge_attr)

        real_edge_index, real_edge_attr, _ = filter_real_subgraph(edge_index, edge_attr, is_real)

        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        h_real = x[is_real]
        h_virt = x[is_virtual] if is_virtual.any() else None
        batch_real = batch[is_real]
        batch_virt = batch[is_virtual] if is_virtual.any() else None

        for layer_idx, conv in enumerate(self.convs):
            h_real = self.layer_activations[layer_idx](
                apply_edge_conv(conv, h_real, real_edge_index, real_edge_attr)
            )
            if is_virtual.any():
                h_virt = self.virtual_update_mlps[layer_idx](h_virt)

        # Probe the final message-passing embeddings on the real subgraph (virtual nodes
        # are MLP-updated, not message-passed) for Dirichlet over-smoothing diagnostics.
        self.dirichlet_probe(h_real, real_edge_index, real_edge_attr)

        return pool_split_embeddings(
            h_real,
            batch_real,
            h_virt,
            batch_virt,
            num_graphs,
            self.hidden_dim,
            pool_fn=self.pool_fn,
        )

    def _uniform_forward(self, x, edge_index, batch, edge_attr=None):
        # Whole-graph hierarchical pooling: no real/virtual split.
        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)
        x, _ = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr, _ = self.edge_encoder(edge_attr)
        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        return self._uniform_pool_forward(x, edge_index, edge_attr, batch, num_graphs)
