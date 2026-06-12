import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import GATv2Conv, GINEConv

from gnn.shared.models.gnn_backbones import (
    EDGE_AWARE_ARCHITECTURE_NAMES,
    EdgeFeatureEncoder,
    NodeFeatureEncoder,
    _gin_mlp,
    apply_edge_conv,
    coalesce_edge_attr,
    filter_real_subgraph,
    pool_split_embeddings,
)

# Edge feature dim at which the relation-type column (index 2) is present and the
# edge encoder can embed it.
ENRICHED_EDGE_DIM = 4


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

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        features = self.backbone(x, edge_index, batch, global_features, edge_attr=edge_attr)
        if global_features is not None:
            global_features = global_features.view(features.size(0), -1)
            features = torch.cat([features, global_features], dim=-1)
        return self.classifier(features)


class TestGraphNetwork(nn.Module):
    """
    Supervised graph classifier with split real/virtual pooling and native edge features.
    Supports GATv2 (attention over edges) and GINE (edge attrs in update MLP).
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
        use_feature_encoder=True,
    ):
        super().__init__()
        if architecture not in EDGE_AWARE_ARCHITECTURE_NAMES:
            raise ValueError(
                f"Unsupported architecture {architecture!r}; "
                f"expected one of {EDGE_AWARE_ARCHITECTURE_NAMES}"
            )

        self.input_dim = input_dim
        self.architecture = architecture
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.activation = LeakyReLU()
        self.num_layers = 3

        # Categorical features (node_type @ col 0, label_id @ col 1; edge
        # relation_type @ col 2) are integer codes. Feeding them as raw floats
        # imposes a spurious ordinal scale (e.g. Log=17 ≈ 17 × Plus=3), which
        # discards the most discriminative signal — operator/function identity.
        # The encoders embed these IDs and linearly project the remaining
        # continuous columns, which also normalises their disparate scales.
        self.use_node_encoder = use_feature_encoder
        self.use_edge_encoder = use_feature_encoder and edge_dim == ENRICHED_EDGE_DIM

        if self.use_node_encoder:
            self.node_encoder = NodeFeatureEncoder(input_dim, hidden_dim, activation=LeakyReLU())
            conv_in_dim = hidden_dim
        else:
            self.node_encoder = None
            conv_in_dim = input_dim
        self.conv_in_dim = conv_in_dim

        # Edge encoder keeps the conv edge_dim unchanged so the conv layers below
        # are constructed identically; it only swaps the raw relation code for a
        # learned relation embedding fused with the continuous edge columns.
        if self.use_edge_encoder:
            self.edge_encoder = EdgeFeatureEncoder(edge_dim, edge_dim, activation=LeakyReLU())
        else:
            self.edge_encoder = None

        if architecture == "gine_stack":
            self.conv1 = GINEConv(_gin_mlp(conv_in_dim, hidden_dim, self.activation), edge_dim=edge_dim)
            self.conv2 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, self.activation), edge_dim=edge_dim)
            self.conv3 = GINEConv(_gin_mlp(hidden_dim, hidden_dim, self.activation), edge_dim=edge_dim)
        else:
            self.conv1 = GATv2Conv(conv_in_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
            self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)

        self.convs = nn.ModuleList([self.conv1, self.conv2, self.conv3])
        self.virtual_update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._virtual_mlp_input_dim(layer_idx), self._layer_output_dim(layer_idx)),
                    nn.BatchNorm1d(self._layer_output_dim(layer_idx)),
                    LeakyReLU(),
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_dim, hidden_dim),
            LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs):
        input_dim = pipeline.input_dim
        global_dim = getattr(pipeline, "global_dim", 0)
        edge_dim = getattr(pipeline, "edge_dim", 4)
        architecture = getattr(pipeline, "architecture", "gatv2_stack")
        # The categorical encoder assumes node_type/label_id live at columns 0/1
        # of the full feature schema. When an explicit active-feature subset is
        # selected those columns may be absent or reordered, so fall back to the
        # raw-linear path in that case.
        use_feature_encoder = getattr(pipeline, "active_features", None) is None
        kwargs.setdefault("use_feature_encoder", use_feature_encoder)
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
        # Derive virtual/real partition from the raw node_type column before any
        # encoding (the encoder consumes that column).
        node_types = x[:, 0].round().long()
        is_virtual = (node_types == 6) | (node_types == 9) | (node_types == 10)
        is_real = ~is_virtual

        edge_attr = coalesce_edge_attr(edge_attr, edge_index, self.edge_dim, x.device, x.dtype)

        if self.node_encoder is not None:
            x, _ = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        real_edge_index, real_edge_attr, _ = filter_real_subgraph(edge_index, edge_attr, is_real)

        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        h_real = x[is_real]
        h_virt = x[is_virtual] if is_virtual.any() else None
        batch_real = batch[is_real]
        batch_virt = batch[is_virtual] if is_virtual.any() else None

        for layer_idx, conv in enumerate(self.convs):
            h_real = self.activation(apply_edge_conv(conv, h_real, real_edge_index, real_edge_attr))
            if is_virtual.any():
                h_virt = self.virtual_update_mlps[layer_idx](h_virt)

        x_pooled = pool_split_embeddings(
            h_real,
            batch_real,
            h_virt,
            batch_virt,
            num_graphs,
            self.hidden_dim,
        )

        if global_features is not None:
            global_features = global_features.view(x_pooled.size(0), -1)
            x_pooled = torch.cat([x_pooled, global_features], dim=-1)

        return self.classifier(x_pooled)
