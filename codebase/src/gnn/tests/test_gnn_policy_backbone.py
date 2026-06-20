import torch
from feature_layout import (
    FeatureLayout,
    GNN_ARCHITECTURE_CHOICES,
    GNN_ACTIVATION_CHOICES,
    GNN_VARIANT_CHOICES,
    POOL_TYPE_CHOICES,
)
from gnn_backbones import (
    build_graph_policy_backbone,
    filter_real_subgraph,
)


def test_filter_real_subgraph_drops_virtual_edges():
    is_real = torch.tensor([True, True, False, True], dtype=torch.bool)
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 3],
            [1, 2, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float)

    real_edge_index, real_edge_attr, real_idx = filter_real_subgraph(edge_index, edge_attr, is_real)

    assert real_idx.tolist() == [0, 1, 3]
    assert real_edge_index.tolist() == [[0, 2], [1, 0]]
    assert real_edge_attr.squeeze(-1).tolist() == [1.0, 5.0]


def test_supernode_participates_in_message_passing():
    """The structural aggregator roots (old node_type codes 6/9/10 = f_root/d1_root/d2_root)
    were removed: ``VIRTUAL_NODE_TYPES`` is now empty and the supernode (code 5) is an ordinary
    message-passing node. So an edge incident to the supernode DOES influence the readout via
    node-feature message passing — the supernode's features propagate to its neighbours.
    (Edge features are now dropped; perturbation is done on node features instead.)"""
    layout = FeatureLayout(node_input_dim=4, global_input_dim=6, edge_input_dim=4)
    hidden_dim = 16
    heads = 2
    num_layers = 2

    torch.manual_seed(0)
    x = torch.randn(4, layout.padded_node_feature_count)
    # Node 2 is the supernode (one-hot: node_type_supernode col=3); edge 2->0 connects it.
    # Set one-hot node_type columns (indices 0-3): operator=1, root=2, supernode=3
    x[:, 0:4] = 0.0
    x[0, 1] = 1.0  # operator
    x[1, 2] = 1.0  # root
    x[2, 3] = 1.0  # supernode
    x[3, 1] = 1.0  # operator
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 0, 0, 3]], dtype=torch.long)
    edge_attr = torch.zeros(edge_index.size(1), layout.padded_edge_feature_count)
    batch_index = torch.zeros(4, dtype=torch.long)
    global_features = torch.randn(1, 8)

    backbone = build_graph_policy_backbone(
        layout=layout,
        architecture="gine_stack",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
    )
    backbone.eval()

    with torch.no_grad():
        out_base = backbone(x, edge_index, batch_index, global_features, edge_attr=edge_attr)

        # Perturb the supernode's node features in a way that changes LayerNorm output.
        # LayerNorm is shift- and scale-invariant, so we use a ramp (non-constant diff)
        # across the non-one-hot feature dimensions to break the relative distribution.
        x_perturbed = x.clone()
        ramp = torch.linspace(0.0, 1000.0, x.size(1))
        x_perturbed[2] = x_perturbed[2] + ramp
        out_perturbed = backbone(x_perturbed, edge_index, batch_index, global_features, edge_attr=edge_attr)

    assert out_base.shape == (1, hidden_dim)
    # The supernode is NOT excluded: its node features propagate via message passing to the readout.
    assert not torch.allclose(out_base, out_perturbed, atol=1e-5)


def test_backbone_forward_pass_all_combinations():
    layout = FeatureLayout(node_input_dim=4, global_input_dim=6, edge_input_dim=4)
    
    num_nodes = 10
    num_graphs = 2

    x = torch.randn(num_nodes, layout.padded_node_feature_count)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]
    ], dtype=torch.long)
    num_edges = edge_index.size(1)
    edge_attr = torch.randn(num_edges, layout.padded_edge_feature_count)
    batch_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    global_features = torch.randn(num_graphs, 8)
    
    # Test all architectures and all activations
    for arch in GNN_ARCHITECTURE_CHOICES:
        for activation in GNN_ACTIVATION_CHOICES:
            hidden_dim = 64
            num_layers = 2
            heads = 2
            
            backbone = build_graph_policy_backbone(
                layout=layout,
                architecture=arch,
                activation=activation,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads
            )
            
            # Forward pass with global features
            out = backbone(x, edge_index, batch_index, global_features, edge_attr=edge_attr)
            
            # Check shape: output should be [num_graphs, hidden_dim]
            assert out.shape == (num_graphs, hidden_dim)
            assert backbone.architecture == arch

            # Forward pass without global features
            out_no_global = backbone(x, edge_index, batch_index, None, edge_attr=edge_attr)
            assert out_no_global.shape == (num_graphs, hidden_dim)


def _pool_inputs():
    layout = FeatureLayout(node_input_dim=4, global_input_dim=6, edge_input_dim=4)
    x = torch.randn(10, layout.padded_node_feature_count)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]], dtype=torch.long
    )
    edge_attr = torch.randn(edge_index.size(1), layout.padded_edge_feature_count)
    batch_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    global_features = torch.randn(2, 8)
    return layout, x, edge_index, edge_attr, batch_index, global_features


def test_variant_pool_forward_shapes():
    layout, x, edge_index, edge_attr, batch_index, global_features = _pool_inputs()
    hidden_dim = 64
    num_graphs = 2

    for variant in ("pooling", "pooling_skip"):
        for pool_type in POOL_TYPE_CHOICES:
            backbone = build_graph_policy_backbone(
                layout=layout,
                architecture="gatv2_stack",
                activation="prelu",
                hidden_dim=hidden_dim,
                num_layers=2,
                heads=2,
                variant=variant,
                pool_type=pool_type,
            )
            backbone.eval()
            with torch.no_grad():
                out = backbone(x, edge_index, batch_index, global_features, edge_attr=edge_attr)
            assert out.shape == (num_graphs, hidden_dim), (variant, pool_type)
            assert torch.isfinite(out).all(), (variant, pool_type)


def test_legacy_variant_is_default_and_deterministic():
    layout, x, edge_index, edge_attr, batch_index, global_features = _pool_inputs()
    assert "legacy" in GNN_VARIANT_CHOICES

    torch.manual_seed(0)
    default_backbone = build_graph_policy_backbone(
        layout=layout, architecture="gine_stack", hidden_dim=32, num_layers=2, heads=2
    )
    torch.manual_seed(0)
    legacy_backbone = build_graph_policy_backbone(
        layout=layout, architecture="gine_stack", hidden_dim=32, num_layers=2, heads=2,
        variant="legacy",
    )
    default_backbone.eval()
    legacy_backbone.eval()
    with torch.no_grad():
        out_default = default_backbone(x, edge_index, batch_index, global_features, edge_attr=edge_attr)
        out_legacy = legacy_backbone(x, edge_index, batch_index, global_features, edge_attr=edge_attr)
    assert default_backbone.variant == "legacy"
    assert torch.allclose(out_default, out_legacy, atol=1e-6)
