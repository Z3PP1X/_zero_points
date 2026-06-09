import torch
from feature_layout import FeatureLayout, GNN_ARCHITECTURE_CHOICES, GNN_ACTIVATION_CHOICES
from gnn_backbones import (
    NODE_TYPE_COL,
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


def test_virtual_nodes_excluded_from_message_passing():
    layout = FeatureLayout(node_input_dim=4, global_input_dim=6, edge_input_dim=4)
    hidden_dim = 16
    heads = 2
    num_layers = 2

    x = torch.randn(4, layout.padded_node_feature_count)
    x[:, NODE_TYPE_COL] = torch.tensor([1.0, 2.0, 5.0, 8.0])
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 0, 0, 3]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), layout.padded_edge_feature_count)
    batch_index = torch.zeros(4, dtype=torch.long)
    global_features = torch.randn(1, 9)

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

        perturbed = edge_attr.clone()
        perturbed[2] = perturbed[2] + 100.0
        out_perturbed = backbone(x, edge_index, batch_index, global_features, edge_attr=perturbed)

    assert out_base.shape == (1, hidden_dim)
    assert torch.allclose(out_base, out_perturbed, atol=1e-5)


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
    global_features = torch.randn(num_graphs, 9)
    
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
