import torch
from feature_layout import FeatureLayout, GNN_ARCHITECTURE_CHOICES, GNN_ACTIVATION_CHOICES
from gnn_policy_backbone import build_graph_policy_backbone, GraphPolicyBackbone

def test_backbone_forward_pass_all_combinations():
    layout = FeatureLayout(node_input_dim=4, global_input_dim=6)
    
    num_nodes = 10
    num_graphs = 2
    
    # Padded node feature count is 5, padded global feature count is 9
    x = torch.randn(num_nodes, layout.padded_node_feature_count)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]
    ], dtype=torch.long)
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
            out = backbone(x, edge_index, batch_index, global_features)
            
            # Check shape: output should be [num_graphs, hidden_dim]
            assert out.shape == (num_graphs, hidden_dim)
            assert backbone.architecture == arch

            # Forward pass without global features
            out_no_global = backbone(x, edge_index, batch_index, None)
            assert out_no_global.shape == (num_graphs, hidden_dim)
