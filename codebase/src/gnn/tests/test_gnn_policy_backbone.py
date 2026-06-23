import torch
from feature_layout import GNN_ACTIVATION_CHOICES
from gnn.shared.models.gnn_backbones import ExpressionGNN

INPUT_DIM = 28   # NATIVE_NODE_FEATURE_COUNT


def _make_batch(num_nodes=10, num_graphs=2, input_dim=INPUT_DIM):
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]], dtype=torch.long
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    return x, edge_index, batch


def test_sl_mode_output_shape():
    x, edge_index, batch = _make_batch()
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        output_dim=2,
        num_layers=2,
        classify=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"
    assert torch.isfinite(out).all()


def test_rl_mode_output_shape():
    x, edge_index, batch = _make_batch()
    hidden_dim = 64
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=3,
        classify=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch)
    assert out.shape == (2, hidden_dim), f"Expected (2, {hidden_dim}), got {out.shape}"
    assert torch.isfinite(out).all()


def test_all_activations():
    x, edge_index, batch = _make_batch()
    for act in GNN_ACTIVATION_CHOICES:
        model = ExpressionGNN(
            input_dim=INPUT_DIM,
            hidden_dim=32,
            activation=act,
            num_layers=2,
            classify=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        assert out.shape == (2, 32), f"{act}: shape mismatch"
        assert torch.isfinite(out).all(), f"{act}: non-finite output"


def test_node_features_change_output():
    """Different node features produce different output (solver state now in node features)."""
    x, edge_index, batch = _make_batch()
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        num_layers=2,
        classify=False,
    )
    model.eval()
    with torch.no_grad():
        out_a = model(x, edge_index, batch)
        out_b = model(torch.zeros_like(x), edge_index, batch)
    assert not torch.allclose(out_a, out_b, atol=1e-5), "node feature change had no effect"


def test_supernode_participates_in_message_passing():
    """Supernode is encoded as node_type_global and participates in message passing."""
    input_dim = INPUT_DIM
    x = torch.randn(4, input_dim)
    x[:, 0:3] = 0.0
    x[0, 1] = 1.0   # operator
    x[1, 2] = 1.0   # function
    x[2, 0] = 1.0   # supernode → encoded as global (col 0)
    x[3, 1] = 1.0   # operator
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 0, 0, 3]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)

    torch.manual_seed(0)
    model = ExpressionGNN(
        input_dim=input_dim,
        hidden_dim=16,
        num_layers=2,
        classify=False,
    )
    model.eval()
    with torch.no_grad():
        out_base = model(x, edge_index, batch)
        x_perturbed = x.clone()
        ramp = torch.linspace(0.0, 1000.0, input_dim)
        x_perturbed[2] = x_perturbed[2] + ramp
        out_perturbed = model(x_perturbed, edge_index, batch)

    assert out_base.shape == (1, 16)
    assert not torch.allclose(out_base, out_perturbed, atol=1e-5), "supernode perturbation had no effect"
