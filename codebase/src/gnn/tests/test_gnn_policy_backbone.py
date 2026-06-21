import torch
from feature_layout import GNN_ARCHITECTURE_CHOICES, GNN_ACTIVATION_CHOICES
from gnn.shared.models.gnn_backbones import ExpressionGNN, ARCHITECTURE_NAMES

INPUT_DIM = 28   # NATIVE_NODE_FEATURE_COUNT
GLOBAL_DIM = 8   # PADDED_GLOBAL_FEATURE_COUNT
GLOBAL_HIDDEN = 6


def _make_batch(num_nodes=10, num_graphs=2, input_dim=INPUT_DIM, global_dim=GLOBAL_DIM):
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]], dtype=torch.long
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    global_features = torch.randn(num_graphs, global_dim)
    return x, edge_index, batch, global_features


def test_sl_mode_output_shape():
    x, edge_index, batch, gf = _make_batch()
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        global_dim=GLOBAL_DIM,
        global_hidden_dim=GLOBAL_HIDDEN,
        output_dim=2,
        architecture="gatv2_stack",
        num_layers=2,
        heads=2,
        classify=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch, gf)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"
    assert torch.isfinite(out).all()


def test_rl_mode_output_shape():
    x, edge_index, batch, gf = _make_batch()
    hidden_dim = 64
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        global_dim=GLOBAL_DIM,
        global_hidden_dim=GLOBAL_HIDDEN,
        architecture="gine_stack",
        num_layers=3,
        classify=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch, gf)
    assert out.shape == (2, hidden_dim), f"Expected (2, {hidden_dim}), got {out.shape}"
    assert torch.isfinite(out).all()


def test_both_architectures_all_activations():
    x, edge_index, batch, gf = _make_batch()
    for arch in GNN_ARCHITECTURE_CHOICES:
        for act in GNN_ACTIVATION_CHOICES:
            model = ExpressionGNN(
                input_dim=INPUT_DIM,
                hidden_dim=32,
                global_dim=GLOBAL_DIM,
                global_hidden_dim=GLOBAL_HIDDEN,
                architecture=arch,
                activation=act,
                num_layers=2,
                heads=2,
                classify=False,
            )
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index, batch, gf)
            assert out.shape == (2, 32), f"{arch}/{act}: shape mismatch"
            assert torch.isfinite(out).all(), f"{arch}/{act}: non-finite output"


def test_no_global_features():
    """global_dim=0: tail takes hidden_dim only, global_features ignored."""
    x, edge_index, batch, _ = _make_batch()
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        global_dim=0,
        architecture="gatv2_stack",
        num_layers=2,
        heads=2,
        classify=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch, global_features=None)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_global_features_change_output():
    """Global features are actually used: different globals → different output."""
    x, edge_index, batch, gf = _make_batch()
    model = ExpressionGNN(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        global_dim=GLOBAL_DIM,
        global_hidden_dim=GLOBAL_HIDDEN,
        architecture="gine_stack",
        num_layers=2,
        classify=False,
    )
    model.eval()
    with torch.no_grad():
        out_a = model(x, edge_index, batch, gf)
        out_b = model(x, edge_index, batch, torch.zeros_like(gf))
    assert not torch.allclose(out_a, out_b, atol=1e-5), "global_features had no effect"


def test_supernode_participates_in_message_passing():
    """Supernode (node_type_supernode = col 3) is a regular MP node; perturbing it changes readout."""
    input_dim = INPUT_DIM
    x = torch.randn(4, input_dim)
    x[:, 0:4] = 0.0
    x[0, 1] = 1.0   # operator
    x[1, 2] = 1.0   # root
    x[2, 3] = 1.0   # supernode
    x[3, 1] = 1.0   # operator
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 0, 0, 3]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    gf = torch.randn(1, GLOBAL_DIM)

    torch.manual_seed(0)
    model = ExpressionGNN(
        input_dim=input_dim,
        hidden_dim=16,
        global_dim=GLOBAL_DIM,
        global_hidden_dim=GLOBAL_HIDDEN,
        architecture="gine_stack",
        num_layers=2,
        classify=False,
    )
    model.eval()
    with torch.no_grad():
        out_base = model(x, edge_index, batch, gf)
        x_perturbed = x.clone()
        ramp = torch.linspace(0.0, 1000.0, input_dim)
        x_perturbed[2] = x_perturbed[2] + ramp
        out_perturbed = model(x_perturbed, edge_index, batch, gf)

    assert out_base.shape == (1, 16)
    assert not torch.allclose(out_base, out_perturbed, atol=1e-5), "supernode perturbation had no effect"


def test_invalid_architecture_raises():
    try:
        ExpressionGNN(input_dim=4, architecture="gcn_stack")
        assert False, "should have raised"
    except ValueError:
        pass
