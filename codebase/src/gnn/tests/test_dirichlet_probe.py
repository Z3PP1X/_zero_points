"""Dirichlet-energy probe is wired into ExpressionGNN for over-smoothing diagnostics."""

import math

import torch
from torch_geometric.loader import DataLoader

from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
    compute_normalized_dirichlet_energy,
)
from gnn.shared.models.gnn_backbones import DirichletProbe, ExpressionGNN, _MPCapture


def _build_batch():
    raw = {
        "id": "P-dir",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "x", "type": "variable", "value": None},
            {"id": "f3", "label": "Sin", "type": "function", "value": None},
            {"id": "f4", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [
            {"source": "global", "target": "f1", "type": "belongs_to_f"},
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f1", "target": "f3", "type": "child_of"},
            {"source": "f3", "target": "f4", "type": "child_of"},
        ],
    }
    data = ExpressionGraphConverter().convert(raw, mode="graph")
    data.global_features = torch.zeros((1, 5), dtype=torch.float)
    data.y = torch.tensor([1], dtype=torch.long)
    return next(iter(DataLoader([data], batch_size=1)))


def _find_probe(model):
    return next(m for m in model.modules() if isinstance(m, DirichletProbe))


def _run_with_probe(model):
    batch = _build_batch()
    captured = []

    def hook(_module, _inputs, outputs):
        captured.append((outputs.x.detach(), outputs.edge_index.detach()))

    handle = _find_probe(model).register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(batch.x, batch.edge_index, batch.batch, batch.global_features)
    handle.remove()

    assert captured, "probe forward hook never fired"
    x, edge_index = captured[0]
    return compute_normalized_dirichlet_energy(x, edge_index)


def test_probe_returns_mpcapture():
    x = torch.randn(4, 3)
    ei = torch.tensor([[0, 1], [1, 2]])
    out = DirichletProbe()(x, ei, None)
    assert isinstance(out, _MPCapture)
    assert torch.equal(out.x, x)
    assert torch.equal(out.edge_index, ei)
    assert out.edge_attr is None


def test_probe_discoverable_in_backbone():
    model = ExpressionGNN(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=16,
        global_dim=5,
        classify=True,
    )
    assert isinstance(_find_probe(model), DirichletProbe)
    assert sum(p.numel() for p in _find_probe(model).parameters()) == 0


def test_energy_measured():
    model = ExpressionGNN(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=16,
        global_dim=5,
        classify=True,
    )
    energy = _run_with_probe(model)
    assert not math.isnan(energy)
    assert energy >= 0.0
