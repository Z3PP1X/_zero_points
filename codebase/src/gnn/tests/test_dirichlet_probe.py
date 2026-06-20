"""Dirichlet-energy probe is wired into the custom expression_classifier backbone.

Before this, the energy callbacks hooked PyG GNN's ``.mp`` stage, which the custom
backbone lacks, so energy was always NaN. The backbone now exposes a parameter-free
``DirichletProbe`` at its message-passing output; these tests confirm a forward hook on
it captures node embeddings and yields a finite, measured energy for every variant.
"""

import math

import torch
from torch_geometric.loader import DataLoader

# Package-style imports throughout: classifiers.py imports DirichletProbe via the
# gnn.shared.models path, so the test must use the same path or isinstance() compares two
# distinct class objects (the repo's dual bare/package import hazard).
from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
    EDGE_FEATURE_SCHEMA,
    compute_normalized_dirichlet_energy,
)
from gnn.shared.models.gnn_backbones import DirichletProbe, _MPCapture
from gnn.shared.models.classifiers import TestGraphNetwork


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
    data = ExpressionGraphConverter().convert(raw,  mode="graph")
    data.global_features = torch.zeros((1, 5), dtype=torch.float)
    data.y = torch.tensor([1], dtype=torch.long)
    return next(iter(DataLoader([data], batch_size=1)))


def _find_probe(model):
    return next(m for m in model.modules() if isinstance(m, DirichletProbe))


def _run_with_probe(model):
    """Forward once with a hook on the probe; return the measured Dirichlet energy."""
    batch = _build_batch()
    captured = []

    def hook(_module, _inputs, outputs):
        captured.append((outputs.x.detach(), outputs.edge_index.detach()))

    handle = _find_probe(model).register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(batch.x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr)
    handle.remove()

    assert captured, "probe forward hook never fired"
    x, edge_index = captured[0]
    return compute_normalized_dirichlet_energy(x, edge_index)


def test_probe_returns_batch_like_capture():
    x = torch.randn(4, 3)
    ei = torch.tensor([[0, 1], [1, 2]])
    out = DirichletProbe()(x, ei, None)
    assert isinstance(out, _MPCapture)
    assert torch.equal(out.x, x)
    assert torch.equal(out.edge_index, ei)
    assert out.edge_attr is None


def test_probe_discoverable_in_backbone():
    model = TestGraphNetwork(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=16,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        variant="legacy",
    )
    assert isinstance(_find_probe(model), DirichletProbe)
    # The probe is parameter-free, so it must not enlarge the checkpoint.
    assert sum(p.numel() for p in _find_probe(model).parameters()) == 0


def test_energy_measured_for_legacy():
    model = TestGraphNetwork(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=16,
        global_dim=5,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        variant="legacy",
    )
    energy = _run_with_probe(model)
    assert not math.isnan(energy)
    assert energy >= 0.0


def test_energy_measured_for_uniform_variants():
    for variant, pool_type in (("pooling", "topk"), ("pooling", "diffpool")):
        model = TestGraphNetwork(
            input_dim=len(NODE_FEATURE_SCHEMA),
            hidden_dim=16,
            global_dim=5,
            edge_dim=len(EDGE_FEATURE_SCHEMA),
            variant=variant,
            pool_type=pool_type,
        )
        energy = _run_with_probe(model)
        assert not math.isnan(energy), f"{variant}/{pool_type} energy is NaN"
        assert energy >= 0.0
