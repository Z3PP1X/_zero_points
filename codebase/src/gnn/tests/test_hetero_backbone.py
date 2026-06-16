"""Spec for the true heterogeneous training path (TDD — drives hetero_backbone.py).

Covers the four building blocks of a real HeteroData run: edge-type metadata collection,
edge-type padding for uniform collation, the to_hetero-based classifier forward (single and
batched), and robustness to graphs that lack a node type. Package-style imports throughout
(the repo's dual bare/package import hazard).
"""

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
)
from gnn.shared.models.hetero_backbone import (
    HETERO_NODE_TYPES,
    HeteroExpressionClassifier,
    build_hetero_metadata,
    collect_edge_types,
    pad_edge_types,
)


# --------------------------------------------------------------------------- #
# Fixtures: real HeteroData from the shared converter
# --------------------------------------------------------------------------- #
def _raw_small():
    return {
        "id": "P-small",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [
            {"source": "global", "target": "f1", "type": "belongs_to_f"},
            {"source": "f1", "target": "f2", "type": "child_of"},
        ],
    }


def _raw_large():
    return {
        "id": "P-large",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Power", "type": "operator", "value": None},
            {"id": "f2", "label": "Sin", "type": "function", "value": None},
            {"id": "f3", "label": "x", "type": "variable", "value": None},
            {"id": "f4", "label": "2", "type": "constant", "value": 2.0},
        ],
        "edges": [
            {"source": "global", "target": "f1", "type": "belongs_to_f"},
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f2", "target": "f3", "type": "child_of"},
            {"source": "f1", "target": "f4", "type": "child_of"},
        ],
    }


def _raw_minimal():
    # global + a single operator (which becomes the tree root) => empty 'operator' store.
    return {
        "id": "P-min",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
        ],
        "edges": [{"source": "global", "target": "f1", "type": "belongs_to_f"}],
    }


def _hetero(raw, label=1):
    data = ExpressionGraphConverter().convert(raw, heterogeneous=True, mode="graph")
    data.y = torch.tensor([label], dtype=torch.long)
    return data


def _dataset():
    return [_hetero(_raw_small(), 0), _hetero(_raw_large(), 1)]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_node_type_vocabulary_is_fixed():
    assert HETERO_NODE_TYPES == ("global", "operator", "root")


def test_collect_edge_types_returns_sorted_union():
    data_list = _dataset()
    edge_types = collect_edge_types(data_list)

    # Every edge type is a (src, relation, dst) triplet over the node vocabulary.
    assert edge_types and all(len(et) == 3 for et in edge_types)
    for src, _rel, dst in edge_types:
        assert src in HETERO_NODE_TYPES and dst in HETERO_NODE_TYPES
    # It is the union: every edge type present in any single graph appears.
    for data in data_list:
        for et in data.edge_types:
            assert et in edge_types
    # Stable ordering (sorted) so model metadata is deterministic across runs.
    assert edge_types == sorted(edge_types)


def test_build_hetero_metadata_shape():
    node_types, edge_types = build_hetero_metadata(_dataset())
    assert node_types == list(HETERO_NODE_TYPES)
    assert edge_types == collect_edge_types(_dataset())


# --------------------------------------------------------------------------- #
# Padding for uniform collation
# --------------------------------------------------------------------------- #
def test_pad_edge_types_adds_missing_as_empty():
    small = _hetero(_raw_small())
    target = collect_edge_types(_dataset())  # union ⊇ small's own edge types

    before = set(small.edge_types)
    padded = pad_edge_types(small, target)

    assert set(padded.edge_types) == set(target)
    for et in target:
        ei = padded[et].edge_index
        assert ei.size(0) == 2
        if et not in before:
            assert ei.size(1) == 0  # newly added types are empty
    # Node stores are untouched.
    for nt in HETERO_NODE_TYPES:
        assert padded[nt].x.shape == small[nt].x.shape


def test_padded_graphs_share_layout_and_batch():
    data_list = _dataset()
    target = collect_edge_types(data_list)
    padded = [pad_edge_types(d, target) for d in data_list]

    # All graphs now expose the identical edge-type set → collatable.
    assert all(set(p.edge_types) == set(target) for p in padded)
    batch = Batch.from_data_list(padded)  # must not raise
    assert batch.num_graphs == 2


# --------------------------------------------------------------------------- #
# The to_hetero classifier
# --------------------------------------------------------------------------- #
def _build_model(data_list, **kw):
    metadata = build_hetero_metadata(data_list)
    # active_features=None -> full NODE_FEATURE_SCHEMA, matching the converter's x layout.
    return HeteroExpressionClassifier(
        metadata, active_features=None, hidden_dim=16, **kw
    )


def test_categoricals_are_embedded_not_raw_ordinal():
    """The hetero model embeds node_type via nn.Embedding (parity with the homo backbone)."""
    model = _build_model(_dataset())
    # node_type is embedded by name, not fed as a raw ordinal code through a plain Linear.
    assert "node_type" in model.node_encoder.embeddings
    assert isinstance(model.node_encoder.embeddings["node_type"], torch.nn.Embedding)
    # The old plain input projection is gone.
    assert not hasattr(model, "lin_in")


def test_classifier_forward_single_graph():
    data_list = _dataset()
    target = collect_edge_types(data_list)
    model = _build_model(data_list)
    model.eval()

    padded = pad_edge_types(_hetero(_raw_large()), target)
    batch = next(iter(DataLoader([padded], batch_size=1)))
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (1, 2)


def test_classifier_forward_batched():
    data_list = _dataset()
    target = collect_edge_types(data_list)
    model = _build_model(data_list)
    model.eval()

    padded = [pad_edge_types(d, target) for d in data_list]
    batch = next(iter(DataLoader(padded, batch_size=2)))
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, 2)


def test_classifier_robust_to_absent_node_type():
    """A graph with an empty node-type store must still classify (to_hetero edge case)."""
    data_list = _dataset()
    target = collect_edge_types(data_list)
    model = _build_model(data_list)
    model.eval()

    minimal = _hetero(_raw_minimal())
    # This graph has no 'operator' nodes — exercises to_hetero with an empty store.
    assert any(minimal[nt].x.size(0) == 0 for nt in HETERO_NODE_TYPES)
    batch = next(iter(DataLoader([pad_edge_types(minimal, target)], batch_size=1)))
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
