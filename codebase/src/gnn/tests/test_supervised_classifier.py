import torch
from torch_geometric.loader import DataLoader

from graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
    EDGE_FEATURE_SCHEMA,
)
from classifiers import TestGraphNetwork


def _sample_raw():
    return {
        "id": "P-clf",
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


def _build_batch():
    converter = ExpressionGraphConverter()
    data = converter.convert(_sample_raw(), heterogeneous=False, mode="graph")
    if hasattr(data, "laplacian"):
        del data.laplacian
    data.global_features = torch.zeros((1, 2), dtype=torch.float)
    data.y = torch.tensor([1], dtype=torch.long)
    loader = DataLoader([data], batch_size=1)
    return next(iter(loader))


def test_classifier_forward_with_feature_encoder():
    batch = _build_batch()
    input_dim = len(NODE_FEATURE_SCHEMA)
    edge_dim = len(EDGE_FEATURE_SCHEMA)

    for arch in ("gatv2_stack", "gine_stack"):
        model = TestGraphNetwork(
            input_dim=input_dim,
            hidden_dim=32,
            global_dim=2,
            edge_dim=edge_dim,
            architecture=arch,
        )
        model.eval()
        assert model.node_encoder is not None
        assert model.edge_encoder is not None
        with torch.no_grad():
            out = model(
                batch.x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
            )
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all()


def test_label_id_embedding_breaks_ordinal_assumption():
    """Two graphs differing only in a function label must be separable through the
    label embedding rather than a linear scaling of the raw id."""
    batch = _build_batch()
    model = TestGraphNetwork(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=32,
        global_dim=2,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        architecture="gatv2_stack",
    )
    # The label embedding table must cover the full canonical vocabulary.
    assert (
        model.node_encoder.embeddings["label_id"].num_embeddings
        >= int(batch.x[:, 1].max().item()) + 1
    )


def test_classifier_forward_with_feature_subset():
    """The encoder locates categoricals BY NAME, so a reordered active-feature subset
    that does not start with node_type/label_id still embeds them and forwards."""
    batch = _build_batch()
    # Reordered subset: node_type is no longer at column 0.
    active_features = ["value", "node_type", "label_id", "depth"]
    sub_x = batch.x[:, [NODE_FEATURE_SCHEMA.index(f) for f in active_features]]
    model = TestGraphNetwork(
        input_dim=len(active_features),
        hidden_dim=32,
        global_dim=2,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        architecture="gatv2_stack",
        active_features=active_features,
    )
    model.eval()
    # Both categoricals are embedded despite the reorder (no plain-linear fallback).
    assert set(model.node_encoder.embeddings.keys()) == {"node_type", "label_id"}
    with torch.no_grad():
        out = model(
            sub_x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
        )
    assert out.shape == (1, 2)
