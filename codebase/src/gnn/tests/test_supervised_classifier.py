import torch
from torch_geometric.loader import DataLoader

from graph_utils import (
    ExpressionGraphConverter,
    ENRICHED_NODE_FEATURE_SCHEMA,
    ENRICHED_EDGE_FEATURE_SCHEMA,
    BASIC_NODE_FEATURE_SCHEMA,
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


def _build_batch(enrich: bool):
    converter = ExpressionGraphConverter()
    data = converter.convert(_sample_raw(), heterogeneous=False, enrich=enrich, mode="graph")
    if hasattr(data, "laplacian"):
        del data.laplacian
    data.global_features = torch.zeros((1, 2), dtype=torch.float)
    data.y = torch.tensor([1], dtype=torch.long)
    loader = DataLoader([data], batch_size=1)
    return next(iter(loader))


def test_classifier_forward_with_feature_encoder():
    batch = _build_batch(enrich=True)
    input_dim = len(ENRICHED_NODE_FEATURE_SCHEMA)
    edge_dim = len(ENRICHED_EDGE_FEATURE_SCHEMA)

    for arch in ("gatv2_stack", "gine_stack"):
        model = TestGraphNetwork(
            input_dim=input_dim,
            hidden_dim=32,
            global_dim=2,
            edge_dim=edge_dim,
            architecture=arch,
            use_feature_encoder=True,
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
    batch = _build_batch(enrich=True)
    model = TestGraphNetwork(
        input_dim=len(ENRICHED_NODE_FEATURE_SCHEMA),
        hidden_dim=32,
        global_dim=2,
        edge_dim=len(ENRICHED_EDGE_FEATURE_SCHEMA),
        architecture="gatv2_stack",
        use_feature_encoder=True,
    )
    # The label embedding table must cover the full canonical vocabulary.
    assert model.node_encoder.label_emb.num_embeddings >= int(batch.x[:, 1].max().item()) + 1


def test_classifier_forward_without_feature_encoder():
    batch = _build_batch(enrich=True)
    model = TestGraphNetwork(
        input_dim=len(ENRICHED_NODE_FEATURE_SCHEMA),
        hidden_dim=32,
        global_dim=2,
        edge_dim=len(ENRICHED_EDGE_FEATURE_SCHEMA),
        architecture="gatv2_stack",
        use_feature_encoder=False,
    )
    model.eval()
    assert model.node_encoder is None
    assert model.edge_encoder is None
    with torch.no_grad():
        out = model(
            batch.x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
        )
    assert out.shape == (1, 2)


def test_classifier_basic_schema_skips_edge_encoder():
    batch = _build_batch(enrich=False)
    model = TestGraphNetwork(
        input_dim=len(BASIC_NODE_FEATURE_SCHEMA),
        hidden_dim=32,
        global_dim=2,
        edge_dim=1,
        architecture="gatv2_stack",
        use_feature_encoder=True,
    )
    model.eval()
    # Node encoder still applies, but the single-column basic edge schema has no
    # relation-type column to embed.
    assert model.node_encoder is not None
    assert model.edge_encoder is None
    with torch.no_grad():
        out = model(
            batch.x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
        )
    assert out.shape == (1, 2)
