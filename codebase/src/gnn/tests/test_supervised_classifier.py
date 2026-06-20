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
    data = converter.convert(_sample_raw(), mode="graph")
    data.global_features = torch.zeros((1, 5), dtype=torch.float)
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
            global_dim=5,
            edge_dim=edge_dim,
            architecture=arch,
        )
        model.eval()
        assert model.node_encoder is not None
        assert model.edge_encoder is None
        with torch.no_grad():
            out = model(
                batch.x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
            )
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all()


def test_variant_pool_classifier():
    """The pooling / pooling_skip variants (TopK + DiffPool) forward to class logits."""
    batch = _build_batch()
    input_dim = len(NODE_FEATURE_SCHEMA)
    edge_dim = len(EDGE_FEATURE_SCHEMA)

    for variant in ("pooling", "pooling_skip"):
        for pool_type in ("topk", "diffpool"):
            model = TestGraphNetwork(
                input_dim=input_dim,
                hidden_dim=32,
                global_dim=5,
                edge_dim=edge_dim,
                architecture="gatv2_stack",
                activation="prelu",
                variant=variant,
                pool_type=pool_type,
            )
            model.eval()
            with torch.no_grad():
                out = model(
                    batch.x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
                )
            assert out.shape == (1, 2), (variant, pool_type)
            assert torch.isfinite(out).all(), (variant, pool_type)


def test_node_encoder_is_layernorm_linear():
    """node_type and root_color are now one-hot columns, so node_encoder is a plain
    LayerNorm→Linear projection (no nn.Embedding tables)."""
    model = TestGraphNetwork(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=32,
        global_dim=5,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        architecture="gatv2_stack",
    )
    enc = model.node_encoder
    assert isinstance(enc, torch.nn.Sequential)
    assert isinstance(enc[0], torch.nn.LayerNorm)
    assert isinstance(enc[1], torch.nn.Linear)
    assert enc[1].in_features == len(NODE_FEATURE_SCHEMA)


def test_classifier_forward_with_feature_subset():
    """A reordered one-hot feature subset still passes through the plain linear encoder."""
    batch = _build_batch()
    active_features = ["subtree_size", "node_type_operator", "root_color_f", "subtree_depth"]
    sub_x = batch.x[:, [NODE_FEATURE_SCHEMA.index(f) for f in active_features]]
    model = TestGraphNetwork(
        input_dim=len(active_features),
        hidden_dim=32,
        global_dim=5,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        architecture="gatv2_stack",
        active_features=active_features,
    )
    model.eval()
    assert isinstance(model.node_encoder, torch.nn.Sequential)
    with torch.no_grad():
        out = model(
            sub_x, batch.edge_index, batch.batch, batch.global_features, edge_attr=batch.edge_attr
        )
    assert out.shape == (1, 2)
