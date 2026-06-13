"""Unit tests for the shared TwoWayFeatureEncoder (categorical + linear embeddings)."""

import torch

from gnn.shared.models.feature_encoders import TwoWayFeatureEncoder
from gnn.shared.utils.feature_config import (
    EDGE_CATEGORICAL_REGISTRY,
    NODE_CATEGORICAL_REGISTRY,
    full_node_schema,
)
from gnn.shared.utils.graph_utils import (
    EDGE_FEATURE_SCHEMA,
    NODE_FEATURE_SCHEMA,
    NUM_EDGE_TYPES,
)


def test_full_schema_node_encoder_shapes_and_node_types():
    enc = TwoWayFeatureEncoder(full_node_schema(), 64, NODE_CATEGORICAL_REGISTRY)
    x = torch.randn(5, len(NODE_FEATURE_SCHEMA))
    x[:, 0] = torch.tensor([0.0, 1.0, 2.0, 5.0, 5.0])   # node_type (global/op/root/supernode)
    x[:, 1] = torch.tensor([0.0, 0.0, 1.0, 2.0, 4.0])   # root_color (none/f/d1/kappa)
    out, node_types = enc(x)
    assert out.shape == (5, 64)
    # node_type ids are returned for RL routing.
    assert node_types.tolist() == [0, 1, 2, 5, 5]
    assert set(enc.embeddings.keys()) == {"node_type", "root_color"}


def test_subset_reorder_still_embeds_categoricals():
    """Categoricals are located BY NAME, so a reordered subset that does not start
    with node_type/root_color still embeds them (no plain-linear fallback)."""
    names = ["root_color", "subtree_size", "node_type", "hist_additive"]
    enc = TwoWayFeatureEncoder(names, 32, NODE_CATEGORICAL_REGISTRY)
    assert set(enc.embeddings.keys()) == {"node_type", "root_color"}
    x = torch.zeros(3, 4)
    x[:, 0] = torch.tensor([0.0, 1.0, 2.0])   # root_color at col 0
    x[:, 2] = torch.tensor([1.0, 2.0, 0.0])   # node_type at col 2
    out, node_types = enc(x)
    assert out.shape == (3, 32)
    assert node_types.tolist() == [1, 2, 0]  # read from col 2, not col 0


def test_subset_without_categoricals_is_pure_continuous():
    names = ["depth", "height", "value"]
    enc = TwoWayFeatureEncoder(names, 16, NODE_CATEGORICAL_REGISTRY)
    assert len(enc.embeddings) == 0
    out, node_types = enc(torch.randn(4, 3))
    assert out.shape == (4, 16)
    # No node_type column -> zeros placeholder.
    assert node_types.tolist() == [0, 0, 0, 0]


def test_edge_encoder_embeds_relation_type():
    enc = TwoWayFeatureEncoder(list(EDGE_FEATURE_SCHEMA), 8, EDGE_CATEGORICAL_REGISTRY)
    assert set(enc.embeddings.keys()) == {"relation_type"}
    ea = torch.randn(7, len(EDGE_FEATURE_SCHEMA))
    ea[:, EDGE_FEATURE_SCHEMA.index("relation_type")] = torch.randint(
        0, NUM_EDGE_TYPES, (7,)
    ).float()
    out, _ = enc(ea)
    assert out.shape == (7, 8)


def test_out_of_range_categorical_id_is_clamped():
    enc = TwoWayFeatureEncoder(["node_type", "value"], 16, NODE_CATEGORICAL_REGISTRY)
    x = torch.zeros(2, 2)
    x[:, 0] = torch.tensor([999.0, -5.0])  # out-of-range ids must not crash
    out, node_types = enc(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
