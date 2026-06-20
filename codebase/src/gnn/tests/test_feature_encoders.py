"""Unit tests for TwoWayFeatureEncoder (categorical + linear embeddings).

These tests supply their own custom categorical registry so they are decoupled
from the production NODE_CATEGORICAL_REGISTRY, which is now empty because all
node features are one-hot encoded directly.
"""

import torch

from gnn.shared.models.feature_encoders import TwoWayFeatureEncoder
from gnn.shared.utils.feature_config import NODE_CATEGORICAL_REGISTRY
from gnn.shared.utils.graph_utils import NODE_FEATURE_SCHEMA

# Custom registry for TwoWayFeatureEncoder unit tests.
_CAT_REGISTRY = {
    "cat_a": (6, 8),   # vocab_size=6, emb_dim=8
    "cat_b": (3, 4),
}


def test_full_schema_no_categoricals_is_pure_linear():
    """With the production empty registry, the encoder is a plain linear projection."""
    enc = TwoWayFeatureEncoder(list(NODE_FEATURE_SCHEMA), 64, NODE_CATEGORICAL_REGISTRY)
    assert len(enc.embeddings) == 0
    x = torch.randn(5, len(NODE_FEATURE_SCHEMA))
    out, node_types = enc(x)
    assert out.shape == (5, 64)
    assert torch.isfinite(out).all()
    # No node_type column in the empty registry → placeholder zeros.
    assert node_types.tolist() == [0, 0, 0, 0, 0]


def test_custom_registry_embeds_categoricals():
    """Categoricals are located BY NAME in a custom registry."""
    names = ["cat_b", "continuous_x", "cat_a", "continuous_y"]
    enc = TwoWayFeatureEncoder(names, 32, _CAT_REGISTRY)
    assert set(enc.embeddings.keys()) == {"cat_a", "cat_b"}
    x = torch.zeros(3, 4)
    x[:, 0] = torch.tensor([0.0, 1.0, 2.0])   # cat_b at col 0
    x[:, 2] = torch.tensor([1.0, 3.0, 0.0])   # cat_a at col 2
    out, node_types = enc(x)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()
    # node_types defaults to zeros (no "node_type" key in custom registry).
    assert node_types.tolist() == [0, 0, 0]


def test_subset_without_categoricals_is_pure_continuous():
    """A schema with no categorical columns uses only linear layers."""
    names = ["depth", "height", "value"]
    enc = TwoWayFeatureEncoder(names, 16, _CAT_REGISTRY)
    assert len(enc.embeddings) == 0
    out, node_types = enc(torch.randn(4, 3))
    assert out.shape == (4, 16)
    assert node_types.tolist() == [0, 0, 0, 0]


def test_out_of_range_categorical_id_is_clamped():
    """Out-of-range category ids are clamped and must not crash."""
    enc = TwoWayFeatureEncoder(["cat_a", "value"], 16, _CAT_REGISTRY)
    x = torch.zeros(2, 2)
    x[:, 0] = torch.tensor([999.0, -5.0])  # out-of-range cat_a ids
    out, node_types = enc(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
