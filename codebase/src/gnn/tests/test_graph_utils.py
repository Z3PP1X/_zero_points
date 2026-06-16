import json
import pytest
import torch
import numpy as np
from pathlib import Path

from graph_utils import (
    ExpressionGraphConverter,
    TopologicalFeatureExtractor,
    NODE_FEATURE_SCHEMA,
)
from feature_layout import NATIVE_NODE_FEATURE_COUNT


def test_convert_ignores_legacy_taylor_coeff_fields(tmp_path):
    raw = {
        "id": "P-test",
        "taylorCoeffs": [1.0, 2.0, 3.0],
        "inverseTaylorCoeffs": [4.0, 5.0],
        "nodes": [
            {
                "id": "n1",
                "label": "x",
                "type": "variable",
                "value": {"mantissa": 1.0, "exponent": 0},
            }
        ],
        "edges": [],
    }
    graph_path = tmp_path / "P-test_meta.json"
    graph_path.write_text(json.dumps(raw), encoding="utf-8")

    data = ExpressionGraphConverter().convert(raw)

    assert data.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    assert data.x.shape[1] == len(NODE_FEATURE_SCHEMA)
    assert not hasattr(data, "taylor_coeffs")
    assert not hasattr(data, "inv_taylor_coeffs")


def test_enriched_graph_features(tmp_path):
    raw = {
        "id": "P-enriched-test",
        "nodes": [
            {
                "id": "f1",
                "label": "Plus",
                "type": "operator",
                "value": None
            },
            {
                "id": "f2",
                "label": "x",
                "type": "variable",
                "value": None
            },
            {
                "id": "f3",
                "label": "2",
                "type": "constant",
                "value": {"mantissa": 0.2, "exponent": 1}
            }
        ],
        "edges": [
            {
                "source": "f1",
                "target": "f2",
                "type": "child_of"
            },
            {
                "source": "f1",
                "target": "f3",
                "type": "child_of"
            }
        ]
    }

    converter = ExpressionGraphConverter()
    data = converter.convert(
        raw, heterogeneous=False, mode="tree", edge_direction="bidirectional"
    )

    assert data.nodes == 3
    assert data.num_nodes == 3
    assert data.edges == 2
    assert data.num_edges == 2
    assert data.tree_depth == 1
    assert data.tree_width == 2

    assert data.x.shape == (3, NATIVE_NODE_FEATURE_COUNT)
    assert getattr(data, "edge_attr", None) is None

    root_idx = 0
    child1_idx = 1
    child2_idx = 2

    st_size = NODE_FEATURE_SCHEMA.index("subtree_size")
    st_depth = NODE_FEATURE_SCHEMA.index("subtree_depth")
    hist_add = NODE_FEATURE_SCHEMA.index("hist_additive")
    hist_var = NODE_FEATURE_SCHEMA.index("hist_variables")
    hist_const = NODE_FEATURE_SCHEMA.index("hist_constants")

    root_features = data.x[root_idx].tolist()
    assert root_features[0] == 1.0           # node_type=1 (operator; no global → Plus not marked root)
    assert root_features[1] == 0.0           # root_color=0 (none)
    assert root_features[st_size] == 3.0     # subtree_size: Plus + x + 2
    assert root_features[st_depth] == 1.0    # subtree_depth=1 (height)
    assert root_features[hist_add] == 1.0    # hist_additive: Plus
    assert root_features[hist_var] == 1.0    # hist_variables: x
    assert root_features[hist_const] == 1.0  # hist_constants: 2

    child2_features = data.x[child2_idx].tolist()
    assert child2_features[0] == 1.0         # node_type=1 (all non-global/non-root → operator)
    assert child2_features[1] == 0.0         # root_color=0
    assert child2_features[st_size] == 1.0   # subtree_size=1 (leaf)
    assert child2_features[st_depth] == 0.0  # subtree_depth=0
    assert child2_features[hist_const] == 1.0  # hist_constants: constant leaf

    # Anchor positional encoding (the 5 anchor_* columns): proximity 1/(1+hops) to the
    # nearest operator anchor of each semantic group, within the node's own function.
    # f1 is Plus -> an additive anchor, so it scores 1.0 on anchor_additive and its two
    # children (1 hop away) score 0.5; all other anchor groups are absent here.
    child1_features = data.x[child1_idx].tolist()
    add_col = NODE_FEATURE_SCHEMA.index("anchor_additive")
    assert root_features[add_col] == pytest.approx(1.0)
    assert child1_features[add_col] == pytest.approx(0.5)
    assert child2_features[add_col] == pytest.approx(0.5)
    for name in (
        "anchor_scaling",
        "anchor_periodic",
        "anchor_exponential",
        "anchor_transcendental",
    ):
        col = NODE_FEATURE_SCHEMA.index(name)
        assert root_features[col] == 0.0
        assert child1_features[col] == 0.0
        assert child2_features[col] == 0.0

    assert data.edge_index.shape == (2, 4)


def test_edge_direction_top_down_has_parent_to_child_only():
    raw = {
        "id": "P-direction-test",
        "nodes": [
            {"id": "root", "label": "Plus", "type": "operator", "value": None},
            {"id": "leaf", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [{"source": "root", "target": "leaf", "type": "child_of"}],
    }
    data = ExpressionGraphConverter().convert(
        raw, heterogeneous=False, mode="tree", edge_direction="top_down"
    )
    assert data.edge_index.shape == (2, 1)
    src, dst = data.edge_index[:, 0].tolist()
    assert src == 0 and dst == 1


def test_edge_direction_bottom_up_has_child_to_parent_only():
    raw = {
        "id": "P-direction-test",
        "nodes": [
            {"id": "root", "label": "Plus", "type": "operator", "value": None},
            {"id": "leaf", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [{"source": "root", "target": "leaf", "type": "child_of"}],
    }
    data = ExpressionGraphConverter().convert(
        raw, heterogeneous=False, mode="tree", edge_direction="bottom_up"
    )
    assert data.edge_index.shape == (2, 1)
    src, dst = data.edge_index[:, 0].tolist()
    assert src == 1 and dst == 0


def test_ast_edge_direction_respected_for_belongs_to_edges():
    """With virtual task nodes removed, AST/aggregator edges follow edge_direction."""
    raw = {
        "id": "P-direction-aggregator-test",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "x1", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [
            {"source": "global", "target": "x1", "type": "child_of"},
        ],
    }
    data = ExpressionGraphConverter().convert(
        raw, heterogeneous=False, mode="graph", edge_direction="top_down"
    )
    assert "virtual_current_x" not in data.node_ids
    x1_idx = data.node_ids.index("x1")
    global_idx = data.node_ids.index("global")
    ast_edge_count = sum(
        1
        for edge_idx in range(data.edge_index.size(1))
        if {
            int(data.edge_index[0, edge_idx].item()),
            int(data.edge_index[1, edge_idx].item()),
        }
        == {global_idx, x1_idx}
    )
    assert ast_edge_count == 1
