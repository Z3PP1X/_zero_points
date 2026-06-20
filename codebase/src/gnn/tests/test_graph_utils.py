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
        raw,  mode="tree", edge_direction="bidirectional"
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
    child1_idx = 1   # x (variable)
    child2_idx = 2   # 2 (constant)

    nt_operator = NODE_FEATURE_SCHEMA.index("node_type_operator")
    nt_global   = NODE_FEATURE_SCHEMA.index("node_type_global")
    rc_none     = NODE_FEATURE_SCHEMA.index("root_color_none")
    st_size     = NODE_FEATURE_SCHEMA.index("subtree_size")
    st_depth    = NODE_FEATURE_SCHEMA.index("subtree_depth")
    hist_trig   = NODE_FEATURE_SCHEMA.index("hist_trigonometric")
    hist_var    = NODE_FEATURE_SCHEMA.index("hist_variables")
    hist_const  = NODE_FEATURE_SCHEMA.index("hist_constants")

    root_features   = data.x[root_idx].tolist()
    child1_features = data.x[child1_idx].tolist()
    child2_features = data.x[child2_idx].tolist()

    # node_type: Plus / x / 2 are all code-1 (operator) since there is no global node
    assert root_features[nt_operator] == 1.0
    assert root_features[nt_global]   == 0.0
    assert root_features[rc_none]     == 1.0   # no root_color marking without global

    assert root_features[st_size]  == 3.0   # Plus + x + 2
    assert root_features[st_depth] == 1.0   # height = 1
    assert root_features[hist_trig]  == 0.0  # no trig node in subtree
    assert root_features[hist_var]   == 1.0  # x in subtree
    # Plus (operator, not in HISTOGRAM_GROUP_BY_LABEL) → hist_constants; plus literal 2
    assert root_features[hist_const] == 2.0

    assert child2_features[nt_operator] == 1.0
    assert child2_features[rc_none]     == 1.0
    assert child2_features[st_size]  == 1.0   # leaf
    assert child2_features[st_depth] == 0.0
    assert child2_features[hist_const] == 1.0  # constant leaf

    # Anchor PE (new 3-group scheme): proximity 1/(1+hops) to the nearest node in each
    # anchor group. The tree is Plus[x, 2].  x is an anchor (anchor_variable group);
    # E/Log/Sin/Cos/Tan are absent so anchor_trigonometric and anchor_exponential = 0.
    anc_trig = NODE_FEATURE_SCHEMA.index("anchor_trigonometric")
    anc_exp  = NODE_FEATURE_SCHEMA.index("anchor_exponential")
    anc_var  = NODE_FEATURE_SCHEMA.index("anchor_variable")

    # Plus: 1 hop to x → anchor_variable = 0.5; no trig/exp anchors
    assert root_features[anc_var]  == pytest.approx(0.5)
    assert root_features[anc_trig] == 0.0
    assert root_features[anc_exp]  == 0.0

    # x: is the variable anchor → anchor_variable = 1.0
    assert child1_features[anc_var]  == pytest.approx(1.0)
    assert child1_features[anc_trig] == 0.0
    assert child1_features[anc_exp]  == 0.0

    # 2 (constant): 2 hops to x → anchor_variable = 1/3
    assert child2_features[anc_var]  == pytest.approx(1.0 / 3.0)
    assert child2_features[anc_trig] == 0.0
    assert child2_features[anc_exp]  == 0.0

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
        raw,  mode="tree", edge_direction="top_down"
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
        raw,  mode="tree", edge_direction="bottom_up"
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
        raw,  mode="graph", edge_direction="top_down"
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
