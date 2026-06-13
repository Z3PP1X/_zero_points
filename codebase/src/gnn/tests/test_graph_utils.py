import json
import pytest
import torch
import numpy as np
from pathlib import Path

from graph_utils import (
    ExpressionGraphConverter,
    TopologicalFeatureExtractor,
    NODE_FEATURE_SCHEMA,
    EDGE_FEATURE_SCHEMA,
    CANONICAL_LABEL_VOCAB,
)
from feature_layout import NATIVE_NODE_FEATURE_COUNT, NATIVE_EDGE_FEATURE_COUNT


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
    assert data.treewidth == 2 or data.tree_width == 2

    assert data.x.shape == (3, NATIVE_NODE_FEATURE_COUNT)
    assert data.edge_attr.shape == (4, NATIVE_EDGE_FEATURE_COUNT)
    assert data.edge_attr.shape[1] == len(EDGE_FEATURE_SCHEMA)

    root_idx = 0
    child1_idx = 1
    child2_idx = 2

    root_features = data.x[root_idx].tolist()
    assert root_features[0] == 1.0
    assert root_features[1] == float(CANONICAL_LABEL_VOCAB["Plus"])
    assert root_features[2] == 0.0
    assert root_features[3] == 1.0
    assert root_features[4] == 3.0
    assert root_features[5] == 2.0
    assert root_features[6] > 0.0
    assert root_features[7] == 0.0
    assert root_features[8] == 0.0

    child2_features = data.x[child2_idx].tolist()
    assert child2_features[0] == 2.0
    assert child2_features[1] == float(CANONICAL_LABEL_VOCAB["<CONSTANT>"])
    assert child2_features[2] == 1.0
    assert child2_features[3] == 0.0
    assert child2_features[4] == 1.0
    assert child2_features[5] == 0.0
    assert child2_features[6] == 0.0
    # value is now emitted RAW (signed_log normalization removed; the model's
    # learnable linear embedding handles scaling).
    assert child2_features[7] == pytest.approx(2.0)
    assert child2_features[8] == 1.0

    for idx in range(3):
        node_features = data.x[idx].tolist()
        lpe = node_features[9:13]
        rwpe = node_features[13:17]
        assert len(lpe) == 4
        assert len(rwpe) == 4
        # Lazy random-walk return probability (step 2) is strictly positive for
        # every node, including on bipartite trees. This guards against the
        # earlier regression where odd-step RWPE dims were identically zero.
        assert rwpe[0] > 0.0

    assert hasattr(data, "laplacian")
    assert data.laplacian.shape == (3, 3)
    assert torch.allclose(data.laplacian, data.laplacian.T)
    diag = torch.diag(data.laplacian).tolist()
    assert diag == [2.0, 1.0, 1.0]

    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, 4)

    directions = data.edge_attr[:, 1].tolist()
    assert directions.count(0.0) == 2
    assert directions.count(1.0) == 2

    child_indices = data.edge_attr[:, 0].tolist()
    assert child_indices.count(0.0) == 2
    assert child_indices.count(1.0) == 2

    eb = data.edge_attr[:, 3].tolist()
    assert all(val > 0.0 for val in eb)


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
