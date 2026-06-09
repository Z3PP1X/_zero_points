import json
import pytest
import torch
import numpy as np
from pathlib import Path

from graph_utils import (
    ExpressionGraphConverter,
    TopologicalFeatureExtractor,
    ENRICHED_NODE_FEATURE_SCHEMA,
    ENRICHED_EDGE_FEATURE_SCHEMA,
    CANONICAL_LABEL_VOCAB,
    signed_log_value,
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
    assert data.x.shape[1] == len(ENRICHED_NODE_FEATURE_SCHEMA)
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
    data = converter.convert(raw, heterogeneous=False, mode="tree")

    assert data.nodes == 3
    assert data.num_nodes == 3
    assert data.edges == 2
    assert data.num_edges == 2
    assert data.tree_depth == 1
    assert data.treewidth == 2 or data.tree_width == 2

    assert data.x.shape == (3, NATIVE_NODE_FEATURE_COUNT)
    assert data.edge_attr.shape == (4, NATIVE_EDGE_FEATURE_COUNT)
    assert data.edge_attr.shape[1] == len(ENRICHED_EDGE_FEATURE_SCHEMA)

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
    assert child2_features[7] == pytest.approx(signed_log_value(2.0))
    assert child2_features[8] == 1.0

    for idx in range(3):
        node_features = data.x[idx].tolist()
        lpe = node_features[9:13]
        rwpe = node_features[13:17]
        assert len(lpe) == 4
        assert len(rwpe) == 4
        assert rwpe[0] == 0.0

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
