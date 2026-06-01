import json
import torch
import numpy as np
from pathlib import Path

from graph_utils import ExpressionGraphConverter, TopologicalFeatureExtractor
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
    assert data.x.shape[1] == 19
    assert not hasattr(data, "taylor_coeffs")
    assert not hasattr(data, "inv_taylor_coeffs")


def test_enriched_graph_features(tmp_path):
    # Construct a simple expression tree representing: Plus[x, 2]
    # Root (f1: Plus) -> Children (f2: x, f3: 2)
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

    # 1. Global graph feature assertions
    assert data.nodes == 3
    assert data.num_nodes == 3
    assert data.edges == 2
    assert data.num_edges == 2
    assert data.tree_depth == 1
    assert data.treewidth == 2 or data.tree_width == 2

    # 2. Node feature shape and values
    # Node features: [node_type, depth, height, subtree_size, out_degree, betweenness, label_id, value, LPE1-4, RWPE1-4, cx, fx, yt]
    assert data.x.shape == (3, 19)

    root_idx = 0  # since order of nodes is preserved in networkx
    child1_idx = 1
    child2_idx = 2

    # Verify Root node (f1)
    root_features = data.x[root_idx].tolist()
    assert root_features[0] == 1.0  # type (operator)
    assert root_features[1] == 0.0  # depth
    assert root_features[2] == 1.0  # height
    assert root_features[3] == 3.0  # subtree_size
    assert root_features[4] == 2.0  # out_degree
    assert root_features[5] > 0.0   # root betweenness centrality should be non-zero (since it connects children)
    assert root_features[7] == 0.0  # value

    # Verify Child 2 node (f3)
    child2_features = data.x[child2_idx].tolist()
    assert child2_features[0] == 2.0  # type (constant)
    assert child2_features[1] == 1.0  # depth
    assert child2_features[2] == 0.0  # height
    assert child2_features[3] == 1.0  # subtree_size
    assert child2_features[4] == 0.0  # out_degree
    assert child2_features[5] == 0.0  # leaf betweenness centrality should be 0.0
    assert child2_features[7] == 2.0  # value (0.2 * 10^1 = 2.0)

    # 3. LPE and RWPE Assertions
    # Check that they exist and have valid values (LPE columns are 8-11, RWPE columns are 12-15)
    for idx in range(3):
        node_features = data.x[idx].tolist()
        lpe = node_features[8:12]
        rwpe = node_features[12:16]
        assert len(lpe) == 4
        assert len(rwpe) == 4
        # Verify first step RWPE is exactly 1/degree or 0.0
        # G_und has f1 (deg 2), f2 (deg 1), f3 (deg 1)
        # Transition probabilities: f1->f2 (0.5), f1->f3 (0.5), f2->f1 (1.0), f3->f1 (1.0)
        # Power 1 diagonal should be P_ii which is 0.0 for all nodes (since no self-loops)
        # Power 2 diagonal is D^-1 A D^-1 A
        # For f2: f2->f1 (1.0) -> f2 (0.5). So P^2_{2,2} = 0.5.
        # Let's verify RWPE step 1 (power 2 since step index starts at 0 for step=1 which is P^2)
        # Wait, step index 0 corresponds to P^1, which has diagonal = 0.0
        assert rwpe[0] == 0.0

    # 4. Laplacian Matrix Assertion
    assert hasattr(data, "laplacian")
    assert data.laplacian.shape == (3, 3)
    # Check Laplacian symmetry
    assert torch.allclose(data.laplacian, data.laplacian.T)
    # Undirected degree list is [2, 1, 1], so diagonal of Laplacian should be degrees
    diag = torch.diag(data.laplacian).tolist()
    assert diag == [2.0, 1.0, 1.0]

    # 5. Bidirectional edge assertions
    # 2 forward edges, 2 backward edges = 4 total edges in homogenous graph
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, 4)  # [child_index, direction, relation_type, edge_betweenness]

    # Verify edge features
    directions = data.edge_attr[:, 1].tolist()
    assert directions.count(0.0) == 2
    assert directions.count(1.0) == 2

    # Check child indices
    child_indices = data.edge_attr[:, 0].tolist()
    assert child_indices.count(0.0) == 2
    assert child_indices.count(1.0) == 2

    # Check edge betweenness centrality (column index 3)
    eb = data.edge_attr[:, 3].tolist()
    assert all(val > 0.0 for val in eb)  # All edges in this graph are bridges, betweenness should be non-zero
