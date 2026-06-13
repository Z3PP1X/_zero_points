"""Tests for the opt-in fully-connected virtual supernode (add_virtual_supernode).

The supernode is a selectable graph augmentation that adds a single node with
bidirectional edges to every other node, shortening message-passing paths. It is
threaded through the loaders exactly like add_kappa and toggled per training run in
the RL, supervised and GraphGym workflows.
"""

import pytest

from graph_utils import (
    EDGE_FEATURE_SCHEMA,
    ExpressionGraphConverter,
    SUPERNODE_NODE_ID,
    SUPERNODE_NODE_TYPE,
    encode_edge_type,
)


@pytest.fixture
def sample_raw():
    # Expression: Plus[x, 2]
    return {
        "id": "P-supernode-test-1",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "x", "type": "variable", "value": None},
            {
                "id": "f3",
                "label": "2",
                "type": "constant",
                "value": {"mantissa": 0.2, "exponent": 1},
            },
        ],
        "edges": [
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f1", "target": "f3", "type": "child_of"},
            {"source": "global", "target": "f1", "type": "belongs_to_f"},
        ],
    }


def test_supernode_absent_by_default(sample_raw):
    """Without the flag the graph is unchanged (backward compatible)."""
    converter = ExpressionGraphConverter()
    data = converter.convert(sample_raw, mode="graph", edge_direction="top_down")
    assert SUPERNODE_NODE_ID not in data.node_ids


def test_supernode_added_and_typed(sample_raw):
    converter = ExpressionGraphConverter()
    base = converter.convert(sample_raw, mode="graph", edge_direction="top_down")
    data = converter.convert(
        sample_raw,
        mode="graph",
        edge_direction="top_down",
        add_virtual_supernode=True,
    )

    # Exactly one extra node (the supernode), appended last.
    assert SUPERNODE_NODE_ID in data.node_ids
    assert data.node_ids[-1] == SUPERNODE_NODE_ID
    assert len(data.node_ids) == len(base.node_ids) + 1

    # The supernode carries its dedicated node_type code (distinct from 6/9/10).
    sn_idx = data.node_ids.index(SUPERNODE_NODE_ID)
    assert int(data.node_type[sn_idx]) == SUPERNODE_NODE_TYPE
    # Feature/tensor alignment holds across all node-derived tensors.
    assert data.x.size(0) == len(data.node_ids)
    assert data.node_type.size(0) == len(data.node_ids)
    assert data.num_nodes == len(data.node_ids)


def test_supernode_bidirectionally_connected_to_all(sample_raw):
    converter = ExpressionGraphConverter()
    data = converter.convert(
        sample_raw,
        mode="graph",
        edge_direction="top_down",
        add_virtual_supernode=True,
    )
    sn_idx = data.node_ids.index(SUPERNODE_NODE_ID)
    edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))

    other_indices = [i for i in range(len(data.node_ids)) if i != sn_idx]
    for i in other_indices:
        assert (sn_idx, i) in edges, f"missing supernode->{data.node_ids[i]}"
        assert (i, sn_idx) in edges, f"missing {data.node_ids[i]}->supernode"

    # The supernode edges use the reserved supernode_connection relation types.
    fwd = float(encode_edge_type("supernode_connection"))
    rev = float(encode_edge_type("supernode_connection_reverse"))
    rel_col = EDGE_FEATURE_SCHEMA.index("relation_type")
    edge_list = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    rels = {
        (u, v): float(data.edge_attr[k, rel_col])
        for k, (u, v) in enumerate(edge_list)
    }
    a_idx = other_indices[0]
    assert rels[(sn_idx, a_idx)] == fwd
    assert rels[(a_idx, sn_idx)] == rev


@pytest.mark.parametrize("mode", ["graph", "tree", "tree_derivatives"])
def test_supernode_works_in_all_modes(sample_raw, mode):
    converter = ExpressionGraphConverter()
    data = converter.convert(
        sample_raw, mode=mode, edge_direction="top_down", add_virtual_supernode=True
    )
    assert SUPERNODE_NODE_ID in data.node_ids


def test_supernode_heterogeneous(sample_raw):
    """In the heterogeneous representation the supernode joins the 'virtual' type and
    its edges form supernode_connection metapaths."""
    converter = ExpressionGraphConverter()
    data = converter.convert(
        sample_raw,
        heterogeneous=True,
        mode="graph",
        edge_direction="top_down",
        add_virtual_supernode=True,
    )
    assert SUPERNODE_NODE_ID in data["virtual"].node_ids
    metapaths = [rel for (_, rel, _) in data.edge_types]
    assert "supernode_connection" in metapaths
    assert "supernode_connection_reverse" in metapaths
