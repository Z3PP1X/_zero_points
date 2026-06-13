"""Tests for the 1-WL distinguishability study core (gnn.weisfeiler_lehman)."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnn.weisfeiler_lehman.wl_runner import (
    distinguishability_matrix,
    global_color_histogram,
    run_wl,
)


def _path_graph(num_nodes: int, label: int = 0) -> Data:
    """Undirected-style path graph with a constant node label."""
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    data.label_id = torch.full((num_nodes,), label, dtype=torch.long)
    data.node_type = torch.zeros(num_nodes, dtype=torch.long)
    return data


def _star_graph(num_leaves: int, label: int = 0) -> Data:
    num_nodes = num_leaves + 1
    src = [0] * num_leaves
    dst = list(range(1, num_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    data.label_id = torch.full((num_nodes,), label, dtype=torch.long)
    data.node_type = torch.zeros(num_nodes, dtype=torch.long)
    return data


def test_isomorphic_graphs_collide() -> None:
    # Two identical path graphs must land in the same equivalence class.
    graphs = {"a": _path_graph(5), "b": _path_graph(5)}
    result = run_wl(graphs, coloring="constant")
    assert result.num_graphs == 2
    assert result.num_classes == 1
    assert result.fingerprints["a"] == result.fingerprints["b"]
    assert result.distinguishability_rate == 0.5


def test_structurally_different_graphs_are_distinguished() -> None:
    # A path and a star on the same node count differ under 1-WL.
    graphs = {"path": _path_graph(5), "star": _star_graph(4)}
    result = run_wl(graphs, coloring="constant")
    assert result.num_classes == 2
    assert result.fingerprints["path"] != result.fingerprints["star"]

    ordered, matrix = distinguishability_matrix(result)
    i, j = ordered.index("path"), ordered.index("star")
    assert matrix[i, j]  # distinguishable
    assert not matrix[i, i]  # a graph is never distinguishable from itself


def test_label_coloring_separates_otherwise_identical_structure() -> None:
    # Same structure, different semantic labels -> distinguishable by 'label'.
    graphs = {"x": _path_graph(4, label=1), "y": _path_graph(4, label=2)}
    by_label = run_wl(graphs, coloring="label")
    assert by_label.num_classes == 2
    # Pure structure (constant colour) cannot tell them apart.
    by_struct = run_wl(graphs, coloring="constant")
    assert by_struct.num_classes == 1


def test_summary_table_exact_format() -> None:
    from gnn.weisfeiler_lehman.main import format_summary_table

    summaries = [
        {
            "label": "synthetic-graph",
            "num_classes": 27,
            "num_graphs": 100,
            "num_colliding_graphs": 83,
            "distinguishability_rate": 0.27,
        },
        {
            "label": "synthetic-tree",
            "num_classes": 24,
            "num_graphs": 100,
            "num_colliding_graphs": 87,
            "distinguishability_rate": 0.24,
        },
        {
            "label": "synthetic-tree-derivative",
            "num_classes": 27,
            "num_graphs": 100,
            "num_colliding_graphs": 83,
            "distinguishability_rate": 0.27,
        },
    ]
    table = format_summary_table(summaries)
    lines = table.split("\n")
    separator = "├" + "─" * 27 + "┼" + "─" * 25 + "┼" + "─" * 18 + "┼" + "─" * 6 + "┤"

    # Header + 3 * (separator + row) = 7 lines.
    assert len(lines) == 7
    assert lines[1] == separator
    assert lines[2] == (
        "│ synthetic-graph           │ 27 / 100                "
        "│ 83               │ 0.27 │"
    )
    assert lines[4] == (
        "│ synthetic-tree            │ 24 / 100                "
        "│ 87               │ 0.24 │"
    )
    assert lines[6] == (
        "│ synthetic-tree-derivative │ 27 / 100                "
        "│ 83               │ 0.27 │"
    )
    # Header carries the four column titles.
    for title in ("mode", "distinguishable classes", "colliding graphs", "rate"):
        assert title in lines[0]


def test_global_histogram_shapes() -> None:
    graphs = {"path": _path_graph(5), "star": _star_graph(4)}
    result = run_wl(graphs, coloring="constant")
    color_ids, matrix = global_color_histogram(result, round_index=-1)
    assert set(matrix) == {"path", "star"}
    for vec in matrix.values():
        assert vec.shape[0] == len(color_ids)
    # Node counts are conserved by the histogram.
    assert int(matrix["path"].sum()) == 5
    assert int(matrix["star"].sum()) == 5
