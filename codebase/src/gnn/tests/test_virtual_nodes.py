import json
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx

from graph_utils import ExpressionGraphConverter, NODE_FEATURE_SCHEMA
from feature_layout import NATIVE_NODE_FEATURE_COUNT
from reinforcement_learning.preprocessor import Preprocessor
from supervised_learning.preprocessing import ProblemRunDataset

# One-hot column indices in the 32-col schema
_NT_OP   = NODE_FEATURE_SCHEMA.index("node_type_operator")    # 1
_NT_FUNC = NODE_FEATURE_SCHEMA.index("node_type_function")    # 2
_RC_F    = NODE_FEATURE_SCHEMA.index("root_color_f")          # 5


def test_node_counts_and_task_features_on_aggregator():
    raw = {
        "id": "P-node-count-test",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "x", "type": "variable", "value": None},
            {"id": "f3", "label": "2", "type": "constant", "value": {"mantissa": 0.2, "exponent": 1}}
        ],
        "edges": [
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f1", "target": "f3", "type": "child_of"},
            {"source": "global", "target": "f1", "type": "belongs_to_f"}
        ]
    }

    converter = ExpressionGraphConverter()
    data = converter.convert(raw, mode="tree_derivatives")

    # 4 nodes: global + f1 + f2 + f3 (no aggregator nodes)
    assert data.num_nodes == 4
    assert data.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    assert len(data.node_ids) == 4
    assert "f_root" not in data.node_ids
    assert "virtual_current_x" not in data.node_ids

    # f1 (Plus) is the root node — operator type, identified by root_color_f
    idx_f1 = data.node_ids.index("f1")
    assert data.x[idx_f1, _NT_OP].item() == 1.0    # node_type_operator (Plus)
    assert data.x[idx_f1, _RC_F].item() == 1.0      # root_color_f


def test_reinforcement_learning_preprocessor_dynamic_updates(tmp_path):
    raw = {
        "id": "P-dynamic",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "x", "type": "variable", "value": None}
        ],
        "edges": []
    }

    meta_path = tmp_path / "P-dynamic_meta.json"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    preprocessor_graph = Preprocessor(graphs_dir=str(tmp_path), mode="tree_derivatives")

    message = {
        "id": "P-dynamic",
        "currentX": 1.5,
        "fx": -2.7,
        "yTarget": 0.0,
        # kappa is now a required live-state key (drives per-step augmentation);
        # 0 = no kappa subgraph, so the base graph is unchanged (num_nodes == 2).
        "kappa": 0,
        "uuid": "test-uuid-123"
    }

    data_graph, extracted = preprocessor_graph.process(message)

    assert "f_root" not in data_graph.node_ids
    assert data_graph.num_nodes == 2   # global + f1
    assert data_graph.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    idx_f1 = data_graph.node_ids.index("f1")
    assert data_graph.x[idx_f1, _NT_FUNC].item() == 1.0   # node_type_function (x)
    assert data_graph.x[idx_f1, _RC_F].item() == 1.0       # root_color_f

    preprocessor_tree = Preprocessor(graphs_dir=str(tmp_path), mode="tree")
    data_tree, extracted = preprocessor_tree.process(message)
    assert data_tree.num_nodes == 2
    assert "f_root" not in data_tree.node_ids


def test_supervised_learning_preprocessor_static_initialization():
    raw = {
        "id": "P-supervised",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "x", "type": "variable", "value": None}
        ],
        "edges": []
    }

    converter = ExpressionGraphConverter()

    base_graph_graph = converter.convert(raw, mode="tree_derivatives")
    base_graphs_graph = {"P-supervised": base_graph_graph}

    assert "f_root" not in base_graph_graph.node_ids
    assert base_graph_graph.num_nodes == 2
    idx_f1 = base_graph_graph.node_ids.index("f1")
    assert base_graph_graph.x[idx_f1, _NT_FUNC].item() == 1.0   # node_type_function (x)
    assert base_graph_graph.x[idx_f1, _RC_F].item() == 1.0       # root_color_f

    df_no_fx = pd.DataFrame([{
        "problem_id": "P-supervised",
        "x0": 2.5,
        "y_target": 4.0,
        "faster_algorithm": 1
    }])

    dataset_no_fx_graph = ProblemRunDataset(df_no_fx, base_graphs_graph, mode="tree_derivatives")
    data_no_fx_graph = dataset_no_fx_graph[0]

    assert data_no_fx_graph.x.shape == base_graph_graph.x.shape
    # Without scalar_features no global vector is attached (pure structural graph).
    assert not hasattr(data_no_fx_graph, "global_features") or data_no_fx_graph.global_features is None


def _single_node_base_graphs(pid: str):
    raw = {
        "id": pid,
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [],
    }
    return {pid: ExpressionGraphConverter().convert(raw, mode="tree_derivatives")}


def test_supervised_scalar_features_attach_and_collate():
    """Scalars are attached as [1, k] and PyG collate stacks them to [num_graphs, k]."""
    from torch_geometric.loader import DataLoader
    from supervised_learning.preprocessing import _parse_scalar

    base_graphs = _single_node_base_graphs("P-scalars")
    scalar_cols = ["x0", "y_target", "fx", "d1x", "d2x"]
    df = pd.DataFrame([
        {"problem_id": "P-scalars", "faster_algorithm": 1,
         "x0": 2.5, "y_target": 4.0, "fx": "1/2", "d1x": -3.0, "d2x": None},
        {"problem_id": "P-scalars", "faster_algorithm": 0,
         "x0": 1.0, "y_target": 0.0, "fx": 9.0, "d1x": 0.5, "d2x": 7.0},
    ])

    dataset = ProblemRunDataset(df, base_graphs, mode="tree_derivatives", scalar_features=scalar_cols)
    item = dataset[0]
    assert tuple(item.global_features.shape) == (1, len(scalar_cols))
    # Fraction string "1/2" -> 0.5 ; missing None -> 0.0.
    assert torch.allclose(
        item.global_features, torch.tensor([[2.5, 4.0, 0.5, -3.0, 0.0]])
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert tuple(batch.global_features.shape) == (2, len(scalar_cols))

    assert _parse_scalar("1/4") == 0.25
    assert _parse_scalar(None) == 0.0
    assert _parse_scalar("") == 0.0
    assert _parse_scalar(float("nan")) == 0.0


def test_supervised_scalar_features_missing_column_raises():
    """A requested scalar column absent from the dataframe fails loudly (no silent skip)."""
    base_graphs = _single_node_base_graphs("P-missing")
    df = pd.DataFrame([{"problem_id": "P-missing", "faster_algorithm": 1, "x0": 1.0}])
    with pytest.raises(RuntimeError, match="scalar_features"):
        ProblemRunDataset(df, base_graphs, mode="tree_derivatives", scalar_features=["x0", "fx"])


def test_dynamic_feature_slicing_and_selection(tmp_path):
    raw = {
        "id": "P-slice",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "x", "type": "variable", "value": None}
        ],
        "edges": []
    }

    converter = ExpressionGraphConverter()

    # Use new one-hot column names
    active_feats = ["node_type_operator", "root_color_f", "subtree_size"]

    data_full = converter.convert(raw, mode="tree_derivatives")
    assert data_full.x.shape[1] == NATIVE_NODE_FEATURE_COUNT

    from graph_utils import slice_active_features
    data_sliced = data_full.clone()
    data_sliced.x = slice_active_features(data_full.x, active_feats)

    assert data_sliced.x.shape[1] == 3

    meta_path = tmp_path / "P-slice_meta.json"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    preprocessor = Preprocessor(graphs_dir=str(tmp_path), mode="tree_derivatives", active_features=active_feats)

    message = {
        "id": "P-slice",
        "currentX": 3.14,
        "fx": -1.2,
        "yTarget": 0.5,
        "kappa": 0,  # required live-state key; 0 = no kappa subgraph
        "uuid": "uuid-slice"
    }

    data, extracted = preprocessor.process(message)
    assert data.x.shape[1] == 3
