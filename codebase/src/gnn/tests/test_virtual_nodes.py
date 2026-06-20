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

# One-hot column indices in the 28-col schema
_NT_ROOT = NODE_FEATURE_SCHEMA.index("node_type_root")        # 2
_RC_F = NODE_FEATURE_SCHEMA.index("root_color_f")              # 5


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
    data = converter.convert(raw, mode="graph")

    # 4 nodes: global + f1 + f2 + f3 (no aggregator nodes)
    assert data.num_nodes == 4
    assert data.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    assert len(data.node_ids) == 4
    assert "f_root" not in data.node_ids
    assert "virtual_current_x" not in data.node_ids

    # f1 (Plus) is now the root node (node_type_root=1.0, root_color_f=1.0)
    idx_f1 = data.node_ids.index("f1")
    assert data.x[idx_f1, _NT_ROOT].item() == 1.0   # node_type_root
    assert data.x[idx_f1, _RC_F].item() == 1.0       # root_color_f


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

    preprocessor_graph = Preprocessor(graphs_dir=str(tmp_path), mode="graph")

    message = {
        "id": "P-dynamic",
        "currentX": 1.5,
        "fx": -2.7,
        "yTarget": 0.0,
        "uuid": "test-uuid-123"
    }

    data_graph, extracted = preprocessor_graph.process(message)

    assert "f_root" not in data_graph.node_ids
    assert data_graph.num_nodes == 2   # global + f1
    assert data_graph.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    idx_f1 = data_graph.node_ids.index("f1")
    assert data_graph.x[idx_f1, _NT_ROOT].item() == 1.0   # node_type_root
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

    base_graph_graph = converter.convert(raw, mode="graph")
    base_graphs_graph = {"P-supervised": base_graph_graph}

    assert "f_root" not in base_graph_graph.node_ids
    assert base_graph_graph.num_nodes == 2
    idx_f1 = base_graph_graph.node_ids.index("f1")
    assert base_graph_graph.x[idx_f1, _NT_ROOT].item() == 1.0   # node_type_root
    assert base_graph_graph.x[idx_f1, _RC_F].item() == 1.0       # root_color_f

    df_no_fx = pd.DataFrame([{
        "problem_id": "P-supervised",
        "x0": 2.5,
        "y_target": 4.0,
        "faster_algorithm": 1
    }])

    dataset_no_fx_graph = ProblemRunDataset(df_no_fx, base_graphs_graph, mode="graph")
    data_no_fx_graph = dataset_no_fx_graph[0]

    assert data_no_fx_graph.x.shape == base_graph_graph.x.shape


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

    data_full = converter.convert(raw, mode="graph")
    assert data_full.x.shape[1] == NATIVE_NODE_FEATURE_COUNT

    from graph_utils import slice_active_features
    data_sliced = data_full.clone()
    data_sliced.x = slice_active_features(data_full.x, active_feats)

    assert data_sliced.x.shape[1] == 3

    meta_path = tmp_path / "P-slice_meta.json"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    preprocessor = Preprocessor(graphs_dir=str(tmp_path), mode="graph", active_features=active_feats)

    message = {
        "id": "P-slice",
        "currentX": 3.14,
        "fx": -1.2,
        "yTarget": 0.5,
        "uuid": "uuid-slice"
    }

    data, extracted = preprocessor.process(message)
    assert data.x.shape[1] == 3
