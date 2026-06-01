import json
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from graph_utils import ExpressionGraphConverter
from reinforcement_learning.preprocessor import Preprocessor
from supervised_learning.preprocessing import ProblemRunDataset


def test_virtual_nodes_injection_and_connections():
    # Construct a simple expression representing: Plus[x, 2]
    # Root (f1: Plus) -> Children (f2: x [variable], f3: 2 [constant])
    # Also includes standard global node
    raw = {
        "id": "P-virtual-test",
        "nodes": [
            {
                "id": "global",
                "label": "GLOBAL",
                "type": "global",
                "value": None
            },
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
            },
            {
                "source": "global",
                "target": "f1",
                "type": "belongs_to_f"
            }
        ]
    }

    # Test basic (supervised, enrich=False) conversion in Graph Mode
    converter = ExpressionGraphConverter()
    data_basic_graph = converter.convert(raw, heterogeneous=False, enrich=False, mode="graph")

    # We expect 4 original nodes + 3 virtual nodes = 7 nodes, with 8 feature dimensions
    assert data_basic_graph.num_nodes == 7
    assert data_basic_graph.x.shape[1] == 8
    assert len(data_basic_graph.node_ids) == 7
    assert "virtual_current_x" in data_basic_graph.node_ids
    assert "virtual_f_x" in data_basic_graph.node_ids
    assert "virtual_y_target" in data_basic_graph.node_ids

    idx_cx = data_basic_graph.node_ids.index("virtual_current_x")
    idx_fx = data_basic_graph.node_ids.index("virtual_f_x")
    idx_yt = data_basic_graph.node_ids.index("virtual_y_target")

    # In basic mode (8 features): [node_type, label_id, value, has_value, degree_centrality, cx, fx, yt]
    assert data_basic_graph.x[idx_cx, 0].item() == 5.0
    assert data_basic_graph.x[idx_fx, 0].item() == 6.0
    assert data_basic_graph.x[idx_yt, 0].item() == 7.0

    # Test basic conversion in Tree Mode
    data_basic_tree = converter.convert(raw, heterogeneous=False, enrich=False, mode="tree")
    assert data_basic_tree.num_nodes == 4
    assert data_basic_tree.x.shape[1] == 8
    assert len(data_basic_tree.node_ids) == 4
    assert "virtual_current_x" not in data_basic_tree.node_ids

    # Test rich (RL, enrich=True) conversion in Graph Mode
    data_rich_graph = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph")
    assert data_rich_graph.num_nodes == 7
    assert data_rich_graph.x.shape[1] == 19
    assert data_rich_graph.x[idx_cx, 0].item() == 5.0

    # Test rich conversion in Tree Mode
    data_rich_tree = converter.convert(raw, heterogeneous=False, enrich=True, mode="tree")
    assert data_rich_tree.num_nodes == 4
    assert data_rich_tree.x.shape[1] == 19


def test_reinforcement_learning_preprocessor_dynamic_updates(tmp_path):
    # Setup raw graph files inside mock graphs_dir
    raw = {
        "id": "P-dynamic",
        "nodes": [
            {
                "id": "global",
                "label": "GLOBAL",
                "type": "global",
                "value": None
            },
            {
                "id": "f1",
                "label": "x",
                "type": "variable",
                "value": None
            }
        ],
        "edges": []
    }
    
    meta_path = tmp_path / "P-dynamic_meta.json"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")
    
    # 1. Test Preprocessor in Graph Mode
    preprocessor_graph = Preprocessor(graphs_dir=str(tmp_path), mode="graph")
    
    message = {
        "id": "P-dynamic",
        "currentX": 1.5,
        "fx": -2.7,
        "yTarget": 0.0,
        "uuid": "test-uuid-123"
    }
    
    data_graph, extracted = preprocessor_graph.process(message)
    
    idx_cx = data_graph.node_ids.index("virtual_current_x")
    idx_fx = data_graph.node_ids.index("virtual_f_x")
    idx_yt = data_graph.node_ids.index("virtual_y_target")
    
    # In preprocessor, enrich=True is used by default (node features = 19)
    # Value column is index 7
    assert data_graph.x[idx_cx, 7].item() == pytest.approx(1.5)
    assert data_graph.x[idx_fx, 7].item() == pytest.approx(-2.7)
    assert data_graph.x[idx_yt, 7].item() == pytest.approx(0.0)

    # 2. Test Preprocessor in Tree Mode
    preprocessor_tree = Preprocessor(graphs_dir=str(tmp_path), mode="tree")
    data_tree, extracted = preprocessor_tree.process(message)
    assert data_tree.num_nodes == 2  # global + variable
    
    idx_global = data_tree.node_ids.index("global")
    # Tree mode: custom slots are at indices 16, 17, 18
    assert data_tree.x[idx_global, 16].item() == pytest.approx(1.5)
    assert data_tree.x[idx_global, 17].item() == pytest.approx(-2.7)
    assert data_tree.x[idx_global, 18].item() == pytest.approx(0.0)


def test_supervised_learning_preprocessor_static_initialization():
    # Setup parent dataset row and a base graph PyG template
    raw = {
        "id": "P-supervised",
        "nodes": [
            {
                "id": "global",
                "label": "GLOBAL",
                "type": "global",
                "value": None
            },
            {
                "id": "f1",
                "label": "x",
                "type": "variable",
                "value": None
            }
        ],
        "edges": []
    }
    
    converter = ExpressionGraphConverter()
    
    # 1. Test Supervised Initialization in Graph Mode
    base_graph_graph = converter.convert(raw, heterogeneous=False, enrich=False, mode="graph")
    base_graphs_graph = {"P-supervised": base_graph_graph}
    
    idx_cx = base_graph_graph.node_ids.index("virtual_current_x")
    idx_fx = base_graph_graph.node_ids.index("virtual_f_x")
    idx_yt = base_graph_graph.node_ids.index("virtual_y_target")
    
    df_no_fx = pd.DataFrame([{
        "problem_id": "P-supervised",
        "startwert": 2.5,
        "zielwert": 4.0,
        "faster_algorithm": 1
    }])
    
    dataset_no_fx_graph = ProblemRunDataset(df_no_fx, base_graphs_graph, mode="graph")
    data_no_fx_graph = dataset_no_fx_graph[0]
    
    assert data_no_fx_graph.x[idx_cx, 2].item() == 2.5
    assert data_no_fx_graph.x[idx_fx, 2].item() == 0.0
    assert data_no_fx_graph.x[idx_yt, 2].item() == 4.0
    
    assert data_no_fx_graph.x[idx_cx, 3].item() == 1.0
    assert data_no_fx_graph.x[idx_fx, 3].item() == 1.0
    assert data_no_fx_graph.x[idx_yt, 3].item() == 1.0

    df_with_fx = pd.DataFrame([{
        "problem_id": "P-supervised",
        "startwert": 2.5,
        "zielwert": 4.0,
        "fx": 10.12,
        "faster_algorithm": 1
    }])
    
    dataset_with_fx_graph = ProblemRunDataset(df_with_fx, base_graphs_graph, mode="graph")
    data_with_fx_graph = dataset_with_fx_graph[0]
    assert data_with_fx_graph.x[idx_fx, 2].item() == pytest.approx(10.12)

    # 2. Test Supervised Initialization in Tree Mode
    base_graph_tree = converter.convert(raw, heterogeneous=False, enrich=False, mode="tree")
    base_graphs_tree = {"P-supervised": base_graph_tree}
    
    idx_global = base_graph_tree.node_ids.index("global")
    
    dataset_no_fx_tree = ProblemRunDataset(df_no_fx, base_graphs_tree, mode="tree")
    data_no_fx_tree = dataset_no_fx_tree[0]
    
    # Tree mode: custom slots are indices 5, 6, 7 on global node
    assert data_no_fx_tree.x[idx_global, 5].item() == 2.5
    assert data_no_fx_tree.x[idx_global, 6].item() == 0.0
    assert data_no_fx_tree.x[idx_global, 7].item() == 4.0

    dataset_with_fx_tree = ProblemRunDataset(df_with_fx, base_graphs_tree, mode="tree")
    data_with_fx_tree = dataset_with_fx_tree[0]
    assert data_with_fx_tree.x[idx_global, 6].item() == pytest.approx(10.12)
