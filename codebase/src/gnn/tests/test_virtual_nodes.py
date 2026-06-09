import json
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from graph_utils import ExpressionGraphConverter, signed_log_value, ENRICHED_NODE_FEATURE_SCHEMA, BASIC_NODE_FEATURE_SCHEMA
from feature_layout import NATIVE_NODE_FEATURE_COUNT, BASIC_NODE_FEATURE_COUNT
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

    # 4 AST nodes + f_root aggregator + 2 task virtual nodes + supernode = 8 nodes
    assert data_basic_graph.num_nodes == 8
    assert data_basic_graph.x.shape[1] == BASIC_NODE_FEATURE_COUNT
    assert len(data_basic_graph.node_ids) == 8
    assert "virtual_supernode" in data_basic_graph.node_ids
    assert "virtual_current_x" in data_basic_graph.node_ids
    assert "f_root" in data_basic_graph.node_ids
    assert "virtual_y_target" in data_basic_graph.node_ids
    assert "virtual_f_x" not in data_basic_graph.node_ids

    idx_cx = data_basic_graph.node_ids.index("virtual_current_x")
    idx_f_root = data_basic_graph.node_ids.index("f_root")
    idx_yt = data_basic_graph.node_ids.index("virtual_y_target")

    assert data_basic_graph.x[idx_cx, 0].item() == 5.0
    assert data_basic_graph.x[idx_f_root, 0].item() == 6.0
    assert data_basic_graph.x[idx_yt, 0].item() == 7.0

    # Test basic conversion in Tree Mode
    data_basic_tree = converter.convert(raw, heterogeneous=False, enrich=False, mode="tree")
    assert data_basic_tree.num_nodes == 5
    assert data_basic_tree.x.shape[1] == BASIC_NODE_FEATURE_COUNT
    assert len(data_basic_tree.node_ids) == 5
    assert "virtual_current_x" not in data_basic_tree.node_ids

    # Test rich (RL, enrich=True) conversion in Graph Mode
    data_rich_graph = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph")
    assert data_rich_graph.num_nodes == 8
    assert data_rich_graph.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    assert "virtual_supernode" in data_rich_graph.node_ids
    assert data_rich_graph.x[idx_cx, 0].item() == 5.0

    # Test rich conversion in Tree Mode
    data_rich_tree = converter.convert(raw, heterogeneous=False, enrich=True, mode="tree")
    assert data_rich_tree.num_nodes == 5
    assert data_rich_tree.x.shape[1] == NATIVE_NODE_FEATURE_COUNT


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
    idx_f_root = data_graph.node_ids.index("f_root")
    idx_yt = data_graph.node_ids.index("virtual_y_target")

    cx_idx = ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_current_x_val")
    assert data_graph.x[idx_cx, cx_idx].item() == pytest.approx(signed_log_value(1.5))
    assert data_graph.x[idx_f_root, ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_delta_target_val")].item() == pytest.approx(signed_log_value(2.7))
    assert data_graph.x[idx_yt, ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_delta_target_val")].item() == pytest.approx(signed_log_value(2.7))

    # 2. Test Preprocessor in Tree Mode
    preprocessor_tree = Preprocessor(graphs_dir=str(tmp_path), mode="tree")
    data_tree, extracted = preprocessor_tree.process(message)
    assert data_tree.num_nodes == 3  # global + f_root + variable

    idx_f_root_tree = data_tree.node_ids.index("f_root")
    cx_idx = ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_current_x_val")
    assert data_tree.x[idx_f_root_tree, cx_idx].item() == pytest.approx(signed_log_value(1.5))
    assert data_tree.x[idx_f_root_tree, ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_delta_target_val")].item() == pytest.approx(signed_log_value(2.7))


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
    idx_f_root = base_graph_graph.node_ids.index("f_root")
    idx_yt = base_graph_graph.node_ids.index("virtual_y_target")
    
    df_no_fx = pd.DataFrame([{
        "problem_id": "P-supervised",
        "x0": 2.5,
        "y_target": 4.0,
        "faster_algorithm": 1
    }])
    
    dataset_no_fx_graph = ProblemRunDataset(df_no_fx, base_graphs_graph, mode="graph")
    data_no_fx_graph = dataset_no_fx_graph[0]
    
    cx_idx = BASIC_NODE_FEATURE_SCHEMA.index("virtual_current_x_val")
    dt_idx = BASIC_NODE_FEATURE_SCHEMA.index("virtual_delta_target_val")
    d1_idx = BASIC_NODE_FEATURE_SCHEMA.index("virtual_d1_x_val")
    d2_idx = BASIC_NODE_FEATURE_SCHEMA.index("virtual_d2_x_val")
    has_idx = BASIC_NODE_FEATURE_SCHEMA.index("has_value")
    
    assert data_no_fx_graph.x[idx_cx, cx_idx].item() == pytest.approx(signed_log_value(2.5))
    assert data_no_fx_graph.x[idx_f_root, dt_idx].item() == pytest.approx(signed_log_value(4.0))
    assert data_no_fx_graph.x[idx_yt, dt_idx].item() == pytest.approx(signed_log_value(4.0))

    assert data_no_fx_graph.x[idx_cx, has_idx].item() == 1.0
    assert data_no_fx_graph.x[idx_f_root, has_idx].item() == 1.0
    assert data_no_fx_graph.x[idx_yt, has_idx].item() == 1.0

    df_with_fx = pd.DataFrame([{
        "problem_id": "P-supervised",
        "x0": 2.5,
        "y_target": 4.0,
        "fx": 10.12,
        "faster_algorithm": 1
    }])
    
    dataset_with_fx_graph = ProblemRunDataset(df_with_fx, base_graphs_graph, mode="graph")
    data_with_fx_graph = dataset_with_fx_graph[0]
    assert data_with_fx_graph.x[idx_f_root, dt_idx].item() == pytest.approx(signed_log_value(-6.12))

    # 2. Test Supervised Initialization in Tree Mode
    base_graph_tree = converter.convert(raw, heterogeneous=False, enrich=False, mode="tree")
    base_graphs_tree = {"P-supervised": base_graph_tree}

    idx_f_root_tree = base_graph_tree.node_ids.index("f_root")

    dataset_no_fx_tree = ProblemRunDataset(df_no_fx, base_graphs_tree, mode="tree")
    data_no_fx_tree = dataset_no_fx_tree[0]

    cx_idx = BASIC_NODE_FEATURE_SCHEMA.index("virtual_current_x_val")
    dt_idx = BASIC_NODE_FEATURE_SCHEMA.index("virtual_delta_target_val")

    assert data_no_fx_tree.x[idx_f_root_tree, cx_idx].item() == pytest.approx(signed_log_value(2.5))
    assert data_no_fx_tree.x[idx_f_root_tree, dt_idx].item() == pytest.approx(signed_log_value(4.0))

    dataset_with_fx_tree = ProblemRunDataset(df_with_fx, base_graphs_tree, mode="tree")
    data_with_fx_tree = dataset_with_fx_tree[0]
    assert data_with_fx_tree.x[idx_f_root_tree, dt_idx].item() == pytest.approx(signed_log_value(-6.12))


def test_dynamic_feature_slicing_and_selection(tmp_path):
    # Construct a simple graph
    raw = {
        "id": "P-slice",
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
    
    # 1. Test slicing in Graph Mode with enriched (RL) features
    converter = ExpressionGraphConverter()
    
    # Slice active features: only node_type, value, virtual_current_x_val
    active_feats = ["node_type", "value", "virtual_current_x_val"]
    
    data_full = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph")
    assert data_full.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    
    from graph_utils import slice_active_features
    data_sliced = data_full.clone()
    data_sliced.x = slice_active_features(data_full.x, active_feats, enrich=True)
    
    # Node features tensor must be sliced to 3 features
    assert data_sliced.x.shape[1] == 3
    # Check that feature indices align correctly:
    # In sliced: index 0 is node_type, index 1 is value, index 2 is virtual_current_x_val
    idx_cx = data_sliced.node_ids.index("virtual_current_x")
    
    # 2. Test preprocessor with active_features sliced
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
    
    idx_cx_pre = data.node_ids.index("virtual_current_x")
    # In active_feats, "virtual_current_x_val" is at index 2
    assert data.x[idx_cx_pre, 2].item() == pytest.approx(signed_log_value(3.14))

    # 3. Test supervised dataset with active_features sliced (basic features)
    base_graph_basic = converter.convert(raw, heterogeneous=False, enrich=False, mode="graph")
    base_graphs_basic = {"P-slice": base_graph_basic}
    
    active_feats_basic = ["node_type", "virtual_delta_target_val", "has_value"]
    
    df = pd.DataFrame([{
        "problem_id": "P-slice",
        "x0": 2.5,
        "y_target": 9.9,
        "faster_algorithm": 0
    }])
    
    dataset = ProblemRunDataset(df, base_graphs_basic, mode="graph", enrich=False, active_features=active_feats_basic)
    data_sup = dataset[0]
    
    assert data_sup.x.shape[1] == 3
    idx_yt = data_sup.node_ids.index("virtual_y_target")
    # In active_feats_basic, "virtual_delta_target_val" is at index 1
    assert data_sup.x[idx_yt, 1].item() == pytest.approx(signed_log_value(9.9))

    # 4. Test preprocessor in Tree Mode with sliced features
    active_feats_tree = ["node_type", "virtual_current_x_val", "virtual_delta_target_val"]
    preprocessor_tree = Preprocessor(graphs_dir=str(tmp_path), mode="tree", active_features=active_feats_tree)

    data_tree, extracted = preprocessor_tree.process(message)
    assert data_tree.x.shape[1] == 3
    idx_f_root_tree = data_tree.node_ids.index("f_root")
    assert data_tree.x[idx_f_root_tree, 1].item() == pytest.approx(signed_log_value(3.14))
    assert data_tree.x[idx_f_root_tree, 2].item() == pytest.approx(signed_log_value(1.7))
