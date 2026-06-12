import json
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx

from graph_utils import ExpressionGraphConverter, signed_log_value, ENRICHED_NODE_FEATURE_SCHEMA, BASIC_NODE_FEATURE_SCHEMA
from feature_layout import NATIVE_NODE_FEATURE_COUNT, BASIC_NODE_FEATURE_COUNT
from reinforcement_learning.preprocessor import Preprocessor
from supervised_learning.preprocessing import ProblemRunDataset


def test_augmented_math_graph_edges():
    # Construct an expression: Plus[x, x] (reused variable x)
    raw = {
        "id": "P-augmented-test-1",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "x", "type": "variable", "value": None},
            {"id": "f3", "label": "x", "type": "variable", "value": None}
        ],
        "edges": [
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f1", "target": "f3", "type": "child_of"},
            {"source": "global", "target": "f1", "type": "belongs_to_f"}
        ]
    }

    # Test in bidirectional mode
    converter = ExpressionGraphConverter()
    data = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph", edge_direction="bidirectional")

    # Verify that NextUse and NextUseBackward edges are created between f2 and f3
    # G_enriched is converted to PyG data. Let's inspect node ids and edge_index.
    assert "virtual_current_x" not in data.node_ids
    assert "f_root" in data.node_ids

    # Let's verify edge types present
    # We can reconstruct G_enriched logic or check data.edge_index and relationship types.
    # In homogeneous, data.edge_index contains all edges. Let's inspect their relation types.
    f2_idx = data.node_ids.index("f2")
    f3_idx = data.node_ids.index("f3")
    
    # We expect a NextUse edge: f2 -> f3
    # We expect a NextUseBackward edge: f3 -> f2
    edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    assert (f2_idx, f3_idx) in edges
    assert (f3_idx, f2_idx) in edges


def test_augmented_math_graph_nesting_edges():
    # Construct a nested expression: Plus[sin[x], 2]
    # Root (f1: Plus) -> Children (f2: sin [function], f4: 2 [constant])
    # f2 (sin) -> Child (f3: x [variable])
    raw = {
        "id": "P-augmented-test-2",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "sin", "type": "function", "value": None},
            {"id": "f3", "label": "x", "type": "variable", "value": None},
            {"id": "f4", "label": "2", "type": "constant", "value": {"mantissa": 0.2, "exponent": 1}}
        ],
        "edges": [
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f2", "target": "f3", "type": "child_of"},
            {"source": "f1", "target": "f4", "type": "child_of"},
            {"source": "global", "target": "f1", "type": "belongs_to_f"}
        ]
    }

    converter = ExpressionGraphConverter()
    
    # Test top_down mode
    data_td = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph", edge_direction="top_down")
    f1_idx = data_td.node_ids.index("f1")
    f2_idx = data_td.node_ids.index("f2")
    edges_td = list(zip(data_td.edge_index[0].tolist(), data_td.edge_index[1].tolist()))
    
    # In top_down, we should have OuterToInner_Arg0: f1 -> f2
    assert (f1_idx, f2_idx) in edges_td
    assert (f2_idx, f1_idx) not in edges_td

    # Test bottom_up mode
    data_bu = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph", edge_direction="bottom_up")
    edges_bu = list(zip(data_bu.edge_index[0].tolist(), data_bu.edge_index[1].tolist()))
    
    # In bottom_up, we should have InnerToOuter_Arg0: f2 -> f1
    assert (f2_idx, f1_idx) in edges_bu
    assert (f1_idx, f2_idx) not in edges_bu


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

    # Test graph mode conversion
    converter = ExpressionGraphConverter()
    data = converter.convert(raw, heterogeneous=False, enrich=False, mode="graph")

    # 4 AST nodes + f_root aggregator = 5 nodes
    assert data.num_nodes == 5
    assert data.x.shape[1] == BASIC_NODE_FEATURE_COUNT
    assert len(data.node_ids) == 5
    assert "f_root" in data.node_ids
    assert "virtual_current_x" not in data.node_ids

    idx_f_root = data.node_ids.index("f_root")
    # Node type of f_root is 6
    assert data.x[idx_f_root, 0].item() == 6.0


def test_reinforcement_learning_preprocessor_dynamic_updates(tmp_path):
    # Setup raw graph files inside mock graphs_dir
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
    
    assert "f_root" in data_graph.node_ids
    idx_f_root = data_graph.node_ids.index("f_root")

    cx_idx = ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_current_x_val")
    dt_idx = ENRICHED_NODE_FEATURE_SCHEMA.index("virtual_delta_target_val")
    
    assert data_graph.x[idx_f_root, cx_idx].item() == pytest.approx(signed_log_value(1.5))
    assert data_graph.x[idx_f_root, dt_idx].item() == pytest.approx(signed_log_value(2.7))

    # 2. Test Preprocessor in Tree Mode
    preprocessor_tree = Preprocessor(graphs_dir=str(tmp_path), mode="tree")
    data_tree, extracted = preprocessor_tree.process(message)
    assert data_tree.num_nodes == 3  # global + f_root + variable

    idx_f_root_tree = data_tree.node_ids.index("f_root")
    assert data_tree.x[idx_f_root_tree, cx_idx].item() == pytest.approx(signed_log_value(1.5))
    assert data_tree.x[idx_f_root_tree, dt_idx].item() == pytest.approx(signed_log_value(2.7))


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
    
    # 1. Test Supervised Initialization in Graph Mode
    base_graph_graph = converter.convert(raw, heterogeneous=False, enrich=False, mode="graph")
    base_graphs_graph = {"P-supervised": base_graph_graph}
    
    assert "f_root" in base_graph_graph.node_ids
    idx_f_root = base_graph_graph.node_ids.index("f_root")
    
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
    
    assert data_no_fx_graph.x[idx_f_root, cx_idx].item() == pytest.approx(signed_log_value(2.5))
    assert data_no_fx_graph.x[idx_f_root, dt_idx].item() == pytest.approx(signed_log_value(4.0))

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
    
    active_feats = ["node_type", "value", "virtual_current_x_val"]
    
    data_full = converter.convert(raw, heterogeneous=False, enrich=True, mode="graph")
    assert data_full.x.shape[1] == NATIVE_NODE_FEATURE_COUNT
    
    from graph_utils import slice_active_features
    data_sliced = data_full.clone()
    data_sliced.x = slice_active_features(data_full.x, active_feats, enrich=True)
    
    assert data_sliced.x.shape[1] == 3
    idx_f_root = data_sliced.node_ids.index("f_root")
    
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
    
    idx_f_root_pre = data.node_ids.index("f_root")
    assert data.x[idx_f_root_pre, 2].item() == pytest.approx(signed_log_value(3.14))
