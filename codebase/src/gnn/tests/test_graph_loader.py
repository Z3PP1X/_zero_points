import json
import pytest
from pathlib import Path
from torch_geometric.data import Data

from unified_loader import UnifiedDataLoader


@pytest.fixture(autouse=True)
def clear_loader_cache():
    UnifiedDataLoader.clear_instances()


@pytest.fixture
def sample_graph_data():
    return {
        "id": "P-test-1",
        "nodes": [
            {"id": "n1", "label": "Plus", "type": "operator", "value": None},
            {"id": "n2", "label": "x", "type": "variable", "value": None},
            {"id": "n3", "label": "1", "type": "constant", "value": 1.0}
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "child_of"},
            {"source": "n1", "target": "n3", "type": "child_of"}
        ]
    }


def test_load_from_single_json_dict(tmp_path, sample_graph_data):
    # Dictionary of graphs
    raw_dict = {
        "graph_A": sample_graph_data,
        "graph_B": {**sample_graph_data, "id": "P-test-2"}
    }
    
    file_path = tmp_path / "dataset_dict.json"
    file_path.write_text(json.dumps(raw_dict), encoding="utf-8")
    
    # Use mode="tree" to avoid injecting virtual nodes, keeping node count to exactly 3
    loader = UnifiedDataLoader.get_instance(dataset_name="test", base_dir=file_path, mode="tree").graph_loader
    
    assert loader.list_graph_ids() == {"graph_A", "graph_B"}
    assert loader.has_graph("graph_A")
    assert loader.has_graph("graph_B")
    
    graph_a = loader.get_graph("graph_A")
    assert isinstance(graph_a, Data)
    assert graph_a.num_nodes == 3


def test_load_from_single_json_list(tmp_path, sample_graph_data):
    # List of graphs
    raw_list = [
        sample_graph_data,
        {**sample_graph_data, "id": "P-test-2"}
    ]
    
    file_path = tmp_path / "dataset_list.json"
    file_path.write_text(json.dumps(raw_list), encoding="utf-8")
    
    loader = UnifiedDataLoader.get_instance(dataset_name="test", base_dir=file_path, mode="tree").graph_loader
    
    assert loader.list_graph_ids() == {"P-test-1", "P-test-2"}
    assert loader.has_graph("P-test-1")
    assert loader.has_graph("P-test-2")
    
    graph_1 = loader.get_graph("P-test-1")
    assert isinstance(graph_1, Data)
    assert graph_1.num_nodes == 3


def test_load_from_single_graph_direct(tmp_path, sample_graph_data):
    # Single graph (not nested)
    file_path = tmp_path / "single_graph.json"
    file_path.write_text(json.dumps(sample_graph_data), encoding="utf-8")
    
    loader = UnifiedDataLoader.get_instance(dataset_name="my_graph", base_dir=file_path, mode="tree").graph_loader
    
    assert loader.list_graph_ids() == {"P-test-1"}
    assert loader.has_graph("P-test-1")
    
    graph = loader.get_graph("P-test-1")
    assert isinstance(graph, Data)
    assert graph.num_nodes == 3


def test_load_from_directory_structure(tmp_path, sample_graph_data):
    # Create directory with normal and meta json files
    dir_path = tmp_path / "graphs_dir"
    dir_path.mkdir()
    
    # normal json
    (dir_path / "P1.json").write_text(json.dumps(sample_graph_data), encoding="utf-8")
    
    # meta json (should take precedence)
    meta_data = {**sample_graph_data, "id": "P2-meta", "meta_flag": True}
    (dir_path / "P2_meta.json").write_text(json.dumps(meta_data), encoding="utf-8")
    (dir_path / "P2.json").write_text(json.dumps(sample_graph_data), encoding="utf-8")
    
    loader = UnifiedDataLoader.get_instance(dataset_name="test", base_dir=dir_path, mode="tree").graph_loader
    
    # Should resolve P1 and P2 (using P2_meta.json)
    assert loader.list_graph_ids() == {"P1", "P2"}
    
    # Verify P2 loaded the meta one
    graph_p2 = loader.get_graph("P2")
    assert graph_p2.num_nodes == 3


def test_caching_and_cloning(tmp_path, sample_graph_data):
    file_path = tmp_path / "single_graph.json"
    file_path.write_text(json.dumps(sample_graph_data), encoding="utf-8")
    
    loader = UnifiedDataLoader.get_instance(dataset_name="my_graph", base_dir=file_path, mode="tree").graph_loader
    
    graph_1 = loader.get_graph("P-test-1")
    graph_2 = loader.get_graph("P-test-1")
    
    # They should be separate object instances (cloned)
    assert graph_1 is not graph_2
    
    # Modifying graph_1 should not affect graph_2
    graph_1.x[0, 0] = 999.0
    assert graph_2.x[0, 0] != 999.0
