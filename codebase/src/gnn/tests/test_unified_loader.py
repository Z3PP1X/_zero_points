from unittest.mock import MagicMock, patch
import pytest
from unified_loader import UnifiedDataLoader


def test_unified_data_loader_multiton_behavior():
    UnifiedDataLoader.clear_instances()

    # Get instance for configuration A
    loader_a1 = UnifiedDataLoader.get_instance(
        dataset_name="run_20260408_160456/dataset_4",
        mode="tree_derivatives",
    )
    # Get instance for identical configuration A again
    loader_a2 = UnifiedDataLoader.get_instance(
        dataset_name="run_20260408_160456/dataset_4",
        mode="tree_derivatives",
    )
    # Must be the exact same object
    assert loader_a1 is loader_a2

    # Get instance for configuration B (different dataset)
    loader_b = UnifiedDataLoader.get_instance(
        dataset_name="run_20260408_160456/dataset_5",
        mode="tree_derivatives",
    )
    assert loader_a1 is not loader_b

    # Get instance for configuration C (different GNN mode)
    loader_c = UnifiedDataLoader.get_instance(
        dataset_name="run_20260408_160456/dataset_4",
        mode="tree",
    )
    assert loader_a1 is not loader_c

    # Clear instances and verify a new one is created
    UnifiedDataLoader.clear_instances()
    loader_a3 = UnifiedDataLoader.get_instance(
        dataset_name="run_20260408_160456/dataset_4",
        mode="tree_derivatives",
    )
    assert loader_a1 is not loader_a3


@patch("unified_loader.DatasetLoader")
@patch("unified_loader.GraphDataLoader")
def test_unified_data_loader_forwards_methods(MockGraphDataLoader, MockDatasetLoader):
    # Setup mock instances
    mock_dataset_inst = MagicMock()
    mock_graph_inst = MagicMock()
    
    MockDatasetLoader.return_value = mock_dataset_inst
    MockGraphDataLoader.return_value = mock_graph_inst
    
    UnifiedDataLoader.clear_instances()
    
    loader = UnifiedDataLoader.get_instance(
        dataset_name="test_dataset",
        run_key="test_run",
        mode="tree_derivatives",
    )
    
    # Test forwarding data property
    mock_dataset_inst.data = "mock_pandas_dataframe"
    assert loader.data == "mock_pandas_dataframe"
    
    # Test forwarding add_column
    loader.add_column("col1", [1, 2])
    mock_dataset_inst.add_column.assert_called_once_with("col1", [1, 2])
    
    # Test forwarding has_graph
    mock_graph_inst.has_graph.return_value = True
    assert loader.has_graph("G1") is True
    mock_graph_inst.has_graph.assert_called_once_with("G1")
    
    # Test forwarding get_graph
    mock_graph_inst.get_graph.return_value = "mock_pyg_graph"
    assert loader.get_graph("G1") == "mock_pyg_graph"
    mock_graph_inst.get_graph.assert_called_once_with("G1")
    
    # Test forwarding list_graph_ids
    mock_graph_inst.list_graph_ids.return_value = {"G1", "G2"}
    assert loader.list_graph_ids() == {"G1", "G2"}
    mock_graph_inst.list_graph_ids.assert_called_once()
    
    # Test forwarding load_all
    mock_graph_inst.load_all.return_value = {"G1": "mock_graph"}
    assert loader.load_all() == {"G1": "mock_graph"}
    mock_graph_inst.load_all.assert_called_once()


def test_unified_data_loader_auto_enrichment():
    import pandas as pd
    # Setup mock data frame lacking x0
    mock_df = pd.DataFrame([
        {"problem_id": "P1", "y_target": 0.5},
        {"problem_id": "P2", "y_target": 0.2, "x0": None},
        {"problem_id": "P3", "y_target": 0.1, "x0": 5.0}
    ])
    
    with patch("unified_loader.DatasetLoader") as MockDatasetLoader, \
         patch("unified_loader.GraphDataLoader") as MockGraphDataLoader:
         
        mock_dataset_inst = MagicMock()
        mock_dataset_inst.data = mock_df
        MockDatasetLoader.return_value = mock_dataset_inst
        
        mock_graph_inst = MagicMock()
        mock_graph_inst.has_graph.side_effect = lambda pid: pid in ["P1", "P2", "P3"]
        mock_graph_inst._raw_sources = {
            "P1": {"id": "P1", "x0": 1.5},
            "P2": {"id": "P2", "startwert": 2.5},
            "P3": {"id": "P3", "x0": 9.9} # Already has x0=5.0, should NOT be overwritten
        }
        MockGraphDataLoader.return_value = mock_graph_inst
        
        UnifiedDataLoader.clear_instances()
        
        # Instantiate loader
        loader = UnifiedDataLoader.get_instance(
            dataset_name="test_dataset",
            run_key="test_run",
            mode="tree_derivatives",
        )
        
        # Check that x0 was enriched correctly
        df = loader.data
        assert df.loc[df["problem_id"] == "P1", "x0"].values[0] == 1.5
        assert df.loc[df["problem_id"] == "P2", "x0"].values[0] == 2.5
        assert df.loc[df["problem_id"] == "P3", "x0"].values[0] == 5.0 # Preserved original value

