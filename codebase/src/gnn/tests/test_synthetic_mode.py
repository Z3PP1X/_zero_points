from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import torch
from pathlib import Path
import sys
import numpy as np

# Add python path
current = Path(__file__).resolve()
src_dir = current.parents[2]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocessing import GraphPipeline
from unified_loader import UnifiedDataLoader
from torch_geometric.data import Data


@pytest.fixture(autouse=True)
def clear_loader_cache():
    UnifiedDataLoader.clear_instances()


@patch("preprocessing.UnifiedDataLoader")
def test_synthetic_mode_split(MockUnifiedDataLoader):
    # Mock curated loader
    mock_curated_unified = MagicMock()
    mock_curated_dataset = MagicMock()
    curated_data = pd.DataFrame([
        {"problem_id": "P_curated_1", "Newton_absTime": 0.1, "GMGF_absTime": 0.2, "Newton_iterSteps": 5, "GMGF_iterSteps": 6, "x0": 1.0, "y_target": 2.0},
        {"problem_id": "P_curated_2", "Newton_absTime": 0.3, "GMGF_absTime": 0.1, "Newton_iterSteps": 7, "GMGF_iterSteps": 4, "x0": 2.0, "y_target": 1.0}
    ])
    mock_curated_dataset.data = curated_data
    mock_curated_unified.dataset_loader = mock_curated_dataset
    
    # Use properties to dynamically return the potentially modified df
    type(mock_curated_unified).data = property(lambda self: mock_curated_dataset.data)
    
    def add_column_curated(name, values):
        mock_curated_dataset.data = mock_curated_dataset.data.copy()
        mock_curated_dataset.data[name] = values
    mock_curated_dataset.add_column.side_effect = add_column_curated
    mock_curated_unified.add_column.side_effect = add_column_curated
    
    # Create simple graphs for curated
    g_curated_1 = Data(num_nodes=3)
    g_curated_1.x = torch.zeros((3, 8))
    g_curated_1.edge_index = torch.empty((2, 0), dtype=torch.long)
    g_curated_2 = Data(num_nodes=3)
    g_curated_2.x = torch.zeros((3, 8))
    g_curated_2.edge_index = torch.empty((2, 0), dtype=torch.long)
    
    mock_curated_unified.load_all.return_value = {
        "P_curated_1": g_curated_1,
        "P_curated_2": g_curated_2
    }
    
    # Mock synthetic loader
    mock_synth_unified = MagicMock()
    mock_synth_dataset = MagicMock()
    synth_data = pd.DataFrame([
        {"problem_id": "P_synth_1", "Newton_absTime": 0.05, "GMGF_absTime": 0.15, "Newton_iterSteps": 4, "GMGF_iterSteps": 5, "x0": 0.5, "y_target": 1.5},
        {"problem_id": "P_synth_2", "Newton_absTime": 0.25, "GMGF_absTime": 0.08, "Newton_iterSteps": 6, "GMGF_iterSteps": 3, "x0": 1.5, "y_target": 0.5}
    ])
    mock_synth_dataset.data = synth_data
    mock_synth_unified.dataset_loader = mock_synth_dataset
    
    type(mock_synth_unified).data = property(lambda self: mock_synth_dataset.data)
    
    def add_column_synth(name, values):
        mock_synth_dataset.data = mock_synth_dataset.data.copy()
        mock_synth_dataset.data[name] = values
    mock_synth_dataset.add_column.side_effect = add_column_synth
    mock_synth_unified.add_column.side_effect = add_column_synth
    
    # Create simple graphs for synthetic
    g_synth_1 = Data(num_nodes=3)
    g_synth_1.x = torch.zeros((3, 8))
    g_synth_1.edge_index = torch.empty((2, 0), dtype=torch.long)
    g_synth_2 = Data(num_nodes=3)
    g_synth_2.x = torch.zeros((3, 8))
    g_synth_2.edge_index = torch.empty((2, 0), dtype=torch.long)
    
    mock_synth_unified.load_all.return_value = {
        "P_synth_1": g_synth_1,
        "P_synth_2": g_synth_2
    }
    
    # Configure multiton behavior
    def get_instance_side_effect(dataset_name, mode, enrich, **kwargs):
        if dataset_name == "curated_dataset":
            return mock_curated_unified
        elif dataset_name == "synthetic_dataset":
            return mock_synth_unified
        raise ValueError(f"Unknown mock dataset name {dataset_name}")
        
    MockUnifiedDataLoader.get_instance.side_effect = get_instance_side_effect
    
    # Initialize pipeline
    pipeline = GraphPipeline(
        dataset_name="curated_dataset",
        mode="graph",
        enrich=False,
        synthetic=True,
        synthetic_dataset_name="synthetic_dataset",
        unified_loader=mock_curated_unified
    )
    # Manually configure the synthetic loader since unified_loader param overrides it in pipeline init
    pipeline.synthetic_unified_loader = mock_synth_unified
    
    train_loader, test_loader, weights = pipeline.pipe(batch_size=1)
    
    assert train_loader is not None
    assert test_loader is not None
    
    # Check that datasets are correctly isolated
    train_problems = set(pipeline.train_dataset.df["problem_id"].unique())
    test_problems = set(pipeline.test_dataset.df["problem_id"].unique())
    
    assert train_problems == {"P_synth_1", "P_synth_2"}
    assert test_problems == {"P_curated_1", "P_curated_2"}
