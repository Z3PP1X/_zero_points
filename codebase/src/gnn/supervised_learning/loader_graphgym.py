import sys
from pathlib import Path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader, register_config
from yacs.config import CfgNode as CN

# Dynamic sys.path resolution
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.preprocessing import GraphPipeline
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.register import register_act

# Register additional activation functions for GraphGym compatibility
register_act('gelu', nn.GELU)
register_act('leaky_relu', nn.LeakyReLU)
register_act('tanh', nn.Tanh)

import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer

@register_layer('gatv2conv')
class GATv2Conv(torch.nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATv2Conv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


import math
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.graphgym.register import register_scheduler

def cosine_with_warmup_scheduler(optimizer, max_epoch):
    from torch_geometric.graphgym.config import cfg
    warmup_epochs = getattr(cfg.train, 'epoch_warmup', 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, max_epoch - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    return LambdaLR(optimizer, lr_lambda)

register_scheduler('cosine_with_warmup', cosine_with_warmup_scheduler)


class ExpressionGraphDataset(InMemoryDataset):
    def __init__(self, data_list, train_idx, val_idx, test_idx):
        super().__init__()
        # Use private _data attribute to prevent PyG warnings
        self._data, self.slices = self.collate(data_list)
        self._data.train_graph_index = torch.tensor(train_idx, dtype=torch.long)
        self._data.val_graph_index = torch.tensor(val_idx, dtype=torch.long)
        self._data.test_graph_index = torch.tensor(test_idx, dtype=torch.long)


def set_custom_cfg(cfg):
    """
    Registers custom expression graph config parameters inside GraphGym.
    Allows specifying experimental variables inside YAML configuration.
    """
    print("--- Executing set_custom_cfg ---")
    cfg.expression_graph = CN()
    cfg.expression_graph.mode = "graph"            # "graph", "tree", or "tree_derivatives"
    cfg.expression_graph.enrich = False             # True or False
    cfg.expression_graph.active_features = ""       # Comma-separated list or empty for all
    cfg.train.mode = "custom"                       # Add custom train mode option
    cfg.train.epoch_warmup = 0                      # Custom warmup epoch config
    cfg.train.epochs = 100                          # Custom epochs config
    cfg.params = 0                                  # Global params key to avoid logger AttributeError


register_config("expression_graph", set_custom_cfg)


def load_custom_expression_graphs(format, name, dataset_dir):
    """
    GraphGym Loader for custom expression graphs.
    Uses Dependency Injection by reading dataset properties dynamically from GraphGym's global config:
    - cfg.dataset.name: e.g., "run_20260408_160456/dataset_4" (injects dataset selection)
    - cfg.train.batch_size: injects batch size
    - cfg.seed: injects seed (defaults to 42001 if not set)
    - cfg.expression_graph.mode: injects GNN mode ("graph", "tree", or "tree_derivatives")
    - cfg.expression_graph.enrich: injects GNN feature enrichment (True or False)
    - cfg.expression_graph.active_features: injects list of active features
    """
    dataset_name = cfg.dataset.name
    batch_size = cfg.train.batch_size
    seed = getattr(cfg, "seed", 42001)

    # Read custom injected expression graph properties
    mode = getattr(cfg.expression_graph, "mode", "graph")
    enrich = getattr(cfg.expression_graph, "enrich", False)
    active_features_str = getattr(cfg.expression_graph, "active_features", "")

    active_features = None
    if active_features_str:
        active_features = [f.strip() for f in active_features_str.split(",") if f.strip()]

    # Resolve run_key prefix to find the matching datasets folder
    if "/" in dataset_name:
        run_key, _ = dataset_name.split("/", 1)
    else:
        run_key = dataset_name

    repo_root = Path(__file__).resolve().parents[4]
    experiments_dir = repo_root / "_datasets" / run_key / "graphs"

    print(f"\n--- [GraphGym Dependency Injection] ---")
    print(f"  Injecting Dataset Name:  {dataset_name}")
    print(f"  Resolved Directory:      {experiments_dir}")
    print(f"  Injected Batch Size:     {batch_size}")
    print(f"  Injected Random Seed:    {seed}")
    print(f"  Injected Mode:           {mode}")
    print(f"  Injected Enrich:         {enrich}")
    if active_features is not None:
        print(f"  Injected Features ({len(active_features)}): {active_features}")
    else:
        print(f"  Injected Features:       ALL")
    print(f"----------------------------------------\n")

    from gnn.shared.utils.graph_loader import GraphDataLoader
    loader = GraphDataLoader(
        name=dataset_name,
        mode=mode,
        enrich=enrich,
        heterogeneous=False,
    )

    # Instantiate the GraphPipeline using the injected dependencies
    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        seed=seed,
        mode=mode,
        enrich=enrich,
        active_features=active_features,
        graph_loader=loader,
    )

    # Use pipeline loaders to trigger train/test split inside pipeline
    pipeline.pipe(
        test_size=0.2,
        batch_size=batch_size,
    )

    # Convert the datasets in GraphPipeline to a list of PyG Data objects
    train_data_list = [pipeline.train_dataset[i] for i in range(len(pipeline.train_dataset))]
    test_data_list = [pipeline.test_dataset[i] for i in range(len(pipeline.test_dataset))]

    all_data_list = train_data_list + test_data_list

    train_indices = list(range(len(train_data_list)))
    val_indices = list(range(len(train_data_list), len(all_data_list)))
    test_indices = list(range(len(train_data_list), len(all_data_list)))

    # Return the ExpressionGraphDataset to GraphGym
    return ExpressionGraphDataset(all_data_list, train_indices, val_indices, test_indices)


# Register the dependency-injected loader in the GraphGym global registry
register_loader("custom_expression_graphs", load_custom_expression_graphs)
