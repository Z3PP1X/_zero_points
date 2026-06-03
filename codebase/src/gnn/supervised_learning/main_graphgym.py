import sys
import argparse
from pathlib import Path

# Dynamic sys.path resolution
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# Import loader_graphgym so it automatically registers our loader inside the global GraphGym registry
import gnn.supervised_learning.loader_graphgym

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, set_run_dir
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train

# Re-run set_cfg(cfg) to incorporate custom configurations registered in loader_graphgym
set_cfg(cfg)


def main():
    # 1. Parse CLI arguments provided by GraphGym
    args = parse_args()
    
    # 2. Load configuration from YAML into GraphGym's global config object
    load_cfg(cfg, args)
    import torch
    if cfg.accelerator == 'auto':
        cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_run_dir(cfg.out_dir)
    
    print("\n[GraphGym Command Center] Launching training run...")
    # 3. Start training loop using PyG's standard train helper and Lightning model/loaders
    datamodule = GraphGymDataModule()
    model = create_model()
    train(model, datamodule, logger=True)


if __name__ == "__main__":
    main()
