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
from torch_geometric.graphgym.config import set_cfg, load_cfg
from torch_geometric.graphgym.utils.comp_exec import set_run_dir
from torch_geometric.graphgym.train import train_runner
import torch_geometric.graphgym.register as register


def main():
    # 1. Parse CLI arguments provided by GraphGym
    args = parse_args()
    
    # 2. Load configuration from YAML into GraphGym's global config object
    load_cfg(set_cfg(), args)
    set_run_dir(set_cfg(), args.cfg_file)
    
    print("\n[GraphGym Command Center] Launching training run...")
    # 3. Start training loop using GraphGym's built-in train_runner
    train_runner(register.train_dict[set_cfg().train.mode])


if __name__ == "__main__":
    main()
