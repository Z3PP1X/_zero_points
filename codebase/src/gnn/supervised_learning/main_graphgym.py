import sys
from pathlib import Path
import torch
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, set_run_dir
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train


gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import gnn.supervised_learning.loader_graphgym  # noqa
from gnn.supervised_learning.preprocessing import GraphPipeline # noqa

set_cfg(cfg)


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        
    args = parse_args()
    load_cfg(cfg, args)
    if cfg.accelerator == "auto":
        cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.optim.max_epoch = cfg.train.epochs
    set_run_dir(cfg.out_dir)

    print("\n[GraphGym Command Center] Launching training run...")
    datamodule = GraphGymDataModule()
    model = create_model()
    train(model, datamodule, logger=True)


if __name__ == "__main__":
    main()
