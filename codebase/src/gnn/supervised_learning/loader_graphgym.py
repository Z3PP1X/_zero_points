import sys
from pathlib import Path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader, register_config
from yacs.config import CfgNode as CN
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.register import register_act
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv import GINEConv
import math
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.graphgym.register import register_scheduler

gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.preprocessing import GraphPipeline  # noqa


# Monkey patch GraphGym Logger to compute PR-AUC dynamically on any system/environment (e.g. Cloud GPU)
import torch_geometric.graphgym.logger as pyg_logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

def custom_classification_binary(self):
    true, pred_score = torch.cat(self._true), torch.cat(self._pred)
    pred_int = self._get_pred_int(pred_score)
    try:
        r_a_score = roc_auc_score(true, pred_score)
    except ValueError:
        r_a_score = 0.0
        
    try:
        if len(pred_score.shape) > 1 and pred_score.shape[1] > 1:
            scores = pred_score[:, 1].cpu().numpy() if hasattr(pred_score, 'cpu') else pred_score[:, 1]
        else:
            scores = pred_score.cpu().numpy() if hasattr(pred_score, 'cpu') else pred_score
        y_true_np = true.cpu().numpy() if hasattr(true, 'cpu') else true
        precision_pts, recall_pts, _ = precision_recall_curve(y_true_np, scores)
        pr_auc_score = auc(recall_pts, precision_pts)
    except Exception:
        pr_auc_score = 0.0

    return {
        'accuracy': round(accuracy_score(true, pred_int), pyg_logger.cfg.round),
        'precision': round(precision_score(true, pred_int, zero_division=0), pyg_logger.cfg.round),
        'recall': round(recall_score(true, pred_int, zero_division=0), pyg_logger.cfg.round),
        'f1': round(f1_score(true, pred_int, zero_division=0), pyg_logger.cfg.round),
        'auc': round(r_a_score, pyg_logger.cfg.round),
        'pr_auc': round(pr_auc_score, pyg_logger.cfg.round),
    }

pyg_logger.Logger.classification_binary = custom_classification_binary


register_act("gelu", nn.GELU)
register_act("leaky_relu", nn.LeakyReLU)
register_act("tanh", nn.Tanh)


@register_layer("gatv2conv")
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


@register_layer("gineconv")
def GINEConv_layer(layer_config, *args, **kwargs):
    nn = torch.nn.Linear(layer_config.dim_in, layer_config.dim_out)
    return GINEConv(nn=nn)


def cosine_with_warmup_scheduler(optimizer, max_epoch):
    from torch_geometric.graphgym.config import cfg

    warmup_epochs = getattr(cfg.train, "epoch_warmup", 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, max_epoch - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


register_scheduler("cosine_with_warmup", cosine_with_warmup_scheduler)


class ExpressionGraphDataset(InMemoryDataset):
    def __init__(self, data_list, train_idx, val_idx, test_idx):
        super().__init__()
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
    cfg.expression_graph.mode = "graph"  # "graph", "tree", or "tree_derivatives"
    cfg.expression_graph.enrich = False
    cfg.expression_graph.active_features = ""  # Comma-separated list or empty for all
    cfg.expression_graph.synthetic = False
    cfg.expression_graph.synthetic_dataset = ""  # Synthetic dataset name
    cfg.train.mode = "custom"
    cfg.train.epoch_warmup = 0
    cfg.train.epochs = 100
    cfg.params = 0


register_config("expression_graph", set_custom_cfg)


def load_custom_expression_graphs(format, name, dataset_dir):
    """
    GraphGym Loader for custom expression graphs.
    Uses Dependency Injection by reading dataset properties dynamically from GraphGym's
    global config:
    - cfg.dataset.name: e.g., "run_20260408_160456/dataset_4"
        (injects dataset selection)
    - cfg.train.batch_size: injects batch size
    - cfg.seed: injects seed (defaults to 42001 if not set)
    - cfg.expression_graph.mode: injects GNN mode ("graph", "tree",
        or "tree_derivatives")
    - cfg.expression_graph.enrich: injects GNN feature enrichment (True or False)
    - cfg.expression_graph.active_features: injects list of active features
    - cfg.expression_graph.synthetic: injects synthetic mode (True or False)
    - cfg.expression_graph.synthetic_dataset: injects synthetic dataset name
    """
    dataset_name = cfg.dataset.name
    batch_size = cfg.train.batch_size
    seed = getattr(cfg, "seed", 42001)

    mode = getattr(cfg.expression_graph, "mode", "graph")
    enrich = getattr(cfg.expression_graph, "enrich", False)
    active_features_str = getattr(cfg.expression_graph, "active_features", "")

    synthetic = getattr(cfg.expression_graph, "synthetic", False)
    synthetic_dataset = getattr(cfg.expression_graph, "synthetic_dataset", "")

    active_features = None
    if active_features_str:
        active_features = [
            f.strip() for f in active_features_str.split(",") if f.strip()
        ]

    if "/" in dataset_name:
        run_key, _ = dataset_name.split("/", 1)
    else:
        run_key = dataset_name

    repo_root = Path(__file__).resolve().parents[4]
    experiments_dir = repo_root / "_datasets" / run_key / "graphs"

    print("\n--- [GraphGym Dependency Injection] ---")
    print(f"  Injecting Dataset Name:  {dataset_name}")
    print(f"  Resolved Directory:      {experiments_dir}")
    print(f"  Injected Batch Size:     {batch_size}")
    print(f"  Injected Random Seed:    {seed}")
    print(f"  Injected Mode:           {mode}")
    print(f"  Injected Enrich:         {enrich}")
    print(f"  Injected Synthetic:      {synthetic}")
    if synthetic:
        print(f"  Injected Synth Dataset:  {synthetic_dataset}")
    if active_features is not None:
        print(f"  Injected Features ({len(active_features)}): {active_features}")
    else:
        print("  Injected Features:       ALL")
    print("----------------------------------------\n")

    from gnn.shared.utils.graph_loader import GraphDataLoader

    loader = GraphDataLoader(
        name=dataset_name,
        mode=mode,
        enrich=enrich,
        heterogeneous=False,
    )

    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        seed=seed,
        mode=mode,
        enrich=enrich,
        active_features=active_features,
        graph_loader=loader,
        synthetic=synthetic,
        synthetic_dataset_name=synthetic_dataset,
    )

    pipeline.pipe(
        test_size=0.2,
        batch_size=batch_size,
        stratify=True,
        num_workers=getattr(cfg, 'num_workers', 0),
    )

    if synthetic:
        train_data_list = [
            pipeline.train_dataset[i] for i in range(len(pipeline.train_dataset))
        ]
        val_data_list = [  # unseen synthetic test data — evaluated every eval_period for checkpoint selection
            pipeline.test_dataset[i] for i in range(len(pipeline.test_dataset))
        ]
        curated_data_list = [  # curated real-world problems — evaluated ONCE at the end with the best model
            pipeline.curated_dataset[i] for i in range(len(pipeline.curated_dataset))
        ]

        all_data_list = train_data_list + val_data_list + curated_data_list

        train_indices = list(range(len(train_data_list)))
        # val = unseen synthetic (20%): runs every eval_period epoch, used for checkpoint selection
        val_indices = list(range(len(train_data_list), len(train_data_list) + len(val_data_list)))
        # test = curated real-world: runs ONCE after training ends with the best saved model
        test_indices = list(range(len(train_data_list) + len(val_data_list), len(all_data_list)))
    else:
        train_data_list = [
            pipeline.train_dataset[i] for i in range(len(pipeline.train_dataset))
        ]
        test_data_list = [
            pipeline.test_dataset[i] for i in range(len(pipeline.test_dataset))
        ]

        all_data_list = train_data_list + test_data_list

        train_indices = list(range(len(train_data_list)))
        val_indices = list(range(len(train_data_list), len(all_data_list)))
        test_indices = list(range(len(train_data_list), len(all_data_list)))

    return ExpressionGraphDataset(
        all_data_list, train_indices, val_indices, test_indices
    )


register_loader("custom_expression_graphs", load_custom_expression_graphs)
