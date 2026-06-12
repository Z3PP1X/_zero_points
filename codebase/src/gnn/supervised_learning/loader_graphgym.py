import sys
from pathlib import Path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader, register_config
from yacs.config import CfgNode as CN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.register import register_act
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer, register_loss
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

from gnn.supervised_learning.preprocessing import GraphPipeline  # noqa: F401
from gnn.supervised_learning.run_results.eval_metrics import compute_confidence_metrics
from gnn.supervised_learning.supervised_config import (
    resolve_edge_dim,
    validate_layer_type,
)

_CLASS_WEIGHTS = torch.tensor([1.0, 1.0], dtype=torch.float)


def configure_class_weights(class_weights: torch.Tensor) -> None:
    """Update the shared class weights used by ``weighted_cross_entropy`` loss."""
    global _CLASS_WEIGHTS
    _CLASS_WEIGHTS = class_weights.detach().float().clone()


@register_loss("weighted_cross_entropy")
def weighted_cross_entropy_loss(pred, true):
    device = pred.device
    w = _CLASS_WEIGHTS.to(device)
    if pred.ndim > 1 and true.ndim == 1:
        log_pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(log_pred, true.long(), weight=w), log_pred
    true = true.float()
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=w[1] / w[0])
    return bce_loss(pred, true), torch.sigmoid(pred)


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


def set_pos_label_from_train_labels(train_labels: torch.Tensor) -> int:
    """Pick the minority class from training labels for metric evaluation."""
    counts = torch.bincount(train_labels.long(), minlength=2)
    pos_label = int(counts.argmin().item())
    cfg.expression_graph.pos_label = pos_label
    print(
        f"[GraphGym] Minority class for metrics: {pos_label} "
        f"(counts: class0={counts[0].item()}, class1={counts[1].item()})"
    )
    return pos_label


def get_pos_label() -> int:
    return int(getattr(cfg.expression_graph, "pos_label", 1))


def _to_numpy(tensor):
    return tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor


def _positive_class_scores(pred_score, pos_label: int):
    """Return continuous scores for the minority (positive) class."""
    if len(pred_score.shape) > 1 and pred_score.shape[1] > 1:
        col = pred_score[:, pos_label]
        return _to_numpy(col)
    scores = _to_numpy(pred_score)
    if pos_label == 1:
        return scores
    return 1.0 - scores


def _hard_predictions(pred_score, pos_label: int, thresh: float):
    if len(pred_score.shape) > 1 and pred_score.shape[1] > 1:
        return pred_score.argmax(dim=1)
    scores = _to_numpy(pred_score)
    if pos_label == 1:
        return torch.tensor((scores > thresh).astype(int))
    return torch.tensor((scores <= (1.0 - thresh)).astype(int))


def compute_binary_metrics(true, pred_score, round_digits=None):
    """
    Compute classification metrics with the minority class as positive.
    Used by both LoggerCallback (stats.json) and ValMetricLogger (checkpointing).
    """
    pos_label = get_pos_label()
    thresh = getattr(cfg.model, "thresh", 0.5)
    rnd = round_digits if round_digits is not None else cfg.round

    true_t = true if isinstance(true, torch.Tensor) else torch.tensor(true)
    pred_t = pred_score if isinstance(pred_score, torch.Tensor) else torch.tensor(pred_score)
    pred_int = _hard_predictions(pred_t, pos_label, thresh)
    scores = _positive_class_scores(pred_t, pos_label)
    y_true_np = _to_numpy(true_t)

    try:
        r_a_score = roc_auc_score(y_true_np, scores)
    except ValueError:
        r_a_score = 0.0

    try:
        precision_pts, recall_pts, _ = precision_recall_curve(
            y_true_np, scores, pos_label=pos_label
        )
        pr_auc_score = auc(recall_pts, precision_pts)
    except Exception:
        pr_auc_score = 0.0

    metrics = {
        "accuracy": round(accuracy_score(y_true_np, _to_numpy(pred_int)), rnd),
        "precision": round(
            precision_score(
                y_true_np, _to_numpy(pred_int), pos_label=pos_label, zero_division=0
            ),
            rnd,
        ),
        "recall": round(
            recall_score(
                y_true_np, _to_numpy(pred_int), pos_label=pos_label, zero_division=0
            ),
            rnd,
        ),
        "f1": round(
            f1_score(
                y_true_np, _to_numpy(pred_int), pos_label=pos_label, zero_division=0
            ),
            rnd,
        ),
        "auc": round(r_a_score, rnd),
        "pr_auc": round(pr_auc_score, rnd),
    }
    metrics.update(
        compute_confidence_metrics(
            true_t,
            pred_t,
            pos_label=pos_label,
            round_digits=rnd,
        )
    )
    return metrics


def custom_classification_binary(self):
    true, pred_score = torch.cat(self._true), torch.cat(self._pred)
    return compute_binary_metrics(true, pred_score, round_digits=pyg_logger.cfg.round)


_orig_logger_basic = pyg_logger.Logger.basic


def custom_logger_basic(self):
    if self._size_current == 0:
        return {
            "loss": 0.0,
            "lr": round(self._lr, cfg.round),
            "params": self._params,
            "time_iter": 0.0,
        }
    stats = _orig_logger_basic(self)
    stats["base_lr"] = round(float(getattr(cfg.optim, "base_lr", 0.0)), cfg.round)
    return stats


pyg_logger.Logger.classification_binary = custom_classification_binary
pyg_logger.Logger.basic = custom_logger_basic


# Patch LoggerCallback to support multiple validation loaders
def custom_on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    if outputs is None:
        return
    stats = self._get_stats(self._val_epoch_start_time, outputs, trainer)
    if dataloader_idx == 0:
        self.val_logger.update_stats(**stats)
    elif dataloader_idx == 1 and len(self._logger) > 2:
        self.test_logger.update_stats(**stats)

def custom_on_validation_epoch_end(self, trainer, pl_module):
    if self.val_logger._size_current > 0:
        self.val_logger.write_epoch(trainer.current_epoch)
    # Curated holdout (test split) is logged on a schedule via CuratedEvalCallback,
    # not during every synthetic validation epoch.

pyg_logger.LoggerCallback.on_validation_batch_end = custom_on_validation_batch_end
pyg_logger.LoggerCallback.on_validation_epoch_end = custom_on_validation_epoch_end


register_act("gelu", nn.GELU)
register_act("leaky_relu", nn.LeakyReLU)
register_act("tanh", nn.Tanh)


from torch_geometric.graphgym.register import register_node_encoder
from gnn.shared.models.gnn_backbones import NUM_NODE_TYPES, NUM_LABELS


@register_node_encoder("ExpressionNodeEncoder")
class ExpressionNodeEncoder(torch.nn.Module):
    """GraphGym node encoder for expression graphs.

    The raw feature matrix stores ``node_type`` (col 0) and ``label_id`` (col 1)
    as integer codes. Feeding them as continuous values imposes a meaningless
    ordinal scale (e.g. ``Log=17`` ≈ 17 × ``Plus=3``) and discards the single
    most discriminative signal: operator / function identity. This encoder embeds
    both categorical columns and projects the remaining continuous columns
    (which also normalises their disparate magnitudes), emitting ``dim_emb``
    (= ``cfg.gnn.dim_inner``) features.

    When an explicit ``active_features`` subset is configured the categorical
    column positions are no longer guaranteed, so the encoder falls back to a
    plain linear projection.
    """

    NODE_TYPE_COL = 0
    LABEL_ID_COL = 1

    def __init__(self, dim_emb: int):
        super().__init__()
        active = getattr(cfg.expression_graph, "active_features", "")
        self.categorical = not bool(active)
        self.activation = nn.GELU()

        dim_in = cfg.share.dim_in
        if self.categorical:
            node_type_emb_dim = 8
            label_emb_dim = 16
            self.node_type_emb = nn.Embedding(NUM_NODE_TYPES, node_type_emb_dim)
            self.label_emb = nn.Embedding(NUM_LABELS, label_emb_dim)
            cont_hidden = max(dim_emb - node_type_emb_dim - label_emb_dim, 8)
            cont_in = max(dim_in - 2, 1)
            self.continuous_encoder = nn.Linear(cont_in, cont_hidden)
            self.fusion = nn.Linear(
                cont_hidden + node_type_emb_dim + label_emb_dim, dim_emb
            )
        else:
            self.proj = nn.Linear(dim_in, dim_emb)

    def forward(self, batch):
        x = batch.x
        if self.categorical:
            node_types = (
                x[:, self.NODE_TYPE_COL].round().long().clamp(0, NUM_NODE_TYPES - 1)
            )
            label_ids = (
                x[:, self.LABEL_ID_COL].round().long().clamp(0, NUM_LABELS - 1)
            )
            cont_cols = [
                i for i in range(x.size(1)) if i not in (self.NODE_TYPE_COL, self.LABEL_ID_COL)
            ]
            x_cont = x[:, cont_cols] if cont_cols else x.new_zeros((x.size(0), 1))
            fused = torch.cat(
                [
                    self.continuous_encoder(x_cont),
                    self.node_type_emb(node_types),
                    self.label_emb(label_ids),
                ],
                dim=-1,
            )
            batch.x = self.activation(self.fusion(fused))
        else:
            batch.x = self.activation(self.proj(x))
        return batch


def _resolve_edge_attr(batch, edge_dim: int):
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is None:
        raise ValueError(
            "edge_attr is required on every graph batch, but edge_attr is missing"
        )
    return edge_attr


@register_layer("gatv2conv")
class GATv2Conv(torch.nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        edge_dim = getattr(layer_config, "edge_dim", 4)
        self.model = pyg.nn.GATv2Conv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
            edge_dim=edge_dim,
        )

    def forward(self, batch):
        edge_attr = _resolve_edge_attr(
            batch,
            getattr(self.model, "edge_dim", 4),
        )
        batch.x = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)
        return batch


@register_layer("gineconv")
class GINEConvLayer(torch.nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        edge_dim = getattr(layer_config, "edge_dim", 4)
        nn_layer = torch.nn.Sequential(
            torch.nn.Linear(layer_config.dim_in, layer_config.dim_out),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_config.dim_out, layer_config.dim_out),
        )
        self.edge_dim = edge_dim
        self.model = GINEConv(nn=nn_layer, edge_dim=edge_dim)

    def forward(self, batch):
        edge_attr = _resolve_edge_attr(batch, self.edge_dim)
        batch.x = self.model(batch.x, batch.edge_index, edge_attr)
        return batch


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
    cfg.expression_graph.features = CN()
    cfg.expression_graph.features.node = True
    cfg.expression_graph.features.topology = True
    cfg.expression_graph.features.positional = CN()
    cfg.expression_graph.features.positional.enabled = True
    cfg.expression_graph.features.positional.encodings = ["lpe", "rwpe"]
    cfg.expression_graph.features.edge = True
    cfg.expression_graph.active_features = ""  # Explicit override list, or empty for grouped toggles
    cfg.expression_graph.synthetic = False
    cfg.expression_graph.synthetic_dataset = ""  # Synthetic dataset name
    cfg.expression_graph.edge_direction = "top_down"  # top_down | bottom_up | bidirectional
    cfg.expression_graph.heterogeneous = False  # heterogeneous or homogeneous graph representation
    cfg.expression_graph.pos_label = 1  # Overwritten from training class counts at load time
    cfg.train.mode = "custom"
    cfg.train.epochs = 100
    cfg.train.epoch_warmup = 5
    cfg.train.curated_eval_period = 5
    cfg.train.curated_eval_on_test_highscore = True
    cfg.train.num_workers = 0
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
    - cfg.expression_graph.features: grouped node/edge feature toggles
    - cfg.expression_graph.active_features: optional explicit feature override list
    - cfg.expression_graph.synthetic: injects synthetic mode (True or False)
    - cfg.expression_graph.synthetic_dataset: injects synthetic dataset name
    - cfg.expression_graph.edge_direction: AST message-passing direction
    """
    from gnn.shared.utils.graph_utils import validate_edge_direction

    dataset_name = cfg.dataset.name
    batch_size = cfg.train.batch_size
    seed = getattr(cfg, "seed", 42001)

    mode = getattr(cfg.expression_graph, "mode", "graph")
    layer_type = validate_layer_type(cfg.gnn.layer_type)
    cfg.dataset.edge_dim = resolve_edge_dim()
    from gnn.supervised_learning.supervised_config import resolve_expression_graph_features

    synthetic = getattr(cfg.expression_graph, "synthetic", False)
    synthetic_dataset = getattr(cfg.expression_graph, "synthetic_dataset", "")
    edge_direction = validate_edge_direction(
        getattr(cfg.expression_graph, "edge_direction", "top_down")
    )
    heterogeneous = getattr(cfg.expression_graph, "heterogeneous", False)

    feature_selection, active_features = resolve_expression_graph_features(
        {
            "features": getattr(cfg.expression_graph, "features", {}),
            "active_features": getattr(cfg.expression_graph, "active_features", ""),
        },
    )

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
    print(f"  Injected Layer Type:     {layer_type}")
    print(f"  Injected Edge Dim:       {cfg.dataset.edge_dim}")
    print(f"  Injected Synthetic:      {synthetic}")
    print(f"  Injected Edge Direction: {edge_direction}")
    if synthetic:
        print(f"  Injected Synth Dataset:  {synthetic_dataset}")
    print(f"  Injected Heterogeneous:  {heterogeneous}")
    print(f"  Injected Feature Groups: {feature_selection.enabled_groups()}")
    print(f"  Injected Positional:     {list(feature_selection.positional_encodings)}")
    print(f"  Injected Features:       {feature_selection.summary()}")
    print("----------------------------------------\n")

    from gnn.shared.utils.graph_loader import GraphDataLoader

    loader = GraphDataLoader(
        name=dataset_name,
        mode=mode,
        heterogeneous=heterogeneous,
        edge_direction=edge_direction,
    )

    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        seed=seed,
        mode=mode,
        active_features=active_features,
        graph_loader=loader,
        synthetic=synthetic,
        synthetic_dataset_name=synthetic_dataset,
        layer_type=layer_type,
        heterogeneous=heterogeneous,
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
        curated_data_list = [  # curated real-world holdout — scheduled during training, final test at end
            pipeline.curated_dataset[i] for i in range(len(pipeline.curated_dataset))
        ]

        all_data_list = train_data_list + val_data_list + curated_data_list

        train_indices = list(range(len(train_data_list)))
        # val = unseen synthetic (20%): runs every eval_period epoch, used for checkpoint selection
        val_indices = list(range(len(train_data_list), len(train_data_list) + len(val_data_list)))
        # test = curated real-world holdout (periodic during training + final run with best ckpt)
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

    # --- Compute class weights from training data and register weighted cross entropy loss ---
    try:
        train_labels = torch.tensor(
            [pipeline.train_dataset[i].y.item() for i in range(len(pipeline.train_dataset))]
        )
        class_counts = torch.bincount(train_labels.long(), minlength=2)
        total_train = len(train_labels)
        num_classes = len(class_counts)
        # Same formula as main.py / preprocessing.py: total / (num_classes * count_per_class)
        class_weights = total_train / (num_classes * class_counts.float().clamp(min=1))
        class_weights = class_weights.to(torch.float)

        set_pos_label_from_train_labels(train_labels)

        print(f"[GraphGym] Computed class weights from {total_train} training samples:")
        print(f"  Class 0 (gMGF):   count={class_counts[0].item()}, weight={class_weights[0].item():.4f}")
        print(f"  Class 1 (Newton): count={class_counts[1].item()}, weight={class_weights[1].item():.4f}")

        configure_class_weights(class_weights)

    except Exception as e:
        print(f"[Warning] Failed to compute class weights: {e}")
        print("[Warning] Falling back to uniform class weights.")

    # Calculate and save class balance information for validation synthetic and curated datasets
    try:
        import json
        
        # Calculate counts for validation synthetic (always present)
        val_syn_df = pipeline.test_dataset.df
        val_syn_0 = int((val_syn_df["faster_algorithm"] == 0).sum())
        val_syn_1 = int((val_syn_df["faster_algorithm"] == 1).sum())
        val_syn_total = len(val_syn_df)
        
        if synthetic:
            # Calculate counts for validation curated
            val_cur_df = pipeline.curated_dataset.df
            val_cur_0 = int((val_cur_df["faster_algorithm"] == 0).sum())
            val_cur_1 = int((val_cur_df["faster_algorithm"] == 1).sum())
            val_cur_total = len(val_cur_df)
        else:
            val_cur_0 = 0
            val_cur_1 = 0
            val_cur_total = 0
            
        balance_info = {
            "validation_synthetic": {
                "0": val_syn_0,
                "1": val_syn_1,
                "total": val_syn_total
            },
            "validation_curated": {
                "0": val_cur_0,
                "1": val_cur_1,
                "total": val_cur_total
            }
        }
        
        # Write to results/agg/class_balance.json
        out_path = Path(cfg.out_dir)
        agg_dir = out_path.parent / "agg"
        agg_dir.mkdir(parents=True, exist_ok=True)
        with open(agg_dir / "class_balance.json", "w", encoding="utf-8") as f:
            json.dump(balance_info, f, indent=4)
        print(f"[GraphGym] Saved class balance info to {agg_dir / 'class_balance.json'}")
    except Exception as e:
        print(f"[Warning] Failed to calculate or save class balance info: {e}")

    return ExpressionGraphDataset(
        all_data_list, train_indices, val_indices, test_indices
    )


register_loader("custom_expression_graphs", load_custom_expression_graphs)
