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
from torch.optim import AdamW
from torch_geometric.graphgym.register import register_scheduler, register_optimizer

from gnn.supervised_learning.preprocessing import GraphPipeline  # noqa: F401
from gnn.supervised_learning.run_results.eval_metrics import compute_confidence_metrics
from gnn.supervised_learning.supervised_config import (
    LAYERS_WITHOUT_EDGE_FEATURES,
    architecture_from_layer_type,
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
from gnn.shared.utils.graph_utils import NODE_FEATURE_SCHEMA


@register_node_encoder("ExpressionNodeEncoder")
class ExpressionNodeEncoder(torch.nn.Module):
    """GraphGym node encoder: LayerNorm → Linear → GELU projection of one-hot node features."""

    def __init__(self, dim_emb: int):
        super().__init__()
        n_features = len(NODE_FEATURE_SCHEMA)
        self.encoder = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, dim_emb),
            nn.GELU(),
        )

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
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


# --------------------------------------------------------------------------- #
# Custom network: routes the supervised pipeline through ExpressionGNN.
# Selected via cfg.model.type = "expression_classifier".
# --------------------------------------------------------------------------- #
from torch_geometric.graphgym.register import register_network
from gnn.shared.models.gnn_backbones import ExpressionGNN


@register_network("expression_classifier")
class ExpressionClassifierNetwork(torch.nn.Module):
    """GraphGym adapter around the shared :class:`ExpressionGNN` (classify=True).

    A custom network receives the RAW ``batch.x`` (the ``ExpressionNodeEncoder``
    is never auto-applied by GraphGym for custom network types). ExpressionGNN
    does its own LayerNorm→Linear node encoding. GraphGym batches carry no
    graph-level global features, so global_dim=0.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        self._last_aux_loss = torch.zeros(())

        layer_type = validate_layer_type(cfg.gnn.layer_type)
        if layer_type in LAYERS_WITHOUT_EDGE_FEATURES:
            raise ValueError(
                f"expression_classifier supports only edge-aware layer types "
                f"(not {sorted(LAYERS_WITHOUT_EDGE_FEATURES)}); got {layer_type!r}"
            )
        names = list(getattr(cfg.expression_graph, "active_feature_names", []) or [])
        self.net = ExpressionGNN(
            input_dim=(len(names) or dim_in),
            hidden_dim=cfg.gnn.dim_inner,
            global_dim=0,  # GraphGym batches carry no graph-level global features
            output_dim=dim_out,
            heads=getattr(cfg.gnn, "att_heads", 4),
            architecture=architecture_from_layer_type(layer_type),
            activation=cfg.gnn.act,
            num_layers=cfg.gnn.layers_mp,
            dropout=cfg.gnn.dropout,
            graph_pooling=cfg.model.graph_pooling,
            classify=True,
        )

    def forward(self, batch):
        logits = self.net(
            batch.x,
            batch.edge_index,
            batch.batch,
            global_features=None,
        )
        if logits.size(-1) == 1:  # match the stock single-logit BCE path
            logits = logits.view(-1)
        return logits, batch.y


# DiffPool exposes a link+entropy auxiliary loss on the network; fold it into the
# training loss. Patching _shared_step mirrors the Logger/LoggerCallback monkeypatches
# above. No-op for the stock GNN (no _last_aux_loss) and for topk/legacy (aux == 0).
import time as _time
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.loss import compute_loss as _compute_loss


def _shared_step_with_aux(self, batch, split):
    batch.split = split
    pred, true = self(batch)
    loss, pred_score = _compute_loss(pred, true)
    if split == "train":
        aux = getattr(self.model, "_last_aux_loss", None)
        if aux is not None:
            loss = loss + float(getattr(cfg.gnn, "aux_loss_weight", 1.0)) * aux
    return dict(
        loss=loss,
        true=true,
        pred_score=pred_score.detach(),
        step_end_time=_time.time(),
    )


GraphGymModule._shared_step = _shared_step_with_aux


def _make_lr_lambda(warmup_epochs: int, post_fn):
    """Linear ramp from 0→1 over warmup_epochs, then delegates to post_fn(epoch)."""
    _warmup_f = float(max(1, warmup_epochs))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / _warmup_f
        return post_fn(epoch)

    return lr_lambda


def cosine_with_warmup_scheduler(optimizer, max_epoch):
    warmup_epochs = getattr(cfg.train, "epoch_warmup", 5)
    _scale = float(max(1, max_epoch - warmup_epochs))

    def post(epoch: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * float(epoch - warmup_epochs) / _scale))

    return LambdaLR(optimizer, _make_lr_lambda(warmup_epochs, post))


register_scheduler("cosine_with_warmup", cosine_with_warmup_scheduler)


# from_config matches the parameter names (base_lr, weight_decay) against
# cfg.optim keys, so no new config keys are needed for AdamW. AdamW differs
# from Adam only in applying *decoupled* weight decay.
def adamw_optimizer(params, base_lr, weight_decay):
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)


register_optimizer("adamw", adamw_optimizer)


def cosine_with_restarts_scheduler(optimizer, max_epoch):
    """Linear warmup then cosine annealing with warm restarts (SGDR).

    Restarts the cosine every cfg.train.restart_period epochs (default 20).
    """
    warmup_epochs = getattr(cfg.train, "epoch_warmup", 5)
    period = max(1, int(getattr(cfg.train, "restart_period", 20)))
    _period_f = float(period)

    def post(epoch: int) -> float:
        progress = float(epoch - warmup_epochs) % period / _period_f
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, _make_lr_lambda(warmup_epochs, post))


register_scheduler("cosine_with_restarts", cosine_with_restarts_scheduler)



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
    # Uniform flat per-category form: each leaf defaults to None ("all members").
    # YAML may override with a bool (true/false) or a list subset, e.g.
    #   node: [node_type, value]   topology: false   positional: [anchor_periodic]
    # None defaults are required so YACS accepts either a bool or a list override.
    cfg.expression_graph.features.node = None
    cfg.expression_graph.features.topology = None
    cfg.expression_graph.features.positional = None
    cfg.expression_graph.features.edge = None
    cfg.expression_graph.active_features = ""  # Explicit override list, or empty for grouped toggles
    cfg.expression_graph.synthetic = False
    cfg.expression_graph.synthetic_dataset = ""  # Synthetic dataset name
    cfg.expression_graph.edge_direction = "top_down"  # top_down | bottom_up | bidirectional
    cfg.expression_graph.heterogeneous = False  # heterogeneous or homogeneous graph representation
    cfg.expression_graph.add_kappa = False  # merge kappa (h-function) subgraphs from datasets/kappas/
    cfg.expression_graph.add_virtual_supernode = False  # add a fully-connected virtual supernode
    cfg.expression_graph.pos_label = 1  # Overwritten from training class counts at load time
    # Resolved ordered node-feature names, stashed by the loader so the custom
    # ExpressionClassifierNetwork (built later by create_model) can locate categorical
    # columns BY NAME. Empty => fall back to the full node schema.
    cfg.expression_graph.active_feature_names = []
    # Edge-type metadata for the heterogeneous to_hetero model, stashed by the loader as a
    # list of [src, relation, dst] lists. Empty unless cfg.expression_graph.heterogeneous.
    cfg.expression_graph.hetero_edge_types = []
    # Structural / pooling axes for the ExpressionClassifierNetwork backbone. Declared here
    # so YACS accepts grid.yaml overrides; ignored when cfg.model.type == "gnn".
    cfg.gnn.variant = "legacy"        # legacy | standard | pooling | pooling_skip
    cfg.gnn.pool_type = "topk"        # topk | diffpool (only used by pooling* variants)
    cfg.gnn.aux_loss_weight = 1.0     # weight for the DiffPool link+entropy aux loss
    cfg.train.mode = "custom"
    cfg.train.epochs = 100
    cfg.train.epoch_warmup = 5
    cfg.train.restart_period = 20      # period (epochs) for cosine_with_restarts
    cfg.train.curated_eval_period = 5
    cfg.train.curated_eval_on_test_highscore = True
    cfg.train.num_workers = 0
    # Early stopping (opt-in). Validation runs every epoch in this custom Lightning
    # path, so patience is counted in epochs. Monitored metric must be one always
    # logged by ValMetricLogger every validation epoch: val_pr_auc (mode max) or
    # val_loss (mode min) — the curated-holdout metrics are logged only on a schedule
    # and would break EarlyStopping.
    cfg.train.early_stopping = False          # master switch
    cfg.train.early_stopping_monitor = "val_pr_auc"  # val_pr_auc | val_loss
    cfg.train.early_stopping_patience = 10    # epochs without improvement before stopping
    cfg.train.early_stopping_min_delta = 0.0  # min change that counts as an improvement
    cfg.params = 0
    # set_run_dir() writes cfg.run_dir at runtime and dump_cfg persists it to config.yaml.
    # Pre-register the key here so load_cfg can merge saved configs without a YACS
    # 'Non-existent config key: run_dir' error during diagnostic/eval reloads.
    cfg.run_dir = ""



register_config("expression_graph", set_custom_cfg)


def load_custom_expression_graphs(format, name, dataset_dir):
    """
    GraphGym Loader for custom expression graphs.
    Uses Dependency Injection by reading dataset properties dynamically from GraphGym's
    global config:
    - cfg.dataset.name: e.g., "run_20260408_160456/dataset_4"
        (injects dataset selection)
    - cfg.train.batch_size: injects batch size
    - cfg.seed: injects the split seed (set explicitly in config_supervised.yaml;
        GraphGym's own unset default is 0, so the 42001 getattr fallback below only
        fires if the cfg node itself lacks the key, which it never does in practice)
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
    # Fail fast (before the dataset load) if expression_classifier was paired with a
    # non-edge-aware layer type; otherwise ExpressionClassifierNetwork.__init__ would
    # only raise after the full data load/cache. Mirrors the check in that constructor.
    is_expression_classifier = cfg.model.type == "expression_classifier"
    if is_expression_classifier and layer_type in LAYERS_WITHOUT_EDGE_FEATURES:
        raise ValueError(
            f"model.type 'expression_classifier' supports only edge-aware layer types "
            f"(not {sorted(LAYERS_WITHOUT_EDGE_FEATURES)}); got {layer_type!r}"
        )
    cfg.dataset.edge_dim = resolve_edge_dim()
    from gnn.supervised_learning.supervised_config import resolve_expression_graph_features

    synthetic = getattr(cfg.expression_graph, "synthetic", False)
    synthetic_dataset = getattr(cfg.expression_graph, "synthetic_dataset", "")
    edge_direction = validate_edge_direction(
        getattr(cfg.expression_graph, "edge_direction", "top_down")
    )
    heterogeneous = getattr(cfg.expression_graph, "heterogeneous", False)
    add_kappa = getattr(cfg.expression_graph, "add_kappa", False)
    add_virtual_supernode = getattr(
        cfg.expression_graph, "add_virtual_supernode", False
    )

    feature_selection, active_features = resolve_expression_graph_features(
        {
            "features": getattr(cfg.expression_graph, "features", {}),
            "active_features": getattr(cfg.expression_graph, "active_features", ""),
        },
    )
    # Anchor positional encoding and the fully-connected supernode are mutually exclusive.
    from gnn.shared.utils.feature_config import validate_positional_supernode_compatibility

    validate_positional_supernode_compatibility(feature_selection, add_virtual_supernode)
    # Expose the resolved ordered node-feature names so the custom
    # ExpressionClassifierNetwork (built later by create_model) can locate categorical
    # columns BY NAME. Empty list => the network falls back to the full node schema.
    cfg.expression_graph.active_feature_names = list(active_features) if active_features else []

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
    print(f"  Injected Add Kappa:      {add_kappa}")
    print(f"  Injected Add Supernode:  {add_virtual_supernode}")
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
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
    )

    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        seed=seed,
        mode=mode,
        active_features=active_features,
        graph_loader=loader,
        synthetic=synthetic,
        synthetic_dataset_name=synthetic_dataset,
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
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
        # The 20% holdout is split into two DISJOINT halves so checkpoint selection
        # (val) never sees the generalization set (test). A deterministic interleave
        # (even -> val, odd -> test) keeps class proportions roughly balanced without
        # needing labels here. Previously val and test pointed at the SAME indices,
        # which leaked the test set into model selection.
        holdout = [
            pipeline.test_dataset[i] for i in range(len(pipeline.test_dataset))
        ]
        val_data_list = holdout[0::2]
        test_data_list = holdout[1::2]

        all_data_list = train_data_list + val_data_list + test_data_list

        n_train = len(train_data_list)
        n_val = len(val_data_list)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, len(all_data_list)))

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

    # Save per-run split DataFrames as CSV training artifacts for reproducibility
    try:
        split_dir = Path(cfg.out_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        pipeline.train_dataset.df.to_csv(split_dir / "split_train.csv", index=False)
        if synthetic:
            pipeline.test_dataset.df.to_csv(split_dir / "split_val.csv", index=False)
            pipeline.curated_dataset.df.to_csv(split_dir / "split_test.csv", index=False)
        else:
            holdout_df = pipeline.test_dataset.df
            holdout_df.iloc[0::2].reset_index(drop=True).to_csv(split_dir / "split_val.csv", index=False)
            holdout_df.iloc[1::2].reset_index(drop=True).to_csv(split_dir / "split_test.csv", index=False)
        print(f"[GraphGym] Saved split CSVs to {split_dir}/split_{{train,val,test}}.csv")
    except Exception as e:
        print(f"[Warning] Failed to save split CSVs: {e}")

    if heterogeneous:
        all_data_list, edge_types = prepare_hetero_data_list(all_data_list)
        # Stash the edge-type metadata so the network can build its to_hetero model.
        cfg.expression_graph.hetero_edge_types = [list(et) for et in edge_types]
        print(f"[GraphGym] Heterogeneous mode: {len(edge_types)} edge types collected")

    return ExpressionGraphDataset(
        all_data_list, train_indices, val_indices, test_indices
    )


register_loader("custom_expression_graphs", load_custom_expression_graphs)
