import sys
import warnings
from pathlib import Path

# Ensure codebase/src is on sys.path when invoked as a subprocess (e.g. from run_all.py).
_src_root = str(Path(__file__).resolve().parent.parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

# Suppress harmless PyG internal GraphGym InMemoryDataset deprecation warning
warnings.filterwarnings("ignore", message=".*InMemoryDataset.*")
import torch
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, set_run_dir, dump_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.imports import pl


import gnn.supervised_learning.loader_graphgym  # noqa
from gnn.supervised_learning.run_results.feature_importance import (
    run_post_training_feature_importance,
)
from gnn.supervised_learning.preprocessing import GraphPipeline # noqa
from gnn.supervised_learning.curated_eval_schedule import (
    CuratedEvalSchedule,
    parse_curated_eval_schedule,
    should_evaluate_curated,
)
from gnn.supervised_learning.supervised_config import (
    resolve_edge_dim,
    validate_layer_type,
)

set_cfg(cfg)


def dump_summary_cfg(cfg, out_dir: Path) -> None:
    """Write a human-readable summary of the key hyperparameters to summary_config.yaml."""
    import yaml

    eg = cfg.expression_graph
    active = eg.get("active_feature_names", []) or []
    if not active:
        # Fall back to the old boolean flags
        feat = eg.get("features", {})
        active = [k for k, v in (feat.items() if hasattr(feat, "items") else []) if v]

    summary = {
        "model": {
            "layer_type": cfg.gnn.layer_type,
            "dim_inner": cfg.gnn.dim_inner,
            "layers_mp": cfg.gnn.layers_mp,
            "layers_pre_mp": cfg.gnn.layers_pre_mp,
            "layers_post_mp": cfg.gnn.layers_post_mp,
            "dropout": cfg.gnn.dropout,
        },
        "graph": {
            "mode": eg.mode,
            "edge_direction": eg.edge_direction,
        },
        "features": {
            "active_feature_names": active,
            "edge_dim": cfg.dataset.edge_dim,
        },
        "training": {
            "base_lr": cfg.optim.base_lr,
            "optimizer": cfg.optim.optimizer,
            "scheduler": cfg.optim.scheduler,
            "max_epoch": cfg.optim.max_epoch,
            "batch_size": cfg.train.batch_size,
            "weight_decay": cfg.optim.weight_decay,
        },
    }

    path = out_dir / "summary_config.yaml"
    with open(path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)


def _register_mp_hook(pl_module, hook_fn):
    """Register a forward hook on the message-passing stage, if the model has one.

    The Dirichlet-energy callbacks hook the message-passing output. PyG's stock ``GNN``
    exposes it as ``pl_module.model.mp``; the custom ``expression_classifier`` backbone
    has no ``.mp`` stage, so it exposes a parameter-free ``DirichletProbe`` submodule at
    its message-passing output instead (see ``gnn_backbones.DirichletProbe``). Hook
    whichever is present. Returns the handle, or ``None`` when neither exists (energy is
    then reported NaN — not measured — rather than crashing).
    """
    model = getattr(pl_module, "model", None)
    if model is None:
        return None
    mp = getattr(model, "mp", None)
    if mp is None:
        from gnn.shared.models.gnn_backbones import DirichletProbe

        mp = next((m for m in model.modules() if isinstance(m, DirichletProbe)), None)
    if mp is None:
        return None
    return mp.register_forward_hook(hook_fn)


class CuratedEvalCallback(pl.callbacks.Callback):
    """Run curated holdout evaluation on a schedule, not every validation epoch."""

    def __init__(self, curated_loader, schedule: CuratedEvalSchedule):
        super().__init__()
        self.curated_loader = curated_loader
        self.schedule = schedule
        self._best_val_pr_auc = float("-inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.curated_loader is None:
            return
        if getattr(trainer, "sanity_checking", False):
            return

        val_pr_auc_tensor = trainer.callback_metrics.get("val_pr_auc")
        if val_pr_auc_tensor is None:
            return
        val_pr_auc = float(val_pr_auc_tensor)
        is_new_highscore = val_pr_auc > self._best_val_pr_auc
        if is_new_highscore:
            self._best_val_pr_auc = val_pr_auc

        should_run, reason = should_evaluate_curated(
            trainer.current_epoch,
            self.schedule,
            is_new_test_highscore=is_new_highscore,
        )
        if not should_run:
            return

        metrics = self._evaluate_curated(pl_module)
        pl_module.log(
            "val_loss_curated",
            metrics["loss"],
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log("val_auc_curated", metrics["auc"], prog_bar=True, sync_dist=True)
        pl_module.log(
            "val_pr_auc_curated",
            metrics["pr_auc"],
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            "val_dirichlet_energy_curated",
            metrics.get("dirichlet_energy", 0.0),
            prog_bar=True,
            sync_dist=True,
        )
        self._write_test_logger_stats(trainer, metrics)
        print(
            f"[GraphGym] Curated holdout eval (epoch {trainer.current_epoch + 1}, "
            f"reason={reason}) | PR-AUC={metrics['pr_auc']:.4f} | Dirichlet Energy={metrics.get('dirichlet_energy', 0.0):.6f}"
        )

    def _evaluate_curated(self, pl_module) -> dict[str, float]:
        from gnn.supervised_learning.loader_graphgym import compute_binary_metrics
        from gnn.shared.utils.graph_utils import compute_normalized_dirichlet_energy

        pl_module.eval()
        loss_sum = 0.0
        count = 0
        true_parts = []
        pred_parts = []

        mp_embeddings = []
        def hook_fn(module, inputs, outputs):
            mp_embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu(), getattr(outputs, 'edge_attr', None)))

        hook_handle = _register_mp_hook(pl_module, hook_fn)

        with torch.no_grad():
            for batch in self.curated_loader:
                batch = batch.to(pl_module.device)
                outputs = pl_module._shared_step(batch, split="test")
                if outputs is None:
                    continue
                loss = float(outputs["loss"])
                batch_size = outputs["true"].size(0)
                loss_sum += loss * batch_size
                count += batch_size
                true_parts.append(outputs["true"].detach().cpu())
                pred_parts.append(outputs["pred_score"].detach().cpu())

        if hook_handle is not None:
            hook_handle.remove()

        if count == 0:
            return {
                "loss": 0.0,
                "auc": 0.0,
                "pr_auc": 0.0,
                "true": None,
                "pred": None,
                "dirichlet_energy": float("nan"),
            }

        true = torch.cat(true_parts)
        pred = torch.cat(pred_parts)
        metric_values = compute_binary_metrics(true, pred)

        # Calculate dirichlet energy
        energies = []
        for x, edge_index, edge_attr in mp_embeddings:
            energy = compute_normalized_dirichlet_energy(x, edge_index)
            energies.append(energy)
        # Empty => no .mp stage hooked (custom backbone): not measured, not a real 0.0.
        avg_energy = sum(energies) / len(energies) if energies else float("nan")

        return {
            "loss": loss_sum / count,
            "true": true,
            "pred": pred,
            "dirichlet_energy": avg_energy,
            **metric_values,
        }

    @staticmethod
    def _write_test_logger_stats(trainer, metrics: dict) -> None:
        true = metrics.get("true")
        pred = metrics.get("pred")
        if true is None or pred is None or true.numel() == 0:
            return

        for callback in trainer.callbacks:
            if not isinstance(callback, LoggerCallback):
                continue
            if len(callback._logger) <= 2:
                return
            test_logger = callback.test_logger
            test_logger.reset()
            test_logger.update_stats(
                true=true,
                pred=pred,
                loss=float(metrics["loss"]),
                lr=0.0,
                time_used=0.0,
                params=cfg.params,
                dirichlet_energy=float(metrics.get("dirichlet_energy", 0.0)),
            )
            test_logger.write_epoch(trainer.current_epoch)
            return


class DirichletLoggerCallback(LoggerCallback):
    """
    Subclasses GraphGym's standard LoggerCallback to intercept GNN message-passing
    embeddings, compute normalized Dirichlet energy for each batch, and log it
    directly to the stats.json outputs as a custom metric.
    """
    def __init__(self):
        super().__init__()
        self._embeddings = []
        self._hook_handle = None

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self._embeddings = []
        def hook_fn(module, inputs, outputs):
            self._embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu()))
        self._hook_handle = _register_mp_hook(pl_module, hook_fn)

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        self._embeddings = []
        def hook_fn(module, inputs, outputs):
            self._embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu()))
        self._hook_handle = _register_mp_hook(pl_module, hook_fn)

    def on_test_epoch_start(self, trainer, pl_module):
        super().on_test_epoch_start(trainer, pl_module)
        self._embeddings = []
        def hook_fn(module, inputs, outputs):
            self._embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu()))
        self._hook_handle = _register_mp_hook(pl_module, hook_fn)

    def _get_dirichlet_energy_for_batch(self):
        if not self._embeddings:
            # No embeddings means the model has no .mp stage (e.g. the custom
            # expression_classifier backbone): the metric was NOT measured. Report
            # NaN so it stays distinguishable from a genuinely measured energy of 0.0.
            return float("nan")
        x, edge_index = self._embeddings.pop(0)
        from gnn.shared.utils.graph_utils import compute_normalized_dirichlet_energy
        return compute_normalized_dirichlet_energy(x, edge_index)

    def _get_stats(self, epoch_start_time: int, outputs: dict, trainer: 'pl.Trainer') -> dict:
        stats = super()._get_stats(epoch_start_time, outputs, trainer)
        stats['dirichlet_energy'] = self._get_dirichlet_energy_for_batch()
        return stats

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def on_train_epoch_end(self, trainer, pl_module):
        self._remove_hook()
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._remove_hook()
        super().on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._remove_hook()
        super().on_test_epoch_end(trainer, pl_module)


class ValMetricLogger(pl.callbacks.Callback):
    """
    Bridges GraphGym's custom Logger (which writes to stats.json) with
    PyTorch Lightning's metric system (which ModelCheckpoint monitors).
    
    On each validation batch, accumulates loss and prediction data per dataloader.
    On validation epoch end, computes metrics and calls trainer.callback_metrics
    via pl_module.log() so that ModelCheckpoint can monitor them.
    """
    def __init__(self):
        super().__init__()
        self._val_data = {}
        self._hook_handle = None
        self._embeddings = []
        
    def _init_data(self, idx):
        if idx not in self._val_data:
            self._val_data[idx] = {
                'loss_sum': 0.0,
                'count': 0,
                'true': [],
                'pred': []
            }
            
    def on_validation_epoch_start(self, trainer, pl_module):
        self._embeddings = []
        def hook_fn(module, inputs, outputs):
            self._embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu(), getattr(outputs, 'edge_attr', None)))
        self._hook_handle = _register_mp_hook(pl_module, hook_fn)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        self._init_data(dataloader_idx)
        loss = float(outputs['loss'])
        batch_size = outputs['true'].size(0)
        self._val_data[dataloader_idx]['loss_sum'] += loss * batch_size
        self._val_data[dataloader_idx]['count'] += batch_size
        self._val_data[dataloader_idx]['true'].append(outputs['true'].detach().cpu())
        self._val_data[dataloader_idx]['pred'].append(outputs['pred_score'].detach().cpu())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            
        # Calculate average dirichlet energy
        from gnn.shared.utils.graph_utils import compute_normalized_dirichlet_energy
        energies = []
        for x, edge_index, edge_attr in self._embeddings:
            energy = compute_normalized_dirichlet_energy(x, edge_index)
            energies.append(energy)
        # Empty => no .mp stage hooked (custom backbone): not measured, not a real 0.0.
        avg_energy = sum(energies) / len(energies) if energies else float("nan")

        for idx, data in self._val_data.items():
            if data['count'] == 0:
                continue

            avg_loss = data['loss_sum'] / data['count']
            true = torch.cat(data['true'])
            pred = torch.cat(data['pred'])
            from gnn.supervised_learning.loader_graphgym import compute_binary_metrics

            metrics = compute_binary_metrics(true, pred)

            if idx == 0:
                pl_module.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
                pl_module.log('val_auc', metrics['auc'], prog_bar=True, sync_dist=True)
                pl_module.log('val_pr_auc', metrics['pr_auc'], prog_bar=True, sync_dist=True)
                pl_module.log('val_dirichlet_energy', avg_energy, prog_bar=True, sync_dist=True)
                print(f"[GraphGym] Validation Normalized Dirichlet Energy: {avg_energy:.6f}")

        self._val_data = {}


# Validation metrics ValMetricLogger logs every epoch, with the direction that counts
# as "better". Only these are safe to monitor for early stopping; the curated-holdout
# metrics are logged on a schedule and would intermittently vanish from callback_metrics.
_EARLY_STOPPING_MONITORS = {"val_pr_auc": "max", "val_loss": "min"}


class _WarmupAwareEarlyStopping(pl.callbacks.EarlyStopping):
    """EarlyStopping that ignores all epochs before `warmup_epochs`.

    During the LR-warmup ramp the model has not yet converged, so val metrics
    can be anomalously high or low and must not seed the patience counter.
    Monitoring begins at the first validation epoch >= warmup_epochs, at which
    point best_score is initialised from that epoch's metric — not from anything
    seen during warmup.
    """

    def __init__(self, *args, warmup_epochs: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self._warmup_epochs:
            return
        super().on_validation_end(trainer, pl_module)


def _build_early_stopping_callback() -> "_WarmupAwareEarlyStopping":
    """Construct the EarlyStopping callback from cfg.train.early_stopping_* settings."""
    monitor = str(getattr(cfg.train, "early_stopping_monitor", "val_pr_auc"))
    if monitor not in _EARLY_STOPPING_MONITORS:
        raise ValueError(
            f"cfg.train.early_stopping_monitor must be one of "
            f"{sorted(_EARLY_STOPPING_MONITORS)} (metrics logged every validation "
            f"epoch); got {monitor!r}"
        )
    mode = _EARLY_STOPPING_MONITORS[monitor]
    patience = int(getattr(cfg.train, "early_stopping_patience", 10))
    min_delta = float(getattr(cfg.train, "early_stopping_min_delta", 0.0))
    warmup_epochs = int(getattr(cfg.train, "epoch_warmup", 0))
    print(
        f"[GraphGym] Early stopping ENABLED: monitor={monitor} (mode={mode}), "
        f"patience={patience} epochs, min_delta={min_delta}, "
        f"warmup_skip={warmup_epochs} epochs"
    )
    return _WarmupAwareEarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        warmup_epochs=warmup_epochs,
    )


def train_with_best_ckpt(model, datamodule, logger=True):
    """
    Custom training function that replaces PyG's built-in train().
    
    Key difference: Uses ModelCheckpoint with monitor='val_pr_auc' (mode='max')
    so the best model (by validation PR-AUC on unseen synthetic data) is saved.
    After training, the final test (on curated real-world data) uses this
    best checkpoint — NOT the last epoch.
    """
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')
    
    callbacks = []

    if logger:
        callbacks.append(DirichletLoggerCallback())

    callbacks.append(ValMetricLogger())

    ckpt_cbk = None
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=get_ckpt_dir(),
            monitor='val_pr_auc',
            mode='max',
            save_top_k=1,
            save_last=True,
            filename='best-{epoch}-{val_pr_auc:.4f}',
            verbose=True,
        )
        callbacks.append(ckpt_cbk)

    # EarlyStopping patience is in epochs; validation runs every epoch in this path.
    if getattr(cfg.train, "early_stopping", False):
        callbacks.append(_build_early_stopping_callback())

    curated_loader = None
    if cfg.expression_graph.synthetic and len(datamodule.loaders) >= 3:
        curated_loader = datamodule.loaders[2]
        curated_schedule = parse_curated_eval_schedule(cfg.train)
        callbacks.append(CuratedEvalCallback(curated_loader, curated_schedule))
        print(
            "[GraphGym] Curated holdout schedule: "
            f"period={curated_schedule.period}, "
            f"on_test_highscore={curated_schedule.on_test_highscore}"
        )

    trainer = pl.Trainer(
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
    )

    # Train the model (validation runs every epoch on unseen synthetic data; the
    # ModelCheckpoint/EarlyStopping callbacks consume those per-epoch val metrics).
    trainer.fit(model, datamodule=datamodule)
    
    # Test using the BEST checkpoint (by val_pr_auc), not the last epoch
    best_path = (
        ckpt_cbk.best_model_path
        if ckpt_cbk is not None and ckpt_cbk.best_model_path
        else None
    )
    
    if best_path:
        print(f"\n[GraphGym] Loading BEST checkpoint for final test: {best_path}")
        print(f"[GraphGym] Best val_pr_auc: {ckpt_cbk.best_model_score:.4f}")
        trainer.test(model, datamodule=datamodule, ckpt_path=best_path)
    else:
        print("\n[GraphGym] No best checkpoint found, testing with last model weights.")
        trainer.test(model, datamodule=datamodule)

    try:
        run_post_training_feature_importance(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_path,
            out_dir=Path(cfg.out_dir),
            device=next(model.parameters()).device,
        )
    except Exception as exc:
        print(f"[GraphGym] Feature importance analysis failed: {exc}")

    try:
        from gnn.supervised_learning.run_results.training_curves import TrainingCurvePlotter
        plotter = TrainingCurvePlotter(
            results_dir=Path(cfg.out_dir).parent,
            output_dir=Path(cfg.out_dir).parent / "eval_plots",
            experiment_name=Path(cfg.out_dir).parent.name
        )
        run_dir = Path(cfg.out_dir)
        series = plotter._load_run_series(run_dir)
        if series:
            title = f"Training Curves — {run_dir.name} — {Path(cfg.out_dir).parent.name}"
            output_path = run_dir / "training_curves.png"
            plotter._plot_series(
                series,
                title,
                output_path,
                best_epoch=plotter._best_val_epoch(series),
                verbose=True
            )
    except Exception as exc:
        print(f"[GraphGym] Failed to generate training curves for configuration: {exc}")

    return best_path


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        
    args = parse_args()
    load_cfg(cfg, args)
    if cfg.accelerator == "auto":
        cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.optim.max_epoch = cfg.train.epochs

    layer_type = validate_layer_type(cfg.gnn.layer_type)
    cfg.dataset.edge_dim = resolve_edge_dim()

    set_run_dir(cfg.out_dir)

    # Reproducibility: seed weight init, dropout masks, and loader shuffling — not just
    # the data split (which loader_graphgym.py already seeds). Without this the GraphGym
    # grid path is non-deterministic and identical configs diverge. workers=True also
    # seeds DataLoader worker processes. (deterministic=True on the Trainer is left off:
    # several PyG scatter kernels lack deterministic GPU implementations and would raise.)
    pl.seed_everything(cfg.seed, workers=True)

    # Snapshot the fully-resolved config next to this run's outputs so the experiment is
    # recoverable from its own folder, independent of the transient shared configs/ dir
    # (which configs_gen.py overwrites on the next run). Writes <out_dir>/config.yaml.
    dump_cfg(cfg)
    dump_summary_cfg(cfg, Path(cfg.out_dir))

    print("\n[GraphGym Command Center] Launching training run...")
    print(f"[GraphGym] Architecture layer_type={layer_type} (from config YAML)")
    print(f"[GraphGym] Edge dim={cfg.dataset.edge_dim}")
    print(f"[GraphGym] Random seed: {cfg.seed} (seeded weight init / shuffling / split)")
    print(f"[GraphGym] Best-model selection: monitor=val_pr_auc, mode=max")
    print(f"[GraphGym] Final test (curated real data) will use the BEST saved checkpoint.\n")
    datamodule = GraphGymDataModule()
    model = create_model()
    train_with_best_ckpt(model, datamodule, logger=True)


if __name__ == "__main__":
    main()
