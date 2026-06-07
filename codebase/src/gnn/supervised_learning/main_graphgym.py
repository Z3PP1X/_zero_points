import sys
import warnings
# Suppress harmless PyG internal GraphGym InMemoryDataset deprecation warning
warnings.filterwarnings("ignore", message=".*InMemoryDataset.*")

from pathlib import Path
import torch
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, set_run_dir
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.imports import pl


gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import gnn.supervised_learning.loader_graphgym  # noqa
from gnn.supervised_learning.preprocessing import GraphPipeline # noqa

set_cfg(cfg)


class ValMetricLogger(pl.callbacks.Callback):
    """
    Bridges GraphGym's custom Logger (which writes to stats.json) with
    PyTorch Lightning's metric system (which ModelCheckpoint monitors).
    
    On each validation batch, accumulates loss and prediction data.
    On validation epoch end, computes metrics and calls trainer.callback_metrics
    via pl_module.log() so that ModelCheckpoint can monitor them.
    """
    def __init__(self):
        super().__init__()
        self._val_loss_sum = 0.0
        self._val_count = 0
        self._val_true = []
        self._val_pred = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        loss = float(outputs['loss'])
        batch_size = outputs['true'].size(0)
        self._val_loss_sum += loss * batch_size
        self._val_count += batch_size
        self._val_true.append(outputs['true'].detach().cpu())
        self._val_pred.append(outputs['pred_score'].detach().cpu())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_count == 0:
            return
        
        avg_loss = self._val_loss_sum / self._val_count
        
        # Compute AUC and PR-AUC for monitoring
        import numpy as np
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        true = torch.cat(self._val_true).numpy()
        pred = torch.cat(self._val_pred)
        
        # Get probability scores for positive class
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            scores = pred[:, 1].numpy()
        else:
            scores = pred.numpy()
        
        try:
            val_auc = roc_auc_score(true, scores)
        except ValueError:
            val_auc = 0.0
        
        try:
            prec_pts, rec_pts, _ = precision_recall_curve(true, scores)
            val_pr_auc = auc(rec_pts, prec_pts)
        except Exception:
            val_pr_auc = 0.0
        
        # Log metrics so ModelCheckpoint can see them
        pl_module.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        pl_module.log('val_auc', val_auc, prog_bar=True, sync_dist=True)
        pl_module.log('val_pr_auc', val_pr_auc, prog_bar=True, sync_dist=True)
        
        # Reset accumulators
        self._val_loss_sum = 0.0
        self._val_count = 0
        self._val_true = []
        self._val_pred = []


def train_with_best_ckpt(model, datamodule, logger=True):
    """
    Custom training function that replaces PyG's built-in train().
    
    Key difference: Uses ModelCheckpoint with monitor='val_auc' (mode='max')
    so the best model (by validation AUC on unseen synthetic data) is saved.
    After training, the final test (on curated real-world data) uses this
    best checkpoint — NOT the last epoch.
    """
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')
    
    callbacks = []
    
    # 1. GraphGym's built-in LoggerCallback (writes stats.json for train/val/test)
    if logger:
        callbacks.append(LoggerCallback())
    
    # 2. Our ValMetricLogger bridge (makes val metrics visible to Lightning)
    callbacks.append(ValMetricLogger())
    
    # 3. ModelCheckpoint — monitors val_auc to save the BEST model
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=get_ckpt_dir(),
            monitor='val_auc',         # Select best model by validation AUC
            mode='max',                # Higher AUC = better
            save_top_k=1,             # Keep only the single best checkpoint
            save_last=True,            # Also keep the last-epoch checkpoint for comparison
            filename='best-{epoch}-{val_auc:.4f}',
            verbose=True,
        )
        callbacks.append(ckpt_cbk)
    
    trainer = pl.Trainer(
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
    )
    
    # Train the model (validation runs every eval_period epochs on unseen synthetic data)
    trainer.fit(model, datamodule=datamodule)
    
    # Test using the BEST checkpoint (by val_auc), not the last epoch
    best_path = ckpt_cbk.best_model_path if cfg.train.enable_ckpt and ckpt_cbk.best_model_path else None
    
    if best_path:
        print(f"\n[GraphGym] Loading BEST checkpoint for final test: {best_path}")
        print(f"[GraphGym] Best val_auc: {ckpt_cbk.best_model_score:.4f}")
        trainer.test(model, datamodule=datamodule, ckpt_path=best_path)
    else:
        print("\n[GraphGym] No best checkpoint found, testing with last model weights.")
        trainer.test(model, datamodule=datamodule)


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
    print(f"[GraphGym] Best-model selection: monitor=val_auc, mode=max")
    print(f"[GraphGym] Final test (curated real data) will use the BEST saved checkpoint.\n")
    datamodule = GraphGymDataModule()
    model = create_model()
    train_with_best_ckpt(model, datamodule, logger=True)


if __name__ == "__main__":
    main()
