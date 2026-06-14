import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from gnn.supervised_learning.run_results._plot_utils import save_figure
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.getLogger("matplotlib").setLevel(logging.ERROR)

from gnn.supervised_learning.run_results.eval_metrics import (
    expected_calibration_error,
    prediction_probabilities,
)
from gnn.supervised_learning.run_results.significance import save_predictions

CONFIG_COLS = (
    "layer_type",
    "layers_mp",
    "dim_inner",
    "dropout",
    "graph_pooling",
    "act",
    "base_lr",
    "variant",
    "pool_type",
    "aux_loss_weight",
    "mode",
    "edge_direction",
)
def _setup_import_paths():
    script_dir = Path(__file__).resolve().parent
    supervised_dir = script_dir.parent
    gnn_root = supervised_dir.parent
    src_root = gnn_root.parent
    for path in (str(gnn_root), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _config_slug(row: pd.Series, config_cols: list) -> str:
    parts = []
    for col in config_cols:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}={row[col]}")
    return "-".join(parts) if parts else "config"


def _find_seed_dir(run_dir: Path) -> Path | None:
    for child in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and child.name.isdigit():
            return child
    return None


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    patterns = ["**/ckpt/best*.ckpt", "**/ckpt/*.ckpt"]
    candidates = []
    for pattern in patterns:
        candidates.extend(run_dir.glob(pattern))
    if not candidates:
        return None
    best = [p for p in candidates if "best" in p.name.lower()]
    pool = best or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def _find_config_for_run(run_dir: Path, configs_dir: Path | None) -> Path | None:
    # Prefer the per-run config.yaml snapshot (written by main_graphgym.dump_cfg). It is
    # the exact resolved config and removes the dependency on the transient configs/ dir.
    snapshot = run_dir / "config.yaml"
    if snapshot.exists():
        return snapshot

    # Legacy fallback: match a generated config by its out_dir against this run folder.
    if configs_dir is None or not configs_dir.exists():
        return None
    run_name = run_dir.name
    for cfg_file in sorted(configs_dir.glob("*.yaml")):
        with open(cfg_file, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        out_dir = Path(str(data.get("out_dir", "")))
        if out_dir.name == run_name or str(out_dir).endswith(run_name):
            return cfg_file
    return None


class DiagnosticPlotter:
    """Reload top checkpoints and plot confusion matrix, ROC, and PR curves."""

    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        experiment_name: str,
        configs_dir: Path | None = None,
        top_k: int = 5,
        device: str | None = None,
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.configs_dir = Path(configs_dir) if configs_dir else None
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_run_dir(self, row: pd.Series, config_cols: list) -> Path | None:
        # Datetime-named runs carry their folder name in the run_name column.
        if "run_name" in row.index and pd.notna(row["run_name"]):
            direct = self.results_dir / str(row["run_name"])
            if direct.exists():
                return direct

        # Legacy: reconstruct the grid-key=value folder name from the config columns.
        run_name = "grid-" + "-".join(
            f"{col}={row[col]}" for col in config_cols if col in row and pd.notna(row[col])
        )
        direct = self.results_dir / run_name
        if direct.exists():
            return direct
        matches = list(self.results_dir.glob(f"{run_name}*"))
        return matches[0] if matches else None

    def _load_model_and_loaders(self, config_path: Path):
        _setup_import_paths()
        import gnn.supervised_learning.loader_graphgym  # noqa: F401
        from gnn.supervised_learning.loader_graphgym import (
            _hard_predictions,
            _positive_class_scores,
            get_pos_label,
        )
        from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
        from torch_geometric.graphgym.model_builder import create_model
        from torch_geometric.graphgym.train import GraphGymDataModule

        set_cfg(cfg)
        args = argparse.Namespace(cfg_file=str(config_path), opts=[])
        load_cfg(cfg, args)

        datamodule = GraphGymDataModule()
        datamodule.setup()

        model = create_model()
        return model, datamodule, get_pos_label, _hard_predictions, _positive_class_scores

    def _collect_predictions(self, model, loader, split_name: str):
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = model._shared_step(batch, split_name)
                y_true.append(out["true"].detach().cpu())
                y_score.append(out["pred_score"].detach().cpu())
        if not y_true:
            return None, None
        return torch.cat(y_true), torch.cat(y_score)

    def _plot_confusion_matrix(
        self, y_true, y_pred, title: str, output_path: Path, pos_label: int
    ):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        for ax, data, subtitle in zip(
            axes,
            [cm, cm_norm],
            ["Counts", "Normalized (by true class)"],
        ):
            im = ax.imshow(data, cmap=plt.cm.Blues, vmin=0, vmax=data.max() if data.max() else 1)
            ax.set_title(subtitle, fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["0 (gMGF)", "1 (Newton)"])
            ax.set_yticklabels(["0 (gMGF)", "1 (Newton)"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            thresh = data.max() / 2 if data.max() else 0.5
            for i in range(2):
                for j in range(2):
                    val = data[i, j]
                    label = f"{val:.2f}" if subtitle.startswith("Norm") else f"{int(val)}"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        color="white" if val > thresh else "black",
                        fontsize=10,
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{title}\n(positive class = {pos_label})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_figure(output_path)
        plt.close(fig)

    def _plot_roc_curve(self, y_true, y_score, title: str, output_path: Path, pos_label: int):
        scores = _positive_class_scores_np(y_score, pos_label)
        y_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        fpr, tpr, _ = roc_curve(y_np, scores, pos_label=pos_label)
        roc_auc = roc_auc_score(y_np, scores) if len(np.unique(y_np)) > 1 else 0.0

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(fpr, tpr, color="#2A9D8F", linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_figure(output_path)
        plt.close(fig)

    def _plot_pr_curve(self, y_true, y_score, title: str, output_path: Path, pos_label: int):
        scores = _positive_class_scores_np(y_score, pos_label)
        y_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        precision, recall, _ = precision_recall_curve(y_np, scores, pos_label=pos_label)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(recall, precision, color="#E76F51", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower left")
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_figure(output_path)
        plt.close(fig)

    def _plot_reliability(
        self,
        y_true,
        y_score,
        title: str,
        output_path: Path,
        pos_label: int,
        n_bins: int = 10,
    ):
        """Reliability diagram: per-bin accuracy vs mean confidence (+ ECE)."""
        probs = prediction_probabilities(y_score)
        probs_pos = probs[:, int(pos_label)].detach().cpu().numpy()
        y_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        y_pos = (y_np == pos_label).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers, accs, confs, weights = [], [], [], []
        for idx in range(n_bins):
            lo, hi = bin_edges[idx], bin_edges[idx + 1]
            if idx < n_bins - 1:
                mask = (probs_pos > lo) & (probs_pos <= hi)
            else:
                mask = (probs_pos >= lo) & (probs_pos <= hi)
            if not np.any(mask):
                continue
            centers.append((lo + hi) / 2.0)
            accs.append(float(y_pos[mask].mean()))
            confs.append(float(probs_pos[mask].mean()))
            weights.append(float(mask.mean()))
        ece = expected_calibration_error(probs_pos, y_pos, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="#999999",
            label="Perfect calibration",
        )
        if centers:
            ax.bar(
                centers, accs, width=1.0 / n_bins, color="#2A9D8F",
                edgecolor="#264653", alpha=0.8, label="Accuracy",
            )
            ax.plot(
                confs, accs, marker="o", color="#E76F51", linewidth=1.5,
                label="Confidence",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability (positive class)")
        ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_figure(output_path)
        plt.close(fig)

    def run_top_configs(self, leaderboard_csv: Path | None = None):
        csv_path = leaderboard_csv or (self.results_dir / "agg" / "val_bestepoch.csv")
        if not csv_path.exists():
            print(f"    Skipping diagnostics ({csv_path} not found)")
            return

        try:
            ranked = pd.read_csv(csv_path).sort_values("pr_auc", ascending=False).head(self.top_k)
        except Exception as exc:
            print(f"    Skipping diagnostics (failed to read leaderboard): {exc}")
            return

        config_cols = [c for c in CONFIG_COLS if c in ranked.columns]
        if not config_cols:
            config_cols = [
                c
                for c in ranked.columns
                if c
                not in {
                    "run_name",
                    "epoch",
                    "loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auc",
                    "pr_auc",
                    "lr",
                    "params",
                    "time_iter",
                    "gpu_memory",
                }
                and not str(c).endswith("_std")
            ]

        print(f"  Running diagnostics for top {len(ranked)} configs...")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            run_dir = self._resolve_run_dir(row, config_cols)
            if run_dir is None:
                print(f"    [{rank}] Run directory not found for config row")
                continue

            config_path = _find_config_for_run(run_dir, self.configs_dir)
            if config_path is None:
                print(f"    [{rank}] Config YAML not found for {run_dir.name}")
                continue

            ckpt_path = _find_best_checkpoint(run_dir)
            if ckpt_path is None:
                print(f"    [{rank}] Checkpoint not found under {run_dir}")
                continue

            slug = _config_slug(row, config_cols)
            out_dir = self.output_dir / "top_configs" / f"rank_{rank}_{slug}"
            print(f"    [{rank}] {slug} — ckpt: {ckpt_path.name}")

            try:
                model, datamodule, get_pos_label, hard_pred_fn, pos_scores_fn = (
                    self._load_model_and_loaders(config_path)
                )
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)

                pos_label = get_pos_label()
                loaders = datamodule.loaders
                synthetic = bool(getattr(model.cfg.expression_graph, "synthetic", False))
                split_loaders = []
                if synthetic and len(loaders) >= 3:
                    split_loaders = [
                        ("val", loaders[1], "Validation Synthetic"),
                        ("test", loaders[2], "Validation Curated"),
                    ]
                elif len(loaders) >= 2:
                    split_loaders = [("val", loaders[1], "Validation")]

                for split_key, loader, split_title in split_loaders:
                    y_true, y_score = self._collect_predictions(model, loader, split_key)
                    if y_true is None:
                        continue
                    y_pred = hard_pred_fn(y_score, pos_label, getattr(model.cfg.model, "thresh", 0.5))
                    prefix = split_title.lower().replace(" ", "_")
                    self._plot_confusion_matrix(
                        y_true.numpy(),
                        y_pred.numpy(),
                        f"Confusion Matrix — {split_title}",
                        out_dir / f"confusion_{prefix}.png",
                        pos_label,
                    )
                    self._plot_roc_curve(
                        y_true,
                        y_score,
                        f"ROC Curve — {split_title}",
                        out_dir / f"roc_{prefix}.png",
                        pos_label,
                    )
                    self._plot_pr_curve(
                        y_true,
                        y_score,
                        f"PR Curve — {split_title}",
                        out_dir / f"pr_{prefix}.png",
                        pos_label,
                    )
                    self._plot_reliability(
                        y_true,
                        y_score,
                        f"Reliability — {split_title}",
                        out_dir / f"reliability_{prefix}.png",
                        pos_label,
                    )
                    # Dump per-prediction positive-class probabilities so the
                    # report can bootstrap CIs / paired tests without reloading.
                    probs_pos = (
                        prediction_probabilities(y_score)[:, int(pos_label)]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    save_predictions(
                        out_dir / f"predictions_{prefix}.npz",
                        y_true.numpy() if hasattr(y_true, "numpy") else y_true,
                        probs_pos,
                        pos_label=pos_label,
                    )
                print(f"    [{rank}] Saved diagnostics to {out_dir}")
            except Exception as exc:
                print(f"    [{rank}] Diagnostics failed: {exc}")


def _positive_class_scores_np(y_score, pos_label: int):
    if hasattr(y_score, "dim") and y_score.dim() > 1 and y_score.shape[1] > 1:
        scores = y_score[:, pos_label]
    else:
        scores = y_score
    arr = scores.numpy() if hasattr(scores, "numpy") else np.asarray(scores)
    if pos_label == 0:
        arr = 1.0 - arr
    return arr
