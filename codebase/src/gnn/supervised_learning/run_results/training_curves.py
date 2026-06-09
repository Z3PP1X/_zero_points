import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.ERROR)

SPLIT_LABELS = {
    "train": "Train (Synthetic)",
    "val": "Validation Synthetic",
    "test": "Validation Curated",
}
CURVE_METRICS = ["pr_auc", "auc", "loss", "f1"]
SPLIT_COLORS = {
    "train": "#2A9D8F",
    "val": "#E76F51",
    "test": "#264653",
}


def _load_stats_json(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    seen_epochs = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            epoch = row.get("epoch")
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            rows.append(row)
    return sorted(rows, key=lambda r: r.get("epoch", 0))


def _find_seed_dir(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    for name in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if name.is_dir() and name.name.isdigit():
            return name
    return None


def _config_slug(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        parts.append(f"{col}={val}")
    return "-".join(parts) if parts else "config"


class TrainingCurvePlotter:
    """Plot per-epoch metrics from GraphGym stats.json files."""

    def __init__(self, results_dir: Path, output_dir: Path, experiment_name: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

    def _iter_run_dirs(self):
        for path in sorted(self.results_dir.iterdir()):
            if path.is_dir() and path.name.startswith("grid"):
                yield path

    def _load_run_series(self, run_dir: Path) -> dict:
        seed_dir = _find_seed_dir(run_dir)
        if seed_dir is None:
            return {}
        series = {}
        for split in SPLIT_LABELS:
            stats = _load_stats_json(seed_dir / split / "stats.json")
            if stats:
                series[split] = pd.DataFrame(stats)
        return series

    def _best_val_epoch(self, series: dict) -> int | None:
        val_df = series.get("val")
        if val_df is None or val_df.empty or "pr_auc" not in val_df.columns:
            return None
        idx = val_df["pr_auc"].idxmax()
        return int(val_df.loc[idx, "epoch"])

    def _plot_series(
        self,
        series: dict,
        title: str,
        output_path: Path,
        best_epoch: int | None = None,
    ):
        available_metrics = [
            m
            for m in CURVE_METRICS
            if any(m in df.columns for df in series.values())
        ]
        if not available_metrics:
            return False

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(11, 3.2 * n_metrics), dpi=150)
        axes = np.atleast_1d(axes)

        for ax, metric in zip(axes, available_metrics):
            for split, label in SPLIT_LABELS.items():
                df = series.get(split)
                if df is None or metric not in df.columns:
                    continue
                plot_df = df.sort_values("epoch")
                ax.plot(
                    plot_df["epoch"],
                    plot_df[metric],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=label,
                    color=SPLIT_COLORS.get(split, "#457B9D"),
                )
            if best_epoch is not None:
                ax.axvline(
                    best_epoch,
                    color="#264653",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Best val PR-AUC epoch" if metric == available_metrics[0] else None,
                )
            ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if metric == available_metrics[0]:
                ax.legend(fontsize=8, frameon=False)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved training curves: {output_path}")
        return True

    def plot_overview(self, output_path: Path | None = None):
        """Average metrics across all grid configurations per epoch."""
        if output_path is None:
            output_path = self.output_dir / "training_curves_overview.png"

        combined = {split: [] for split in SPLIT_LABELS}
        for run_dir in self._iter_run_dirs():
            series = self._load_run_series(run_dir)
            for split, df in series.items():
                if not df.empty:
                    combined[split].append(df)

        averaged = {}
        for split, frames in combined.items():
            if not frames:
                continue
            merged = pd.concat(frames, ignore_index=True)
            metric_cols = [m for m in CURVE_METRICS if m in merged.columns]
            if not metric_cols:
                continue
            averaged[split] = (
                merged.groupby("epoch")[metric_cols].mean().reset_index()
            )

        if not averaged:
            print("    Skipping training curve overview (no stats.json data found)")
            return False

        title = f"Training Curves Overview — {self.experiment_name}"
        return self._plot_series(averaged, title, output_path)

    def plot_top_configs(self, leaderboard_csv: Path, top_k: int = 5):
        """Plot training curves for the top-K configs from the leaderboard."""
        if not leaderboard_csv.exists():
            print(f"    Skipping per-config curves ({leaderboard_csv} missing)")
            return

        ranked = pd.read_csv(leaderboard_csv).head(top_k)
        config_cols = [
            c
            for c in ranked.columns
            if c
            not in {
                "epoch",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "pr_auc",
                "lr",
                "base_lr",
                "params",
                "time_iter",
                "gpu_memory",
            }
            and not c.endswith("_std")
        ]

        out_root = self.output_dir / "top_configs"
        for idx, row in ranked.iterrows():
            run_name = "grid-" + "-".join(
                f"{col}={row[col]}" for col in config_cols if col in row and pd.notna(row[col])
            )
            run_dir = self.results_dir / run_name
            if not run_dir.exists():
                matches = list(self.results_dir.glob(f"{run_name}*"))
                run_dir = matches[0] if matches else None
            if run_dir is None or not run_dir.exists():
                print(f"    Skipping curves for missing run dir: {run_name}")
                continue

            series = self._load_run_series(run_dir)
            if not series:
                continue

            slug = _config_slug(row[config_cols])
            output_path = out_root / f"rank_{idx + 1}_{slug}" / "training_curves.png"
            title = f"Training Curves — rank {idx + 1} — {slug}"
            self._plot_series(
                series,
                title,
                output_path,
                best_epoch=self._best_val_epoch(series),
            )
