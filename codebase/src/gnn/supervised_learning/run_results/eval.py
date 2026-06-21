import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress warnings and matplotlib's internal log messages (like missing fonts)
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from gnn.supervised_learning.run_results._plot_utils import save_figure
import numpy as np
import pandas as pd

from gnn.supervised_learning.run_results.eval_metrics import (
    CONFIDENCE_METRICS,
    EVAL_WARMUP_EPOCHS,
    LOWER_IS_BETTER_METRICS,
    MIN_CLASSIFICATION_METRIC,
    filter_warmup_epochs_df,
    passes_quality_threshold,
)


class GNNResultEvaluator:
    """
    Evaluates GNN training runs, plotting pivot grid heatmaps and layer summaries.

    In synthetic mode:
      - train_*: Training on synthetic data
      - val_*:   Unseen synthetic holdout (model selection via pr_auc)
      - test_*:  Curated real-world holdout (generalisation only, no training effect)

    MAX heatmaps: per heatmap cell, pick the configuration with the highest pr_auc,
    then plot auc / pr_auc / loss / precision / f1 / recall from that same epoch row.
    MEAN heatmaps: average metrics across configurations in each cell.
    """

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
    METRIC_COLS = {
        "epoch",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "mean_confidence",
        "mean_margin",
        "mean_entropy",
        "brier_score",
        "ece",
        "lr",
        "base_lr",
        "params",
        "time_iter",
        "gpu_memory",
    }
    # mean_margin, mean_entropy, ece require post-training calibration and are
    # excluded from plots. brier_score is kept as the single uncalibrated signal.
    HEATMAP_METRICS = [
        "auc",
        "pr_auc",
        "loss",
        "recall",
        "f1",
        "precision",
        "mean_confidence",
        "brier_score",
    ]
    BOUNDED_METRICS = [
        "auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mean_confidence",
        "brier_score",
    ]
    DEFAULT_RUNS = ["train_bestepoch", "val_bestepoch", "test_bestepoch"]
    ALL_RUNS = [
        "train_best",
        "train_bestepoch",
        "train",
        "test_best",
        "test_bestepoch",
        "test",
        "val_best",
        "val_bestepoch",
        "val",
    ]
    PREMIUM_PALETTE = [
        "#2A9D8F",
        "#E76F51",
        "#264653",
        "#F4A261",
        "#E9C46A",
        "#457B9D",
        "#1D3557",
    ]

    def __init__(
        self,
        naming_var: str,
        base_dir: Path = None,
        runs: list = None,
        skip_slices: bool = False,
        top_k: int = 10,
    ):
        """
        self.naming_var = naming_var
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.data_dir = self.base_dir / self.naming_var / "agg"
        self.output_dir = self.base_dir / self.naming_var / "eval_plots"
        self.runs = runs if runs is not None else list(self.DEFAULT_RUNS)
        self.skip_slices = skip_slices
        self.top_k = top_k

        self.run_labels = {
            "train": "Training (Synthetic)",
            "train_best": "Training Best (Synthetic)",
            "train_bestepoch": "Training Best Epoch (Synthetic)",
            "val": "Validation Synthetic (Unseen Synthetic)",
            "val_best": "Validation Synthetic Best (Unseen Synthetic)",
            "val_bestepoch": "Validation Synthetic Best Epoch (Unseen Synthetic)",
            "test": "Validation Curated (Curated Real)",
            "test_best": "Validation Curated Best (Curated Real)",
            "test_bestepoch": "Validation Curated Best Epoch (Curated Real)",
        }

        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "premium_green", ["#FFFFFF", "#D1E7DD", "#0F5132"]
        )
        self.cmap_loss = mcolors.LinearSegmentedColormap.from_list(
            "premium_red", ["#FFFFFF", "#F8D7DA", "#842029"]
        )

        self._exp_config: dict = {}
        self._load_experiment_config()
        self.class_balance = None
        self.load_class_balance()

    def _load_experiment_config(self):
        """Load base config from run_manifest.json or config_supervised.yaml."""
        manifest_path = self.base_dir / self.naming_var / "run_manifest.json"
        if manifest_path.exists():
            try:
                import json as _json
                manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
                self._exp_config = manifest.get("base_config", {}) or {}
                return
            except Exception:
                pass
        config_file = self.base_dir.parent / "config_supervised.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file, "r", encoding="utf-8") as f:
                    self._exp_config = yaml.safe_load(f) or {}
            except Exception:
                pass

    def _build_footnote(self, df: pd.DataFrame | None = None) -> str:
        """Compose a two-line figure footnote: arch config + dataset context."""
        cfg = self._exp_config
        expr = cfg.get("expression_graph", {}) or {}
        dataset_cfg = cfg.get("dataset", {}) or {}

        arch_parts = []
        if df is not None and "layer_type" in df.columns:
            vals = sorted({str(v) for v in df["layer_type"].dropna()})
            if vals:
                arch_parts.append(f"Architecture: {', '.join(vals)}")
        if df is not None and "layers_mp" in df.columns:
            vals = sorted(df["layers_mp"].dropna().unique())
            if len(vals) == 1:
                arch_parts.append(f"MP-Layers: {int(vals[0])}")
            elif len(vals) > 1:
                arch_parts.append(f"MP-Layers: {int(vals[0])}–{int(vals[-1])}")
        if df is not None and "dim_inner" in df.columns:
            vals = sorted(df["dim_inner"].dropna().unique())
            if len(vals) == 1:
                arch_parts.append(f"dim_inner: {int(vals[0])}")
            elif len(vals) > 1:
                arch_parts.append(f"dim_inner: {int(vals[0])}–{int(vals[-1])}")
        for key, label in [
            ("mode", "Mode"), ("edge_direction", "Edge"),
            ("bidirectional", "Bidirectional"), ("add_kappa", "Kappa"),
        ]:
            val = expr.get(key)
            if val is not None:
                arch_parts.append(f"{label}: {val}")
        line1 = " | ".join(arch_parts) if arch_parts else ""

        synthetic_name = (
            expr.get("synthetic_dataset") or dataset_cfg.get("name") or self.naming_var
        )
        line2 = (
            f"Train: {synthetic_name} (80%-Split, Synthetic)"
            f"  ·  Val: {synthetic_name} (20%-Split, Synthetic)"
            f"  ·  Test: CuratedReal (100%)"
        )

        return "\n".join(p for p in [line1, line2] if p)

    def load_class_balance(self):
        """Loads class_balance.json or dynamically computes it as a fallback."""
        import json

        cb_file = self.data_dir / "class_balance.json"
        if cb_file.exists():
            try:
                with open(cb_file, "r", encoding="utf-8") as f:
                    self.class_balance = json.load(f)
                print(f"Loaded class balance from {cb_file}")
            except Exception as e:
                print(f"Warning: Failed to load class balance from {cb_file}: {e}")
        else:
            print(
                f"class_balance.json not found in {self.data_dir}. "
                "Attempting fallback calculation..."
            )
            try:
                # Dataset names are resolved from the run's config — never hardcoded,
                # otherwise the fallback would compute class balance against the wrong
                # dataset. If the config (or the dataset name) is unavailable we skip the
                # annotation rather than guess.
                config_file = self.base_dir.parent / "config_supervised.yaml"
                if not config_file.exists():
                    print(
                        f"  Cannot compute class balance: config not found at "
                        f"{config_file}. Skipping class-balance annotation."
                    )
                    return

                import yaml

                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
                dataset_name = cfg_data.get("dataset", {}).get("name")
                synthetic_dataset_name = cfg_data.get("expression_graph", {}).get(
                    "synthetic_dataset"
                )
                add_kappa = bool(
                    cfg_data.get("expression_graph", {}).get("add_kappa", False)
                )
                if not dataset_name or not synthetic_dataset_name:
                    print(
                        "  Cannot compute class balance: dataset.name / "
                        "expression_graph.synthetic_dataset missing from "
                        f"{config_file}. Skipping class-balance annotation."
                    )
                    return

                src_path = str(self.base_dir.parents[2])
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)

                from gnn.shared.utils.graph_loader import GraphDataLoader
                from gnn.supervised_learning.preprocessing import GraphPipeline

                print(
                    f"  Loading dataset for class balance: "
                    f"{dataset_name} / {synthetic_dataset_name}"
                )
                loader = GraphDataLoader(
                    name=dataset_name, mode="graph", add_kappa=add_kappa
                )
                pipeline = GraphPipeline(
                    dataset_name=dataset_name,
                    seed=42001,
                    mode="graph",
                    graph_loader=loader,
                    synthetic=True,
                    synthetic_dataset_name=synthetic_dataset_name,
                    add_kappa=add_kappa,
                )
                pipeline.pipe(test_size=0.2, batch_size=256, stratify=True)

                val_syn_df = pipeline.test_dataset.df
                val_syn_0 = int((val_syn_df["faster_algorithm"] == 0).sum())
                val_syn_1 = int((val_syn_df["faster_algorithm"] == 1).sum())
                val_syn_total = len(val_syn_df)

                val_cur_df = pipeline.curated_dataset.df
                val_cur_0 = int((val_cur_df["faster_algorithm"] == 0).sum())
                val_cur_1 = int((val_cur_df["faster_algorithm"] == 1).sum())
                val_cur_total = len(val_cur_df)

                self.class_balance = {
                    "validation_synthetic": {
                        "0": val_syn_0,
                        "1": val_syn_1,
                        "total": val_syn_total,
                    },
                    "validation_curated": {
                        "0": val_cur_0,
                        "1": val_cur_1,
                        "total": val_cur_total,
                    },
                }

                self.data_dir.mkdir(parents=True, exist_ok=True)
                with open(cb_file, "w", encoding="utf-8") as f:
                    json.dump(self.class_balance, f, indent=4)
                print(f"Saved computed class balance to {cb_file}")
            except Exception as e:
                if isinstance(e, ModuleNotFoundError) and any(
                    m in str(e) for m in ["torch", "gnn"]
                ):
                    print(
                        "\n[Warning] Fallback dataset loading failed because PyTorch "
                        "or codebase dependencies are missing in this environment."
                    )
                    print("Please run the script using the correct Conda environment:")
                    print(
                        "  /home/zapp1x/miniconda3/envs/pytorch/bin/python eval.py "
                        "<naming_var>\n"
                    )
                else:
                    print(f"Warning: Fallback class balance calculation failed: {e}")

    def get_class_balance_str(self) -> str:
        """Constructs a formatted string of the class balances."""
        if not self.class_balance:
            return ""

        syn = self.class_balance.get("validation_synthetic", {})
        cur = self.class_balance.get("validation_curated", {})

        parts = []
        if syn and syn.get("total", 0) > 0:
            s0 = syn.get("0", 0)
            s1 = syn.get("1", 0)
            st = syn.get("total", 0)
            p0 = (s0 / st) * 100 if st > 0 else 0
            p1 = (s1 / st) * 100 if st > 0 else 0
            parts.append(
                f"Validation Synthetic: 0 (gMGF): {s0} ({p0:.1f}%) / "
                f"1 (Newton): {s1} ({p1:.1f}%)"
            )

        if cur and cur.get("total", 0) > 0:
            c0 = cur.get("0", 0)
            c1 = cur.get("1", 0)
            ct = cur.get("total", 0)
            cp0 = (c0 / ct) * 100 if ct > 0 else 0
            cp1 = (c1 / ct) * 100 if ct > 0 else 0
            parts.append(
                f"Validation Curated: 0 (gMGF): {c0} ({cp0:.1f}%) / "
                f"1 (Newton): {c1} ({cp1:.1f}%)"
            )

        if not parts:
            return ""
        return " | ".join(parts)

    def load_data(self, run_name: str) -> pd.DataFrame:
        """Loads the CSV file for a given run."""
        file_path = self.data_dir / f"{run_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        return self._prepare_eval_df(pd.read_csv(file_path))

    def _prepare_eval_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude warmup epochs from best/mean evaluation inputs."""
        return filter_warmup_epochs_df(df, warmup_epochs=EVAL_WARMUP_EPOCHS)

    def _config_columns(self, df: pd.DataFrame) -> list:
        skip = set(self.METRIC_COLS)
        skip.add("run_name")  # run identity, not a hyperparameter axis
        skip.update(c for c in df.columns if c.endswith("_std"))
        known = [c for c in self.CONFIG_COLS if c in df.columns]
        extra = [c for c in df.columns if c not in skip and c not in known]
        return known + extra

    def _varying_config_columns(self, df: pd.DataFrame) -> list:
        return [
            c
            for c in self._config_columns(df)
            if c in df.columns and df[c].nunique(dropna=False) > 1
        ]

    def _detect_heatmap_axes(self, df: pd.DataFrame):
        """Pick index/column axes from hyperparameters that actually vary."""
        varying = self._varying_config_columns(df)
        if len(varying) < 2:
            return None, None

        preferred_pairs = [
            ("dim_inner", "dropout"),
            ("layer_type", "graph_pooling"),
            ("layers_mp", "graph_pooling"),
            ("layer_type", "layers_mp"),
            ("act", "base_lr"),
        ]
        for a, b in preferred_pairs:
            if a in varying and b in varying:
                return a, b
        return varying[0], varying[1]

    def _best_pr_auc_rows(self, df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        """Keep one row per group: the configuration with the highest pr_auc."""
        if "pr_auc" not in df.columns or not group_cols:
            return df
        idx = df.groupby(group_cols, dropna=False)["pr_auc"].idxmax().dropna()
        return df.loc[idx]

    def _format_axis_label(self, col: str) -> str:
        labels = {
            "dim_inner": "Dim Inner",
            "dropout": "Dropout",
            "layer_type": "Layer Type",
            "layers_mp": "MP Layers",
            "graph_pooling": "Graph Pooling",
            "act": "Activation",
            "base_lr": "Learning Rate",
        }
        return labels.get(col, col.replace("_", " ").title())

    def _save_figure(
        self,
        fig,
        output_path: Path,
        title: str,
        subtitle_y: float = 0.94,
        footnote_df: pd.DataFrame | None = None,
    ):
        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        balance_info = self.get_class_balance_str()
        if balance_info:
            plt.figtext(
                0.5,
                subtitle_y,
                balance_info,
                ha="center",
                va="center",
                fontsize=10,
                style="italic",
                color="#555555",
            )
        footnote = self._build_footnote(footnote_df)
        if footnote:
            plt.figtext(
                0.5,
                0.01,
                footnote,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#666666",
                style="italic",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#f5f5f5",
                    edgecolor="#cccccc",
                    alpha=0.85,
                ),
            )
        save_figure(output_path)
        plt.close(fig)

    def _plot_single_heatmap(
        self,
        ax,
        plot_df: pd.DataFrame,
        metric: str,
        index_col: str,
        column_col: str,
        agg: str,
    ):
        if metric not in plot_df.columns:
            raise KeyError(f"metric '{metric}' not in data")

        pivot_grid = plot_df.pivot_table(
            values=metric,
            index=index_col,
            columns=column_col,
            aggfunc="first" if agg == "max" else agg,
        )
        pivot_grid = pivot_grid.sort_index(ascending=True)
        pivot_grid = pivot_grid.reindex(sorted(pivot_grid.columns), axis=1)

        values = pivot_grid.values
        if pivot_grid.empty or np.all(np.isnan(values)):
            raise ValueError("no pivot data")

        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        if min_val == max_val:
            vmin_val = min_val - 0.05
            vmax_val = max_val + 0.05
        else:
            vmin_val = min_val
            vmax_val = max_val

        cmap_to_use = self.cmap_loss if metric in LOWER_IS_BETTER_METRICS else self.cmap
        im = ax.imshow(
            values,
            cmap=cmap_to_use,
            aspect="auto",
            origin="lower",
            vmin=vmin_val,
            vmax=vmax_val,
        )

        ax.set_xticks(range(len(pivot_grid.columns)))
        ax.set_xticklabels(
            pivot_grid.columns,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(range(len(pivot_grid.index)))
        ax.set_yticklabels(pivot_grid.index)
        ax.set_xlabel(self._format_axis_label(column_col), fontsize=9, fontweight="bold")
        ax.set_ylabel(self._format_axis_label(index_col), fontsize=9, fontweight="bold")
        # In "max" mode every metric in a cell is read from the single best-pr_auc
        # configuration's row (one coherent row, NOT an independent per-metric max),
        # so label it as such to avoid that misreading.
        metric_title = (
            f"{metric.upper()} @ BEST PR-AUC" if agg == "max" else f"{metric.upper()} (MEAN)"
        )
        ax.set_title(metric_title, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(left=False, bottom=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        range_val = max(max_val - min_val, 1e-5)
        for i in range(len(pivot_grid.index)):
            for j in range(len(pivot_grid.columns)):
                val = pivot_grid.values[i, j]
                if not pd.isna(val):
                    normalized_val = (val - min_val) / range_val
                    text_color = "white" if normalized_val > 0.6 else "#111111"
                    ax.text(
                        j,
                        i,
                        f"{val:.4f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="semibold",
                    )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)

    def generate_heatmaps(
        self,
        df: pd.DataFrame,
        output_path: Path,
        title: str,
        agg: str,
    ) -> bool:
        """Generate a 2x3 heatmap grid for one aggregation mode. Returns False if skipped."""
        index_col, column_col = self._detect_heatmap_axes(df)
        if index_col is None:
            return False

        group_cols = [index_col, column_col]
        plot_df = (
            self._best_pr_auc_rows(df, group_cols)
            if agg == "max" and "pr_auc" in df.columns
            else df
        )

        metrics = [m for m in self.HEATMAP_METRICS if m in plot_df.columns]
        if not metrics:
            return False

        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6.5 * n_cols, 5.2 * n_rows), dpi=150
        )
        axes = np.atleast_1d(axes).flatten()
        # Reserve 14% at top for suptitle + subtitle, 8% at bottom for footnote.
        fig.subplots_adjust(hspace=0.55, wspace=0.42, top=0.84, bottom=0.08)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            try:
                self._plot_single_heatmap(
                    ax, plot_df, metric, index_col, column_col, agg
                )
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"No Data\n{e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax.set_title(f"{metric.upper()} ({agg.upper()}) - N/A", fontsize=10)

        for ax in axes[n_metrics:]:
            ax.axis("off")

        axis_note = (
            f"Axes: {self._format_axis_label(index_col)} × "
            f"{self._format_axis_label(column_col)}"
        )
        agg_note = (
            "best-pr_auc configuration per cell (all metrics from that one row)"
            if agg == "max"
            else "mean across configurations per cell"
        )
        full_title = f"{title}\n{axis_note} — {agg_note}"
        self._save_figure(fig, output_path, full_title, subtitle_y=0.92, footnote_df=plot_df)
        print(f"    Saved heatmap: {output_path}")
        return True

    def _resolve_bar_grouping(self, df: pd.DataFrame, overall_df: pd.DataFrame, group_col: str = None):
        legend_title = ""
        chart_title = ""

        if group_col is not None:
            titles = {
                "layers_mp": ("MP Layers", "MP Layers Comparison"),
                "act": ("Activation Function", "Activation Function Comparison"),
                "graph_pooling": ("Graph Pooling", "Pooling Comparison"),
                "layer_type": ("Model Architecture", "Architecture Comparison"),
                "base_lr": ("Learning Rate", "Learning Rate Comparison"),
            }
            if group_col in titles:
                legend_title, base_title = titles[group_col]
                if "layer_type" in df.columns and df["layer_type"].nunique() == 1:
                    chart_title = f"{base_title} for {df['layer_type'].iloc[0]}"
                else:
                    chart_title = base_title
        else:
            if "layer_type" in overall_df.columns and overall_df["layer_type"].nunique() > 1:
                group_col = "layer_type"
                legend_title = "Model Architecture"
                chart_title = "Architecture Comparison (mean)"
            elif "layer_type" in df.columns and df["layer_type"].nunique() == 1:
                lt = df["layer_type"].iloc[0]
                for candidate, legend, title_tpl in [
                    ("act", "Activation Function", "Activation Function Comparison for {}"),
                    ("layers_mp", "MP Layers", "MP Layers Comparison for {}"),
                    ("graph_pooling", "Graph Pooling", "Pooling Comparison for {}"),
                    ("base_lr", "Learning Rate", "Learning Rate Comparison for {}"),
                ]:
                    if (
                        candidate in overall_df.columns
                        and overall_df[candidate].nunique() > 1
                    ):
                        group_col = candidate
                        legend_title = legend
                        chart_title = title_tpl.format(lt)
                        break
                else:
                    group_col = "layer_type"
                    legend_title = "Model Architecture"
                    chart_title = f"Architecture Comparison for {lt}"
            else:
                for candidate, legend, title in [
                    ("layer_type", "Model Architecture", "Architecture Comparison"),
                    ("act", "Activation Function", "Activation Function Comparison"),
                    ("layers_mp", "MP Layers", "MP Layers Comparison"),
                    ("base_lr", "Learning Rate", "Learning Rate Comparison"),
                ]:
                    if candidate in overall_df.columns:
                        group_col = candidate
                        legend_title = legend
                        chart_title = title
                        break

        return group_col, legend_title, chart_title

    def _grouped_metric_summary(
        self, comparison_df: pd.DataFrame, group_col: str, metrics: list
    ) -> pd.DataFrame:
        """Mean metrics per group, attaching std columns when available."""
        summary = comparison_df.groupby(group_col)[metrics].mean()
        for metric in metrics:
            std_col = f"{metric}_std"
            if std_col in comparison_df.columns:
                summary[f"{metric}_err"] = comparison_df.groupby(group_col)[std_col].mean()
        return summary

    def generate_summary_bars(
        self,
        df: pd.DataFrame,
        overall_df: pd.DataFrame,
        output_path: Path,
        title: str,
        group_col: str = None,
    ) -> bool:
        """Grouped bar chart for bounded metrics; loss shown in a separate panel below."""
        present_bounded = [m for m in self.BOUNDED_METRICS if m in overall_df.columns]
        has_loss = "loss" in overall_df.columns
        if not present_bounded and not has_loss:
            return False

        group_col, legend_title, chart_title = self._resolve_bar_grouping(
            df, overall_df, group_col
        )
        if group_col is None:
            return False

        comparison_df = df if df[group_col].nunique() > 1 else overall_df
        if comparison_df[group_col].nunique() < 1:
            return False

        n_panels = int(bool(present_bounded)) + int(has_loss)
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(12, 4.5 * n_panels), dpi=150, sharex=False
        )
        axes = np.atleast_1d(axes)
        panel_idx = 0
        n_groups = comparison_df[group_col].nunique()
        colors = [
            self.PREMIUM_PALETTE[i % len(self.PREMIUM_PALETTE)]
            for i in range(n_groups)
        ]

        if present_bounded:
            ax = axes[panel_idx]
            layer_summary = self._grouped_metric_summary(
                comparison_df, group_col, present_bounded
            )
            layer_summary[present_bounded].T.plot(
                kind="bar", ax=ax, width=0.8, color=colors, edgecolor="none"
            )
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score", fontsize=10, fontweight="bold")
            ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(
                title=legend_title,
                frameon=True,
                facecolor="#f8f9fa",
                edgecolor="none",
                fontsize=9,
                title_fontsize=10,
            )
            ax.set_xticklabels(
                [m.upper() for m in present_bounded], rotation=45, ha="right", fontsize=9
            )
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(
                            f"{height:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=90,
                        )
            panel_idx += 1

        if has_loss:
            ax = axes[panel_idx]
            loss_summary = self._grouped_metric_summary(
                comparison_df, group_col, ["loss"]
            )
            loss_summary[["loss"]].T.plot(
                kind="bar",
                ax=ax,
                width=0.8,
                color=["#842029"] * len(loss_summary),
                edgecolor="none",
            )
            ax.set_ylabel("Loss", fontsize=10, fontweight="bold", color="#842029")
            ax.set_title("Loss Comparison", fontsize=12, fontweight="bold", pad=8)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(
                title=legend_title,
                frameon=False,
                fontsize=9,
                title_fontsize=10,
            )
            ax.set_xticklabels(["LOSS"], rotation=0, fontsize=10)

        for ax in axes:
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")

        self._save_figure(fig, output_path, title, subtitle_y=0.92, footnote_df=overall_df)
        print(f"    Saved summary bars: {output_path}")
        return True

    def generate_plots_for_df(
        self,
        df: pd.DataFrame,
        overall_df: pd.DataFrame,
        output_dir: Path,
        title: str,
        group_col: str = None,
    ):
        """Generate split outputs: heatmaps_mean, heatmaps_max, and summary_bars."""
        output_dir.mkdir(parents=True, exist_ok=True)

        self.generate_heatmaps(
            df,
            output_dir / "heatmaps_mean.png",
            title,
            agg="mean",
        )
        self.generate_heatmaps(
            df,
            output_dir / "heatmaps_max.png",
            title,
            agg="max",
        )
        self.generate_summary_bars(
            df,
            overall_df,
            output_dir / "summary_bars.png",
            title,
            group_col=group_col,
        )

    def generate_summary_comparison(self, output_path: Path):
        """Grouped bar chart across train / val synthetic / val curated (best epoch)."""
        split_files = {
            "Train (Synthetic)": "train_bestepoch",
            "Validation Synthetic": "val_bestepoch",
            "Validation Curated": "test_bestepoch",
        }

        summary_metrics = self.BOUNDED_METRICS + ["loss"]
        split_data = {}

        for label, filename in split_files.items():
            try:
                df = self.load_data(filename)
                present = [m for m in summary_metrics if m in df.columns]
                split_data[label] = df[present].mean()
            except FileNotFoundError:
                continue

        if len(split_data) < 2:
            return

        summary_df = pd.DataFrame(split_data)
        bounded = [m for m in self.BOUNDED_METRICS if m in summary_df.index]
        has_loss = "loss" in summary_df.index
        n_panels = int(bool(bounded)) + int(has_loss)
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), dpi=150)
        axes = np.atleast_1d(axes)
        colors = [
            self.PREMIUM_PALETTE[i % len(self.PREMIUM_PALETTE)]
            for i in range(len(summary_df.columns))
        ]
        panel_idx = 0

        if bounded:
            ax = axes[panel_idx]
            summary_df.loc[bounded].plot(
                kind="bar", ax=ax, width=0.75, color=colors, edgecolor="none"
            )
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score", fontsize=11, fontweight="bold")
            ax.set_title(
                "Train / Val Synthetic / Val Curated Comparison (Best Epoch Avg)",
                fontsize=15,
                fontweight="bold",
                pad=12,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(
                title="Data Split",
                frameon=True,
                facecolor="#f8f9fa",
                edgecolor="none",
                fontsize=10,
                title_fontsize=11,
            )
            ax.set_xticklabels([m.upper() for m in bounded], rotation=45, ha="right", fontsize=9)
            panel_idx += 1

        if has_loss:
            ax = axes[panel_idx]
            summary_df.loc[["loss"]].plot(
                kind="bar", ax=ax, width=0.75, color=colors, edgecolor="none"
            )
            ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
            ax.set_title("Loss Comparison (Best Epoch Avg)", fontsize=13, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(
                title="Data Split",
                frameon=False,
                fontsize=10,
                title_fontsize=11,
            )
            ax.set_xticklabels(["LOSS"], rotation=0, fontsize=10)

        for ax in axes:
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")

        footnote_df = None
        try:
            footnote_df = self.load_data("val_bestepoch")
        except FileNotFoundError:
            pass
        self._save_figure(
            fig,
            output_path,
            f"Split Comparison — {self.naming_var}",
            subtitle_y=0.96,
            footnote_df=footnote_df,
        )
        print(f"    Saved split comparison plot: {output_path}")

    def generate_generalization_gap(self, output_path: Path):
        """Line chart of pr_auc across splits, grouped by architecture."""
        split_files = {
            "Train": "train_bestepoch",
            "Val Synthetic": "val_bestepoch",
            "Val Curated": "test_bestepoch",
        }

        frames = []
        for split_label, filename in split_files.items():
            try:
                df = self.load_data(filename)
            except FileNotFoundError:
                continue
            if "pr_auc" not in df.columns:
                continue
            chunk = df.copy()
            chunk["split"] = split_label
            frames.append(chunk)

        if not frames:
            return

        combined = pd.concat(frames, ignore_index=True)
        group_col = (
            "layer_type"
            if "layer_type" in combined.columns
            and combined["layer_type"].nunique() > 1
            else None
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        split_order = ["Train", "Val Synthetic", "Val Curated"]
        x = np.arange(len(split_order))

        if group_col:
            for i, (name, group) in enumerate(combined.groupby(group_col)):
                means = []
                for split in split_order:
                    subset = group[group["split"] == split]["pr_auc"]
                    means.append(subset.mean() if len(subset) else np.nan)
                color = self.PREMIUM_PALETTE[i % len(self.PREMIUM_PALETTE)]
                ax.plot(x, means, marker="o", linewidth=2, label=name, color=color)
                for xi, val in zip(x, means):
                    if not np.isnan(val):
                        ax.annotate(f"{val:.3f}", (xi, val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        else:
            means = [
                combined[combined["split"] == s]["pr_auc"].mean() for s in split_order
            ]
            ax.plot(x, means, marker="o", linewidth=2, color=self.PREMIUM_PALETTE[0])
            for xi, val in zip(x, means):
                if not np.isnan(val):
                    ax.annotate(f"{val:.3f}", (xi, val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(split_order)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("PR-AUC", fontsize=11, fontweight="bold")
        ax.set_xlabel("Data Split", fontsize=11, fontweight="bold")
        ax.set_title("Generalization Gap (Best Epoch)", fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if group_col:
            ax.legend(title="Architecture", frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        footnote_df = combined if not combined.empty else None
        self._save_figure(
            fig,
            output_path,
            f"Generalization Gap — {self.naming_var}",
            subtitle_y=0.90,
            footnote_df=footnote_df,
        )
        print(f"    Saved generalization gap plot: {output_path}")

    def generate_leaderboard(self, output_dir: Path):
        """Rank top configs by val_bestepoch pr_auc; save CSV and table PNG."""
        try:
            val_df = self.load_data("val_bestepoch")
        except FileNotFoundError:
            print("    Skipping leaderboard (val_bestepoch.csv not found)")
            return

        if "pr_auc" not in val_df.columns:
            print("    Skipping leaderboard (pr_auc column missing)")
            return

        before_quality = len(val_df)
        val_df = passes_quality_threshold(val_df)
        if val_df.empty:
            print(
                "    Skipping leaderboard (no configs meet recall/f1/precision "
                f"threshold >= {MIN_CLASSIFICATION_METRIC})"
            )
            return
        if len(val_df) < before_quality:
            print(
                f"    Leaderboard: discarded {before_quality - len(val_df)} "
                "trivial configs below quality threshold"
            )

        config_cols = self._config_columns(val_df)
        # mean_margin, mean_entropy, ece excluded: require calibration, not primary metrics.
        metric_cols = [
            m
            for m in [
                "pr_auc",
                "auc",
                "f1",
                "recall",
                "precision",
                "accuracy",
                "loss",
                "mean_confidence",
                "brier_score",
            ]
            if m in val_df.columns
        ]
        display_cols = config_cols + metric_cols + (["epoch"] if "epoch" in val_df.columns else [])

        ranked = val_df.sort_values("pr_auc", ascending=False).head(self.top_k)
        ranked = ranked[[c for c in display_cols if c in ranked.columns]]
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "leaderboard.csv"
        ranked.to_csv(csv_path, index=False)
        print(f"    Saved leaderboard CSV: {csv_path}")

        if ranked.empty:
            return

        n_table_cols = len(ranked.columns)
        fig_width = max(16, 1.6 * n_table_cols)
        fig_height = max(4, 0.50 * len(ranked) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
        ax.axis("off")

        table_data = ranked.copy()
        for col in metric_cols:
            if col in table_data.columns:
                table_data[col] = table_data[col].map(lambda v: f"{v:.4f}")

        # Shorten header labels to prevent cell overflow.
        _HEADER_SHORT = {
            "Layer Type": "Arch",
            "Dim Inner": "dim_in",
            "MP Layers": "MP",
            "Graph Pooling": "Pooling",
            "MEAN_CONFIDENCE": "CONF",
            "BRIER_SCORE": "BRIER",
        }
        cell_text = table_data.values.tolist()
        col_labels = [
            self._format_axis_label(c) if c in self.CONFIG_COLS else c.upper()
            for c in table_data.columns
        ]
        col_labels = [_HEADER_SHORT.get(lbl, lbl) for lbl in col_labels]

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(n_table_cols)))
        table.scale(1.0, 1.5)

        # Style header row.
        for col_idx in range(n_table_cols):
            cell = table[0, col_idx]
            cell.set_facecolor("#264653")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(7)

        self._save_figure(
            fig,
            output_dir / "leaderboard.png",
            f"Top {len(ranked)} Configurations by Val Synthetic PR-AUC — {self.naming_var}",
            subtitle_y=0.02,
            footnote_df=ranked,
        )
        print(f"    Saved leaderboard PNG: {output_dir / 'leaderboard.png'}")

    def _pareto_front(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of non-dominated points (both axes maximize).

        Point i is on the front when no other point is >= on both axes and
        strictly greater on at least one.
        """
        n = len(points)
        on_front = np.ones(n, dtype=bool)
        for i in range(n):
            if not on_front[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                dominates = (
                    points[j, 0] >= points[i, 0]
                    and points[j, 1] >= points[i, 1]
                    and (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])
                )
                if dominates:
                    on_front[i] = False
                    break
        return on_front

    def _plot_pareto_panel(self, ax, df, x_col, y_col, x_label, y_label, maximize_x):
        """Scatter configs and outline the Pareto-optimal trade-off frontier."""
        sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.empty:
            ax.set_visible(False)
            return False

        x_raw = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        # Internally both axes maximize; flip x when smaller-is-better (params, brier_score).
        x_obj = x_raw if maximize_x else -x_raw
        mask = self._pareto_front(np.column_stack([x_obj, y]))

        ax.scatter(
            x_raw[~mask], y[~mask], s=35, color="#B0B0B0",
            alpha=0.7, label="Dominated", zorder=2,
        )
        ax.scatter(
            x_raw[mask], y[mask], s=70, color="#E76F51",
            edgecolor="#264653", linewidth=1.2, label="Pareto front", zorder=3,
        )
        order = np.argsort(x_raw[mask])
        ax.plot(
            x_raw[mask][order], y[mask][order],
            color="#E76F51", linewidth=1.5, alpha=0.6, zorder=1,
        )
        ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(frameon=False, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return True

    def generate_pareto(self, output_path: Path):
        """Multi-metric Pareto fronts over val-synthetic best-epoch configs.

        Left: PR-AUC vs model size (params, smaller better). Right: PR-AUC vs
        brier_score (lower better) as the single uncalibrated signal.
        """
        try:
            df = self.load_data("val_bestepoch")
        except FileNotFoundError:
            print("    Skipping Pareto (val_bestepoch.csv not found)")
            return
        if "pr_auc" not in df.columns:
            print("    Skipping Pareto (pr_auc column missing)")
            return

        panels = []
        has_params = (
            "params" in df.columns
            and pd.to_numeric(df["params"], errors="coerce").gt(0).any()
        )
        if has_params:
            panels.append(("params", "pr_auc", "Params", "PR-AUC", False))
        # ECE removed (needs post-training calibration); brier_score kept as
        # the single uncalibrated signal for the Pareto trade-off view.
        if "brier_score" in df.columns and "pr_auc" in df.columns:
            panels.append(
                ("brier_score", "pr_auc", "Brier Score (↓ better)", "PR-AUC", False)
            )
        if not panels:
            # Fall back to a PR-AUC vs MP-layers trade-off when no cost axis available.
            if "layers_mp" in df.columns:
                panels.append(("layers_mp", "pr_auc", "MP Layers", "PR-AUC", False))
            else:
                print("    Skipping Pareto (no cost axis available)")
                return

        fig, axes = plt.subplots(
            1, len(panels), figsize=(7 * len(panels), 5.5), dpi=150
        )
        axes = np.atleast_1d(axes)
        plotted = False
        for ax, (x_col, y_col, x_label, y_label, max_x) in zip(axes, panels):
            plotted |= self._plot_pareto_panel(
                ax, df, x_col, y_col, x_label, y_label, max_x
            )

        if not plotted:
            plt.close(fig)
            print("    Skipping Pareto (no plottable panels)")
            return

        self._save_figure(
            fig,
            output_path,
            f"Pareto Fronts (Val Synthetic, Best Epoch) — {self.naming_var}",
            subtitle_y=0.92,
            footnote_df=df,
        )
        print(f"    Saved Pareto plot: {output_path}")

    def _evaluate_run_slices(self, run: str, df: pd.DataFrame):
        label = self.run_labels.get(run, run)
        run_out_dir = self.output_dir / run
        run_out_dir.mkdir(parents=True, exist_ok=True)

        self.generate_plots_for_df(
            df=df,
            overall_df=df,
            output_dir=run_out_dir,
            title=f"{label} - Overall ({self.naming_var})",
        )

        if self.skip_slices or "layer_type" not in df.columns:
            return

        for lt in df["layer_type"].dropna().unique():
            arch_df = df[df["layer_type"] == lt]
            if arch_df.empty:
                continue

            arch_dir = run_out_dir / "layer_type" / lt
            self.generate_plots_for_df(
                df=arch_df,
                overall_df=arch_df,
                output_dir=arch_dir,
                title=f"{label} - Layer: {lt} ({self.naming_var})",
            )

            if "layers_mp" in arch_df.columns:
                for mp in arch_df["layers_mp"].dropna().unique():
                    sub_df = arch_df[arch_df["layers_mp"] == mp]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=arch_df,
                            output_dir=arch_dir / "layers_mp" / f"{mp}_layers",
                            title=f"{label} - {lt} - MP Layers: {mp} ({self.naming_var})",
                            group_col="layers_mp",
                        )

            if "act" in arch_df.columns:
                for act in arch_df["act"].dropna().unique():
                    sub_df = arch_df[arch_df["act"] == act]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=arch_df,
                            output_dir=arch_dir / "act" / str(act),
                            title=f"{label} - {lt} - Activation: {act} ({self.naming_var})",
                            group_col="act",
                        )

            if "graph_pooling" in arch_df.columns:
                for pooling in arch_df["graph_pooling"].dropna().unique():
                    sub_df = arch_df[arch_df["graph_pooling"] == pooling]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=arch_df,
                            output_dir=arch_dir / "graph_pooling" / str(pooling),
                            title=f"{label} - {lt} - Pooling: {pooling} ({self.naming_var})",
                            group_col="graph_pooling",
                        )

            if "base_lr" in arch_df.columns:
                lr_values = arch_df["base_lr"].dropna().unique()
                if len(lr_values) > 1:
                    for lr in lr_values:
                        sub_df = arch_df[arch_df["base_lr"] == lr]
                        if not sub_df.empty:
                            self.generate_plots_for_df(
                                df=sub_df,
                                overall_df=arch_df,
                                output_dir=arch_dir / "base_lr" / f"lr_{lr}",
                                title=f"{label} - {lt} - LR: {lr} ({self.naming_var})",
                                group_col="base_lr",
                            )

    def run_all(self):
        """Runs evaluation for configured runs and global summary plots."""
        print(f"Starting GNN Evaluation on '{self.naming_var}'...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Runs: {', '.join(self.runs)}")
        if self.skip_slices:
            print("Slice generation: disabled")

        for run in self.runs:
            label = self.run_labels.get(run, run)
            print(f"  Evaluating run: {run} ({label})...")
            try:
                df = self.load_data(run)
            except FileNotFoundError:
                print(f"    Skipping {run} (file not found)")
                continue
            self._evaluate_run_slices(run, df)

        print("  Generating global summary plots...")
        self.generate_summary_comparison(self.output_dir / "split_comparison.png")
        self.generate_generalization_gap(self.output_dir / "generalization_gap.png")
        self.generate_leaderboard(self.output_dir)
        self.generate_pareto(self.output_dir / "pareto.png")

        print(f"Evaluation complete! Plots saved to {self.output_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots for GNN grid-search results."
    )
    parser.add_argument(
        "naming_var",
        nargs="?",
        help="Experiment folder under run_results/ (e.g. res_with_enrich)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all 9 run CSV variants (default: best-epoch splits only)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="RUN",
        help="Explicit list of run CSV stems to evaluate",
    )
    parser.add_argument(
        "--skip-slices",
        action="store_true",
        help="Only generate top-level run plots (no nested architecture slices)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of configs in the leaderboard (default: 10)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory containing experiment folders (default: run_results/)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    naming_var = args.naming_var
    if not naming_var:
        naming_var = input("Enter naming var (e.g. res_with_enrich): ").strip()
        if not naming_var:
            naming_var = "res_with_enrich"

    if args.runs:
        runs = args.runs
    elif args.full:
        runs = GNNResultEvaluator.ALL_RUNS
    else:
        runs = GNNResultEvaluator.DEFAULT_RUNS

    evaluator = GNNResultEvaluator(
        naming_var=naming_var,
        base_dir=args.base_dir,
        runs=runs,
        skip_slices=args.skip_slices,
        top_k=args.top_k,
    )
    evaluator.run_all()


if __name__ == "__main__":
    main()
