"""Gradient-based node feature importance for GraphGym configurations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from gnn.supervised_learning.run_results._plot_utils import save_figure
import numpy as np
import pandas as pd
import torch
from torch_geometric.graphgym.loss import compute_loss

from gnn.shared.utils.graph_utils import NODE_FEATURE_SCHEMA

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def resolve_feature_names(cfg) -> list[str]:
    """Return active node feature names for the current GraphGym config."""
    from gnn.supervised_learning.supervised_config import resolve_expression_graph_features
    from gnn.shared.utils.feature_config import plain_dict

    try:
        expr_graph_dict = plain_dict(cfg.expression_graph)
        _, active_features = resolve_expression_graph_features(
            expr_graph_dict,
        )
        if active_features is not None:
            return active_features
    except Exception as e:
        logging.warning(f"Failed to resolve active features dynamically: {e}")

    schema = list(NODE_FEATURE_SCHEMA)
    active_features_str = getattr(cfg.expression_graph, "active_features", "") or ""
    if active_features_str.strip():
        return [part.strip() for part in active_features_str.split(",") if part.strip()]
    return list(schema)


def compute_gradient_feature_importance(
    model,
    loader,
    *,
    device: str | torch.device,
    pos_label: int,
    max_batches: int = 32,
) -> np.ndarray | None:
    """
    Compute mean |grad * input| saliency per node feature dimension.

    Aggregates over graphs and nodes on the provided loader.
    """
    if loader is None:
        return None

    model.eval()
    importance_sum = None
    graph_count = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch = batch.to(device)
        x = batch.x.detach().clone().requires_grad_(True)
        batch.x = x

        pred, _true = model(batch)
        _loss, pred_score = compute_loss(pred, _true)
        if pred_score.ndim > 1:
            target = pred_score[:, int(pos_label)].sum()
        else:
            target = pred_score.sum()

        model.zero_grad(set_to_none=True)
        target.backward()

        if x.grad is None:
            continue

        saliency = (x * x.grad).abs().detach().cpu()
        batch_vec = batch.batch.detach().cpu()
        num_graphs = int(batch_vec.max().item()) + 1

        for graph_idx in range(num_graphs):
            node_mask = batch_vec == graph_idx
            if not node_mask.any():
                continue
            graph_importance = saliency[node_mask].mean(dim=0).numpy()
            importance_sum = (
                graph_importance
                if importance_sum is None
                else importance_sum + graph_importance
            )
            graph_count += 1

    if importance_sum is None or graph_count == 0:
        return None

    importance = importance_sum / graph_count
    total = float(importance.sum())
    if total > 0:
        importance = importance / total
    return importance


def save_feature_importance_artifacts(
    feature_names: list[str],
    importance: np.ndarray,
    output_dir: Path,
    *,
    split_label: str,
    top_k: int = 15,
) -> Path:
    """Persist feature importance as JSON and a bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    importance = np.asarray(importance, dtype=float)
    if len(feature_names) != importance.shape[0]:
        feature_names = [f"feature_{idx}" for idx in range(importance.shape[0])]

    ranked_idx = np.argsort(importance)[::-1]
    payload = {
        "split": split_label,
        "feature_names": feature_names,
        "importance": {
            feature_names[idx]: round(float(importance[idx]), 6) for idx in range(len(feature_names))
        },
        "top_features": [
            {
                "feature": feature_names[idx],
                "importance": round(float(importance[idx]), 6),
            }
            for idx in ranked_idx[:top_k]
        ],
    }

    json_path = output_dir / f"feature_importance_{split_label}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    plot_top = min(top_k, len(feature_names))
    top_idx = ranked_idx[:plot_top]
    labels = [feature_names[idx] for idx in top_idx][::-1]
    values = [importance[idx] for idx in top_idx][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * plot_top + 1.5)), dpi=150)
    ax.barh(labels, values, color="#2A9D8F")
    ax.set_xlabel("Normalized importance (|grad × input|)")
    ax.set_title(f"Feature Importance — {split_label.replace('_', ' ').title()}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    png_path = output_dir / f"feature_importance_{split_label}.png"
    save_figure(png_path)
    plt.close(fig)
    return json_path


def run_post_training_feature_importance(
    model,
    datamodule,
    *,
    ckpt_path: str | Path | None,
    out_dir: str | Path,
    device: str | torch.device | None = None,
    max_batches: int = 32,
) -> list[Path]:
    """
    Load the best checkpoint and compute feature importance per validation split.
    """
    from torch_geometric.graphgym.config import cfg

    from gnn.supervised_learning.loader_graphgym import get_pos_label

    out_dir = Path(out_dir)
    device = device or next(model.parameters()).device
    saved_paths: list[Path] = []

    if ckpt_path:
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    feature_names = resolve_feature_names(model.cfg if hasattr(model, "cfg") else cfg)
    pos_label = get_pos_label()
    synthetic = bool(getattr(cfg.expression_graph, "synthetic", False))
    loaders = getattr(datamodule, "loaders", [])
    split_loaders = []
    if synthetic and len(loaders) >= 3:
        split_loaders = [
            ("val_synthetic", loaders[1]),
            ("val_curated", loaders[2]),
        ]
    elif len(loaders) >= 2:
        split_loaders = [("val", loaders[1])]

    for split_label, loader in split_loaders:
        importance = compute_gradient_feature_importance(
            model,
            loader,
            device=device,
            pos_label=pos_label,
            max_batches=max_batches,
        )
        if importance is None:
            print(f"[FeatureImportance] Skipped {split_label} (no batches)")
            continue

        saved_paths.append(
            save_feature_importance_artifacts(
                feature_names,
                importance,
                out_dir,
                split_label=split_label,
            )
        )
        print(f"[FeatureImportance] Saved {split_label} importance to {out_dir}")

    return saved_paths


def _load_feature_importance_records(results_dir: Path) -> list[dict]:
    records = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "agg":
            continue
        if not run_dir.name.startswith("grid"):
            continue

        for json_path in sorted(run_dir.glob("feature_importance_*.json")):
            with open(json_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            split = payload.get("split", json_path.stem.replace("feature_importance_", ""))
            importance_map = payload.get("importance", {})
            record = {"run_dir": run_dir.name, "split": split}
            for feature, value in importance_map.items():
                record[feature] = value
            records.append(record)
    return records


def aggregate_feature_importance_plots(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "val_synthetic",
    top_features: int = 10,
) -> bool:
    """Build a cross-configuration heatmap of top feature importances."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    records = _load_feature_importance_records(results_dir)
    if not records:
        print("    Skipping feature-importance aggregation (no JSON files found)")
        return False

    df = pd.DataFrame(records)
    if "split" in df.columns:
        split_df = df[df["split"] == split]
        if split_df.empty:
            split_df = df
    else:
        split_df = df

    meta_cols = {"run_dir", "split"}
    feature_cols = [col for col in split_df.columns if col not in meta_cols]
    if not feature_cols:
        return False

    mean_importance = split_df[feature_cols].mean().sort_values(ascending=False)
    selected = mean_importance.head(top_features).index.tolist()
    matrix = split_df.set_index("run_dir")[selected]
    matrix = matrix.fillna(0.0)

    fig_height = max(4.0, 0.35 * len(matrix.index) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=150)
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlGn", origin="upper")
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(
        f"Feature Importance Across Configurations ({split})",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"feature_importance_heatmap_{split}.png"
    save_figure(out_path)
    plt.close(fig)

    csv_path = output_dir / f"feature_importance_matrix_{split}.csv"
    matrix.to_csv(csv_path)
    print(f"    Saved feature-importance heatmap: {out_path}")
    print(f"    Saved feature-importance matrix: {csv_path}")
    return True
