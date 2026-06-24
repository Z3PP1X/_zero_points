"""Shared evaluation constants and helpers for post-training analysis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EVAL_WARMUP_EPOCHS = 3
MIN_CLASSIFICATION_METRIC = 0.25
QUALITY_THRESHOLD_METRICS = ("recall", "f1", "precision")
LOWER_IS_BETTER_METRICS = frozenset({"loss", "brier_score", "ece", "mean_entropy"})


def filter_post_warmup(stats_list: list[dict], warmup_epochs: int = EVAL_WARMUP_EPOCHS) -> list[dict]:
    """Drop warmup epochs; fall back to the full list if nothing remains."""
    eligible = [s for s in stats_list if s.get("epoch", 0) >= warmup_epochs]
    return eligible if eligible else list(stats_list)


def select_best_epoch_row(
    stats_list: list[dict],
    metric: str,
    agg: str = "argmax",
    warmup_epochs: int = EVAL_WARMUP_EPOCHS,
) -> dict:
    """Return the stats row with the best metric after excluding warmup epochs."""
    eligible = filter_post_warmup(stats_list, warmup_epochs=warmup_epochs)
    performance_np = np.array([stats[metric] for stats in eligible])
    if metric in LOWER_IS_BETTER_METRICS and agg == "argmax":
        agg = "argmin"
    idx = int(eval(f"performance_np.{agg}()"))
    return eligible[idx]


def select_best_epoch(
    stats_list: list[dict],
    metric: str,
    agg: str = "argmax",
    warmup_epochs: int = EVAL_WARMUP_EPOCHS,
) -> int:
    """Return the best epoch number after excluding warmup epochs."""
    return int(select_best_epoch_row(stats_list, metric, agg, warmup_epochs)["epoch"])


def filter_warmup_epochs_df(df, warmup_epochs: int = EVAL_WARMUP_EPOCHS):
    """Exclude warmup epochs from evaluation dataframes when an epoch column exists."""
    if df is None or df.empty or "epoch" not in df.columns:
        return df
    filtered = df[df["epoch"] >= warmup_epochs]
    return filtered if not filtered.empty else df


def passes_quality_threshold(df, min_metric: float = MIN_CLASSIFICATION_METRIC):
    """Drop rows where recall, f1, or precision fall below the minimum threshold."""
    if df is None or df.empty:
        return df

    mask = np.ones(len(df), dtype=bool)
    for metric in QUALITY_THRESHOLD_METRICS:
        if metric in df.columns:
            mask &= df[metric].to_numpy() >= min_metric
    filtered = df.loc[mask]
    return filtered if not filtered.empty else df.iloc[0:0]


def _as_tensor(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.as_tensor(values)


def prediction_probabilities(pred_score) -> torch.Tensor:
    """Return class probabilities [N, C] from logits, log-softmax, or sigmoid scores."""
    pred_t = _as_tensor(pred_score).float()
    if pred_t.ndim > 1 and pred_t.shape[1] > 1:
        if torch.all(pred_t <= 1e-5):
            return pred_t.exp()
        return F.softmax(pred_t, dim=-1)

    scores = torch.sigmoid(pred_t.squeeze(-1) if pred_t.ndim > 1 else pred_t)
    if scores.ndim == 0:
        scores = scores.unsqueeze(0)
    return torch.stack([1.0 - scores, scores], dim=-1)


def expected_calibration_error(
    probs_pos,
    labels_pos,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for binary positive-class probabilities."""
    probs_pos = np.asarray(probs_pos, dtype=float)
    labels_pos = np.asarray(labels_pos, dtype=float)
    if probs_pos.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx < n_bins - 1:
            mask = (probs_pos > lower) & (probs_pos <= upper)
        else:
            mask = (probs_pos >= lower) & (probs_pos <= upper)
        if not np.any(mask):
            continue
        bin_acc = float(labels_pos[mask].mean())
        bin_conf = float(probs_pos[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
    return ece


def compute_confidence_metrics(
    true,
    pred_score,
    pos_label: int,
    round_digits: int = 4,
) -> dict[str, float]:
    """Aggregate confidence / calibration metrics from model scores."""
    probs = prediction_probabilities(pred_score)
    true_t = _as_tensor(true).long().view(-1)

    confidence = probs.max(dim=1).values
    pos_probs = probs[:, int(pos_label)]
    margin = (pos_probs - 0.5).abs() * 2.0
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
    max_entropy = float(np.log(max(probs.shape[1], 2)))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else entropy

    y_pos = (true_t == int(pos_label)).float()
    brier = float(((pos_probs - y_pos) ** 2).mean().item())
    ece = expected_calibration_error(pos_probs.detach().cpu().numpy(), y_pos.numpy())

    return {
        "mean_confidence": round(float(confidence.mean().item()), round_digits),
        "mean_margin": round(float(margin.mean().item()), round_digits),
        "mean_entropy": round(float(norm_entropy.mean().item()), round_digits),
        "brier_score": round(brier, round_digits),
        "ece": round(ece, round_digits),
    }
