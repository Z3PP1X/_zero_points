"""Shared evaluation constants and helpers for post-training analysis."""

from __future__ import annotations

import numpy as np

EVAL_WARMUP_EPOCHS = 3
MIN_CLASSIFICATION_METRIC = 0.25
QUALITY_THRESHOLD_METRICS = ("recall", "f1", "precision")


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
