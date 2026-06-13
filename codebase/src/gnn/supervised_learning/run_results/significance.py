"""Bootstrap significance over a single evaluation set.

With single-seed runs there is no across-run variance to test, so confidence
comes from resampling the *predictions* of one fixed eval set. We bootstrap the
held-out predictions (sampling rows with replacement) to put a confidence
interval on a metric, and use a *paired* bootstrap — the same resampled row
indices applied to two models' scores — to test whether one config beats another
on the very same examples.

Inputs are per-prediction arrays dumped by the diagnostics pass
(``predictions_<split>.npz`` with ``y_true`` and ``probs_pos``). Pure
numpy + sklearn; no torch, no model reload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_N_BOOT = 1000
DEFAULT_ALPHA = 0.05


def pr_auc_metric(y_true: np.ndarray, probs_pos: np.ndarray) -> float:
    """Average precision (PR-AUC); 0.0 when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, probs_pos))


def roc_auc_metric(y_true: np.ndarray, probs_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, probs_pos))


def accuracy_metric(
    y_true: np.ndarray, probs_pos: np.ndarray, thresh: float = 0.5
) -> float:
    preds = (probs_pos >= thresh).astype(int)
    return float((preds == y_true).mean())


METRIC_FNS = {
    "pr_auc": pr_auc_metric,
    "auc": roc_auc_metric,
    "accuracy": accuracy_metric,
}


def _resample_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``[n_boot, n]`` matrix of with-replacement row indices."""
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_metric_ci(
    y_true,
    probs_pos,
    metric: str = "pr_auc",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict:
    """Point estimate and percentile bootstrap CI for one metric.

    Returns ``{"metric", "point", "lo", "hi", "alpha", "n_boot", "n"}``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    probs_pos = np.asarray(probs_pos, dtype=float).ravel()
    metric_fn = METRIC_FNS[metric]
    n = y_true.shape[0]

    point = metric_fn(y_true, probs_pos)
    if n == 0:
        return {
            "metric": metric, "point": point, "lo": point, "hi": point,
            "alpha": alpha, "n_boot": 0, "n": 0,
        }

    rng = np.random.default_rng(seed)
    idx = _resample_indices(n, n_boot, rng)
    samples = np.array([metric_fn(y_true[i], probs_pos[i]) for i in idx])
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {
        "metric": metric, "point": float(point), "lo": lo, "hi": hi,
        "alpha": alpha, "n_boot": n_boot, "n": int(n),
    }


def paired_bootstrap_diff(
    y_true,
    probs_a,
    probs_b,
    metric: str = "pr_auc",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of ``metric(a) - metric(b)`` on a shared eval set.

    Both models are resampled with the *same* row indices each iteration, so the
    interval reflects the per-example difference and cancels eval-set noise. The
    two-sided p-value is the bootstrap fraction whose sign opposes the observed
    difference (doubled), clipped to [0, 1].

    Returns ``{"metric", "diff", "lo", "hi", "p_value", "significant", ...}``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    probs_a = np.asarray(probs_a, dtype=float).ravel()
    probs_b = np.asarray(probs_b, dtype=float).ravel()
    metric_fn = METRIC_FNS[metric]
    n = y_true.shape[0]

    diff_point = metric_fn(y_true, probs_a) - metric_fn(y_true, probs_b)
    if n == 0:
        return {
            "metric": metric, "diff": diff_point, "lo": diff_point,
            "hi": diff_point, "p_value": 1.0, "significant": False,
            "alpha": alpha, "n_boot": 0, "n": 0,
        }

    rng = np.random.default_rng(seed)
    idx = _resample_indices(n, n_boot, rng)
    diffs = np.array(
        [
            metric_fn(y_true[i], probs_a[i]) - metric_fn(y_true[i], probs_b[i])
            for i in idx
        ]
    )
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    # Two-sided bootstrap p-value: fraction on the opposite side of 0, doubled.
    if diff_point >= 0:
        p_value = float(2.0 * np.mean(diffs <= 0.0))
    else:
        p_value = float(2.0 * np.mean(diffs >= 0.0))
    p_value = min(1.0, p_value)
    return {
        "metric": metric, "diff": float(diff_point), "lo": lo, "hi": hi,
        "p_value": p_value, "significant": bool(p_value < alpha),
        "alpha": alpha, "n_boot": n_boot, "n": int(n),
    }


def save_predictions(path: Path, y_true, probs_pos, pos_label: int = 1) -> Path:
    """Persist per-prediction arrays for later bootstrap analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        y_true=np.asarray(y_true).astype(int).ravel(),
        probs_pos=np.asarray(probs_pos, dtype=float).ravel(),
        pos_label=int(pos_label),
    )
    return path


def load_predictions(path: Path) -> dict | None:
    """Load a ``predictions_*.npz`` dump; ``None`` if missing/unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except (OSError, ValueError):
        return None
    return {
        "y_true": data["y_true"],
        "probs_pos": data["probs_pos"],
        "pos_label": int(data["pos_label"]) if "pos_label" in data else 1,
    }


def discover_prediction_dumps(eval_dir: Path, split: str) -> list[tuple[str, Path]]:
    """Find ``predictions_<split>.npz`` under ``eval_dir/top_configs/rank_*``.

    Returns ``[(rank_dir_name, npz_path), ...]`` sorted by rank, so callers can
    line up the best config (rank 1) against the runner-up.
    """
    top_dir = Path(eval_dir) / "top_configs"
    if not top_dir.exists():
        return []
    found = []
    for rank_dir in sorted(top_dir.glob("rank_*")):
        npz = rank_dir / f"predictions_{split}.npz"
        if npz.exists():
            found.append((rank_dir.name, npz))
    return found
