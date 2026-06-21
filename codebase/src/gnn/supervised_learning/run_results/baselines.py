"""Trivial-classifier baselines from class balance.

Single-seed runs have no across-seed variance, so the only honest reference for
"is this model doing anything?" is a non-learning baseline. Given the per-split
class counts (``agg/class_balance.json``) we derive, in closed form, the metrics
of two reference classifiers plus the no-skill PR line:

  * **majority** — always predict the majority class.
  * **random**   — predict the positive class with probability equal to its
                   prevalence (stratified random guessing).
  * **no_skill_pr_auc** — PR-AUC of any constant-score classifier equals the
                   positive-class prevalence; this is the reference line a real
                   model must beat on the leaderboard.

Everything here is pure arithmetic over the counts — no model, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path

# faster_algorithm: 0 = gMGF, 1 = Newton. The pipeline treats class 1 as the
# positive class for pr_auc / precision / recall, so baselines mirror that.
DEFAULT_POS_LABEL = 1

# class_balance.json split key -> the run CSV stem it corresponds to.
SPLIT_KEYS = {
    "validation_synthetic": "val_bestepoch",
    "validation_curated": "test_bestepoch",
}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def split_baselines(counts: dict, pos_label: int = DEFAULT_POS_LABEL) -> dict:
    """Closed-form baseline metrics for one split given its class counts.

    ``counts`` is ``{"0": n0, "1": n1, "total": n}`` (``total`` optional).
    Returns a dict with ``prevalence``, ``majority`` and ``random`` blocks plus
    ``no_skill_pr_auc``, or an empty dict when the split is empty.
    """
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    total = int(counts.get("total", n0 + n1)) or (n0 + n1)
    if total <= 0:
        return {}

    n_pos = n1 if pos_label == 1 else n0
    n_neg = total - n_pos
    prevalence = _safe_div(n_pos, total)  # positive-class rate == no-skill PR-AUC

    # Majority classifier: always predict whichever class is larger.
    majority_is_pos = n_pos >= n_neg
    majority_acc = _safe_div(max(n_pos, n_neg), total)
    if majority_is_pos:
        maj_precision = prevalence  # every sample predicted positive
        maj_recall = 1.0
    else:
        maj_precision = 0.0  # no positive predictions at all
        maj_recall = 0.0
    majority = {
        "predicts": "positive" if majority_is_pos else "negative",
        "accuracy": round(majority_acc, 4),
        "precision": round(maj_precision, 4),
        "recall": round(maj_recall, 4),
        "f1": round(_f1(maj_precision, maj_recall), 4),
        "pr_auc": round(prevalence, 4),
        "auc": 0.5,
    }

    # Stratified random: predict positive w.p. == prevalence p.
    # Expected accuracy = p^2 + (1-p)^2; precision == p; recall == p.
    p = prevalence
    random_acc = p * p + (1.0 - p) * (1.0 - p)
    random = {
        "accuracy": round(random_acc, 4),
        "precision": round(p, 4),
        "recall": round(p, 4),
        "f1": round(p, 4),
        "pr_auc": round(p, 4),
        "auc": 0.5,
    }

    return {
        "counts": {"0": n0, "1": n1, "total": total, "pos_label": pos_label},
        "prevalence": round(prevalence, 4),
        "no_skill_pr_auc": round(prevalence, 4),
        "majority": majority,
        "random": random,
    }


def load_class_balance(agg_dir: Path) -> dict | None:
    """Read ``class_balance.json`` from an experiment's agg directory."""
    cb_file = Path(agg_dir) / "class_balance.json"
    if not cb_file.exists():
        return None
    try:
        with open(cb_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def compute_baselines(
    agg_dir: Path, pos_label: int = DEFAULT_POS_LABEL
) -> dict:
    """Baselines for every split present in ``class_balance.json``.

    Returns ``{run_stem: split_baselines(...)}`` keyed by the run CSV stem
    (e.g. ``val_bestepoch``) so the report can line baselines up with metrics.
    Empty dict when no class balance file is available.
    """
    balance = load_class_balance(agg_dir)
    if not balance:
        return {}

    out: dict = {}
    for split_key, run_stem in SPLIT_KEYS.items():
        counts = balance.get(split_key)
        if not counts:
            continue
        result = split_baselines(counts, pos_label=pos_label)
        if result:
            out[run_stem] = result
    return out
