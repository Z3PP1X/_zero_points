#!/usr/bin/env python3
"""Re-evaluate trained supervised-GNN runs with the FIXED metric code — NO retraining.

Why this exists
---------------
Two bugs lived purely in the metric/reporting layer of the final_model_selection runs:

  Bug 1 — ``loader_graphgym._hard_predictions`` returned a positive-class *indicator*
          (1 == "predicted class 0") for ``pos_label == 0`` instead of a class *label*,
          inverting every hard-label metric (accuracy/precision/recall/f1). Symptom:
          leaderboard ``accuracy ≈ 1 - true_accuracy`` (e.g. 0.1999 instead of ~0.80).

  Bug 2 — ``eval_metrics.prediction_probabilities`` applied ``sigmoid`` to the single-logit
          score, which the loss had ALREADY sigmoided — a double sigmoid that squashed
          every probability toward 0.5 and corrupted ECE / Brier / the reliability diagram
          and the saved ``predictions_*.npz``.

Neither bug touched training or model selection: the loss is ``BCEWithLogitsLoss`` on raw
logits, and checkpoint/early-stopping selection used ``val_auc`` (rank-based, unaffected).
So the *weights are correct* and the AUC ranking is unchanged — only the reported numbers
need recomputing. This script reloads each run's best checkpoint, re-runs inference with
the fixed code, and writes corrected plots, ``predictions_*.npz`` and a metrics JSON, plus
a combined corrected leaderboard CSV.

Usage (from the repo root, with the run-time conda env active — ``pytorch`` here)
--------------------------------------------------------------------------------
  python -m gnn.supervised_learning.run_results.reeval_metrics <DIR> [<DIR> ...]

  <DIR> may be a single run directory (contains ``config.yaml`` + a checkpoint) or an
  experiment directory containing many such runs — they are discovered recursively.

Options
-------
  --in-place        Overwrite each run's existing ``diagnostics/`` (default: write to a
                    sibling ``diagnostics_corrected/`` so the original buggy artifacts are
                    kept for comparison).
  --device cpu|cuda Force a device (default: cuda if available, else cpu).
  --glob PATTERN    Override checkpoint glob (default: best-*.ckpt, fallback last.ckpt).

Outputs (per run)
-----------------
  <run>/diagnostics_corrected/predictions_<split>.npz   corrected probabilities
  <run>/diagnostics_corrected/{confusion,roc,pr,reliability}_<split>.png
  <run>/diagnostics_corrected/metrics_corrected.json     full corrected metric set + a
                                                         before/after delta vs the old npz
And one ``reeval_corrected_leaderboard.csv`` per top-level <DIR> argument.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless remote-safe; must precede any pyplot import

import numpy as np
import torch

# Make ``gnn`` importable however this file is launched (``-m`` with PYTHONPATH, a bare
# ``python .../reeval_metrics.py``, or from any cwd) — mirrors diagnostics._setup_import_paths.
_HERE = Path(__file__).resolve()
for _p in (str(_HERE.parents[3]), str(_HERE.parents[2])):  # codebase/src and codebase/src/gnn
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _find_runs(root: Path) -> list[Path]:
    """A run dir has a config.yaml and at least one checkpoint beneath it."""
    root = Path(root)
    if (root / "config.yaml").exists() and _find_checkpoint(root) is not None:
        return [root]
    runs = []
    for cfg_path in sorted(root.rglob("config.yaml")):
        run_dir = cfg_path.parent
        if _find_checkpoint(run_dir) is not None:
            runs.append(run_dir)
    return runs


def _find_checkpoint(run_dir: Path, pattern: str = "best-*.ckpt") -> Path | None:
    """Best checkpoint by parsed val_auc; fall back to last.ckpt."""
    best = list(run_dir.rglob(pattern))

    def _auc(p: Path) -> float:
        # filename like best-epoch=53-val_auc=0.8413.ckpt
        for tok in p.stem.split("-"):
            if tok.startswith("val_auc="):
                try:
                    return float(tok.split("=", 1)[1])
                except ValueError:
                    return -1.0
        return -1.0

    if best:
        return max(best, key=_auc)
    fallback = list(run_dir.rglob("last.ckpt"))
    return fallback[0] if fallback else None


def _old_npz_metrics(npz_path: Path) -> dict | None:
    """Recompute hard metrics from a previously-saved (buggy) npz, for a before/after delta.

    The old npz stored a double-sigmoided probs_pos; we just report what acc/f1 those
    files imply at threshold 0.5 so the user can see the size of the correction.
    """
    if not npz_path.exists():
        return None
    from sklearn.metrics import accuracy_score, f1_score

    d = np.load(npz_path)
    y = d["y_true"]
    pos = int(d["pos_label"])
    p_pos = d["probs_pos"]  # P(positive class), per the OLD (buggy) pipeline
    # The old hard rule predicted the positive class when p_pos >= 0.5; map to a label.
    pred_pos = (p_pos >= 0.5).astype(int)
    pred = pred_pos if pos == 1 else 1 - pred_pos
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "f1": round(float(f1_score(y, pred, pos_label=pos, zero_division=0)), 6),
    }


def _reeval_run(run_dir: Path, in_place: bool, device: str, ckpt_glob: str) -> list[dict]:
    """Reload one run, recompute corrected metrics, rewrite diagnostics. Returns CSV rows."""
    from gnn.supervised_learning.loader_graphgym import compute_binary_metrics
    from gnn.supervised_learning.run_results.diagnostics import (
        DiagnosticPlotter,
        _split_pos_label,
    )

    config_path = run_dir / "config.yaml"
    ckpt_path = _find_checkpoint(run_dir, ckpt_glob)
    if ckpt_path is None:
        print(f"  [skip] no checkpoint under {run_dir}")
        return []

    out_dir = run_dir / ("diagnostics" if in_place else "diagnostics_corrected")
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = DiagnosticPlotter(
        results_dir=run_dir,
        output_dir=run_dir,
        experiment_name=run_dir.name,
        device=device,
    )
    # Build the model + data loaders exactly as training did, then load the best weights.
    model, datamodule, get_pos_label, hard_pred_fn, _ = plotter._load_model_and_loaders(
        config_path
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.to(device)

    # Plots + corrected predictions_*.npz (uses the fixed _hard_predictions /
    # prediction_probabilities). Returns the split prefixes actually written.
    prefixes = plotter.render_split_diagnostics(
        model, datamodule, out_dir, get_pos_label, hard_pred_fn, log_prefix="[reeval]"
    )

    rows, per_split = [], {}
    for prefix in prefixes:
        npz = np.load(out_dir / f"predictions_{prefix}.npz")
        y_true = torch.tensor(npz["y_true"])
        pos_label = int(npz["pos_label"])
        # Reconstruct the single-logit P(class 1) from the stored P(positive class), then
        # run the full (fixed) metric stack — hard metrics + AUC/PR-AUC + ECE/Brier.
        p_pos = npz["probs_pos"]
        p1 = p_pos if pos_label == 1 else 1.0 - p_pos
        metrics = compute_binary_metrics(
            y_true, torch.tensor(p1), pos_label=pos_label, round_digits=6
        )
        before = _old_npz_metrics(run_dir / "diagnostics" / f"predictions_{prefix}.npz")
        per_split[prefix] = {
            "pos_label": pos_label,
            "n": int(len(y_true)),
            "corrected": metrics,
            "old_buggy": before,
        }
        rows.append(
            {
                "run": run_dir.name,
                "split": prefix,
                "n": int(len(y_true)),
                "pos_label": pos_label,
                **{k: metrics.get(k) for k in
                   ("accuracy", "precision", "recall", "f1", "auc", "pr_auc",
                    "brier_score", "ece")},
                "old_accuracy": (before or {}).get("accuracy"),
                "old_f1": (before or {}).get("f1"),
            }
        )
        d = "" if before is None else (
            f"  (was acc={before['accuracy']:.4f} f1={before['f1']:.4f})"
        )
        print(
            f"  {prefix:24s} pos={pos_label} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f} ece={metrics['ece']:.4f}{d}"
        )

    (out_dir / "metrics_corrected.json").write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "checkpoint": str(ckpt_path.relative_to(run_dir)),
                "splits": per_split,
            },
            indent=2,
        )
    )
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    import csv

    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote corrected leaderboard: {path}  ({len(rows)} split-rows)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="run or experiment directories")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite each run's diagnostics/ (default: diagnostics_corrected/)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--glob", default="best-*.ckpt", help="checkpoint glob")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Re-evaluating with FIXED metric code on device={device}\n")

    failures = []
    for top in args.dirs:
        runs = _find_runs(top)
        if not runs:
            print(f"[warn] no runs (config.yaml + checkpoint) found under {top}")
            continue
        print(f"== {top}  ({len(runs)} run(s)) ==")
        all_rows = []
        for run_dir in runs:
            print(f"- {run_dir}")
            try:
                all_rows.extend(_reeval_run(run_dir, args.in_place, device, args.glob))
            except Exception as exc:  # keep going; one bad run shouldn't abort the sweep
                failures.append((run_dir, repr(exc)))
                print(f"  [FAIL] {exc!r}")
        _write_csv(all_rows, Path(top) / "reeval_corrected_leaderboard.csv")

    if failures:
        print("\nFailures:")
        for run_dir, err in failures:
            print(f"  {run_dir}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
