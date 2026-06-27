"""Post-hoc Platt calibration + F1-optimal decision threshold for a trained classifier.

Loads a pickled checkpoint (.pth state_dict or GraphGym .ckpt), rebuilds the exact model
+ validation loaders it was trained with, then on the chosen split:

  1. fits Platt scaling  p_cal = sigmoid(a * logit + b)  so the reported probabilities
     (ECE / Brier / NLL) become trustworthy; and
  2. replaces the 0.5 threshold with the F1-optimal one -- swept, or supplied directly
     via --threshold when the operating point for this model is already known.

Model/loader reconstruction reuses DiagnosticPlotter so the rebuilt network matches the
trained one (edge_dim, accelerator, synthetic-holdout vs curated split, sigmoid scores).

Outputs (under --output, default <run-dir>/calibration/):
  * calibration.json -- Platt (a, b), threshold + its scale, val metrics vs 0.5.
  * reliability.png  -- reliability diagram, uncalibrated vs Platt-calibrated.

Batch mode (--experiment-dir): instead of one run, point at a whole experiment folder
(run_results/<exp>). It ranks the top-K models from agg/val_bestepoch.csv (the same
leaderboard the diagnostics use), fits Platt + threshold on each, and re-renders the
confusion / ROC / PR / reliability diagnostics with the calibration applied into
eval_plots/top_configs/<model>/calibrated/ -- next to the uncalibrated diagnostics --
plus a top-level calibration_summary.csv / .json.

Examples:
    python calibration.py --run-dir ../run_results/stage1/run_20260624_082939_000
    python calibration.py --run-dir <run> --threshold 0.63
    python calibration.py --config <cfg.yaml> --checkpoint <ckpt> --calibration none
    python calibration.py --experiment-dir ../run_results/stage1 --top-k 5
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def _setup_import_paths():
    """Put src/ (and codebase/) on sys.path so `import gnn...` resolves as a script."""
    here = Path(__file__).resolve()
    for path in (str(here.parents[3]), str(here.parents[4])):  # src/, codebase/
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_import_paths()

from gnn.supervised_learning.run_results.diagnostics import (  # noqa: E402
    CONFIG_COLS,
    DiagnosticPlotter,
    _config_slug,
    _dataset_label_for_split,
    _find_best_checkpoint,
    _find_config_for_run,
    _positive_class_probs,
    _provenance_subtitle,
)
from gnn.supervised_learning.run_results.eval_metrics import (  # noqa: E402
    expected_calibration_error,
)
from gnn.supervised_learning.run_results.significance import (  # noqa: E402
    save_predictions,
)

CLASS_NAMES = {0: "gMGF", 1: "Newton"}
EPS = 1e-6  # logit clamp so sigmoid(logit(p)) stays finite at p in {0, 1}


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #
def resolve_inputs(args) -> tuple[Path, Path]:
    """Return (config_path, checkpoint_path) from an explicit pair or a run dir."""
    if args.config and args.checkpoint:
        return Path(args.config), Path(args.checkpoint)
    if not args.run_dir:
        raise SystemExit("Provide --run-dir, or both --config and --checkpoint.")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir is not a directory: {run_dir}")

    config_path = (Path(args.config) if args.config
                   else _find_config_for_run(run_dir, None))
    if config_path is None or not config_path.exists():
        raise SystemExit(f"No config.yaml under {run_dir} (pass --config explicitly).")

    ckpt = Path(args.checkpoint) if args.checkpoint else _find_best_checkpoint(run_dir)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"No checkpoint under {run_dir} (pass --checkpoint).")
    return config_path, ckpt


# --------------------------------------------------------------------------- #
# Load model + collect positive-class scores on the requested split
# --------------------------------------------------------------------------- #
def collect_scores(config_path, ckpt_path, split, device, pos_label_override):
    """Rebuild the trained model + loaders, load the pickle, and return per-graph
    P(positive class) plus the positive-class indicator on the chosen split.

    Returns (y_indicator[N], s_pos[N], pos_label, n_graphs). ``s_pos`` is the raw
    (uncalibrated) probability of the positive class, derived the SAME way the
    diagnostics plots derive it (``prediction_probabilities(...)[:, pos_label]``) — for
    this 2-logit log_softmax head that is ``exp(log_softmax)``, a proper [0, 1]
    probability. The previous ``_positive_class_scores`` returned the raw log-prob, the
    wrong scale to fit Platt on.
    """
    # DiagnosticPlotter only stores paths + device in __init__; instantiate it purely to
    # reuse its verified model-rebuild + prediction-collection helpers on a single run.
    plotter = DiagnosticPlotter(
        results_dir=config_path.parent.parent,
        output_dir=config_path.parent,
        experiment_name=config_path.parent.name,
        device=device,
    )
    model, datamodule, get_pos_label, _, _ = (
        plotter._load_model_and_loaders(config_path)
    )

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    incompatible = model.load_state_dict(state_dict, strict=False)
    # strict=False silently tolerates a total key mismatch (loads nothing) -- guard
    # against calibrating a randomly-initialised model by requiring a real overlap.
    target_keys = set(model.state_dict().keys())
    missing = set(getattr(incompatible, "missing_keys", []))
    if not (target_keys - missing):
        raise SystemExit(
            f"Checkpoint {ckpt_path.name} shares no parameter keys with the rebuilt "
            "model -- wrong config/checkpoint pairing?"
        )
    if missing:
        print(f"  [warn] {len(missing)} model params not in checkpoint "
              f"(loaded {len(target_keys - missing)}/{len(target_keys)}).")
    model.to(device)

    split_index = {"val": 1, "test": 2}[split]
    loaders = datamodule.loaders
    if split_index >= len(loaders):
        raise SystemExit(
            f"Split '{split}' (loader index {split_index}) unavailable: run exposes "
            f"{len(loaders)} loader(s). Curated 'test' exists only in synthetic mode."
        )

    y_true, pred_score, pids = plotter._collect_predictions(
        model, loaders[split_index], split
    )
    if y_true is None:
        raise SystemExit(f"Split '{split}' produced no predictions (empty loader).")

    pos_label = (pos_label_override if pos_label_override is not None
                 else int(get_pos_label()))
    s_pos = np.asarray(_positive_class_probs(pred_score, pos_label), dtype=float).ravel()
    y_raw = y_true.numpy() if hasattr(y_true, "numpy") else y_true
    y_ind = (np.asarray(y_raw).astype(int) == pos_label).astype(int)
    n_graphs = len(set(pids)) if pids else None
    return y_ind, s_pos, pos_label, n_graphs


# --------------------------------------------------------------------------- #
# Platt scaling (logistic regression of labels on logits) -- fit in pure torch
# --------------------------------------------------------------------------- #
def fit_platt(s_pos, y_ind, max_iter: int = 200) -> tuple[float, float]:
    """Fit a, b in ``p_cal = sigmoid(a * logit(s_pos) + b)`` minimising BCE via LBFGS.

    Operates on the positive-class logit, so it is correct regardless of which class is
    positive: ``logit(P(pos)) == a * z + b``. a=1, b=0 recovers the uncalibrated output.
    """
    s = torch.tensor(s_pos, dtype=torch.float64).clamp(EPS, 1.0 - EPS)
    z = torch.log(s / (1.0 - s))  # recover the positive-class logit
    y = torch.tensor(y_ind, dtype=torch.float64)

    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=max_iter,
                                  line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(a * z + b, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(a.detach()), float(b.detach())


def apply_platt(s_pos, a: float, b: float) -> np.ndarray:
    s = np.clip(s_pos, EPS, 1.0 - EPS)
    z = np.log(s / (1.0 - s))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


# --------------------------------------------------------------------------- #
# Calibration quality + threshold metrics
# --------------------------------------------------------------------------- #
def calibration_metrics(probs_pos, y_ind) -> dict:
    """ECE / Brier / NLL of the positive-class probabilities (lower is better)."""
    p = np.clip(probs_pos, EPS, 1.0 - EPS)
    return {
        "ece": float(expected_calibration_error(probs_pos, y_ind.astype(float))),
        "brier": float(np.mean((probs_pos - y_ind) ** 2)),
        "nll": float(-np.mean(y_ind * np.log(p) + (1 - y_ind) * np.log(1.0 - p))),
    }


def select_threshold(scores, y_ind) -> tuple[float, float]:
    """Sweep candidate thresholds and return (best_threshold, best_f1).

    Predict positive iff ``score >= threshold``. Candidates are the unique scores (the
    only points where the confusion matrix changes), plus 0.0 (predict all positive).
    """
    candidates = np.unique(np.concatenate([[0.0], scores]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_ind, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def threshold_report(scores, y_ind, threshold) -> dict:
    """Classification metric suite at one operating point (positive-class framing)."""
    pred = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_ind, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_ind, pred)),
        "precision": float(precision_score(y_ind, pred, zero_division=0)),
        "recall": float(recall_score(y_ind, pred, zero_division=0)),
        "f1": float(f1_score(y_ind, pred, zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# Reliability diagram (uncalibrated vs calibrated)
# --------------------------------------------------------------------------- #
def plot_reliability(s_raw, p_cal, y_ind, pos_label, out_path: Path, n_bins: int = 10):
    import matplotlib.pyplot as plt

    def _bins(probs):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        xs, ys = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
            if np.any(mask):
                xs.append(float(probs[mask].mean()))
                ys.append(float(y_ind[mask].mean()))
        return xs, ys

    ece_raw = calibration_metrics(s_raw, y_ind)["ece"]
    ece_cal = calibration_metrics(p_cal, y_ind)["ece"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="perfect calibration")
    for probs, color, label in (
        (s_raw, "#E76F51", f"uncalibrated (ECE={ece_raw:.4f})"),
        (p_cal, "#2A9D8F", f"Platt (ECE={ece_cal:.4f})"),
    ):
        xs, ys = _bins(probs)
        ax.plot(xs, ys, marker="o", color=color, linewidth=1.6, label=label)
    name = CLASS_NAMES.get(pos_label, pos_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted P(class {pos_label} = {name})")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle("Reliability -- calibration set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Batch mode: calibrate every top-K leaderboard model in one experiment folder
# --------------------------------------------------------------------------- #
def _split_loaders(model, datamodule):
    """Mirror DiagnosticPlotter.render_split_diagnostics' split selection so the
    calibrated plots cover exactly the same splits as the uncalibrated diagnostics."""
    loaders = datamodule.loaders
    synthetic = bool(getattr(model.cfg.expression_graph, "synthetic", False))
    if synthetic and len(loaders) >= 3:
        return [
            ("val", loaders[1], "Validation Synthetic"),
            ("test", loaders[2], "Validation Curated"),
        ]
    if len(loaders) >= 2:
        return [("val", loaders[1], "Validation")]
    return []


def discover_top_models(experiment_dir: Path, top_k: int, plotter: DiagnosticPlotter):
    """Top-K runs by val-synthetic ROC-AUC, resolved to (rank, slug, config, ckpt).

    Single source of truth: ``agg/val_bestepoch.csv`` (the same leaderboard
    DiagnosticPlotter.run_top_configs ranks on). No fallback chain.
    """
    csv_path = experiment_dir / "agg" / "val_bestepoch.csv"
    if not csv_path.exists():
        raise SystemExit(f"No leaderboard at {csv_path} (run post_eval first).")
    ranked = pd.read_csv(csv_path)
    if "auc" not in ranked.columns:
        raise SystemExit(f"{csv_path} has no 'auc' column to rank on.")
    ranked = ranked.sort_values("auc", ascending=False).head(top_k)
    config_cols = [c for c in CONFIG_COLS if c in ranked.columns]

    models = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        slug = _config_slug(row, config_cols)
        run_dir = plotter._resolve_run_dir(row, config_cols)
        if run_dir is None:
            print(f"  [{rank}] {slug}: run dir not found — skipped")
            continue
        config_path = _find_config_for_run(run_dir, None)
        ckpt_path = _find_best_checkpoint(run_dir)
        if config_path is None or ckpt_path is None:
            print(f"  [{rank}] {slug}: config/checkpoint missing — skipped")
            continue
        models.append((rank, slug, config_path, ckpt_path))
    return models


def _load_checkpoint_into(model, ckpt_path, device):
    """Load a .pth/.ckpt pickle into ``model`` and guard against a total key mismatch."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    incompatible = model.load_state_dict(state_dict, strict=False)
    target_keys = set(model.state_dict().keys())
    missing = set(getattr(incompatible, "missing_keys", []))
    if not (target_keys - missing):
        raise RuntimeError("checkpoint shares no parameter keys with the rebuilt model")
    model.to(device)


def calibrate_model(plotter, config_path, ckpt_path, out_dir, args, device, rank, slug):
    """Fit Platt + threshold on the selection split, then re-render the confusion /
    ROC / PR / reliability diagnostics for every split with the calibration applied.

    Writes the plots + a per-model calibration.json into ``out_dir`` and returns a flat
    summary row for the batch CSV.
    """
    model, datamodule, get_pos_label, _, _ = plotter._load_model_and_loaders(config_path)
    _load_checkpoint_into(model, ckpt_path, device)

    splits = _split_loaders(model, datamodule)
    if not splits:
        raise RuntimeError("no usable validation split")

    pos_label = (args.pos_label if args.pos_label is not None else int(get_pos_label()))
    data_cfg = getattr(model.cfg, "data", None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Fit Platt + threshold on the selection split (default: synthetic val) ----
    fit_entry = next((e for e in splits if e[0] == args.split), splits[0])
    fit_key, fit_loader, _ = fit_entry
    y_true_f, y_score_f, _ = plotter._collect_predictions(model, fit_loader, fit_key)
    if y_true_f is None:
        raise RuntimeError(f"fit split '{args.split}' produced no predictions")
    probs_f = _positive_class_probs(y_score_f, pos_label)
    y_ind_f = (np.asarray(y_true_f).astype(int) == pos_label).astype(int)

    cal_before = calibration_metrics(probs_f, y_ind_f)
    if args.calibration == "platt":
        a, b = fit_platt(probs_f, y_ind_f)
        probs_f_cal = apply_platt(probs_f, a, b)
    else:
        a, b, probs_f_cal = 1.0, 0.0, probs_f  # identity transform
    cal_after = calibration_metrics(probs_f_cal, y_ind_f)

    if args.threshold is not None:
        threshold, thr_source = float(args.threshold), "user-supplied"
    else:
        threshold, best_f1 = select_threshold(probs_f_cal, y_ind_f)
        thr_source = f"swept (max f1={best_f1:.4f})"
    thr_scale = "calibrated" if args.calibration == "platt" else "raw"

    if args.calibration == "platt" and not args.no_plot:
        plot_reliability(probs_f, probs_f_cal, y_ind_f, pos_label,
                         out_dir / "reliability_comparison.png")

    # --- 2. Re-render the diagnostics for every split, calibration applied ----------
    per_split = {}
    for split_key, loader, split_title in splits:
        y_true, y_score, pids = plotter._collect_predictions(model, loader, split_key)
        if y_true is None:
            continue
        probs_raw = _positive_class_probs(y_score, pos_label)
        probs_cal = apply_platt(probs_raw, a, b) if args.calibration == "platt" else probs_raw
        y_pred = (probs_cal >= threshold).astype(int)
        y_ind = (np.asarray(y_true).astype(int) == pos_label).astype(int)
        n_graphs = len(set(pids)) if pids else None
        dataset_label = _dataset_label_for_split(split_title, data_cfg)
        provenance = (
            _provenance_subtitle(y_true, pos_label, dataset_label, n_graphs)
            + f"  ·  Platt a={a:.2f} b={b:.2f}  ·  thr={threshold:.2f} ({thr_scale})"
        )
        prefix = split_title.lower().replace(" ", "_")

        if not args.no_plot:
            plotter._plot_confusion_matrix(
                np.asarray(y_true).astype(int), y_pred,
                f"Confusion Matrix (calibrated) — {split_title}",
                out_dir / f"confusion_{prefix}.png", pos_label, provenance,
            )
            plotter._plot_roc_curve(
                y_true, y_score, f"ROC Curve (calibrated) — {split_title}",
                out_dir / f"roc_{prefix}.png", pos_label, provenance, probs_pos=probs_cal,
            )
            plotter._plot_pr_curve(
                y_true, y_score, f"PR Curve (calibrated) — {split_title}",
                out_dir / f"pr_{prefix}.png", pos_label, provenance, probs_pos=probs_cal,
            )
            plotter._plot_reliability(
                y_true, y_score, f"Reliability (calibrated) — {split_title}",
                out_dir / f"reliability_{prefix}.png", pos_label,
                provenance=provenance, probs_pos=probs_cal,
            )
        save_predictions(
            out_dir / f"predictions_{prefix}.npz",
            np.asarray(y_true).astype(int), probs_cal, pos_label=pos_label,
        )
        per_split[prefix] = {
            "n_samples": int(len(y_ind)),
            "n_graphs": n_graphs,
            "calibration": calibration_metrics(probs_cal, y_ind),
            "metrics_at_threshold": threshold_report(probs_cal, y_ind, threshold),
            "metrics_at_0.5": threshold_report(probs_cal, y_ind, 0.5),
        }

    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "rank": rank,
        "slug": slug,
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "fit_split": fit_key,
        "pos_label": pos_label,
        "pos_class_name": CLASS_NAMES.get(pos_label, str(pos_label)),
        "calibration": {
            "method": args.calibration,
            "platt_a": a,
            "platt_b": b,
            "before": cal_before,
            "after": cal_after,
        },
        "threshold": {"value": threshold, "scale": thr_scale, "source": thr_source},
        "decision_rule": (
            f"predict class {pos_label} iff "
            f"{'sigmoid(a*logit(P_pos)+b)' if thr_scale == 'calibrated' else 'P_pos'} "
            f">= {threshold:.6f}"
        ),
        "splits": per_split,
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")

    val_split = per_split.get("validation_synthetic") or per_split.get("validation") or {}
    return {
        "rank": rank,
        "slug": slug,
        "platt_a": a,
        "platt_b": b,
        "threshold": threshold,
        "threshold_scale": thr_scale,
        "val_ece_before": cal_before["ece"],
        "val_ece_after": cal_after["ece"],
        "val_f1_at_0.5": val_split.get("metrics_at_0.5", {}).get("f1"),
        "val_f1_at_thr": val_split.get("metrics_at_threshold", {}).get("f1"),
        "out_dir": str(out_dir),
    }


def _write_batch_summary(top_root: Path, summaries: list[dict]):
    if not summaries:
        print("\nNo models calibrated; nothing to summarise.")
        return
    top_root.mkdir(parents=True, exist_ok=True)
    csv_path = top_root / "calibration_summary.csv"
    json_path = top_root / "calibration_summary.json"
    pd.DataFrame(summaries).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nWrote {csv_path}\nWrote {json_path}")


def run_batch(args, device):
    """Calibrate the top-K leaderboard models of one experiment folder, dropping the
    calibrated diagnostics next to the existing ones (top_configs/<model>/calibrated/)."""
    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"--experiment-dir is not a directory: {experiment_dir}")
    top_root = experiment_dir / "eval_plots" / "top_configs"

    plotter = DiagnosticPlotter(
        results_dir=experiment_dir,
        output_dir=experiment_dir / "eval_plots",
        experiment_name=experiment_dir.name,
        device=device,
    )
    models = discover_top_models(experiment_dir, args.top_k, plotter)
    if not models:
        raise SystemExit("No top-K models could be resolved for calibration.")

    print(f"Experiment : {experiment_dir}")
    print(f"Models     : {len(models)} (top-{args.top_k} by val-synthetic ROC-AUC)")
    print(f"Fit split  : {args.split}   Calibration: {args.calibration}   Device: {device}\n")

    summaries = []
    for rank, slug, config_path, ckpt_path in models:
        out_dir = top_root / f"rank_{rank}_{slug}" / "calibrated"
        print(f"[{rank}] {slug}\n    ckpt: {ckpt_path.name} -> {out_dir}")
        try:
            summary = calibrate_model(
                plotter, config_path, ckpt_path, out_dir, args, device, rank, slug
            )
            summaries.append(summary)
            print(
                f"    a={summary['platt_a']:.3f} b={summary['platt_b']:.3f} "
                f"thr={summary['threshold']:.3f}  ·  val ECE "
                f"{summary['val_ece_before']:.4f} -> {summary['val_ece_after']:.4f}"
            )
        except Exception as exc:  # one bad run must not abort the whole batch
            print(f"    [warn] calibration failed: {exc}")
    _write_batch_summary(top_root, summaries)
    return summaries


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fit Platt calibration + an F1-optimal decision threshold for a "
                    "trained solver classifier on its validation split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("model inputs")
    src.add_argument("--run-dir", help="Trained run dir (auto-finds config + ckpt).")
    src.add_argument("--config", help="Explicit config.yaml (overrides --run-dir).")
    src.add_argument("--checkpoint", help="Explicit pickle: .pth state_dict or .ckpt.")
    src.add_argument(
        "--experiment-dir",
        help="Batch mode: stage/experiment folder (run_results/<exp>). Calibrates the "
             "top-K leaderboard models and re-renders their diagnostics, calibrated, "
             "into eval_plots/top_configs/<model>/calibrated/.",
    )
    src.add_argument(
        "--top-k", type=int, default=5,
        help="Batch mode: number of top-AUC leaderboard models to calibrate.",
    )

    cal = p.add_argument_group("calibration & threshold")
    cal.add_argument("--split", choices=("val", "test"), default="val",
                     help="val = synthetic holdout (selection set); test = curated.")
    cal.add_argument("--calibration", choices=("platt", "none"), default="platt",
                     help="Fit Platt scaling on the split, or skip calibration.")
    cal.add_argument("--threshold", type=float, default=None,
                     help="Use this threshold directly (skips the F1 sweep).")
    cal.add_argument("--threshold-scale", choices=("raw", "calibrated"), default="raw",
                     help="Scale the threshold uses: raw output or Platt prob.")
    cal.add_argument("--pos-label", type=int, choices=(0, 1), default=None,
                     help="Positive class (default: training minority).")

    out = p.add_argument_group("output")
    out.add_argument("--output", help="Output dir (default: <run-dir>/calibration/).")
    out.add_argument("--no-plot", action="store_true", help="Skip reliability plot.")
    out.add_argument("--device", default=None, help="cpu|cuda (default: auto).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment_dir:
        return run_batch(args, device)

    config_path, ckpt_path = resolve_inputs(args)
    out_dir = Path(args.output) if args.output else (
        Path(args.run_dir) / "calibration" if args.run_dir else Path.cwd()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config     : {config_path}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Split      : {args.split}   Device: {device}")

    y_ind, s_pos, pos_label, n_graphs = collect_scores(
        config_path, ckpt_path, args.split, device, args.pos_label
    )
    n = len(y_ind)
    prevalence = float(y_ind.mean())
    name = CLASS_NAMES.get(pos_label, pos_label)
    print(f"Positive   : class {pos_label} ({name})  ·  "
          f"N={n} rows" + (f" / {n_graphs} graphs" if n_graphs else "")
          + f"  ·  prevalence={prevalence:.4f}")

    # --- 1. Calibration -----------------------------------------------------
    cal_before = calibration_metrics(s_pos, y_ind)
    if args.calibration == "platt":
        a, b = fit_platt(s_pos, y_ind)
        p_cal = apply_platt(s_pos, a, b)
        cal_after = calibration_metrics(p_cal, y_ind)
        print(f"\nPlatt fit  : a={a:.4f}  b={b:.4f}")
        for key in ("ece", "brier", "nll"):
            print(f"  {key.upper():<5} {cal_before[key]:.4f} -> {cal_after[key]:.4f}")
    else:
        a, b, p_cal, cal_after = None, None, s_pos, cal_before
        print("\nCalibration: skipped (--calibration none)")

    # The threshold is evaluated on whichever probability scale the user selected. With
    # 'raw' (default) keeps it on the current model output -- the same scale
    # cfg.model.thresh uses and the one any externally-known optimum was measured on.
    if args.threshold_scale == "calibrated" and args.calibration == "none":
        raise SystemExit("--threshold-scale calibrated requires --calibration platt.")
    scores = p_cal if args.threshold_scale == "calibrated" else s_pos

    # --- 2. Threshold -------------------------------------------------------
    if args.threshold is not None:
        threshold, source = float(args.threshold), "user-supplied"
    else:
        threshold, best_f1 = select_threshold(scores, y_ind)
        source = f"swept (max f1={best_f1:.4f})"
    print(f"\nThreshold  : {threshold:.4f} on {args.threshold_scale} scale  [{source}]")

    at_new = threshold_report(scores, y_ind, threshold)
    at_half = threshold_report(scores, y_ind, 0.5)
    print(f"  {'metric':<19}{'@0.50':>10}{'@%.3f' % threshold:>12}")
    for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        print(f"  {k:<19}{at_half[k]:>10.4f}{at_new[k]:>12.4f}")

    # --- 3. Persist ---------------------------------------------------------
    rule_score = ("sigmoid(a*logit(P_pos)+b)"
                  if args.threshold_scale == "calibrated" else "P_pos")
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_samples": n,
        "n_graphs": n_graphs,
        "pos_label": pos_label,
        "pos_class_name": CLASS_NAMES.get(pos_label, str(pos_label)),
        "prevalence": prevalence,
        "calibration": {
            "method": args.calibration,
            "platt_a": a,
            "platt_b": b,
            "before": cal_before,
            "after": cal_after,
        },
        "threshold": {
            "value": threshold,
            "scale": args.threshold_scale,
            "source": source,
            "optimize": None if args.threshold is not None else "f1",
            "metrics_at_threshold": at_new,
            "metrics_at_0.5": at_half,
        },
        "decision_rule": (
            f"predict class {pos_label} iff {rule_score} >= {threshold:.6f}"
        ),
    }
    json_path = out_dir / "calibration.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {json_path}")

    if not args.no_plot and args.calibration == "platt":
        plot_path = out_dir / "reliability.png"
        plot_reliability(s_pos, p_cal, y_ind, pos_label, plot_path)
        print(f"Wrote {plot_path}")

    return result


if __name__ == "__main__":
    main()
