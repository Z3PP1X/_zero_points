"""Auto-generated experiment summary (``summary.md`` + ``summary.json``).

One glance answer to "how did this grid search go?": the best config by val
synthetic PR-AUC, a top-K leaderboard, the synthetic→curated generalization gap
per architecture, calibration, how far the best model sits above a no-skill
baseline, and (when prediction dumps exist) a bootstrap CI / paired test. Pure
pandas over the aggregated CSVs and ``leaderboard.csv`` — no model reload.

Split semantics (do not mislabel): ``val_bestepoch`` = validation **synthetic**
(model selection), ``test_bestepoch`` = validation **curated** (generalization),
``train_bestepoch`` = training (synthetic).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gnn.supervised_learning.run_results import baselines as baselines_mod
from gnn.supervised_learning.run_results import significance as sig_mod

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
    "epoch", "loss", "accuracy", "precision", "recall", "f1", "auc", "pr_auc",
    "mean_confidence", "mean_margin", "mean_entropy", "brier_score", "ece",
    "lr", "base_lr", "params", "time_iter", "gpu_memory",
}
# ece excluded from leaderboard: requires post-training calibration; brier_score
# kept as the single uncalibrated signal alongside mean_confidence.
LEADERBOARD_METRICS = (
    "pr_auc", "auc", "f1", "recall", "precision", "accuracy", "loss",
    "mean_confidence", "brier_score",
)
CALIBRATION_METRICS = ("ece", "brier_score", "mean_confidence", "mean_entropy")

SPLIT_STEMS = {
    "train": "train_bestepoch",
    "val_synthetic": "val_bestepoch",
    "val_curated": "test_bestepoch",
}
SPLIT_LABELS = {
    "train": "Train (Synthetic)",
    "val_synthetic": "Validation Synthetic",
    "val_curated": "Validation Curated",
}

# PNGs the report links to, relative to eval_plots/ where summary.md is written.
LINKED_PLOTS = {
    "Leaderboard": "leaderboard.png",
    "Split comparison": "split_comparison.png",
    "Generalization gap": "generalization_gap.png",
    "Pareto front": "pareto.png",
}


def _read_csv(agg_dir: Path, stem: str) -> pd.DataFrame | None:
    path = agg_dir / f"{stem}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    return df if not df.empty else None


def _config_columns(df: pd.DataFrame) -> list[str]:
    skip = set(METRIC_COLS)
    skip.add("run_name")  # run identity column, not a hyperparameter
    skip.update(c for c in df.columns if c.endswith("_std"))
    known = [c for c in CONFIG_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in skip and c not in known]
    return known + extra


def _config_label(row: pd.Series, config_cols: list[str]) -> str:
    parts = [
        f"{c}={row[c]}"
        for c in config_cols
        if c in row.index and pd.notna(row[c])
    ]
    return ", ".join(parts) if parts else "config"


def _to_float(value) -> float | None:
    """Coerce a possibly-stringy scalar to float, or None when not numeric."""
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(coerced) else float(coerced)


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Hand-rolled GitHub-flavoured markdown table (avoids a tabulate dep)."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body]) if rows else head + "\n" + sep


def _best_config(val_df: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    config_cols = _config_columns(val_df)
    if "pr_auc" not in val_df.columns:
        return None, config_cols
    numeric = pd.to_numeric(val_df["pr_auc"], errors="coerce")
    if numeric.dropna().empty:
        return None, config_cols
    return val_df.loc[numeric.idxmax()], config_cols


def _leaderboard_section(agg_dir: Path, eval_dir: Path, top_k: int) -> tuple[str, list]:
    csv_path = eval_dir / "leaderboard.csv"
    df = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            df = None
    if df is None:
        df = _read_csv(agg_dir, "val_bestepoch")
        if df is not None and "pr_auc" in df.columns:
            df = df.sort_values("pr_auc", ascending=False).head(top_k)
    if df is None or df.empty:
        return "_No leaderboard available._", []

    config_cols = [c for c in _config_columns(df) if c in df.columns]
    metric_cols = [m for m in LEADERBOARD_METRICS if m in df.columns]
    headers = ["#"] + config_cols + [m.upper() for m in metric_cols]
    rows, json_rows = [], []
    for rank, (_, row) in enumerate(df.head(top_k).iterrows(), start=1):
        cells = [str(rank)]
        record: dict = {"rank": rank}
        for c in config_cols:
            cells.append(_fmt(row.get(c)))
            record[c] = None if pd.isna(row.get(c)) else row.get(c)
        for m in metric_cols:
            val = pd.to_numeric(pd.Series([row.get(m)]), errors="coerce").iloc[0]
            cells.append(_fmt(float(val) if pd.notna(val) else None))
            record[m] = None if pd.isna(val) else float(val)
        rows.append(cells)
        json_rows.append(record)
    return _md_table(headers, rows), json_rows


def _generalization_section(agg_dir: Path) -> tuple[str, list]:
    val = _read_csv(agg_dir, "val_bestepoch")
    cur = _read_csv(agg_dir, "test_bestepoch")
    if val is None or cur is None or "pr_auc" not in val.columns:
        return "_Generalization gap unavailable (missing splits)._", []
    if "layer_type" not in val.columns:
        return "_Generalization gap unavailable (no layer_type column)._", []

    val_g = val.groupby("layer_type")["pr_auc"].mean()
    cur_g = (
        cur.groupby("layer_type")["pr_auc"].mean()
        if "pr_auc" in cur.columns and "layer_type" in cur.columns
        else pd.Series(dtype=float)
    )
    rows, records = [], []
    for arch in val_g.index:
        syn = float(val_g.loc[arch])
        curated = float(cur_g.loc[arch]) if arch in cur_g.index else float("nan")
        delta = curated - syn if pd.notna(curated) else float("nan")
        rows.append([
            str(arch), _fmt(syn), _fmt(curated), _fmt(delta),
        ])
        records.append({
            "layer_type": str(arch),
            "val_synthetic_pr_auc": syn,
            "val_curated_pr_auc": None if pd.isna(curated) else curated,
            "delta": None if pd.isna(delta) else delta,
        })
    headers = ["Architecture", "Val Synth PR-AUC", "Val Curated PR-AUC", "Δ"]
    return _md_table(headers, rows), records


def _calibration_section(agg_dir: Path) -> tuple[str, dict]:
    records: dict = {}
    rows = []
    for split_key, stem in SPLIT_STEMS.items():
        df = _read_csv(agg_dir, stem)
        if df is None:
            continue
        present = [m for m in CALIBRATION_METRICS if m in df.columns]
        if not present:
            continue
        means = {
            m: float(pd.to_numeric(df[m], errors="coerce").mean()) for m in present
        }
        records[split_key] = means
        row = [SPLIT_LABELS[split_key]]
        row += [_fmt(means.get(m)) for m in CALIBRATION_METRICS]
        rows.append(row)
    if not rows:
        return "_No calibration metrics logged (ece / brier_score absent)._", {}
    headers = ["Split"] + [m.upper() for m in CALIBRATION_METRICS]
    return _md_table(headers, rows), records


def _baseline_section(
    agg_dir: Path, best_row: pd.Series | None
) -> tuple[str, dict]:
    base = baselines_mod.compute_baselines(agg_dir)
    if not base:
        return "_No baselines (class_balance.json absent)._", {}

    rows = []
    best_pr = None
    if best_row is not None and "pr_auc" in best_row.index:
        best_pr = _to_float(best_row["pr_auc"])

    for stem, info in base.items():
        label = SPLIT_LABELS.get(
            next((k for k, v in SPLIT_STEMS.items() if v == stem), stem), stem
        )
        no_skill = info.get("no_skill_pr_auc")
        maj = info.get("majority", {})
        delta = None
        if best_pr is not None and stem == "val_bestepoch":
            delta = best_pr - no_skill
        rows.append([
            label,
            _fmt(no_skill),
            _fmt(maj.get("accuracy")),
            _fmt(maj.get("f1")),
            _fmt(delta) if delta is not None else "—",
        ])
    headers = [
        "Split", "No-skill PR-AUC", "Majority Acc", "Majority F1",
        "Best Δ vs no-skill",
    ]
    return _md_table(headers, rows), base


def _significance_section(eval_dir: Path) -> tuple[str, dict]:
    dumps = sig_mod.discover_prediction_dumps(eval_dir, "validation_synthetic")
    if not dumps:
        return (
            "_No prediction dumps found — run diagnostics with checkpoints "
            "present to enable bootstrap significance._",
            {},
        )

    lines, record = [], {}
    best_name, best_path = dumps[0]
    best = sig_mod.load_predictions(best_path)
    if best is None:
        return "_Prediction dump unreadable._", {}

    ci = sig_mod.bootstrap_metric_ci(
        best["y_true"], best["probs_pos"], metric="pr_auc"
    )
    lines.append(
        f"- **{best_name}** PR-AUC = {ci['point']:.4f} "
        f"(95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}], n={ci['n']}, "
        f"{ci['n_boot']} bootstraps)"
    )
    record["best_ci"] = {"config": best_name, **ci}

    if len(dumps) >= 2:
        second_name, second_path = dumps[1]
        second = sig_mod.load_predictions(second_path)
        if second is not None:
            n = min(len(best["probs_pos"]), len(second["probs_pos"]))
            diff = sig_mod.paired_bootstrap_diff(
                best["y_true"][:n],
                best["probs_pos"][:n],
                second["probs_pos"][:n],
                metric="pr_auc",
            )
            verdict = "significant" if diff["significant"] else "not significant"
            lines.append(
                f"- **{best_name} vs {second_name}**: ΔPR-AUC = {diff['diff']:.4f} "
                f"(95% CI [{diff['lo']:.4f}, {diff['hi']:.4f}], "
                f"p={diff['p_value']:.3f}, {verdict})"
            )
            record["paired_diff"] = {
                "config_a": best_name, "config_b": second_name, **diff,
            }
    return "\n".join(lines), record


def _provenance_section(results_dir: Path) -> tuple[str, dict]:
    """Pull reproducibility provenance from run_manifest.json (written by run_all.py).

    Surfaces in the eval artifact the context the leaderboard alone hides: seed, git
    commit, dataset/feature schema, and key fixed training knobs. Without the manifest
    (older experiments) the section degrades to a note rather than failing.
    """
    manifest_path = results_dir / "run_manifest.json"
    if not manifest_path.exists():
        return (
            "_No run_manifest.json — experiment predates manifest emission or was not "
            "launched via run_all.py; full config/seed/git provenance unavailable._",
            {},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "_run_manifest.json present but unreadable._", {}

    base = manifest.get("base_config", {}) or {}
    expr = base.get("expression_graph", {}) or {}
    dataset = base.get("dataset", {}) or {}
    train = base.get("train", {}) or {}
    git = manifest.get("git", {}) or {}

    record = {
        "manifest": "../run_manifest.json",
        "seed": manifest.get("seed"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "dataset": dataset.get("name"),
        "synthetic_dataset": expr.get("synthetic_dataset"),
        "mode": expr.get("mode"),
        "edge_direction": expr.get("edge_direction"),
        "add_kappa": expr.get("add_kappa"),
        "add_virtual_supernode": expr.get("add_virtual_supernode"),
        "features": expr.get("features"),
        "active_features": expr.get("active_features"),
        "epochs": train.get("epochs"),
        "batch_size": train.get("batch_size"),
        "versions": manifest.get("versions"),
    }

    commit = git.get("commit")
    commit_str = (commit[:10] if isinstance(commit, str) else "—") + (
        " (dirty)" if git.get("dirty") else ""
    )
    rows = [
        ["Seed", _fmt(record["seed"])],
        ["Git commit", commit_str],
        ["Dataset", _fmt(record["dataset"])],
        ["Synthetic dataset", _fmt(record["synthetic_dataset"])],
        ["Graph mode", _fmt(record["mode"])],
        ["Edge direction", _fmt(record["edge_direction"])],
        ["add_kappa", _fmt(record["add_kappa"])],
        ["add_virtual_supernode", _fmt(record["add_virtual_supernode"])],
        ["Epochs", _fmt(record["epochs"])],
        ["Batch size", _fmt(record["batch_size"])],
    ]
    md = _md_table(["Field", "Value"], rows)
    md += "\n\nFull resolved config, grid, and package versions: [`run_manifest.json`](../run_manifest.json)."
    return md, record


def _plots_section(eval_dir: Path) -> tuple[str, dict]:
    rows, record = [], {}
    for label, fname in LINKED_PLOTS.items():
        if (eval_dir / fname).exists():
            rows.append(f"- [{label}]({fname})")
            record[label] = fname
    if not rows:
        return "_No plots found._", {}
    return "\n".join(rows), record


def generate_report(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    top_k: int = 10,
) -> Path:
    """Build ``summary.md`` + ``summary.json`` for one experiment.

    ``results_dir`` must contain ``agg/`` and ``eval_plots/``. Output defaults to
    ``<results_dir>/eval_plots``. Returns the path to ``summary.md``.
    """
    results_dir = Path(results_dir).resolve()
    experiment = results_dir.name
    agg_dir = results_dir / "agg"
    eval_dir = Path(output_dir).resolve() if output_dir else results_dir / "eval_plots"
    eval_dir.mkdir(parents=True, exist_ok=True)

    val_df = _read_csv(agg_dir, "val_bestepoch")
    best_row, config_cols = (None, [])
    if val_df is not None:
        best_row, config_cols = _best_config(val_df)

    summary: dict = {"experiment": experiment}

    if best_row is not None:
        best_pr = _to_float(best_row.get("pr_auc"))
        best_label = _config_label(best_row, config_cols)
        summary["best_config"] = {
            "label": best_label,
            "config": {
                c: (None if pd.isna(best_row.get(c)) else best_row.get(c))
                for c in config_cols
            },
            "val_synthetic_pr_auc": best_pr,
        }
        n_configs = len(val_df) if val_df is not None else 0
        best_block = (
            f"**Best config (Val Synthetic PR-AUC):** `{best_label}`  \n"
            f"**Val Synthetic PR-AUC:** {_fmt(best_pr)}  \n"
            f"**Configurations evaluated:** {n_configs}"
        )
    else:
        best_block = "_No val_bestepoch.csv found — cannot pick a best config._"

    leaderboard_md, leaderboard_json = _leaderboard_section(agg_dir, eval_dir, top_k)
    gap_md, gap_json = _generalization_section(agg_dir)
    calib_md, calib_json = _calibration_section(agg_dir)
    baseline_md, baseline_json = _baseline_section(agg_dir, best_row)
    sig_md, sig_json = _significance_section(eval_dir)
    provenance_md, provenance_json = _provenance_section(results_dir)
    plots_md, plots_json = _plots_section(eval_dir)

    summary["provenance"] = provenance_json
    summary["leaderboard"] = leaderboard_json
    summary["generalization_gap"] = gap_json
    summary["calibration"] = calib_json
    summary["baselines"] = baseline_json
    summary["significance"] = sig_json
    summary["plots"] = plots_json

    md = f"""# Experiment summary — {experiment}

{best_block}

## Reproducibility provenance

{provenance_md}

## Top configurations (Val Synthetic PR-AUC)

{leaderboard_md}

## Generalization gap (synthetic → curated)

Δ is curated minus synthetic PR-AUC per architecture; large negative Δ means the
architecture overfits the synthetic holdout.

{gap_md}

## Calibration

{calib_md}

## Baselines

{baseline_md}

## Significance (single-set bootstrap)

{sig_md}

## Plots

{plots_md}
"""

    md_path = eval_dir / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    with open(eval_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"    Saved summary report: {md_path}")
    return md_path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment summary report.")
    parser.add_argument("results_dir", help="Experiment folder (name or path)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    from gnn.supervised_learning.run_results.post_eval import resolve_results_dir

    results_dir = resolve_results_dir(args.results_dir)
    generate_report(results_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
