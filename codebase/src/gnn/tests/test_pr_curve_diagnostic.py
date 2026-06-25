"""The per-model PR curve is a threshold-selection diagnostic.

PR-AUC is demoted from the leaderboard/comparison panels (recorded-only), but the
precision-recall curve itself is still plotted per model so the decision threshold
can be read off it — it marks the default 0.5 operating point and the F1-optimal
threshold.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from gnn.supervised_learning.run_results.diagnostics import DiagnosticPlotter


def test_diagnostic_plotter_exposes_pr_curve():
    assert hasattr(DiagnosticPlotter, "_plot_pr_curve")


def test_pr_curve_renders_a_figure(tmp_path):
    rng = np.random.RandomState(0)
    y = np.concatenate([np.ones(60), np.zeros(240)]).astype(int)  # imbalanced
    p1 = np.clip(rng.normal(np.where(y == 1, 0.55, 0.35), 0.12), 1e-4, 1 - 1e-4)
    y_score = torch.tensor(np.log(np.stack([1 - p1, p1], 1) + 1e-9), dtype=torch.float)

    # _plot_pr_curve only needs the model-free args; bypass __init__.
    plotter = DiagnosticPlotter.__new__(DiagnosticPlotter)
    out = tmp_path / "pr_val.png"
    DiagnosticPlotter._plot_pr_curve(
        plotter, torch.tensor(y), y_score, "PR Curve — test", out, pos_label=1
    )
    assert out.exists() and out.stat().st_size > 0


def test_diagnostics_loader_does_not_call_broken_setup():
    """Regression: GraphGymDataModule builds self.loaders in __init__; the inherited
    LightningDataModule.setup(stage) requires a positional `stage`, so calling
    datamodule.setup() raised "missing 1 required positional argument: 'stage'" and
    crashed every top_configs diagnostics run (no ROC/PR curves ever produced)."""
    import inspect

    from gnn.supervised_learning.run_results.diagnostics import DiagnosticPlotter

    src = inspect.getsource(DiagnosticPlotter._load_model_and_loaders)
    # Strip comments so the explanatory note mentioning setup() doesn't trip the check.
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert ".setup()" not in code


def test_render_split_diagnostics_is_a_reuse_point():
    # main_graphgym's single-run path and run_top_configs both call this method, so it
    # must stay a public method on the plotter (not inlined back into the grid loop).
    assert hasattr(DiagnosticPlotter, "render_split_diagnostics")


def test_run_top_configs_never_silently_empty(tmp_path):
    """When every top-K config is unresolvable, run_top_configs must still create
    top_configs/ and write SKIPPED.txt with a reason — never leave no folder at all
    (the original failure mode that hid diagnostics breakage)."""
    import pandas as pd

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    leaderboard = tmp_path / "val_bestepoch.csv"
    # Rows whose run dirs do not exist on disk → all resolve to None and get skipped.
    pd.DataFrame(
        {"run_name": ["grid-missing-a", "grid-missing-b"], "layer_type": ["gcnconv", "ginconv"], "auc": [0.91, 0.88]}
    ).to_csv(leaderboard, index=False)

    plotter = DiagnosticPlotter.__new__(DiagnosticPlotter)
    plotter.results_dir = results_dir
    plotter.output_dir = tmp_path / "eval_plots"
    plotter.configs_dir = None
    plotter.top_k = 5

    plotter.run_top_configs(leaderboard_csv=leaderboard)

    top_root = plotter.output_dir / "top_configs"
    assert top_root.is_dir(), "top_configs/ must exist even when all configs are skipped"
    skipped = top_root / "SKIPPED.txt"
    assert skipped.exists(), "a SKIPPED.txt must explain why no rank_* folders were produced"
    assert "run directory not found" in skipped.read_text()
