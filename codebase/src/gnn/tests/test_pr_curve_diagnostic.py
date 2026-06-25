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
