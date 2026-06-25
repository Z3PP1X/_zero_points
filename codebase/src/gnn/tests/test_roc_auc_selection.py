"""Lock in the PR-AUC -> ROC-AUC evaluation switch.

The leaderboard, best-epoch aggregation and best-config selection rank by the
ROC-AUC column ('auc'). PR-AUC ('pr_auc') is kept only as a *recorded* scalar
(in the agg CSVs and leaderboard/summary tables, like dirichlet_energy) and is
no longer drawn as a headline curve/heatmap/bar panel.
"""

import pandas as pd

from gnn.supervised_learning.aggregate_graphgym import (
    AGG_METRIC_COLUMNS,
    BEST_METRIC,
    _resolve_best_metric,
)
from gnn.supervised_learning.run_results.eval import GNNResultEvaluator
from gnn.supervised_learning.run_results.report import LEADERBOARD_METRICS, _best_config
from gnn.supervised_learning.run_results.training_curves import CURVE_METRICS


def test_best_metric_is_roc_auc():
    assert BEST_METRIC == "auc"


def test_resolve_best_metric_prefers_roc_auc():
    # 'auto' must prefer the ROC-AUC column over pr_auc when both are present.
    stats = [{"auc": 0.8, "pr_auc": 0.7, "accuracy": 0.6}]
    assert _resolve_best_metric("auto", stats) == "auc"
    assert _resolve_best_metric("auc", stats) == "auc"


def test_pr_auc_demoted_from_plot_panels_but_still_recorded():
    # Not plotted as a headline panel anymore...
    for plotted in (CURVE_METRICS, GNNResultEvaluator.HEATMAP_METRICS,
                    GNNResultEvaluator.BOUNDED_METRICS):
        assert "auc" in plotted
        assert "pr_auc" not in plotted
    # ...but still recorded in the aggregated CSV columns.
    assert "pr_auc" in AGG_METRIC_COLUMNS
    assert "pr_auc_std" in AGG_METRIC_COLUMNS
    assert "auc" in AGG_METRIC_COLUMNS


def test_leaderboard_metrics_lead_with_roc_auc_and_retain_pr_auc():
    # ROC-AUC leads the ranking columns; PR-AUC is retained as a recorded column.
    assert LEADERBOARD_METRICS[0] == "auc"
    assert "pr_auc" in LEADERBOARD_METRICS


def test_best_config_ranks_by_roc_auc_not_pr_auc():
    # Row with the higher ROC-AUC must win even when another row has higher PR-AUC.
    df = pd.DataFrame(
        [
            {"layers_mp": 2, "auc": 0.70, "pr_auc": 0.90},  # higher PR-AUC
            {"layers_mp": 3, "auc": 0.85, "pr_auc": 0.60},  # higher ROC-AUC -> winner
        ]
    )
    best_row, _ = _best_config(df)
    assert best_row is not None
    assert best_row["auc"] == 0.85
    assert best_row["layers_mp"] == 3
