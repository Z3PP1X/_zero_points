import pytest
import numpy as np
from unittest.mock import MagicMock

from gnn.supervised_learning.run_results.eval_metrics import (
    select_best_epoch,
    select_best_epoch_row,
)
from gnn.supervised_learning.run_results.feature_importance import (
    resolve_feature_names,
)


def test_select_best_epoch_higher_is_better():
    stats_list = [
        {"epoch": 0, "pr_auc": 0.5},
        {"epoch": 1, "pr_auc": 0.6},
        {"epoch": 2, "pr_auc": 0.8},
        {"epoch": 3, "pr_auc": 0.75},
        {"epoch": 4, "pr_auc": 0.7},
    ]
    best_epoch = select_best_epoch(stats_list, "pr_auc", agg="argmax", warmup_epochs=0)
    assert best_epoch == 2


def test_select_best_epoch_lower_is_better():
    stats_list = [
        {"epoch": 0, "brier_score": 0.5},
        {"epoch": 1, "brier_score": 0.4},
        {"epoch": 2, "brier_score": 0.2},
        {"epoch": 3, "brier_score": 0.3},
        {"epoch": 4, "brier_score": 0.35},
    ]
    # Under the old implementation, agg="argmax" would select the maximum (0.5 at epoch 0).
    # Under the new implementation, it should override to "argmin" and select 0.2 at epoch 2.
    best_epoch = select_best_epoch(stats_list, "brier_score", agg="argmax", warmup_epochs=0)
    assert best_epoch == 2

    # Should also work for other LOWER_IS_BETTER_METRICS like ece
    ece_stats_list = [
        {"epoch": 0, "ece": 0.1},
        {"epoch": 1, "ece": 0.3},
        {"epoch": 2, "ece": 0.05},
        {"epoch": 3, "ece": 0.2},
    ]
    best_epoch_ece = select_best_epoch(ece_stats_list, "ece", agg="argmax", warmup_epochs=0)
    assert best_epoch_ece == 2


def test_resolve_feature_names_with_custom_pos_encodings():
    cfg = MagicMock()
    cfg.expression_graph = MagicMock()
    cfg.expression_graph.enrich = True
    cfg.expression_graph.active_features = ""
    
    features_dict = {
        "node": True,
        "topology": True,
        "positional": {
            "enabled": True,
            "encodings": ["lpe"]  # lpe only (4 features), no rwpe (4 features)
        },
        "edge": True
    }
    
    cfg.expression_graph.features = features_dict
    cfg.expression_graph.items = lambda: [
        ("enrich", True),
        ("active_features", ""),
        ("features", features_dict)
    ]

    feature_names = resolve_feature_names(cfg)
    # Total enriched schema has 24 features. Since rwpe (4 features) is disabled, it should resolve to 20 features.
    assert len(feature_names) == 20
    assert "lpe_1" in feature_names
    assert "rwpe_1" not in feature_names
