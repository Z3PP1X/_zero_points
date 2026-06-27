"""Regression tests for the reproducibility/validity audit fixes.

Covers:
  F-06 — ROC-AUC and PR-AUC in one metrics row reference the SAME positive class.
  F-10 — diagnostics ROC/PR plots use proper probabilities; curve and legend agree
         (no pos_label==0 inversion); per-split minority pos_label.
  F-08 — add_kappa=True never silently degrades to kappa-less graphs.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_auc_score, roc_curve

from gnn.supervised_learning.loader_graphgym import (
    _hard_predictions,
    compute_binary_metrics,
)
from gnn.supervised_learning.run_results.eval_metrics import prediction_probabilities
from gnn.supervised_learning.run_results.diagnostics import (
    _positive_class_probs,
    _provenance_subtitle,
    _split_pos_label,
)


def _toy_log_softmax(n=600, class1_rate=0.3, sep=1.3, seed=0):
    """Labels with a known class balance + correlated log-softmax model scores."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < class1_rate).astype(int)
    raw = np.stack(
        [
            rng.normal(0.0, 0.7, n) + (y == 0) * sep,
            rng.normal(0.0, 0.7, n) + (y == 1) * sep,
        ],
        axis=1,
    )
    log_probs = F.log_softmax(torch.tensor(raw, dtype=torch.float), dim=1)
    return torch.tensor(y), log_probs


# --------------------------------------------------------------------------- F-06


def test_roc_auc_is_pos_label_invariant_and_value_preserving():
    """ROC-AUC is symmetric: the reported value must not depend on pos_label, and must
    equal the standard positive=class1 AUC (i.e. the fix changes semantics, not numbers)."""
    y, scores = _toy_log_softmax(class1_rate=0.3)  # class 1 is the minority
    reference = roc_auc_score(y.numpy(), scores[:, 1].numpy())
    m0 = compute_binary_metrics(y, scores, pos_label=0, round_digits=6)
    m1 = compute_binary_metrics(y, scores, pos_label=1, round_digits=6)
    assert abs(m0["auc"] - reference) < 1e-4
    assert abs(m1["auc"] - reference) < 1e-4
    assert abs(m0["auc"] - m1["auc"]) < 1e-4
    assert m0["pos_label"] == 0 and m1["pos_label"] == 1


def test_pr_auc_depends_on_pos_label_unlike_roc():
    """PR-AUC is asymmetric, so it must differ between positive classes — confirming the
    row's ROC (symmetric) and PR (pos_label-specific) now describe the same positive class."""
    y, scores = _toy_log_softmax(class1_rate=0.22)
    m0 = compute_binary_metrics(y, scores, pos_label=0, round_digits=6)
    m1 = compute_binary_metrics(y, scores, pos_label=1, round_digits=6)
    assert abs(m0["auc"] - m1["auc"]) < 1e-4          # ROC symmetric
    assert m0["pr_auc"] != m1["pr_auc"]               # PR positive-class specific


# --------------------------------------------------------------------------- F-10


def test_positive_class_probs_are_valid_probabilities():
    """The diagnostics score is a real probability in [0, 1] for both classes (the old
    1 - log_softmax trick produced values > 1 and anti-correlated scores for pos_label=0)."""
    _, scores = _toy_log_softmax(class1_rate=0.4)
    for pos in (0, 1):
        s = _positive_class_probs(scores, pos)
        assert s.min() >= 0.0 and s.max() <= 1.0


# --------------------------------------------------------------- single-logit (binary head)
# The production GraphGym head is a SINGLE logit; pred_score = sigmoid(logit) = P(class 1).
# These guard the two display-layer bugs found in the final_model_selection run:
#   Bug 1 — _hard_predictions returned a positive-class INDICATOR (1 == predicted class 0)
#           for pos_label==0, inverting accuracy/precision/recall/f1.
#   Bug 2 — prediction_probabilities sigmoided the already-sigmoid score a second time,
#           corrupting ECE / Brier / reliability (but not the rank-based AUC).


def _toy_sigmoid_scores(n=600, class1_rate=0.6, sep=1.3, seed=0):
    """Labels + a correlated single-column P(class 1) (what the binary head emits)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < class1_rate).astype(int)
    logit = rng.normal(0.0, 0.7, n) + (y == 1) * sep - (y == 0) * sep
    p1 = torch.sigmoid(torch.tensor(logit, dtype=torch.float))
    return torch.tensor(y), p1


def test_hard_predictions_single_logit_are_class_labels_not_indicators():
    """_hard_predictions must return y_true-encoded labels for BOTH pos_label values.

    Regression for Bug 1: for a good single-logit model the hard accuracy is high (~the
    fraction separated), never its 1 - acc mirror image, regardless of which class is the
    metric positive. The predicted LABEL is pos_label-invariant (only the decision cutoff
    on P(class 1) matters), so both pos_label calls return identical labels at thresh 0.5.
    """
    y, p1 = _toy_sigmoid_scores(class1_rate=0.6)
    pred0 = _hard_predictions(p1, pos_label=0, thresh=0.5).numpy()
    pred1 = _hard_predictions(p1, pos_label=1, thresh=0.5).numpy()
    # labels, not indicators: identical regardless of the reporting positive class
    assert np.array_equal(pred0, pred1)
    # at thresh 0.5 the label is argmax(P): predict class 1 iff P(class 1) > 0.5
    assert np.array_equal(pred0, (p1.numpy() > 0.5).astype(int))
    acc = float((pred0 == y.numpy()).mean())
    assert acc > 0.75  # a separated model — NOT the inverted ~0.20


def test_compute_binary_metrics_single_logit_not_inverted():
    """f1/accuracy on a single-logit score reflect the real (good) model, not its inverse.

    Directly exercises the final_model_selection failure mode: minority = class 0 so
    pos_label defaults to 0, the branch that used to invert.
    """
    y, p1 = _toy_sigmoid_scores(class1_rate=0.7)  # class 0 is the minority -> pos_label 0
    m0 = compute_binary_metrics(y, p1, pos_label=0, round_digits=6)
    # accuracy of the actual labels, computed independently of the metric code
    direct_acc = float(((p1.numpy() > 0.5).astype(int) == y.numpy()).mean())
    assert abs(m0["accuracy"] - direct_acc) < 1e-6
    assert m0["accuracy"] > 0.7 and m0["f1"] > 0.5  # not the inverted mirror


def test_prediction_probabilities_no_double_sigmoid_on_single_logit():
    """A 1-col probability input is passed through unchanged (Bug 2 would re-sigmoid it)."""
    p1 = torch.tensor([0.05, 0.3, 0.5, 0.8, 0.97])
    probs = prediction_probabilities(p1)
    assert torch.allclose(probs[:, 1], p1, atol=1e-6)
    assert torch.allclose(probs[:, 0], 1.0 - p1, atol=1e-6)
    # double sigmoid would have squashed everything toward [0.5, 0.73]
    assert probs[:, 1].min() < 0.1 and probs[:, 1].max() > 0.9


@pytest.mark.parametrize("pos_label", [0, 1])
def test_diagnostics_roc_curve_matches_legend(pos_label):
    """The plotted ROC curve area must equal the legend AUC — previously they diverged by
    ~1-AUC for pos_label==0 (curve inverted under a high legend value)."""
    y, scores = _toy_log_softmax(class1_rate=0.6)  # class 0 minority -> pos_label 0 realistic
    s = _positive_class_probs(scores, pos_label)
    y_bin = (y.numpy() == pos_label).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, s)
    curve_area = sk_auc(fpr, tpr)
    legend = roc_auc_score(y_bin, s)
    assert abs(curve_area - legend) < 1e-9
    assert legend > 0.5  # a discriminating model is ABOVE no-skill, never below


def test_split_pos_label_is_minority_class():
    assert _split_pos_label(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])) == 1
    assert _split_pos_label(torch.tensor([0, 1, 1, 1, 1, 1])) == 0


def test_provenance_subtitle_reports_dataset_class_and_size():
    y = torch.tensor([0, 0, 0, 1])
    text = _provenance_subtitle(y, pos_label=1, dataset_label="curated: bench.csv", n_graphs=7)
    assert "curated: bench.csv" in text
    assert "positive = 1 (Newton)" in text
    assert "N=4 rows" in text
    assert "7 distinct graphs" in text
    assert "prevalence=0.2500" in text  # P(y==1)


# --------------------------------------------------------------------------- F-08


def test_add_kappa_true_without_kappa_column_raises():
    """add_kappa=True on a dataset without a 'kappa' column must fail loudly instead of
    silently loading kappa-less graphs (which would invalidate any kappa ablation)."""
    import pandas as pd

    from unified_loader import UnifiedDataLoader

    mock_df = pd.DataFrame(
        [
            {"problem_id": "P1", "Newton_absTime": 1.0, "GMGF_absTime": 2.0},
            {"problem_id": "P2", "Newton_absTime": 2.0, "GMGF_absTime": 1.0},
        ]
    )
    with patch("unified_loader.DatasetLoader") as MockDatasetLoader, patch(
        "unified_loader.GraphDataLoader"
    ):
        mock_dataset_inst = MagicMock()
        mock_dataset_inst.data = mock_df
        MockDatasetLoader.return_value = mock_dataset_inst

        UnifiedDataLoader.clear_instances()
        with pytest.raises(RuntimeError, match="kappa"):
            UnifiedDataLoader.get_instance(
                dataset_name="test_kappa_missing",
                run_key="test",
                mode="tree_derivatives",
                add_kappa=True,
            )
    UnifiedDataLoader.clear_instances()
