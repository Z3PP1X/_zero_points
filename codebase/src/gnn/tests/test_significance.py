import numpy as np

from gnn.supervised_learning.run_results.significance import (
    bootstrap_metric_ci,
    discover_prediction_dumps,
    load_predictions,
    paired_bootstrap_diff,
    pr_auc_metric,
    save_predictions,
)


def _separable(n=200, seed=0):
    """A well-separated and a near-random predictor on the same labels."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    strong = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    weak = rng.uniform(0.0, 1.0, n)
    return y, strong, weak


def test_bootstrap_ci_brackets_point_and_orders():
    y, strong, weak = _separable()
    ci = bootstrap_metric_ci(y, strong, metric="pr_auc", n_boot=300, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n"] == len(y)
    # A strong predictor's PR-AUC CI should sit well above chance prevalence (0.5).
    assert ci["lo"] > 0.5
    assert pr_auc_metric(y, strong) > pr_auc_metric(y, weak)


def test_paired_bootstrap_identical_is_not_significant():
    y, strong, _ = _separable()
    diff = paired_bootstrap_diff(y, strong, strong, metric="pr_auc", n_boot=300, seed=2)
    assert abs(diff["diff"]) < 1e-9
    assert diff["significant"] is False
    assert diff["p_value"] == 1.0


def test_paired_bootstrap_detects_real_difference():
    y, strong, weak = _separable(n=400)
    diff = paired_bootstrap_diff(y, strong, weak, metric="pr_auc", n_boot=400, seed=3)
    assert diff["diff"] > 0
    assert diff["significant"] is True
    assert diff["p_value"] < 0.05


def test_save_load_and_discover(tmp_path):
    y, strong, _ = _separable(n=20)
    top = tmp_path / "top_configs"
    rank1 = top / "rank_1_a"
    npz = rank1 / "predictions_validation_synthetic.npz"
    save_predictions(npz, y, strong, pos_label=1)

    loaded = load_predictions(npz)
    assert loaded is not None
    assert np.array_equal(loaded["y_true"], y)
    assert loaded["pos_label"] == 1

    dumps = discover_prediction_dumps(tmp_path, "validation_synthetic")
    assert len(dumps) == 1 and dumps[0][0] == "rank_1_a"


def test_load_predictions_missing(tmp_path):
    assert load_predictions(tmp_path / "nope.npz") is None
    assert discover_prediction_dumps(tmp_path, "validation_synthetic") == []
