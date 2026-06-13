import json

from gnn.supervised_learning.run_results.baselines import (
    compute_baselines,
    split_baselines,
)


def test_split_baselines_balanced():
    # 50/50 split: no-skill PR-AUC == prevalence == 0.5; majority predicts positive.
    out = split_baselines({"0": 50, "1": 50, "total": 100})
    assert out["prevalence"] == 0.5
    assert out["no_skill_pr_auc"] == 0.5
    assert out["majority"]["predicts"] == "positive"  # ties go to positive
    assert out["majority"]["accuracy"] == 0.5
    assert out["majority"]["recall"] == 1.0
    assert out["majority"]["auc"] == 0.5


def test_split_baselines_imbalanced_majority_negative():
    # Positive class rare: majority predicts negative -> zero positive recall/precision.
    out = split_baselines({"0": 90, "1": 10, "total": 100})
    assert out["prevalence"] == 0.1
    assert out["majority"]["predicts"] == "negative"
    assert out["majority"]["accuracy"] == 0.9
    assert out["majority"]["precision"] == 0.0
    assert out["majority"]["recall"] == 0.0
    assert out["majority"]["f1"] == 0.0
    # Random stratified accuracy = p^2 + (1-p)^2 = 0.82
    assert abs(out["random"]["accuracy"] - 0.82) < 1e-9
    assert out["random"]["pr_auc"] == 0.1


def test_split_baselines_empty():
    assert split_baselines({"0": 0, "1": 0, "total": 0}) == {}


def test_compute_baselines_from_file(tmp_path):
    agg = tmp_path / "agg"
    agg.mkdir()
    balance = {
        "validation_synthetic": {"0": 60, "1": 40, "total": 100},
        "validation_curated": {"0": 30, "1": 10, "total": 40},
    }
    (agg / "class_balance.json").write_text(json.dumps(balance))

    out = compute_baselines(agg)
    assert set(out.keys()) == {"val_bestepoch", "test_bestepoch"}
    assert out["val_bestepoch"]["prevalence"] == 0.4
    assert out["test_bestepoch"]["prevalence"] == 0.25


def test_compute_baselines_missing_file(tmp_path):
    assert compute_baselines(tmp_path) == {}
