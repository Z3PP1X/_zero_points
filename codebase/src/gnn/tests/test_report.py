import json

import numpy as np
import pandas as pd

from gnn.supervised_learning.run_results.report import generate_report
from gnn.supervised_learning.run_results.significance import save_predictions


def _make_experiment(tmp_path):
    """Minimal experiment layout: agg CSVs + class_balance, no checkpoints."""
    exp = tmp_path / "exp"
    agg = exp / "agg"
    agg.mkdir(parents=True)

    configs = [
        {"layer_type": "gatv2conv", "layers_mp": 2, "dim_inner": 128, "dropout": 0.1},
        {"layer_type": "gineconv", "layers_mp": 3, "dim_inner": 64, "dropout": 0.2},
    ]
    # val synthetic: gatv2 is the winner; curated drops a bit (generalization gap).
    val = pd.DataFrame([
        {**configs[0], "epoch": 10, "pr_auc": 0.81, "auc": 0.80, "f1": 0.7,
         "recall": 0.68, "precision": 0.72, "accuracy": 0.78, "loss": 0.42},
        {**configs[1], "epoch": 8, "pr_auc": 0.74, "auc": 0.73, "f1": 0.65,
         "recall": 0.6, "precision": 0.7, "accuracy": 0.72, "loss": 0.5},
    ])
    test = val.copy()
    test["pr_auc"] = [0.70, 0.69]
    train = val.copy()
    train["pr_auc"] = [0.92, 0.88]
    val.to_csv(agg / "val_bestepoch.csv", index=False)
    test.to_csv(agg / "test_bestepoch.csv", index=False)
    train.to_csv(agg / "train_bestepoch.csv", index=False)

    balance = {
        "validation_synthetic": {"0": 60, "1": 40, "total": 100},
        "validation_curated": {"0": 30, "1": 10, "total": 40},
    }
    (agg / "class_balance.json").write_text(json.dumps(balance))
    return exp


def test_generate_report_writes_md_and_json(tmp_path):
    exp = _make_experiment(tmp_path)
    md_path = generate_report(exp, top_k=5)

    assert md_path.exists()
    text = md_path.read_text()
    assert "Experiment summary" in text
    # Best config by val synthetic pr_auc is gatv2conv (0.81).
    assert "gatv2conv" in text
    assert "Generalization gap" in text
    assert "Baselines" in text

    summary = json.loads((exp / "eval_plots" / "summary.json").read_text())
    assert summary["experiment"] == "exp"
    assert summary["best_config"]["val_synthetic_pr_auc"] == 0.81
    assert len(summary["leaderboard"]) == 2
    # No-skill PR-AUC baseline equals positive-class prevalence (0.4) on val synthetic.
    assert summary["baselines"]["val_bestepoch"]["no_skill_pr_auc"] == 0.4


def test_generate_report_with_prediction_dumps(tmp_path):
    exp = _make_experiment(tmp_path)
    eval_dir = exp / "eval_plots"
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    strong = np.where(y == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100))
    weak = rng.uniform(0.0, 1.0, 100)
    save_predictions(
        eval_dir / "top_configs" / "rank_1_a" / "predictions_validation_synthetic.npz",
        y, strong, pos_label=1,
    )
    save_predictions(
        eval_dir / "top_configs" / "rank_2_b" / "predictions_validation_synthetic.npz",
        y, weak, pos_label=1,
    )

    generate_report(exp, top_k=5)
    summary = json.loads((eval_dir / "summary.json").read_text())
    assert "best_ci" in summary["significance"]
    assert "paired_diff" in summary["significance"]
    # The strong predictor's CI lower bound should beat chance.
    assert summary["significance"]["best_ci"]["lo"] > 0.5


def test_generate_report_missing_agg_is_graceful(tmp_path):
    exp = tmp_path / "empty"
    (exp / "agg").mkdir(parents=True)
    md_path = generate_report(exp)
    assert md_path.exists()
    assert "cannot pick a best config" in md_path.read_text()
