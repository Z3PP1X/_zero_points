import torch

from gnn.supervised_learning.run_results.eval_metrics import (
    compute_confidence_metrics,
    prediction_probabilities,
)


def test_prediction_probabilities_from_log_softmax():
    logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = prediction_probabilities(log_probs)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert probs[1, 0] > probs[1, 1]


def test_confidence_metrics_high_vs_low_margin():
    true = torch.tensor([1, 0, 1, 0])
    confident = torch.log_softmax(
        torch.tensor(
            [
                [-4.0, 4.0],
                [4.0, -4.0],
                [-3.5, 3.5],
                [3.5, -3.5],
            ]
        ),
        dim=-1,
    )
    uncertain = torch.log_softmax(
        torch.tensor(
            [
                [-0.1, 0.1],
                [0.1, -0.1],
                [-0.05, 0.05],
                [0.05, -0.05],
            ]
        ),
        dim=-1,
    )

    confident_metrics = compute_confidence_metrics(true, confident, pos_label=1)
    uncertain_metrics = compute_confidence_metrics(true, uncertain, pos_label=1)

    assert confident_metrics["mean_confidence"] > uncertain_metrics["mean_confidence"]
    assert confident_metrics["mean_margin"] > uncertain_metrics["mean_margin"]
    assert confident_metrics["mean_entropy"] < uncertain_metrics["mean_entropy"]
    assert confident_metrics["brier_score"] < uncertain_metrics["brier_score"]
