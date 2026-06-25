"""Smoke tests: verify the supervised training pipeline runs end-to-end.

These tests exercise the full data → DataLoader → forward → backward path so
regressions in graph loading, feature slicing, or model construction surface
immediately — not just when a multi-hour training run crashes on step 1.
"""
import json
import pytest
import torch
import pandas as pd
from gnn.shared.utils.unified_loader import UnifiedDataLoader


@pytest.fixture(autouse=True)
def clear_loader_cache():
    UnifiedDataLoader.clear_instances()
    yield
    UnifiedDataLoader.clear_instances()


def _make_graphs(problem_ids, label="Plus"):
    """Minimal 3-node expression trees keyed by problem_id."""
    graphs = {}
    for pid in problem_ids:
        graphs[pid] = {
            "id": pid,
            "nodes": [
                {"id": "n1", "label": label, "type": "operator", "value": None},
                {"id": "n2", "label": "x", "type": "variable", "value": None},
                {"id": "n3", "label": "1", "type": "constant", "value": 1.0},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "child_of"},
                {"source": "n1", "target": "n3", "type": "child_of"},
            ],
        }
    return graphs


def _make_csv(problem_ids, tmp_path, name="dataset.csv"):
    """CSV where even/odd-indexed PIDs alternate Newton vs GMGF faster (balanced classes)."""
    rows = [
        {
            "problem_id": pid,
            "Newton_absTime": 1.0 if i % 2 == 0 else 2.0,
            "GMGF_absTime": 2.0 if i % 2 == 0 else 1.0,
            "Newton_iterSteps": 5,
            "GMGF_iterSteps": 5,
        }
        for i, pid in enumerate(problem_ids)
    ]
    csv_path = tmp_path / name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_pipeline_builds_dataloaders(tmp_path):
    """GraphPipeline.pipe() produces non-empty DataLoaders with the full 33-column schema."""
    from gnn.supervised_learning.preprocessing import GraphPipeline

    pids = [f"P_{i:04d}" for i in range(12)]
    graphs_path = tmp_path / "graphs.json"
    graphs_path.write_text(json.dumps(_make_graphs(pids)), encoding="utf-8")
    csv_path = _make_csv(pids, tmp_path)

    pipeline = GraphPipeline(
        dataset_name="smoke",
        seed=42,
        mode="tree_derivatives",
        curated_csv_path=csv_path,
        curated_graphs_path=graphs_path,
    )
    train_loader, test_loader, class_weights = pipeline.pipe(
        test_size=0.2, batch_size=4, stratify=True
    )

    assert len(pipeline.train_dataset) > 0, "Train dataset is empty"
    assert len(pipeline.test_dataset) > 0, "Test dataset is empty"
    assert class_weights.shape == (2,)
    assert torch.all(class_weights > 0), "Class weights must be positive"

    batch = next(iter(train_loader))
    assert batch.x is not None
    assert batch.x.shape[1] == 32, f"Expected 32 node features, got {batch.x.shape[1]}"
    assert batch.y is not None


def test_pipeline_active_features_slice(tmp_path):
    """Passing active_features slices x to exactly the requested subset."""
    from gnn.supervised_learning.preprocessing import GraphPipeline

    pids = [f"P_{i:04d}" for i in range(12)]
    graphs_path = tmp_path / "graphs.json"
    graphs_path.write_text(json.dumps(_make_graphs(pids)), encoding="utf-8")
    csv_path = _make_csv(pids, tmp_path)

    active = ["node_type_global", "node_type_operator", "label_Plus", "label_x"]
    pipeline = GraphPipeline(
        dataset_name="smoke",
        seed=42,
        mode="tree_derivatives",
        active_features=active,
        curated_csv_path=csv_path,
        curated_graphs_path=graphs_path,
    )
    pipeline.pipe(test_size=0.2, batch_size=4)

    batch = next(iter(pipeline.train_loader))
    assert batch.x.shape[1] == len(active), (
        f"Expected {len(active)} features after active_features slice, got {batch.x.shape[1]}"
    )


def test_forward_and_backward_pass(tmp_path):
    """One training step through ExpressionGNN produces finite loss and updates weights."""
    from gnn.supervised_learning.preprocessing import GraphPipeline
    from gnn.shared.models.gnn_backbones import ExpressionGNN

    pids = [f"P_{i:04d}" for i in range(12)]
    graphs_path = tmp_path / "graphs.json"
    graphs_path.write_text(json.dumps(_make_graphs(pids)), encoding="utf-8")
    csv_path = _make_csv(pids, tmp_path)

    pipeline = GraphPipeline(
        dataset_name="smoke",
        seed=42,
        mode="tree_derivatives",
        curated_csv_path=csv_path,
        curated_graphs_path=graphs_path,
    )
    pipeline.pipe(test_size=0.2, batch_size=4)

    model = ExpressionGNN(
        input_dim=pipeline.input_dim,
        hidden_dim=16,
        output_dim=1,
        num_layers=2,
        classify=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    params_before = [p.data.clone() for p in model.parameters()]

    model.train()
    batch = next(iter(pipeline.train_loader))
    optimizer.zero_grad()
    logits = model(batch.x, batch.edge_index, batch.batch).view(-1)
    loss = criterion(logits, batch.y.float())
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"Training loss is not finite: {loss.item()}"
    params_changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, model.parameters())
    )
    assert params_changed, "No model parameters changed after backward + optimizer step"


def test_two_epoch_training_all_losses_finite(tmp_path):
    """Two full epochs of training produce only finite losses."""
    from gnn.supervised_learning.preprocessing import GraphPipeline
    from gnn.shared.models.gnn_backbones import ExpressionGNN

    pids = [f"P_{i:04d}" for i in range(20)]
    graphs_path = tmp_path / "graphs.json"
    graphs_path.write_text(json.dumps(_make_graphs(pids)), encoding="utf-8")
    csv_path = _make_csv(pids, tmp_path)

    pipeline = GraphPipeline(
        dataset_name="smoke",
        seed=42,
        mode="tree_derivatives",
        curated_csv_path=csv_path,
        curated_graphs_path=graphs_path,
    )
    pipeline.pipe(test_size=0.2, batch_size=8)

    model = ExpressionGNN(
        input_dim=pipeline.input_dim,
        hidden_dim=32,
        output_dim=1,
        num_layers=2,
        classify=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(2):
        model.train()
        for batch in pipeline.train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch).view(-1)
            loss = criterion(logits, batch.y.float())
            assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch}: {loss.item()}"
            loss.backward()
            optimizer.step()
