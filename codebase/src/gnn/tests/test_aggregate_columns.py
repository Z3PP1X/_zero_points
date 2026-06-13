"""Config-column recovery for datetime-named runs (aggregate_graphgym)."""

import json

import yaml

from gnn.supervised_learning.aggregate_graphgym import (
    _grid_axes_from_manifest,
    _run_config_columns,
)


def _make_run(tmp_path, name, cfg):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return run_dir


def test_run_config_columns_from_snapshot_all_known(tmp_path):
    cfg = {
        "gnn": {
            "layer_type": "gatv2conv",
            "layers_mp": 3,
            "dim_inner": 256,
            "variant": "pooling",
            "pool_type": "diffpool",
            "aux_loss_weight": 1.0,
        },
        "expression_graph": {"mode": "graph", "edge_direction": "top_down"},
        "optim": {"base_lr": 0.0001},
        "model": {"graph_pooling": "mean"},
    }
    run_dir = _make_run(tmp_path, "run_20260613_120000_000", cfg)

    cols = _run_config_columns(str(run_dir), run_dir.name, axes=None)
    assert cols["layer_type"] == "gatv2conv"
    assert cols["variant"] == "pooling"
    assert cols["pool_type"] == "diffpool"
    assert cols["mode"] == "graph"
    assert cols["edge_direction"] == "top_down"
    assert cols["run_name"] == "run_20260613_120000_000"


def test_run_config_columns_restricted_to_swept_axes(tmp_path):
    cfg = {
        "gnn": {"layer_type": "gatv2conv", "layers_mp": 2, "variant": "legacy"},
        "expression_graph": {"mode": "graph"},
    }
    run_dir = _make_run(tmp_path, "run_t_000", cfg)
    axes = {"layers_mp": "gnn.layers_mp"}  # only this axis was swept

    cols = _run_config_columns(str(run_dir), run_dir.name, axes=axes)
    assert cols == {"layers_mp": 2, "run_name": "run_t_000"}


def test_run_config_columns_legacy_fallback(tmp_path):
    """A grid-key=value folder with no config.yaml falls back to name parsing."""
    run_name = "grid-layer_type=gatv2conv-layers_mp=2"
    run_dir = tmp_path / run_name
    run_dir.mkdir()

    cols = _run_config_columns(str(run_dir), run_name, axes=None)
    assert cols["layer_type"] == "gatv2conv"
    assert cols["layers_mp"] == "2"
    assert cols["run_name"] == run_name


def test_grid_axes_from_manifest(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"grid": {"gnn.layer_type": ["gatv2conv"], "gnn.layers_mp": [2, 3]}}),
        encoding="utf-8",
    )
    axes = _grid_axes_from_manifest(str(tmp_path))
    assert axes == {"layer_type": "gnn.layer_type", "layers_mp": "gnn.layers_mp"}


def test_grid_axes_from_manifest_absent(tmp_path):
    assert _grid_axes_from_manifest(str(tmp_path)) is None
