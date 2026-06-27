"""Unit tests for the RL config layer — stage-based dataset resolution.

The RL pipeline no longer carries its own dataset/graph definition: it consumes a
supervised-learning *stage* (config_settings/<stage>/base_config.yaml), so both
pipelines share one source of truth. These tests pin that wiring down without
needing Mathematica or a GPU.
"""

from pathlib import Path

import pytest

from gnn.reinforcement_learning.rl_config import (
    RL_STAGE_DEFAULT,
    StageDataset,
    read_rl_settings,
    resolve_stage_dataset,
)


def test_read_rl_settings_defaults_to_stage_default():
    settings = read_rl_settings({})
    assert settings["stage"] == RL_STAGE_DEFAULT


def test_read_rl_settings_honours_configured_stage():
    settings = read_rl_settings({"stage": "stage1_pure_ast"})
    assert settings["stage"] == "stage1_pure_ast"


def test_read_rl_settings_no_longer_exposes_legacy_experiment_keys():
    # The dead nur_f/f_fp_roh/kein_inv experiment block is gone; mode/features now
    # come from the stage, not from read_rl_settings.
    settings = read_rl_settings({})
    for legacy_key in ("experiment", "mode", "add_kappa", "add_virtual_supernode"):
        assert legacy_key not in settings


@pytest.mark.parametrize("stage", ["3", "stage3_full_graph"])
def test_resolve_stage3_by_number_and_folder(stage):
    ds = resolve_stage_dataset(stage)
    assert isinstance(ds, StageDataset)
    assert ds.label == "stage3_full_graph"
    assert ds.mode == "tree_derivatives"
    assert ds.add_virtual_supernode is False
    # Stage 3 is the kappa-ON full graph — faithfully mirrored from its base_config.
    assert ds.add_kappa is True
    # Full graph → all features (empty active_features string).
    assert ds.expression_graph.get("active_features", "") == ""
    # Graph dir + curated CSV resolve to absolute, repo-rooted paths.
    assert ds.graphs_path is not None and ds.graphs_path.is_absolute()
    assert ds.graphs_path.parts[-2:] == ("datasets", "graphs")
    assert ds.curated_csv is not None and ds.curated_csv.is_absolute()
    assert ds.curated_csv.suffix == ".csv"


def test_resolve_stage1_has_explicit_feature_subset():
    ds = resolve_stage_dataset("1")
    assert ds.label == "stage1_pure_ast"
    # Stage 1 = pure AST → an explicit, non-empty active_features list.
    active = ds.expression_graph.get("active_features", "")
    assert active and "node_type_global" in active
    # Same shared graphs dir as every other stage.
    assert ds.graphs_path.parts[-2:] == ("datasets", "graphs")


def test_default_stage_is_kappa_free():
    # The default must not silently require kappa data (would hard-fail at load).
    ds = resolve_stage_dataset(RL_STAGE_DEFAULT)
    assert ds.add_kappa is False
    assert ds.add_virtual_supernode is False


def test_resolve_unknown_stage_raises():
    with pytest.raises(SystemExit):
        resolve_stage_dataset("does_not_exist")
