import pytest
import yaml

from gnn.supervised_learning.configs_gen import (
    _canonical_signature,
    _is_valid_config,
    generate_configs,
    validate_grid_keys,
)


def _write_base(path):
    path.write_text(
        yaml.safe_dump(
            {
                "out_dir": "results",
                "seed": 42001,
                "gnn": {
                    "layer_type": "ginconv",
                    "layers_mp": 3,
                    "dim_inner": 128,
                },
                "expression_graph": {"mode": "tree_derivatives", "edge_direction": "top_down"},
            }
        ),
        encoding="utf-8",
    )


def test_generate_configs_datetime_naming(tmp_path):
    base = tmp_path / "config_supervised.yaml"
    grid = tmp_path / "grid.yaml"
    _write_base(base)
    grid.write_text(yaml.safe_dump({"gnn.layers_mp": [2, 3]}), encoding="utf-8")

    out = tmp_path / "configs"
    results = tmp_path / "run_results" / "exp"
    created = generate_configs(
        base, grid, out, results_base_dir=results, run_timestamp="20260613_120000"
    )

    assert [p.name for p in created] == [
        "cfg_20260613_120000_000.yaml",
        "cfg_20260613_120000_001.yaml",
    ]
    # out_dir folders are datetime+index, under the experiment dir, no params in the name.
    out_dirs = [yaml.safe_load(p.read_text())["out_dir"] for p in created]
    assert out_dirs[0].endswith("run_20260613_120000_000")
    assert "layers_mp" not in out_dirs[0]
    # The swept value still lives inside the config (recovered later from the snapshot).
    layers = sorted(yaml.safe_load(p.read_text())["gnn"]["layers_mp"] for p in created)
    assert layers == [2, 3]


def test_canonical_signature_deduplicates_identical_configs():
    swept = ["gnn.layers_mp", "gnn.dim_inner"]
    cfg_a = {"gnn": {"layers_mp": 3, "dim_inner": 128}}
    cfg_b = {"gnn": {"layers_mp": 3, "dim_inner": 128}}
    cfg_c = {"gnn": {"layers_mp": 2, "dim_inner": 128}}
    assert _canonical_signature(cfg_a, swept) == _canonical_signature(cfg_b, swept)
    assert _canonical_signature(cfg_a, swept) != _canonical_signature(cfg_c, swept)


def test_is_valid_config_rejects_supernode_with_positional():
    """Anchor positional encoding and the virtual supernode are mutually exclusive."""
    def cfg(supernode, positional):
        return {
            "expression_graph": {
                "add_virtual_supernode": supernode,
                "features": {"positional": positional},
            }
        }

    # The one invalid quadrant: supernode on AND positional enabled.
    assert _is_valid_config(cfg(True, True)) is False
    # A non-empty list counts as "positional enabled" too.
    assert _is_valid_config(cfg(True, ["anchor_periodic"])) is False
    # None and a missing key both mean "all anchors ON" at runtime -> still invalid with
    # the supernode (mirrors feature_config._resolve_category_value, which treats None as all).
    assert _is_valid_config(cfg(True, None)) is False
    assert _is_valid_config(
        {"expression_graph": {"add_virtual_supernode": True, "features": {}}}
    ) is False
    # Every other combination is valid.
    assert _is_valid_config(cfg(True, False)) is True
    assert _is_valid_config(cfg(False, True)) is True
    assert _is_valid_config(cfg(False, False)) is True
    assert _is_valid_config(cfg(True, [])) is True  # empty list = positional off
    assert _is_valid_config(cfg(False, None)) is True  # positional on, no supernode


def test_generate_configs_skips_supernode_positional(tmp_path):
    """The grid sweeps both axes; the 2 invalid (on, on) points must be dropped."""
    base = tmp_path / "base.yaml"
    grid = tmp_path / "grid.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "out_dir": "results",
                "gnn": {"layer_type": "ginconv"},
                "expression_graph": {
                    "mode": "tree_derivatives",
                    "add_virtual_supernode": False,
                    "features": {"positional": False},
                },
            }
        ),
        encoding="utf-8",
    )
    grid.write_text(
        yaml.safe_dump(
            {
                "expression_graph.add_virtual_supernode": [False, True],
                "expression_graph.features.positional": [False, True],
            }
        ),
        encoding="utf-8",
    )
    created = generate_configs(base, grid, tmp_path / "configs", run_timestamp="t")
    # 4 grid points -> 3 valid (the (supernode=true, positional=true) point is skipped).
    assert len(created) == 3
    for p in created:
        eg = yaml.safe_load(p.read_text())["expression_graph"]
        assert not (eg["add_virtual_supernode"] and eg["features"]["positional"])


def test_validate_grid_keys_rejects_unknown_axis():
    base = {"gnn": {"layer_type": "ginconv"}, "expression_graph": {"mode": "tree_derivatives"}}
    # This is exactly the typo that crashed yacs at train time.
    grid = {"expression_graph.graph": ["graph"]}
    with pytest.raises(ValueError) as exc:
        validate_grid_keys(base, grid)
    msg = str(exc.value)
    assert "expression_graph.graph" in msg
    assert "mode" in msg  # suggests the valid sibling key


def test_generate_configs_raises_on_typo_axis(tmp_path):
    base = tmp_path / "base.yaml"
    grid = tmp_path / "grid.yaml"
    _write_base(base)
    grid.write_text(yaml.safe_dump({"expression_graph.graph": ["graph"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="expression_graph.graph"):
        generate_configs(base, grid, tmp_path / "configs", run_timestamp="t")
