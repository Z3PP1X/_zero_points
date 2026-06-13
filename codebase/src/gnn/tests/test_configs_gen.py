import yaml

from gnn.supervised_learning.configs_gen import (
    _canonical_signature,
    generate_configs,
    get_nested_value,
)


def _write_base(path):
    path.write_text(
        yaml.safe_dump(
            {
                "out_dir": "results",
                "seed": 42001,
                "gnn": {
                    "layer_type": "gatv2conv",
                    "variant": "legacy",
                    "pool_type": "topk",
                    "aux_loss_weight": 1.0,
                },
                "expression_graph": {"mode": "graph", "edge_direction": "top_down"},
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


def test_generate_configs_dedup_inert_pooling(tmp_path):
    """legacy ignores pool_type, so legacy×{topk,diffpool} collapses to one config."""
    base = tmp_path / "base.yaml"
    grid = tmp_path / "grid.yaml"
    _write_base(base)
    grid.write_text(
        yaml.safe_dump(
            {
                "gnn.variant": ["legacy", "pooling"],
                "gnn.pool_type": ["topk", "diffpool"],
            }
        ),
        encoding="utf-8",
    )

    created = generate_configs(
        base, grid, tmp_path / "configs", run_timestamp="t"
    )
    combos = {
        (
            yaml.safe_load(p.read_text())["gnn"]["variant"],
            yaml.safe_load(p.read_text())["gnn"]["pool_type"],
        )
        for p in created
    }
    # 4 grid points -> 3 distinct: legacy (pool_type collapsed) + pooling×{topk,diffpool}.
    assert len(created) == 3
    assert ("pooling", "topk") in combos
    assert ("pooling", "diffpool") in combos
    legacy = [c for c in combos if c[0] == "legacy"]
    assert len(legacy) == 1


def test_canonical_signature_nullifies_inert_axes():
    swept = ["gnn.variant", "gnn.pool_type", "gnn.aux_loss_weight"]
    legacy_topk = {
        "gnn": {"variant": "legacy", "pool_type": "topk", "aux_loss_weight": 1.0}
    }
    legacy_diff = {
        "gnn": {"variant": "legacy", "pool_type": "diffpool", "aux_loss_weight": 0.5}
    }
    # Both legacy → pool_type and aux_loss_weight inert → identical signatures.
    assert _canonical_signature(legacy_topk, swept) == _canonical_signature(
        legacy_diff, swept
    )

    diff_a = {
        "gnn": {"variant": "pooling", "pool_type": "diffpool", "aux_loss_weight": 1.0}
    }
    diff_b = {
        "gnn": {"variant": "pooling", "pool_type": "diffpool", "aux_loss_weight": 0.5}
    }
    # diffpool actually uses aux_loss_weight → signatures differ.
    assert _canonical_signature(diff_a, swept) != _canonical_signature(diff_b, swept)


def test_get_nested_value():
    d = {"a": {"b": {"c": 5}}}
    assert get_nested_value(d, "a.b.c") == 5
    assert get_nested_value(d, "a.x") is None
    assert get_nested_value(d, "a.b.c.d") is None
