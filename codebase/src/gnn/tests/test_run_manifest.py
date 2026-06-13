import json

import yaml

from gnn.supervised_learning.run_all import write_run_manifest


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_write_run_manifest_combines_base_and_grid(tmp_path):
    results_dir = tmp_path / "run_results" / "exp"
    results_dir.mkdir(parents=True)

    config_base = tmp_path / "config_supervised.yaml"
    grid_path = tmp_path / "grid.yaml"
    _write_yaml(
        config_base,
        {
            "seed": 42001,
            "dataset": {"name": "run_X/parallel_benchmark_results"},
            "expression_graph": {"mode": "graph", "features": {"node": True}},
        },
    )
    _write_yaml(grid_path, {"gnn.layer_type": ["gatv2conv", "gineconv"]})

    config_files = [
        tmp_path / "configs" / "config_grid_layer_type_gatv2conv.yaml",
        tmp_path / "configs" / "config_grid_layer_type_gineconv.yaml",
    ]

    manifest_path = write_run_manifest(
        results_dir=results_dir,
        experiment_name="exp",
        config_base=config_base,
        grid_path=grid_path,
        config_files=config_files,
        timestamp="20260613_120000",
        repo_dir=tmp_path,
    )

    assert manifest_path == results_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # The "settings.json" pairing: base config + grid both embedded verbatim.
    assert manifest["seed"] == 42001
    assert manifest["base_config"]["dataset"]["name"] == "run_X/parallel_benchmark_results"
    assert manifest["grid"] == {"gnn.layer_type": ["gatv2conv", "gineconv"]}
    assert manifest["num_configs"] == 2
    assert manifest["config_files"] == [
        "config_grid_layer_type_gatv2conv.yaml",
        "config_grid_layer_type_gineconv.yaml",
    ]
    assert manifest["timestamp"] == "20260613_120000"

    # Provenance keys are always present (values may be None when git/imports fail).
    assert "git" in manifest and "commit" in manifest["git"] and "dirty" in manifest["git"]
    assert "versions" in manifest and "python" in manifest["versions"]


def test_write_run_manifest_git_never_raises(tmp_path):
    """A non-git repo_dir yields null git fields, not an exception."""
    results_dir = tmp_path / "exp"
    results_dir.mkdir(parents=True)
    config_base = tmp_path / "base.yaml"
    grid_path = tmp_path / "grid.yaml"
    _write_yaml(config_base, {"seed": 7})
    _write_yaml(grid_path, {})

    not_a_repo = tmp_path / "isolated"
    not_a_repo.mkdir()

    manifest_path = write_run_manifest(
        results_dir=results_dir,
        experiment_name="exp",
        config_base=config_base,
        grid_path=grid_path,
        config_files=[],
        timestamp="t",
        repo_dir=not_a_repo,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git"]["commit"] is None
    assert manifest["num_configs"] == 0
