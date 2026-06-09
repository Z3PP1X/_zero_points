import argparse
from pathlib import Path
import itertools
import yaml


def set_nested_value(d, key, value):
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def generate_configs(
    config_path: Path,
    grid_path: Path,
    configs_out_dir: Path,
    results_base_dir: Path | None = None,
) -> list[Path]:
    """Generate GraphGym grid configs and return the created YAML paths."""
    import copy

    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found at {config_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found at {grid_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    with open(grid_path, "r", encoding="utf-8") as f:
        grid_config = yaml.safe_load(f)

    if not grid_config:
        raise ValueError("Grid file is empty or invalid.")

    keys = list(grid_config.keys())
    value_lists = [grid_config[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    configs_out_dir.mkdir(parents=True, exist_ok=True)
    for cfg_file in configs_out_dir.glob("*.yaml"):
        try:
            cfg_file.unlink()
        except OSError:
            pass

    created = []
    results_root = (
        Path(results_base_dir)
        if results_base_dir is not None
        else Path(base_config.get("out_dir", "results"))
    )
    if results_root.name.startswith("grid-"):
        results_root = results_root.parent

    for combination in combinations:
        config = copy.deepcopy(base_config)
        param_desc_und = []
        param_desc_pyg = []
        for key, val in zip(keys, combination):
            set_nested_value(config, key, val)
            short_key = key.split(".")[-1]
            param_desc_und.append(f"{short_key}_{val}")
            param_desc_pyg.append(f"{short_key}={val}")

        suffix_und = "_".join(param_desc_und)
        suffix_pyg = "-".join(param_desc_pyg)
        config_name = f"config_grid_{suffix_und}.yaml"
        config["out_dir"] = str(results_root / f"grid-{suffix_pyg}")

        dest_path = configs_out_dir / config_name
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        created.append(dest_path)

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Generate GraphGym grid search configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_supervised.yaml",
        help="Path to the base config file (YAML)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="grid.yaml",
        help="Path to the grid search file (YAML)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="configs",
        help="Directory to save the generated config files",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args.config
    grid_path = script_dir / args.grid
    out_dir = script_dir / args.out_dir

    if not config_path.exists():
        print(f"Error: Base config not found at {config_path}")
        return

    if not grid_path.exists():
        print(f"Error: Grid file not found at {grid_path}")
        return

    print(f"Generating configs from grid {grid_path.name}...")
    created = generate_configs(config_path, grid_path, out_dir)
    print(f"Successfully generated {len(created)} configs in: {out_dir}")
    print("\nRecommended: run the full grid with automatic post-evaluation:")
    print("  python run_all.py --experiment-name <name>")
    print("\nManual training only:")
    print(f"  for conf in {args.out_dir}/*.yaml; do")
    print("    python main_graphgym.py --cfg $conf")
    print("  done")
    print("\nManual post-evaluation after training:")
    print("  python aggregate_graphgym.py <experiment_dir> --eval")


if __name__ == "__main__":
    main()
