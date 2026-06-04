import os
import argparse
from pathlib import Path
import itertools
import yaml

def set_nested_value(d, key, value):
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def main():
    parser = argparse.ArgumentParser(description="Generate GraphGym grid search configuration files")
    parser.add_argument(
        "--config",
        type=str,
        default="config_supervised.yaml",
        help="Path to the base config file (YAML)"
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="grid.yaml",
        help="Path to the grid search file (YAML)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="configs",
        help="Directory to save the generated config files"
    )
    args = parser.parse_args()

    # Resolve paths relative to script directory
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

    # Load base config
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Load grid config
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_config = yaml.safe_load(f)

    if not grid_config:
        print("Error: Grid file is empty or invalid.")
        return

    # Generate combinations
    keys = list(grid_config.keys())
    value_lists = [grid_config[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(combinations)} config files from grid {grid_path.name}...")
    
    # Clean old configs in that dir to avoid mixups
    for f in out_dir.glob("*.yaml"):
        try:
            f.unlink()
        except OSError:
            pass

    for i, combination in enumerate(combinations):
        # Deep copy base config
        import copy
        config = copy.deepcopy(base_config)
        
        # Build suffix name representing parameter values
        param_desc_und = [] # Clean representation for yaml filename
        param_desc_pyg = [] # GraphGym standard naming convention (key=val separated by -)
        for key, val in zip(keys, combination):
            set_nested_value(config, key, val)
            # Use short representations for file naming
            short_key = key.split('.')[-1]
            param_desc_und.append(f"{short_key}_{val}")
            param_desc_pyg.append(f"{short_key}={val}")
        
        suffix_und = "_".join(param_desc_und)
        suffix_pyg = "-".join(param_desc_pyg)
        config_name = f"config_grid_{suffix_und}.yaml"
        
        # Override output directory inside config to store results in subfolders
        # GraphGym stores results in: out_dir/config_name/
        config["out_dir"] = str(Path(base_config.get("out_dir", "results")) / f"grid-{suffix_pyg}")
        
        with open(out_dir / config_name, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully generated {len(combinations)} configs in: {out_dir}")
    print("\nTo run the batch of experiments sequentially:")
    print(f"  for conf in {args.out_dir}/*.yaml; do")
    print(f"    python main_graphgym.py --cfg $conf")
    print("  done")
    print("\nTo run the batch in parallel (e.g., using xargs or parallel):")
    print(f"  ls {args.out_dir}/*.yaml | xargs -n 1 -P 4 python main_graphgym.py --cfg")

if __name__ == "__main__":
    main()
