import argparse
from datetime import datetime
from pathlib import Path
import itertools
import yaml


def set_nested_value(d, key, value):
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_nested_value(d, key):
    cur = d
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _canonical_signature(config: dict, swept_keys: list[str]) -> tuple:
    """Signature of a config over the swept axes, nullifying inert combinations.

    The pooling axes interact: ``variant=legacy`` ignores ``pool_type``, and
    ``pool_type=topk`` ignores ``aux_loss_weight``. Blindly crossing them emits runs that
    train to identical behaviour. Collapsing those inert axes to ``None`` here lets
    generate_configs() skip the duplicate before it costs a full training run.
    """
    gnn = config.get("gnn", {}) or {}
    variant = gnn.get("variant")
    pool_type = gnn.get("pool_type")

    sig = {}
    for key in swept_keys:
        short = key.split(".")[-1]
        value = get_nested_value(config, key)
        if short == "pool_type" and variant == "legacy":
            value = None  # legacy variant ignores pool_type entirely
        elif short == "aux_loss_weight" and (variant == "legacy" or pool_type == "topk"):
            value = None  # aux loss is a DiffPool-only term
        sig[key] = value
    return tuple(sorted((k, str(v)) for k, v in sig.items()))


def generate_configs(
    config_path: Path,
    grid_path: Path,
    configs_out_dir: Path,
    results_base_dir: Path | None = None,
    run_timestamp: str | None = None,
) -> list[Path]:
    """Generate GraphGym grid configs and return the created YAML paths.

    Filenames and run folders are datetime+index based (``cfg_<ts>_<NNN>.yaml`` /
    ``run_<ts>_<NNN>``) rather than param-encoded, so they stay short as the grid grows.
    Each generated config carries the swept values internally and is snapshotted into its
    run folder as config.yaml at train time, so reproducibility does not depend on the
    name. ``run_timestamp`` groups one grid's configs under a shared stamp (run_all passes
    its experiment timestamp); when omitted a fresh stamp is generated.
    """
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

    results_root = (
        Path(results_base_dir)
        if results_base_dir is not None
        else Path(base_config.get("out_dir", "results"))
    )
    if results_root.name.startswith(("grid-", "run_")):
        results_root = results_root.parent

    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    created = []
    seen_signatures: set[tuple] = set()
    index = 0
    for combination in combinations:
        config = copy.deepcopy(base_config)
        for key, val in zip(keys, combination):
            set_nested_value(config, key, val)

        signature = _canonical_signature(config, keys)
        if signature in seen_signatures:
            continue  # behaviourally identical to an already-emitted config
        seen_signatures.add(signature)

        stem = f"{timestamp}_{index:03d}"
        config["out_dir"] = str(results_root / f"run_{stem}")

        dest_path = configs_out_dir / f"cfg_{stem}.yaml"
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        created.append(dest_path)
        index += 1

    skipped = len(combinations) - len(created)
    if skipped:
        print(
            f"  Deduplicated {skipped} inert pooling combination(s) "
            f"({len(combinations)} grid points -> {len(created)} distinct configs)."
        )

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
