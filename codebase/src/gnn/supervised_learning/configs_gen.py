import argparse
import copy
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


def path_exists(d, key) -> bool:
    """True if every segment of the dotted key resolves in the nested mapping."""
    cur = d
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    return True


def validate_grid_keys(base_config: dict, grid_config: dict) -> None:
    """Fail fast if a grid axis names a path absent from the base config.

    GraphGym merges the generated config with yacs in strict mode, so an unknown key
    (e.g. ``expression_graph.graph`` instead of ``expression_graph.mode``) only surfaces
    deep inside ``load_cfg`` at train time as ``Non-existent config key``. Catching it here
    points at the offending grid axis with the closest valid siblings.
    """
    unknown = [key for key in grid_config if not path_exists(base_config, key)]
    if not unknown:
        return

    lines = ["Grid axis/axes not found in the base config:"]
    for key in unknown:
        parent = key.rsplit(".", 1)[0] if "." in key else None
        siblings = get_nested_value(base_config, parent) if parent else base_config
        valid = sorted(siblings) if isinstance(siblings, dict) else []
        hint = f" — valid keys under '{parent}': {valid}" if valid else ""
        lines.append(f"  - {key}{hint}")
    raise ValueError("\n".join(lines))


def _is_valid_config(config: dict) -> bool:
    """Reject semantically invalid combinations a grid can emit before they cost a run.

    The anchor positional encoding and the fully-connected virtual supernode are mutually
    exclusive: the supernode collapses every pairwise distance to <=2 hops, destroying the
    shortest-path anchor distances, so the trainer raises ``PositionalSupernodeConflictError``
    (``validate_positional_supernode_compatibility`` in ``feature_config.py``). A stage grid
    that sweeps ``add_virtual_supernode`` and ``features.positional`` independently would
    otherwise generate configs that crash at load time. Mirror that rule here.
    """
    eg = config.get("expression_graph", {}) or {}
    supernode = bool(eg.get("add_virtual_supernode", False))
    features = eg.get("features", {}) if isinstance(eg.get("features"), dict) else {}
    # Mirror the runtime resolver (feature_config._resolve_category_value): a missing key
    # OR an explicit ``None`` both mean "all anchor members enabled" (positional ON). Only
    # ``False`` / an empty list mean OFF. Getting this wrong would let a supernode + None
    # config slip past the filter and then crash at load with PositionalSupernodeConflictError.
    positional = features.get("positional", None)
    if positional is None:
        positional_on = True
    elif isinstance(positional, list):
        positional_on = len(positional) > 0
    else:
        positional_on = bool(positional)
    return not (supernode and positional_on)


def _canonical_signature(config: dict, swept_keys: list[str]) -> tuple:
    """Signature of a config over the swept axes for deduplication in generate_configs()."""
    sig = {key: get_nested_value(config, key) for key in swept_keys}
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

    validate_grid_keys(base_config, grid_config)

    keys = list(grid_config.keys())
    value_lists = [grid_config[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    configs_out_dir.mkdir(parents=True, exist_ok=True)
    for cfg_file in configs_out_dir.glob("*.yaml"):
        try:
            cfg_file.unlink()
        except OSError:
            pass

    # Run folders nest directly under the results root. When run_all passes a
    # results_base_dir it IS the experiment dir that aggregation later scans, so the run
    # folders must be its children — do NOT strip it. (A previous startswith("run_"/"grid-")
    # strip orphaned run folders as siblings whenever the experiment used the default
    # ``run_<ts>`` name, leaving aggregation with nothing to read.)
    results_root = (
        Path(results_base_dir)
        if results_base_dir is not None
        else Path(base_config.get("out_dir", "results"))
    )

    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    created = []
    seen_signatures: set[tuple] = set()
    invalid_skipped = 0
    index = 0
    for combination in combinations:
        config = copy.deepcopy(base_config)
        for key, val in zip(keys, combination):
            set_nested_value(config, key, val)

        if not _is_valid_config(config):
            invalid_skipped += 1
            continue  # would crash at load time (e.g. supernode + positional)

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

    if invalid_skipped:
        print(
            f"  Skipped {invalid_skipped} invalid supernode+positional combination(s) "
            f"(mutually exclusive)."
        )
    dedup_skipped = len(combinations) - invalid_skipped - len(created)
    if dedup_skipped:
        valid_points = len(combinations) - invalid_skipped
        print(
            f"  Deduplicated {dedup_skipped} inert pooling combination(s) "
            f"({valid_points} valid grid points -> {len(created)} distinct configs)."
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
