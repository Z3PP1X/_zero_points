import os
import re
import sys
from pathlib import Path
import os.path as osp

_script_dir = Path(__file__).resolve().parent
_gnn_root = _script_dir.parent
_src_root = _gnn_root.parent
for _path in (str(_gnn_root), str(_src_root)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Patch agg_runs to support 'test' as an epoch-by-epoch split
import torch_geometric.graphgym.utils.agg_runs as agg_runs_mod

def custom_is_split(s):
    return s in ['train', 'val', 'test']

BEST_METRIC = 'pr_auc'
KEYS_TO_STRIP = ['eta', 'eta_std', 'params_std']

from gnn.supervised_learning.run_results.eval_metrics import select_best_epoch


def _resolve_best_metric(metric_best, stats_list):
    if metric_best == 'auto':
        if 'pr_auc' in stats_list[0]:
            return 'pr_auc'
        if 'auc' in stats_list[0]:
            return 'auc'
        return 'accuracy'
    return metric_best


def custom_agg_runs(dir, metric_best='auto'):
    import numpy as np
    from torch_geometric.graphgym.utils.agg_runs import json_to_dict_list, join_list, agg_dict_list, makedirs_rm_exist, dict_list_to_json, dict_list_to_tb, dict_to_json, SummaryWriter, is_seed
    from torch_geometric.graphgym.config import cfg
    
    results = {'train': None, 'val': None, 'test': None}
    results_best = {'train': None, 'val': None, 'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = osp.join(dir, seed)

            split = 'val'
            if split in os.listdir(dir_seed):
                dir_split = osp.join(dir_seed, split)
                fname_stats = osp.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                metric = _resolve_best_metric(metric_best, stats_list)
                best_epoch = select_best_epoch(
                    stats_list, metric, cfg.metric_agg
                )

            for split in os.listdir(dir_seed):
                if custom_is_split(split):
                    dir_split = osp.join(dir_seed, split)
                    fname_stats = osp.join(dir_split, 'stats.json')
                    if not osp.exists(fname_stats):
                        continue
                    stats_list = json_to_dict_list(fname_stats)
                    if not stats_list:
                        continue
                    # Find stats for best epoch
                    stats_best_candidates = [
                        stats for stats in stats_list
                        if stats['epoch'] == best_epoch
                    ]
                    if not stats_best_candidates:
                        # Fallback to last epoch if best_epoch is not found in stats
                        stats_best = stats_list[-1]
                    else:
                        stats_best = stats_best_candidates[0]
                        
                    stats_list = [[stats] for stats in stats_list]
                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        results[split] = join_list(results[split], stats_list)
                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]
                        
    results = {k: v for k, v in results.items() if v is not None}
    results_best = {k: v for k, v in results_best.items() if v is not None}
    for key in results:
        for i in range(len(results[key])):
            results[key][i] = agg_dict_list(results[key][i])
    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key])
        
    for key, value in results.items():
        dir_out = osp.join(dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = osp.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        if cfg.tensorboard_agg:
            if SummaryWriter is not None:
                writer = SummaryWriter(dir_out)
                dict_list_to_tb(value, writer)
                writer.close()
                
    for key, value in results_best.items():
        dir_out = osp.join(dir, 'agg', key)
        fname = osp.join(dir_out, 'best.json')
        dict_to_json(value, fname)

def custom_agg_batch(dir, metric_best='auto'):
    """Batch aggregation aligned with checkpoint selection (pr_auc) and full logging."""
    import numpy as np
    import pandas as pd
    from torch_geometric.graphgym.utils.agg_runs import (
        name_to_dict,
        json_to_dict_list,
        makedirs_rm_exist,
    )
    from torch_geometric.graphgym.config import cfg

    def _strip_keys(stats):
        for key in KEYS_TO_STRIP:
            stats.pop(key, None)
        return stats

    # best.json → *_best.csv (metrics at best val epoch per seed, then mean-aggregated)
    results = {'train': [], 'val': [], 'test': []}
    for run in os.listdir(dir):
        if run == 'agg':
            continue
        dict_name = name_to_dict(run)
        dir_run = osp.join(dir, run, 'agg')
        if not osp.isdir(dir_run):
            continue
        for split in os.listdir(dir_run):
            fname_stats = osp.join(dir_run, split, 'best.json')
            if not osp.exists(fname_stats):
                continue
            dict_stats = json_to_dict_list(fname_stats)[-1]
            results[split].append({**dict_name, **_strip_keys(dict_stats)})

    dir_out = osp.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    sort_keys = None
    for key in results:
        if len(results[key]) == 0:
            continue
        results[key] = pd.DataFrame(results[key])
        if sort_keys is None and len(results[key].columns):
            sort_keys = [c for c in results[key].columns if c not in (
                'epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1',
                'auc', 'pr_auc', 'lr', 'base_lr', 'params', 'time_iter',
                'gpu_memory', 'loss_std', 'accuracy_std', 'precision_std',
                'recall_std', 'f1_std', 'auc_std', 'pr_auc_std', 'lr_std',
                'base_lr_std', 'params_std', 'time_iter_std',
            )]
        if sort_keys:
            present = [k for k in sort_keys if k in results[key].columns]
            if present:
                results[key] = results[key].sort_values(
                    present, ascending=[True] * len(present)
                )
        results[key].to_csv(osp.join(dir_out, f'{key}_best.csv'), index=False)

    # stats.json → *.csv (last epoch per configuration)
    results = {'train': [], 'val': [], 'test': []}
    for run in os.listdir(dir):
        if run == 'agg':
            continue
        dict_name = name_to_dict(run)
        dir_run = osp.join(dir, run, 'agg')
        if not osp.isdir(dir_run):
            continue
        for split in os.listdir(dir_run):
            fname_stats = osp.join(dir_run, split, 'stats.json')
            if not osp.exists(fname_stats):
                continue
            dict_stats = json_to_dict_list(fname_stats)[-1]
            results[split].append({**dict_name, **_strip_keys(dict_stats)})

    for key in results:
        if len(results[key]) == 0:
            continue
        results[key] = pd.DataFrame(results[key])
        if sort_keys:
            present = [k for k in sort_keys if k in results[key].columns]
            if present:
                results[key] = results[key].sort_values(
                    present, ascending=[True] * len(present)
                )
        results[key].to_csv(osp.join(dir_out, f'{key}.csv'), index=False)

    # stats.json → *_bestepoch.csv (best epoch by pr_auc on synthetic val split)
    results = {'train': [], 'val': [], 'test': []}
    for run in os.listdir(dir):
        if run == 'agg':
            continue
        dict_name = name_to_dict(run)
        dir_run = osp.join(dir, run, 'agg')
        if not osp.isdir(dir_run):
            continue
        val_stats_path = osp.join(dir_run, 'val', 'stats.json')
        if not osp.exists(val_stats_path):
            continue
        val_stats = json_to_dict_list(val_stats_path)
        metric = _resolve_best_metric(metric_best, val_stats)
        best_epoch = select_best_epoch(val_stats, metric, cfg.metric_agg)

        for split in os.listdir(dir_run):
            fname_stats = osp.join(dir_run, split, 'stats.json')
            if not osp.exists(fname_stats):
                continue
            dict_stats_list = json_to_dict_list(fname_stats)
            matches = [s for s in dict_stats_list if s['epoch'] == best_epoch]
            dict_stats = matches[0] if matches else dict_stats_list[-1]
            results[split].append({**dict_name, **_strip_keys(dict_stats)})

    for key in results:
        if len(results[key]) == 0:
            continue
        results[key] = pd.DataFrame(results[key])
        if sort_keys:
            present = [k for k in sort_keys if k in results[key].columns]
            if present:
                results[key] = results[key].sort_values(
                    present, ascending=[True] * len(present)
                )
        results[key].to_csv(osp.join(dir_out, f'{key}_bestepoch.csv'), index=False)

    print(f'Results aggregated across models saved in {dir_out}')


agg_runs_mod.is_split = custom_is_split
agg_runs_mod.agg_runs = custom_agg_runs
agg_runs_mod.agg_batch = custom_agg_batch

from torch_geometric.graphgym.utils.agg_runs import agg_runs, agg_batch
agg_runs = custom_agg_runs
agg_batch = custom_agg_batch

KNOWN_GRID_PARAMS = [
    "layer_type",
    "layers_mp",
    "dim_inner",
    "dropout",
    "graph_pooling",
    "act",
    "base_lr",
]


def _normalize_run_directories(results_dir: Path):
    """Rename grid_* folders to GraphGym's grid-key=value convention."""
    print("Checking and renaming directories to PyG GraphGym convention...")
    for run in os.listdir(results_dir):
        if run == "agg":
            continue

        match = re.match(r"grid_(.+)", run)
        if not match:
            continue

        parts = match.group(1)
        extracted = {}
        for param in KNOWN_GRID_PARAMS:
            pattern = rf"{param}_([a-zA-Z0-9\.-]+)"
            m = re.search(pattern, parts)
            if m:
                extracted[param] = m.group(1)

        if extracted:
            new_name = "grid-" + "-".join(f"{k}={v}" for k, v in extracted.items())
            old_path = results_dir / run
            new_path = results_dir / new_name
            if old_path != new_path and not new_path.exists():
                os.rename(old_path, new_path)
                print(f"  Renamed: {run} -> {new_name}")


def aggregate_results(results_dir: str | Path) -> Path:
    """
    Aggregate GraphGym run outputs into CSVs under {results_dir}/agg/.

    Returns:
        Path to the agg directory.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found at {results_dir}")

    _normalize_run_directories(results_dir)

    print("Directories normalized. Running PyG internal aggregation...")
    for run in os.listdir(results_dir):
        run_dir = results_dir / run
        if run_dir.is_dir() and run != "agg":
            try:
                agg_runs(str(run_dir), metric_best=BEST_METRIC)
            except Exception as e:
                print(f"Warning: Failed to aggregate runs for {run}: {e}")

    try:
        agg_batch(str(results_dir), metric_best=BEST_METRIC)
        print(f"Successfully aggregated batch results inside: {results_dir / 'agg'}")
    except Exception as e:
        print(f"Error running batch aggregation: {e}")

    agg_dir = results_dir / "agg"
    if agg_dir.exists():
        csvs = sorted(agg_dir.glob("*.csv"))
        print(f"\nGenerated {len(csvs)} aggregated CSV files:")
        for csv in csvs:
            print(f"  - {csv.name}")

    return agg_dir


def main():
    import argparse
    import sys

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Aggregate GraphGym grid-search results into CSV files."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Experiment folder name or path (default: results/)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="After aggregation, run the full post-evaluation plot pipeline",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Config YAML directory for top-config diagnostics",
    )
    parser.add_argument("--full", action="store_true", help="Evaluate all 9 run CSVs")
    parser.add_argument(
        "--skip-slices",
        action="store_true",
        help="Skip nested architecture slice plots during evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configs for diagnostics plots",
    )
    args = parser.parse_args(sys.argv[1:] or None)

    if args.target:
        target = args.target
        if (script_dir / "run_results" / target).exists():
            results_dir = script_dir / "run_results" / target
        elif Path(target).exists():
            results_dir = Path(target)
        else:
            results_dir = script_dir / target
    else:
        results_dir = script_dir / "results"

    try:
        aggregate_results(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    if args.eval:
        configs_dir = args.configs_dir
        if configs_dir is None and (script_dir / "configs").exists():
            configs_dir = script_dir / "configs"

        from gnn.supervised_learning.run_results.post_eval import run_post_evaluation

        run_post_evaluation(
            results_dir,
            configs_dir=configs_dir,
            full_runs=args.full,
            skip_slices=args.skip_slices,
            top_k=args.top_k,
            skip_aggregation=True,
        )


if __name__ == "__main__":
    main()
