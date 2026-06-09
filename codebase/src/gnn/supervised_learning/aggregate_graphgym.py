import os
import re
from pathlib import Path
import os.path as osp

# Patch agg_runs to support 'test' as an epoch-by-epoch split
import torch_geometric.graphgym.utils.agg_runs as agg_runs_mod

def custom_is_split(s):
    return s in ['train', 'val', 'test']

BEST_METRIC = 'pr_auc'
KEYS_TO_STRIP = ['eta', 'eta_std', 'params_std']


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
                performance_np = np.array(
                    [stats[metric] for stats in stats_list])
                best_epoch = \
                    stats_list[
                        eval(f"performance_np.{cfg.metric_agg}()")][
                        'epoch']

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
        performance_np = np.array([stats[metric] for stats in val_stats])
        best_epoch = val_stats[eval(f"performance_np.{cfg.metric_agg}()")]['epoch']

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

def main():
    import sys
    script_dir = Path(__file__).resolve().parent
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
        # Check run_results first, then relative/absolute paths
        if (script_dir / "run_results" / target).exists():
            results_dir = script_dir / "run_results" / target
        elif Path(target).exists():
            results_dir = Path(target)
        else:
            results_dir = script_dir / target
    else:
        results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"Error: results directory not found at {results_dir}")
        return

    # 1. Rename directories to the PyG/GraphGym expected format if necessary
    print("Checking and renaming directories to PyG GraphGym convention...")
    for run in os.listdir(results_dir):
        if run == "agg":
            continue
        
        # Match current underscore-based format (flexible: captures all key=value pairs)
        match = re.match(
            r"grid_(.+)",
            run
        )
        if match:
            # Convert underscore-separated key_value pairs to hyphen-separated key=value pairs
            parts = match.group(1)
            # Split into key-value pairs: key1_val1_key2_val2_...
            # The pattern is: word_word_word_value where value can be numeric
            # Use a more robust approach: find known parameter names
            known_params = ['layer_type', 'layers_mp', 'dim_inner', 'dropout', 'graph_pooling', 'act', 'base_lr']
            
            remaining = parts
            extracted = {}
            for param in known_params:
                pattern = rf"{param}_([a-zA-Z0-9\.-]+)"
                m = re.search(pattern, remaining)
                if m:
                    extracted[param] = m.group(1)
            
            if extracted:
                new_name = "grid-" + "-".join(f"{k}={v}" for k, v in extracted.items())
                old_path = results_dir / run
                new_path = results_dir / new_name
                if old_path != new_path and not new_path.exists():
                    os.rename(old_path, new_path)
                    print(f"  Renamed: {run} -> {new_name}")

    print("Directories normalized. Running PyG internal aggregation...")
    
    # 2. Run seed-level aggregation on all run directories
    for run in os.listdir(results_dir):
        run_dir = results_dir / run
        if run_dir.is_dir() and run != "agg":
            try:
                agg_runs(str(run_dir), metric_best=BEST_METRIC)
            except Exception as e:
                print(f"Warning: Failed to aggregate runs for {run}: {e}")

    # 3. Run batch-level aggregation across all configurations
    try:
        agg_batch(str(results_dir), metric_best=BEST_METRIC)
        print(f"Successfully aggregated batch results inside: {results_dir / 'agg'}")
    except Exception as e:
        print(f"Error running batch aggregation: {e}")
    
    # 4. Report which CSV files were generated
    agg_dir = results_dir / "agg"
    if agg_dir.exists():
        csvs = sorted(agg_dir.glob("*.csv"))
        print(f"\nGenerated {len(csvs)} aggregated CSV files:")
        for csv in csvs:
            print(f"  - {csv.name}")

if __name__ == "__main__":
    main()
