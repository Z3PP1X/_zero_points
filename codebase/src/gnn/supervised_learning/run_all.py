#!/usr/bin/env python3
"""
End-to-end GraphGym grid-search orchestrator.

1. Generate grid configs
2. Train each configuration via main_graphgym.py
3. Aggregate results and run the full evaluation pipeline automatically
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphGym grid search with automatic post-evaluation."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Folder name under run_results/ (default: run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_supervised.yaml",
        help="Base GraphGym config YAML",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="grid.yaml",
        help="Grid search YAML",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of training jobs to run in parallel (default: 1 = sequential)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run aggregation + evaluation",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run training/aggregation only, skip plot generation",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Generate plots for all 9 run CSV variants",
    )
    parser.add_argument(
        "--skip-slices",
        action="store_true",
        help="Skip nested architecture slice plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top configs for diagnostics plots",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    gnn_root = script_dir.parent
    src_root = gnn_root.parent
    for path in (str(gnn_root), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"run_{timestamp}"
    results_dir = script_dir / "run_results" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = script_dir / "configs"
    config_base = script_dir / args.config
    grid_path = script_dir / args.grid
    train_script = script_dir / "main_graphgym.py"
    python_exe = sys.executable

    from configs_gen import generate_configs

    print(f"[Orchestrator] Experiment: {experiment_name}")
    print(f"[Orchestrator] Results dir: {results_dir}")

    print("[Orchestrator] Generating grid configuration files...")
    config_files = generate_configs(
        config_base,
        grid_path,
        configs_dir,
        results_base_dir=results_dir,
    )
    print(f"[Orchestrator] Generated {len(config_files)} configs.")

    if not args.skip_training:
        if args.parallel > 1:
            print(
                f"[Orchestrator] Parallel training ({args.parallel} workers) "
                "is not implemented yet; running sequentially."
            )

        for idx, cfg_file in enumerate(config_files, start=1):
            print(
                f"\n[Orchestrator] [{idx}/{len(config_files)}] "
                f"Training {cfg_file.name}..."
            )
            cmd = [python_exe, str(train_script), "--cfg", str(cfg_file)]
            try:
                subprocess.run(cmd, check=True, cwd=str(script_dir))
            except subprocess.CalledProcessError as exc:
                print(
                    f"[Orchestrator] Warning: training failed for {cfg_file.name} "
                    f"(exit code {exc.returncode})"
                )
    else:
        print("[Orchestrator] Skipping training (--skip-training).")

    if args.skip_eval:
        print("[Orchestrator] Skipping post-evaluation (--skip-eval).")
        return

    from gnn.supervised_learning.run_results.post_eval import run_post_evaluation

    run_post_evaluation(
        results_dir,
        configs_dir=configs_dir,
        full_runs=args.full_eval,
        skip_slices=args.skip_slices,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
