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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _run_training_job(python_exe: str, train_script: str, cfg_file: str, script_dir: str) -> tuple[str, int]:
    """Run a single GraphGym training job in an isolated subprocess."""
    cmd = [python_exe, train_script, "--cfg", cfg_file]
    try:
        subprocess.run(cmd, check=True, cwd=script_dir)
        return Path(cfg_file).name, 0
    except subprocess.CalledProcessError as exc:
        return Path(cfg_file).name, exc.returncode


def _train_configs(
    config_files: list[Path],
    python_exe: str,
    train_script: Path,
    script_dir: Path,
    parallel: bool,
    num_workers: int,
):
    total = len(config_files)
    if parallel and num_workers > 1:
        workers = min(num_workers, total)
        print(f"[Orchestrator] Parallel training: {workers} workers, {total} configs")
        jobs = [
            (python_exe, str(train_script), str(cfg_file), str(script_dir))
            for cfg_file in config_files
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_training_job, *job) for job in jobs]
            for future in as_completed(futures):
                name, exit_code = future.result()
                completed += 1
                if exit_code != 0:
                    print(
                        f"[Orchestrator] [{completed}/{total}] "
                        f"Warning: {name} failed (exit code {exit_code})"
                    )
                else:
                    print(f"[Orchestrator] [{completed}/{total}] Finished {name}")
        return

    print(f"[Orchestrator] Sequential training: {total} configs")
    for idx, cfg_file in enumerate(config_files, start=1):
        print(f"\n[Orchestrator] [{idx}/{total}] Training {cfg_file.name}...")
        name, exit_code = _run_training_job(
            python_exe,
            str(train_script),
            str(cfg_file),
            str(script_dir),
        )
        if exit_code != 0:
            print(
                f"[Orchestrator] Warning: {name} failed (exit code {exit_code})"
            )


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
        action="store_true",
        help="Run training jobs in parallel (default: sequential)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel training jobs when --parallel is set (default: 1)",
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

    if args.num < 1:
        parser.error("--num must be at least 1")
    
    if args.parallel and args.num > 2:
        print(f"[Orchestrator] Warning: --num {args.num} requested, but running >2 experiments concurrently can cause CPU starvation. Capping to 2.")
        args.num = 2

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
        _train_configs(
            config_files=config_files,
            python_exe=python_exe,
            train_script=train_script,
            script_dir=script_dir,
            parallel=args.parallel,
            num_workers=args.num,
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
