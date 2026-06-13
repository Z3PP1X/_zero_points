"""
Post-training evaluation pipeline for GraphGym grid-search results.

Runs aggregation, CSV-based plots, training curves, and top-config diagnostics.
"""

import sys
from pathlib import Path


def _ensure_import_paths():
    script_dir = Path(__file__).resolve().parent
    gnn_root = script_dir.parents[1]
    src_root = gnn_root.parent
    for path in (str(gnn_root), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_results_dir(target: str | Path, script_dir: Path | None = None) -> Path:
    """Resolve an experiment results directory from a name or path."""
    target = Path(target)
    if target.exists():
        return target.resolve()

    base = script_dir or Path(__file__).resolve().parent
    supervised_dir = base.parent

    candidates = [
        base / target,
        supervised_dir / "run_results" / target,
        supervised_dir / target,
        supervised_dir / "results" / target,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Results directory not found for '{target}'")


def run_post_evaluation(
    results_dir: str | Path,
    configs_dir: str | Path | None = None,
    *,
    full_runs: bool = False,
    skip_slices: bool = False,
    top_k: int = 5,
    skip_diagnostics: bool = False,
    skip_training_curves: bool = False,
    skip_aggregation: bool = False,
    skip_report: bool = False,
):
    """
    Run the full post-training pipeline on a completed GraphGym experiment.

    Steps:
      1. Aggregate seed/run stats into CSVs (agg/)
      2. Generate heatmaps, summary bars, leaderboard, split comparison
      3. Plot training curves from stats.json (overview + per-config + top-K)
      4. Reload top-K checkpoints for confusion matrix / ROC / PR curves
    """
    results_dir = Path(results_dir).resolve()
    experiment_name = results_dir.name
    eval_output_dir = results_dir / "eval_plots"
    configs_dir = Path(configs_dir).resolve() if configs_dir else None

    _ensure_import_paths()

    print(f"\n{'=' * 72}")
    print(f"[PostEval] Starting pipeline for: {results_dir}")
    print(f"{'=' * 72}")

    if not skip_aggregation:
        from gnn.supervised_learning.aggregate_graphgym import aggregate_results

        aggregate_results(results_dir)

    from gnn.supervised_learning.run_results.eval import GNNResultEvaluator

    runs = (
        GNNResultEvaluator.ALL_RUNS if full_runs else GNNResultEvaluator.DEFAULT_RUNS
    )
    evaluator = GNNResultEvaluator(
        naming_var=experiment_name,
        base_dir=results_dir.parent,
        runs=runs,
        skip_slices=skip_slices,
        top_k=top_k,
    )
    evaluator.output_dir = eval_output_dir
    evaluator.run_all()

    if not skip_training_curves:
        print("  Generating training curves...")
        from gnn.supervised_learning.run_results.training_curves import (
            TrainingCurvePlotter,
        )

        curve_plotter = TrainingCurvePlotter(
            results_dir=results_dir,
            output_dir=eval_output_dir,
            experiment_name=experiment_name,
        )
        curve_plotter.plot_overview()
        curve_plotter.plot_all_configs()
        curve_plotter.plot_top_configs(eval_output_dir / "leaderboard.csv", top_k=top_k)

    if not skip_diagnostics:
        print("  Generating top-config diagnostics (CM / ROC / PR)...")
        from gnn.supervised_learning.run_results.diagnostics import DiagnosticPlotter

        diag_plotter = DiagnosticPlotter(
            results_dir=results_dir,
            output_dir=eval_output_dir,
            experiment_name=experiment_name,
            configs_dir=configs_dir,
            top_k=top_k,
        )
        diag_plotter.run_top_configs()

    print("  Aggregating feature-importance plots across configurations...")
    from gnn.supervised_learning.run_results.feature_importance import (
        aggregate_feature_importance_plots,
    )

    for split_name in ("val_synthetic", "val_curated", "val"):
        aggregate_feature_importance_plots(
            results_dir,
            eval_output_dir / "feature_importance",
            split=split_name,
        )

    if not skip_report:
        print("  Generating auto summary report (summary.md / summary.json)...")
        from gnn.supervised_learning.run_results.report import generate_report

        try:
            generate_report(results_dir, output_dir=eval_output_dir, top_k=top_k)
        except Exception as exc:
            print(f"  Warning: summary report generation failed: {exc}")

    print(f"\n[PostEval] Complete. Outputs in: {eval_output_dir}")
    return eval_output_dir


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run aggregation and full evaluation for GraphGym results."
    )
    parser.add_argument(
        "results_dir",
        help="Experiment folder (name under run_results/ or full path)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Directory with generated grid config YAMLs (for checkpoint reload)",
    )
    parser.add_argument("--full", action="store_true", help="Evaluate all 9 run CSVs")
    parser.add_argument(
        "--skip-slices",
        action="store_true",
        help="Skip nested architecture slice plots",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top configs for diagnostics")
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip confusion matrix / ROC / PR inference plots",
    )
    parser.add_argument(
        "--skip-training-curves",
        action="store_true",
        help="Skip training curve plots",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip the auto summary report (summary.md / summary.json)",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    results_dir = resolve_results_dir(args.results_dir, script_dir)
    configs_dir = args.configs_dir
    if configs_dir is None:
        default_configs = script_dir.parent / "configs"
        if default_configs.exists():
            configs_dir = default_configs

    run_post_evaluation(
        results_dir,
        configs_dir=configs_dir,
        full_runs=args.full,
        skip_slices=args.skip_slices,
        top_k=args.top_k,
        skip_diagnostics=args.skip_diagnostics,
        skip_training_curves=args.skip_training_curves,
        skip_report=args.skip_report,
    )


if __name__ == "__main__":
    main()
