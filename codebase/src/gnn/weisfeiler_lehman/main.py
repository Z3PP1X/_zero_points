"""CLI entry point for the 1-Weisfeiler-Lehman distinguishability study.

Loads the dataset's expression graphs exactly as a normal pipeline run would
(via :class:`UnifiedDataLoader`), but instead of training it runs the 1-WL
colour-refinement test and reports which graphs the test can and cannot tell
apart. One results subdirectory is produced per evaluated graph mode
(``graph`` / ``tree`` / ``tree-derivative``), plus a cross-mode summary.

Examples
--------
    # All three modes, default dataset, no kappa augmentation
    python main.py

    # Only the tree-derivative mode, with kappa (h-function) subgraphs merged in
    python main.py --mode tree_derivatives --add-kappa

    # Tree + tree-derivative, custom dataset and more sample drawings
    python main.py --mode tree tree_derivatives --dataset run_key/name --sample-graphs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Mirror the path bootstrap used by the other runnable scripts so the module can
# be launched directly (``python main.py``) as well as imported as a package.
_GNN_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_SRC_ROOT), str(_GNN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn.weisfeiler_lehman import visualize  # noqa: E402
from gnn.weisfeiler_lehman.wl_runner import (  # noqa: E402
    COLORING_SCHEMES,
    WLRunResult,
    global_color_histogram,
    run_wl,
)

# Internal graph-mode name -> human/results subdirectory name.
MODE_CHOICES: tuple[str, ...] = ("tree", "tree_derivatives")
MODE_TO_SUBDIR: dict[str, str] = {
    "tree": "tree",
    "tree_derivatives": "tree-derivative",
}


def _default_dataset_name(synthetic: bool = False) -> str | None:
    """Read the pipeline's default (synthetic) dataset from the supervised config."""
    cfg_path = (
        _GNN_ROOT
        / "supervised_learning"
        / "config_supervised.yaml"
    )
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if synthetic:
            return (cfg.get("expression_graph") or {}).get("synthetic_dataset")
        return (cfg.get("dataset") or {}).get("name")
    except Exception:
        return None


def _load_graphs(
    dataset_name: str,
    mode: str,
    add_kappa: bool,
    max_graphs: int | None,
    is_synthetic: bool = False,
) -> dict[str, object]:
    """Load all graphs for one mode, mirroring a normal pipeline run."""
    from gnn.shared.utils.unified_loader import UnifiedDataLoader

    try:
        loader = UnifiedDataLoader.get_instance(
            dataset_name=dataset_name,
            mode=mode,
            add_kappa=add_kappa,
            is_synthetic=is_synthetic,
        )
        graph_ids = sorted(loader.list_graph_ids())
        get_graph = loader.get_graph
    except Exception as exc:
        # Fall back to the bare graph loader (no tabular CSV needed for a
        # purely structural study).
        print(f"[wl] UnifiedDataLoader unavailable ({exc}); using GraphDataLoader.")
        from gnn.shared.utils.graph_loader import GraphDataLoader

        loader = GraphDataLoader(
            name=dataset_name,
            mode=mode,
            add_kappa=add_kappa,
            is_synthetic=is_synthetic,
        )
        graph_ids = sorted(loader.list_graph_ids())
        get_graph = loader.get_graph

    if max_graphs is not None:
        graph_ids = graph_ids[:max_graphs]
    return {gid: get_graph(gid) for gid in graph_ids}


def _write_histograms(
    result: WLRunResult, out_dir: Path, per_graph_cap: int
) -> None:
    hist_dir = out_dir / "histograms"
    (hist_dir / "per_graph").mkdir(parents=True, exist_ok=True)

    color_ids, matrix = global_color_histogram(result, round_index=-1)

    # Full per-graph histogram matrix as CSV (every graph, every final colour).
    with open(hist_dir / "histograms_matrix.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["graph_id", *[f"color_{c}" for c in color_ids]])
        for gid in result.graph_ids:
            writer.writerow([gid, *matrix[gid].tolist()])

    visualize.plot_color_heatmap(result, hist_dir / "color_heatmap.png")

    rendered = 0
    for gid in result.graph_ids:
        if rendered >= per_graph_cap:
            break
        visualize.plot_per_graph_histogram(
            gid,
            color_ids,
            matrix[gid],
            hist_dir / "per_graph" / f"{gid}.png",
            round_label="final round",
        )
        rendered += 1
    if rendered < result.num_graphs:
        print(
            f"[wl]   rendered {rendered}/{result.num_graphs} per-graph histograms "
            f"(capped by --per-graph-histograms; full data in histograms_matrix.csv)"
        )


def _write_distinguishability(result: WLRunResult, out_dir: Path) -> None:
    dist_dir = out_dir / "distinguishability"
    dist_dir.mkdir(parents=True, exist_ok=True)

    classes = result.equivalence_classes
    with open(dist_dir / "equivalence_classes.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_graphs": result.num_graphs,
                "num_classes": result.num_classes,
                "distinguishability_rate": result.distinguishability_rate,
                "classes": [
                    {"fingerprint": fp, "size": len(m), "members": m}
                    for fp, m in classes.items()
                ],
            },
            handle,
            indent=2,
        )

    # Explicit list of indistinguishable groups (the graphs that collide).
    with open(dist_dir / "collisions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "size", "members"])
        for idx, (_fp, members) in enumerate(result.colliding_classes().items()):
            writer.writerow([idx, len(members), " ".join(members)])

    visualize.plot_distinguishability_matrix(
        result, dist_dir / "distinguishability_matrix.png"
    )
    visualize.plot_class_size_distribution(result, dist_dir / "class_sizes.png")


def _write_samples(
    result: WLRunResult, graphs: dict[str, object], out_dir: Path, sample_cap: int
) -> None:
    if sample_cap <= 0:
        return
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    final_round = result.history[-1]
    # Prefer one representative from each of the largest classes for variety.
    picks: list[str] = []
    for members in result.equivalence_classes.values():
        picks.append(members[0])
        if len(picks) >= sample_cap:
            break
    for gid in picks:
        colors = final_round[gid].cpu().numpy()
        visualize.plot_sample_graph(
            graphs[gid], colors, gid, sample_dir / f"{gid}.png"
        )


def _run_mode(
    mode: str,
    dataset_name: str,
    add_kappa: bool,
    args: argparse.Namespace,
    base_out: Path,
) -> dict | None:
    print(
        f"\n[wl] === mode '{mode}' "
        f"(add_kappa={add_kappa}, synthetic={args.synthetic}) ==="
    )
    graphs = _load_graphs(
        dataset_name,
        mode,
        add_kappa,
        args.max_graphs,
        is_synthetic=args.synthetic,
    )
    if not graphs:
        print(f"[wl] no graphs found for mode '{mode}'; skipping.")
        return None
    print(f"[wl] loaded {len(graphs)} graphs.")

    result = run_wl(
        graphs,
        coloring=args.coloring,
        symmetrize=not args.directed,
        max_iterations=args.iterations,
    )
    print(
        f"[wl] {result.iterations} refinement rounds -> "
        f"{result.num_classes}/{result.num_graphs} distinguishable classes "
        f"(rate {result.distinguishability_rate:.3f}); "
        f"{result.num_colliding_graphs()} graphs in collisions."
    )

    subdir = (
        ("synthetic-" if args.synthetic else "")
        + MODE_TO_SUBDIR[mode]
        + ("-kappa" if add_kappa else "")
    )
    out_dir = base_out / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_histograms(result, out_dir, args.per_graph_histograms)
    _write_distinguishability(result, out_dir)
    _write_samples(result, graphs, out_dir, args.sample_graphs)

    summary = {
        "label": subdir,
        "mode": mode,
        "add_kappa": add_kappa,
        "synthetic": args.synthetic,
        "dataset": dataset_name,
        "coloring": args.coloring,
        "symmetrized": not args.directed,
        "iterations": result.iterations,
        "num_graphs": result.num_graphs,
        "num_classes": result.num_classes,
        "distinguishability_rate": result.distinguishability_rate,
        "num_colliding_graphs": result.num_colliding_graphs(),
        "colliding_groups": [
            m for m in result.colliding_classes().values()
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


# Column headers and minimum widths for the distinguishability summary table.
_TABLE_HEADERS = ("mode", "distinguishable classes", "colliding graphs", "rate")
_TABLE_MIN_WIDTHS = (27, 25, 18, 6)


def format_summary_table(summaries: list[dict]) -> str:
    """Render the per-mode results as a box-drawing distinguishability table."""
    rows = [
        (
            row["label"],
            f"{row['num_classes']} / {row['num_graphs']}",
            str(row["num_colliding_graphs"]),
            f"{row['distinguishability_rate']:.2f}",
        )
        for row in summaries
    ]
    widths = []
    for col, (header, min_w) in enumerate(zip(_TABLE_HEADERS, _TABLE_MIN_WIDTHS)):
        content_w = max([len(r[col]) + 1 for r in rows], default=0)
        widths.append(max(min_w, content_w, len(header) + 2))

    # The header has no leading border, so widen its first cell by one to keep its
    # separators aligned with the bordered body rows below.
    header_cells = [_TABLE_HEADERS[i].center(widths[i]) for i in range(4)]
    header_cells[0] = _TABLE_HEADERS[0].center(widths[0] + 1)
    header_line = "│".join(header_cells) + "│"
    separator = "├" + "┼".join("─" * w for w in widths) + "┤"

    lines = [header_line]
    for r in rows:
        cells = [(" " + r[i]).ljust(widths[i]) for i in range(4)]
        lines.append(separator)
        lines.append("│" + "│".join(cells) + "│")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="1-Weisfeiler-Lehman distinguishability study over expression graphs.",  # noqa: E501
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (run_key/name). Defaults to the supervised config.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=MODE_CHOICES,
        default=list(MODE_CHOICES),
        help="Graph mode(s) to evaluate; each gets its own results subdirectory.",
    )
    parser.add_argument(
        "--add-kappa",
        action="store_true",
        help="Merge kappa (h-function) subgraphs from datasets/kappas/ per graph.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Load the synthetic dataset (datasets/graphs/synthetic_graphs.json); "
        "results go to synthetic-<mode> subdirs.",
    )
    parser.add_argument(
        "--coloring",
        type=str,
        default="label",
        choices=list(COLORING_SCHEMES),
        help="Initial 1-WL node colouring (label=semantic operator/operand label).",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Refine on directed edges (default: symmetrize to undirected 1-WL).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Max WL refinement rounds (stops early once the partition is stable).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base results directory (default: <module>/results).",
    )
    parser.add_argument(
        "--per-graph-histograms",
        type=int,
        default=50,
        help="Cap on individual per-graph histogram PNGs (the CSV always holds all).",
    )
    parser.add_argument(
        "--sample-graphs",
        type=int,
        default=6,
        help="Number of sample graph drawings (one per largest class). 0 disables.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on number of graphs loaded (for quick runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_name = args.dataset or _default_dataset_name(synthetic=args.synthetic)
    if not dataset_name:
        print("[wl] No dataset given and none found in config_supervised.yaml.")
        return 2

    base_out = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent / "results"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    print(
        f"[wl] dataset='{dataset_name}'  modes={args.mode}  "
        f"add_kappa={args.add_kappa}  synthetic={args.synthetic}"
    )
    print(f"[wl] results -> {base_out}")

    summaries: list[dict] = []
    for mode in args.mode:
        summary = _run_mode(
            mode, dataset_name, args.add_kappa, args, base_out
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("[wl] no modes produced results.")
        return 1

    # Per-dataset slug so synthetic and curated runs into the same results dir
    # do not overwrite each other's base-level artifacts.
    slug = ("synthetic" if args.synthetic else "curated") + (
        "-kappa" if args.add_kappa else ""
    )

    with open(
        base_out / f"study_summary_{slug}.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "synthetic": args.synthetic,
                "add_kappa": args.add_kappa,
                "modes": summaries,
            },
            handle,
            indent=2,
        )

    table = format_summary_table(summaries)
    with open(base_out / f"study_table_{slug}.txt", "w", encoding="utf-8") as handle:
        handle.write(table + "\n")

    if len(summaries) > 1:
        visualize.plot_mode_summary(
            summaries, base_out / f"mode_comparison_{slug}.png"
        )

    print(f"\n[wl] === study summary ({slug}) ===")
    print(table)
    print(f"[wl] done. See {base_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
