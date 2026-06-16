#!/usr/bin/env python
"""
Benchmark: GINConv inference time vs. graph size across 8 structural variants.

Measures median inference time for 107 graphs (7 real + 100 synthetic) under
each structural variant.  Outputs:
  - benchmark_results.csv  — raw per-(graph, structure) measurements
  - benchmark_config.json  — reproducibility metadata
  - inference_time_vs_size.{svg,pdf}  — log-log plot with power-law fits

Usage:
    conda activate pytorch
    python codebase/src/gnn/benchmark_inference_time.py [--output-dir DIR] \\
        [--x-axis processed|base] [--n-warmup 10] [--n-steps 100] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1]   # codebase/src
_REPO = Path(__file__).resolve().parents[3]  # repo root (_zero_points)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

from gnn.shared.utils.graph_loader import GraphDataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Structural variants ───────────────────────────────────────────────────────

STRUCTURES: list[dict] = [
    dict(id=1, label="tree (top-down)",       mode="tree",             edge_direction="top_down",      add_kappa=False, add_virtual_supernode=False),
    dict(id=2, label="tree (bidirectional)",   mode="tree",             edge_direction="bidirectional", add_kappa=False, add_virtual_supernode=False),
    dict(id=3, label="tree + kappa",           mode="tree",             edge_direction="top_down",      add_kappa=True,  add_virtual_supernode=False),
    dict(id=4, label="tree-deriv",             mode="tree_derivatives", edge_direction="top_down",      add_kappa=False, add_virtual_supernode=False),
    dict(id=5, label="tree-deriv + supernode", mode="tree_derivatives", edge_direction="top_down",      add_kappa=False, add_virtual_supernode=True),
    dict(id=6, label="tree-deriv + kappa",     mode="tree_derivatives", edge_direction="top_down",      add_kappa=True,  add_virtual_supernode=False),
    dict(id=7, label="graph",                  mode="graph",            edge_direction="top_down",      add_kappa=False, add_virtual_supernode=False),
    dict(id=8, label="graph + kappa",          mode="graph",            edge_direction="top_down",      add_kappa=True,  add_virtual_supernode=False),
]


# ─── Model ────────────────────────────────────────────────────────────────────

def _gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GINBenchmarkModel(nn.Module):
    """3-layer GINConv + global mean pooling.  Architecture is identical across
    all structural variants; only input_dim may differ when the feature schema
    changes (e.g. supernode disables anchor PE)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            GINConv(_gin_mlp(input_dim, hidden_dim)),
            GINConv(_gin_mlp(hidden_dim, hidden_dim)),
            GINConv(_gin_mlp(hidden_dim, hidden_dim)),
        ])
        self.act = nn.ReLU()

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.float()
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        return global_mean_pool(x, batch)


# ─── Timing ───────────────────────────────────────────────────────────────────

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_inference_time(
    model: nn.Module,
    data: Data,
    device: torch.device,
    n_warmup: int,
    n_steps: int,
) -> tuple[float, float]:
    """Return (t_median_s, t_iqr_s).

    GPU: torch.cuda.synchronize() wraps every step so host timing reflects
    kernel completion, not just kernel launch.
    """
    data = data.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(data)
        _sync(device)

        times: list[float] = []
        for _ in range(n_steps):
            _sync(device)
            t0 = time.perf_counter()
            model(data)
            _sync(device)
            times.append(time.perf_counter() - t0)

    arr = np.array(times)
    t_median = float(np.median(arr))
    t_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return t_median, t_iqr


# ─── Graph loading ─────────────────────────────────────────────────────────────

def _load_kappa_map(csv_path: Path) -> dict[str, float]:
    """Read {Problem_ID: kappa_value} from a benchmark CSV."""
    kappa_map: dict[str, float] = {}
    if not csv_path.exists():
        return kappa_map
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get("Problem_ID") or row.get("problem_id")
            kappa_str = row.get("kappa")
            if pid and kappa_str is not None:
                try:
                    kappa_map[pid] = float(kappa_str)
                except (ValueError, TypeError):
                    pass
    return kappa_map


def _build_kappa_maps() -> tuple[dict[str, float], dict[str, float]]:
    """Return (real_kappa_map, synth_kappa_map) loaded from benchmark CSVs."""
    real_csv = _REPO / "datasets" / "run_20260603_123013" / "parallel_benchmark_results.csv"
    synth_csv = _REPO / "datasets" / "run_20260604_154509" / "parallel_benchmark_results.csv"
    return _load_kappa_map(real_csv), _load_kappa_map(synth_csv)


def _make_loader(
    is_synthetic: bool,
    mode: str,
    edge_direction: str,
    add_kappa: bool,
    add_virtual_supernode: bool,
    kappa_map: dict[str, float] | None = None,
) -> GraphDataLoader:
    return GraphDataLoader(
        "graphs",
        mode=mode,
        edge_direction=edge_direction,
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
        is_synthetic=is_synthetic,
        kappa_map=kappa_map,
    )


def _build_base_sizes(
    real_kappa_map: dict[str, float],
    synth_kappa_map: dict[str, float],
) -> dict[tuple[str, str], int]:
    """Load all graphs in their base form (top_down, no kappa, no supernode)
    and record |V|+|E|.  Keyed by (mode_synth_tag, graph_id)."""
    base_sizes: dict[tuple[str, str], int] = {}
    for mode in ("tree", "tree_derivatives", "graph"):
        for is_synth in (False, True):
            tag = f"{mode}_synth" if is_synth else mode
            try:
                loader = _make_loader(
                    is_synthetic=is_synth,
                    mode=mode,
                    edge_direction="top_down",
                    add_kappa=False,
                    add_virtual_supernode=False,
                )
            except Exception as exc:
                print(f"  [base loader failed] {tag}: {exc}")
                continue
            for gid in loader.list_graph_ids():
                try:
                    data = loader.get_graph(gid)
                    base_sizes[(tag, gid)] = data.num_nodes + data.edge_index.shape[1]
                except Exception as exc:
                    print(f"  [base skip] {tag}/{gid}: {exc}")
    return base_sizes


# ─── Benchmark loop ────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> list[dict]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("Loading per-graph kappa values from benchmark CSVs...")
    real_kappa_map, synth_kappa_map = _build_kappa_maps()
    print(f"  {len(real_kappa_map)} real, {len(synth_kappa_map)} synthetic kappa entries.\n")

    print("Building base-size reference (top_down, no kappa, no supernode)...")
    base_sizes = _build_base_sizes(real_kappa_map, synth_kappa_map)
    print(f"  Cached {len(base_sizes)} base-size entries.\n")

    rows: list[dict] = []

    for struct in STRUCTURES:
        sid = struct["id"]
        label = struct["label"]
        mode = struct["mode"]
        print(f"=== Structure {sid}: {label} ===")

        model: Optional[GINBenchmarkModel] = None
        input_dim: Optional[int] = None

        sources = [False, True] if args.include_synthetic else [False]
        for is_synth in sources:
            synth_tag = "synth" if is_synth else "real"
            mode_key = f"{mode}_synth" if is_synth else mode
            kappa_map = synth_kappa_map if is_synth else real_kappa_map

            try:
                loader = _make_loader(
                    is_synthetic=is_synth,
                    mode=mode,
                    edge_direction=struct["edge_direction"],
                    add_kappa=struct["add_kappa"],
                    add_virtual_supernode=struct["add_virtual_supernode"],
                    kappa_map=kappa_map if struct["add_kappa"] else None,
                )
            except Exception as exc:
                print(f"  [loader failed] {synth_tag}: {exc}")
                continue

            graph_ids = sorted(loader.list_graph_ids())
            print(f"  [{synth_tag}] {len(graph_ids)} graphs ...")

            # Pre-load all graphs so the global warm-up can cycle through them.
            loaded: dict[str, Data] = {}
            for gid in graph_ids:
                try:
                    data = loader.get_graph(gid)
                except Exception as exc:
                    print(f"    [skip] {gid}: {exc}")
                    continue
                if data.num_nodes == 0:
                    print(f"    [skip] {gid}: empty graph")
                    continue
                loaded[gid] = data

            if not loaded:
                continue

            # Build model once — input_dim from first valid graph.
            if model is None:
                first = next(iter(loaded.values()))
                input_dim = int(first.x.shape[1])
                model = GINBenchmarkModel(input_dim, hidden_dim=args.hidden_dim).to(device)
                model.eval()
                print(f"    Model built: input_dim={input_dim}, hidden_dim={args.hidden_dim}")

            # Global pre-warm: cycle through all graphs so model weights and
            # PyTorch's internal caches are hot before any timed measurement.
            print(f"    Pre-warming ...", end=" ", flush=True)
            with torch.no_grad():
                for _ in range(args.n_warmup):
                    for data in loaded.values():
                        model(data.to(device))
            _sync(device)
            print("done")

            for gid, data in loaded.items():
                size_processed = data.num_nodes + int(data.edge_index.shape[1])
                size_base = base_sizes.get((mode_key, gid), size_processed)

                t_median, t_iqr = measure_inference_time(
                    model, data, device,
                    n_warmup=args.n_warmup,
                    n_steps=args.n_steps,
                )

                rows.append({
                    "structure_id": sid,
                    "structure_label": label,
                    "graph_id": gid,
                    "size_base": size_base,
                    "size_processed": size_processed,
                    "t_median": t_median,
                    "t_iqr": t_iqr,
                    "n_steps": args.n_steps,
                })

                print(
                    f"    {gid:20s}  base={size_base:5d}  proc={size_processed:5d}"
                    f"  t={t_median * 1e6:8.1f} µs  iqr={t_iqr * 1e6:6.1f} µs"
                )

        print()

    return rows


# ─── Plot ─────────────────────────────────────────────────────────────────────

_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def _power_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Fit t ≈ a · size^k in log space; return (k, a)."""
    log_k, log_a = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(log_k), float(np.exp(log_a))


def plot_benchmark(rows: list[dict], output_dir: Path, x_axis: str) -> None:
    if not HAS_MPL:
        print("matplotlib not available — skipping plot.")
        return

    x_col = "size_processed" if x_axis == "processed" else "size_base"
    by_struct: dict[int, list[dict]] = {}
    for row in rows:
        by_struct.setdefault(row["structure_id"], []).append(row)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")

    for struct in STRUCTURES:
        sid = struct["id"]
        pts = by_struct.get(sid, [])
        if not pts:
            continue

        xs = np.array([r[x_col] for r in pts], dtype=float)
        ys = np.array([r["t_median"] for r in pts], dtype=float)
        mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            continue

        color = _COLORS[(sid - 1) % len(_COLORS)]
        k, a = _power_fit(xs, ys)
        x_fit = np.linspace(xs.min(), xs.max(), 300)
        y_fit = a * x_fit ** k

        ax.scatter(xs, ys, color=color, alpha=0.45, s=22, zorder=3)
        ax.plot(
            x_fit, y_fit, color=color, linewidth=1.8,
            label=f"#{sid} {struct['label']}  (k≈{k:.2f})",
        )

    x_label = (
        "|V|+|E| nach Augmentierung (size_processed)"
        if x_axis == "processed"
        else "|V|+|E| Ausgangsgraph (size_base)"
    )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Inferenzzeit [s]  (Median über 100 Schritte)", fontsize=12)
    ax.set_title("GINConv Inferenzzeit vs. Graphgröße — Log-Log-Plot", fontsize=14)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.tight_layout()

    for fmt in ("svg", "pdf"):
        path = output_dir / f"inference_time_vs_size.{fmt}"
        plt.savefig(path, format=fmt, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GINConv inference time vs. graph size (8 structural variants).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "results" / "benchmark"),
        help="Directory for CSV, plot, and config output.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["processed", "base"],
        default="processed",
        help="x-axis: size_processed (after augmentation) or size_base (before).",
    )
    parser.add_argument("--n-warmup", type=int, default=10, help="Warm-up forward passes.")
    parser.add_argument("--n-steps",  type=int, default=100, help="Timed forward passes per graph.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GINConv hidden dimension.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--include-synthetic", action="store_true", help="Also include 100 synthetic graphs (default: real graphs only).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    config = {
        "model": {"architecture": "GINConv", "n_layers": 3, "hidden_dim": args.hidden_dim},
        "n_warmup": args.n_warmup,
        "n_steps": args.n_steps,
        "device": device_str,
        "seed": args.seed,
        "x_axis": args.x_axis,
        "include_synthetic": args.include_synthetic,
        "structures": STRUCTURES,
    }
    cfg_path = output_dir / "benchmark_config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"Config saved: {cfg_path}\n")

    rows = run_benchmark(args)

    csv_path = output_dir / "benchmark_results.csv"
    fieldnames = [
        "structure_id", "structure_label", "graph_id",
        "size_base", "size_processed",
        "t_median", "t_iqr", "n_steps",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}  ({len(rows)} rows)")

    plot_benchmark(rows, output_dir, x_axis=args.x_axis)
    print(f"\nDone — {len(rows)} data points across {len(STRUCTURES)} structures.")


if __name__ == "__main__":
    main()
