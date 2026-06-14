#!/usr/bin/env python3
"""
benchmark_inference.py

Measures GNN inference latency across the 6 supervised learning stages and
sweeps over num_layers × hidden_dim to identify which architectural decisions
have the highest impact on production inference time.

Stage overview
--------------
  1  Tree / edge-blind GIN          mode=tree,             homo
  2  Tree-Derivatives / edge-blind  mode=tree_derivatives, homo
  3  Graph / edge-blind GIN         mode=graph,            homo
  4  Graph / edge-aware GATv2       mode=graph,            homo  (TestGraphNetwork)
  5  Heterogeneous / GIN            mode=graph,            hetero (HeteroExpressionClassifier)
  6  Hetero + DiffPool              mode=graph,            hetero (variant=pooling)

Usage
-----
  # Single stage, single config:
  python benchmark_inference.py --stages 4 --hidden-dims 128 --num-layers 3

  # Full grid (all 6 stages, default dims/layers):
  python benchmark_inference.py --sweep

  # Custom grid:
  python benchmark_inference.py --stages 1 2 3 4 5 6 \\
      --hidden-dims 64 128 256 --num-layers 2 3 4 \\
      --out-dir results/bench
"""

import sys
import time
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── path setup ────────────────────────────────────────────────────────────────
_gnn_root = Path(__file__).resolve().parents[1]
_src_root = Path(__file__).resolve().parents[2]
for _p in (_gnn_root, _src_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.models.classifiers import TestGraphNetwork
from gnn.shared.models.hetero_backbone import (
    HeteroExpressionClassifier,
    build_hetero_metadata,
    pad_edge_types,
    collect_edge_attr_dims,
)


# ── Stage registry ────────────────────────────────────────────────────────────
STAGE_DEFS: dict[int, dict] = {
    1: dict(name="Tree / edge-blind",          mode="tree",             hetero=False, default_arch="gin_stack"),
    2: dict(name="Tree-Deriv / edge-blind",    mode="tree_derivatives", hetero=False, default_arch="gin_stack"),
    3: dict(name="Graph / edge-blind",         mode="graph",            hetero=False, default_arch="gin_stack"),
    4: dict(name="Graph / edge-aware (GATv2)", mode="graph",            hetero=False, default_arch="gatv2_stack"),
    5: dict(name="Heterogeneous / GIN",        mode="graph",            hetero=True,  default_arch="hetero"),
    6: dict(name="Hetero / DiffPool",          mode="graph",            hetero=True,  default_arch="hetero_diffpool"),
}

EDGE_BLIND_ARCHS = {"gcn_stack", "gin_stack", "sage_stack"}
EDGE_AWARE_ARCHS = {"gatv2_stack", "gine_stack"}
HETERO_ARCHS    = {"hetero", "hetero_diffpool"}


# ── Lightweight benchmark models ──────────────────────────────────────────────

class _BenchmarkEdgeBlindModel(nn.Module):
    """Variable-depth edge-blind GNN (GCN / GIN / SAGE) for latency benchmarking.

    Mirrors the computational structure used in stages 1-3 but supports
    arbitrary num_layers, unlike the production _EdgeBlindStack (fixed at 3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        architecture: str = "gin_stack",
        global_dim: int = 2,
    ):
        super().__init__()
        convs: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            if architecture == "gcn_stack":
                convs.append(GCNConv(in_dim, hidden_dim))
            elif architecture == "sage_stack":
                convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            else:  # gin_stack (default)
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                convs.append(GINConv(mlp))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(num_layers)])
        self._global_dim = global_dim
        self.head = nn.Linear(hidden_dim + global_dim, 2)

    def forward(self, x, edge_index, batch, global_features=None, edge_attr=None):
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        if global_features is not None:
            gf = global_features.view(x.size(0), -1)
        else:
            gf = torch.zeros(x.size(0), self._global_dim, device=x.device, dtype=x.dtype)
        return self.head(torch.cat([x, gf], dim=-1))


# ── Model factories ───────────────────────────────────────────────────────────

def _build_homo_model(
    arch: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    edge_dim: int,
    global_dim: int,
    device: torch.device,
) -> nn.Module:
    if arch in EDGE_BLIND_ARCHS:
        return _BenchmarkEdgeBlindModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            architecture=arch,
            global_dim=global_dim,
        ).to(device)
    # edge-aware: use the full production model
    return TestGraphNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        global_dim=global_dim,
        architecture=arch,
        edge_dim=edge_dim,
        num_layers=num_layers,
    ).to(device)


def _build_hetero_model(
    arch: str,
    hidden_dim: int,
    num_layers: int,
    metadata,
    device: torch.device,
) -> nn.Module:
    variant   = "pooling" if arch == "hetero_diffpool" else "legacy"
    pool_type = "diffpool" if arch == "hetero_diffpool" else "topk"
    return HeteroExpressionClassifier(
        metadata=metadata,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        variant=variant,
        pool_type=pool_type,
    ).to(device)


# ── Timing primitives ─────────────────────────────────────────────────────────

def _time_homo(
    model: nn.Module,
    data,
    device: torch.device,
    warmup: int,
    runs: int,
) -> tuple[float, float, float]:
    """Time a single-graph homo (x, edge_index, batch, …) forward pass.

    Returns (mean_ms, median_ms, std_ms).
    """
    model.eval()
    data = data.to(device)

    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    gf = getattr(data, "global_features", None)
    if gf is None:
        gf = torch.zeros((1, 2), dtype=torch.float, device=device)
    ea = getattr(data, "edge_attr", None)

    def _fwd():
        return model(data.x, data.edge_index, data.batch, gf, ea)

    with torch.no_grad():
        for _ in range(warmup):
            _fwd()

    times_ms: list[float] = []
    with torch.no_grad():
        if device.type == "cuda":
            for _ in range(runs):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _fwd()
                e.record()
                torch.cuda.synchronize()
                times_ms.append(s.elapsed_time(e))
        else:
            for _ in range(runs):
                t0 = time.perf_counter()
                _fwd()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.median(times_ms)), float(np.std(times_ms))


def _time_hetero(
    model: nn.Module,
    data,
    device: torch.device,
    warmup: int,
    runs: int,
) -> tuple[float, float, float]:
    """Time a single-graph hetero forward pass (model(data)).

    Returns (mean_ms, median_ms, std_ms).
    """
    model.eval()
    data = data.to(device)

    # Add missing batch tensors (needed by global pooling inside the model)
    for node_type in data.node_types:
        nd = data[node_type]
        if not hasattr(nd, "batch") or nd.batch is None:
            nd.batch = torch.zeros(nd.num_nodes, dtype=torch.long, device=device)

    def _fwd():
        return model(data)

    with torch.no_grad():
        for _ in range(warmup):
            _fwd()

    times_ms: list[float] = []
    with torch.no_grad():
        if device.type == "cuda":
            for _ in range(runs):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _fwd()
                e.record()
                torch.cuda.synchronize()
                times_ms.append(s.elapsed_time(e))
        else:
            for _ in range(runs):
                t0 = time.perf_counter()
                _fwd()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.median(times_ms)), float(np.std(times_ms))


# ── Single configuration benchmark ───────────────────────────────────────────

def benchmark_config(
    *,
    dataset_name: str,
    stage_id: int,
    arch: str | None = None,
    hidden_dim: int = 128,
    num_layers: int = 3,
    device: torch.device,
    warmup: int = 50,
    runs: int = 200,
    edge_direction: str = "top_down",
) -> dict | None:
    """Run inference timing for one (stage, arch, hidden_dim, num_layers) combination.

    Returns a result dict, or None on failure.
    """
    stage_def = STAGE_DEFS[stage_id]
    arch = arch or stage_def["default_arch"]
    label = f"Stage {stage_id} | {arch:<18} | L={num_layers} | H={hidden_dim}"
    print(f"  {label}")

    # Load graphs
    try:
        loader = GraphDataLoader(
            name=dataset_name,
            mode=stage_def["mode"],
            heterogeneous=stage_def["hetero"],
            edge_direction=edge_direction,
        )
        graphs = loader.load_all()
    except Exception as exc:
        print(f"    [SKIP] Data load failed: {exc}")
        return None

    if not graphs:
        print(f"    [SKIP] No graphs found.")
        return None

    graph_list = list(graphs.values())
    sample = graph_list[0]

    # Build model + run timing
    try:
        if stage_def["hetero"]:
            metadata = build_hetero_metadata(graph_list)
            edge_attr_dims = collect_edge_attr_dims(graph_list)
            edge_types = metadata[1]  # collect_edge_types already called inside build_hetero_metadata
            graph_list = [pad_edge_types(g, edge_types, edge_attr_dims) for g in graph_list]
            model = _build_hetero_model(arch, hidden_dim, num_layers, metadata, device)
            time_fn = _time_hetero
        else:
            input_dim = sample.x.shape[1]
            ea = getattr(sample, "edge_attr", None)
            edge_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 4
            gf = getattr(sample, "global_features", None)
            global_dim = gf.shape[-1] if gf is not None else 2
            model = _build_homo_model(arch, input_dim, hidden_dim, num_layers, edge_dim, global_dim, device)
            time_fn = _time_homo

        latencies: list[float] = []
        for g in graph_list:
            mean_t, _, _ = time_fn(model, g, device, warmup, runs)
            latencies.append(mean_t)

    except Exception as exc:
        print(f"    [FAIL] {exc}")
        return None

    total_params = sum(p.numel() for p in model.parameters())
    mean_lat    = float(np.mean(latencies))
    median_lat  = float(np.median(latencies))
    std_lat     = float(np.std(latencies))
    throughput  = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"    → {mean_lat:.4f} ms (median {median_lat:.4f}) | {total_params:,} params | {len(graph_list)} graphs")
    return {
        "stage":         stage_id,
        "stage_name":    stage_def["name"],
        "architecture":  arch,
        "num_layers":    num_layers,
        "hidden_dim":    hidden_dim,
        "params":        total_params,
        "num_graphs":    len(graph_list),
        "mean_ms":       mean_lat,
        "median_ms":     median_lat,
        "std_ms":        std_lat,
        "throughput_gps": throughput,
    }


# ── Grid sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    *,
    dataset_name: str,
    stage_ids: list[int],
    hidden_dims: list[int],
    num_layers_list: list[int],
    device: torch.device,
    warmup: int,
    runs: int,
    out_dir: Path,
    use_mlflow: bool = False,
    edge_direction: str = "top_down",
) -> pd.DataFrame:
    """Run a full grid: stage × hidden_dim × num_layers → latency table."""
    print(f"\n{'='*70}")
    print(f"INFERENCE BENCHMARK SWEEP  |  Device: {device.type.upper()}")
    print(f"  Stages:      {stage_ids}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Num layers:  {num_layers_list}")
    n_configs = len(stage_ids) * len(hidden_dims) * len(num_layers_list)
    print(f"  Total configurations: {n_configs}")
    print(f"{'='*70}\n")

    records: list[dict] = []
    for stage_id in stage_ids:
        stage_def = STAGE_DEFS[stage_id]
        arch = stage_def["default_arch"]
        print(f"\n--- Stage {stage_id}: {stage_def['name']} (arch={arch}) ---")
        for hidden_dim, num_layers in itertools.product(hidden_dims, num_layers_list):
            result = benchmark_config(
                dataset_name=dataset_name,
                stage_id=stage_id,
                arch=arch,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=device,
                warmup=warmup,
                runs=runs,
                edge_direction=edge_direction,
            )
            if result:
                records.append(result)

    df = pd.DataFrame(records)
    if df.empty:
        print("[WARNING] No results collected — check dataset path and stage configs.")
        return df

    # Print comparison table
    print(f"\n\n{'='*90}")
    print("SWEEP RESULTS")
    print(f"{'='*90}")
    print(df.to_string(
        columns=["stage", "stage_name", "architecture", "num_layers", "hidden_dim",
                 "params", "mean_ms", "median_ms", "throughput_gps"],
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))
    print(f"{'='*90}\n")

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "benchmark_sweep.csv"
    plot_path = out_dir / "benchmark_sweep.png"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV  → {csv_path}")
    plot_sweep(df, plot_path)

    # MLflow
    if use_mlflow and MLFLOW_AVAILABLE:
        _log_mlflow(dataset_name, device, warmup, runs, df, csv_path, plot_path)

    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

_STAGE_COLORS = {
    1: "#4e79a7",
    2: "#f28e2b",
    3: "#e15759",
    4: "#76b7b2",
    5: "#59a14f",
    6: "#edc948",
}


def plot_sweep(df: pd.DataFrame, plot_path: Path) -> None:
    """Four-panel figure comparing latency across stages, layer counts, and dims."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("GNN Inferenzzeit — Stage-Sweep", fontsize=15, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        present_stages = sorted(df["stage"].unique())

        # ── Panel 1: Mean latency per stage (averaged across all configs) ────
        ax1 = fig.add_subplot(gs[0, 0])
        stage_means = df.groupby("stage")["mean_ms"].mean()
        stage_labels = [str(s) for s in stage_means.index]
        bars = ax1.bar(
            stage_labels,
            stage_means.values,
            color=[_STAGE_COLORS.get(s, "#aaa") for s in stage_means.index],
            edgecolor="gray",
            linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8,
            )
        ax1.set_title("Ø Latenz je Stage (Mittel über Layer & Dim)", fontsize=10, pad=8)
        ax1.set_xlabel("Stage")
        ax1.set_ylabel("Latenz (ms)")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # ── Panel 2: Latency vs num_layers per stage ──────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        for stage_id, group in df.groupby("stage"):
            layer_means = group.groupby("num_layers")["mean_ms"].mean()
            ax2.plot(
                layer_means.index, layer_means.values,
                marker="o", label=f"S{stage_id}",
                color=_STAGE_COLORS.get(stage_id, "#aaa"),
            )
        ax2.set_title("Latenz vs. Anzahl Layer (je Stage)", fontsize=10, pad=8)
        ax2.set_xlabel("num_layers")
        ax2.set_ylabel("Latenz (ms)")
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(linestyle=":", alpha=0.6)

        # ── Panel 3: Latency vs hidden_dim per stage ──────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        for stage_id, group in df.groupby("stage"):
            dim_means = group.groupby("hidden_dim")["mean_ms"].mean()
            ax3.plot(
                dim_means.index, dim_means.values,
                marker="s", label=f"S{stage_id}",
                color=_STAGE_COLORS.get(stage_id, "#aaa"),
            )
        ax3.set_title("Latenz vs. Hidden Dimension (je Stage)", fontsize=10, pad=8)
        ax3.set_xlabel("hidden_dim")
        ax3.set_ylabel("Latenz (ms)")
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(linestyle=":", alpha=0.6)

        # ── Panel 4: Param count vs latency scatter ───────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        # Bubble size encodes hidden_dim
        max_dim = df["hidden_dim"].max()
        sizes = ((df["hidden_dim"] / max_dim) * 200 + 30).values
        ax4.scatter(
            df["params"] / 1e3,
            df["mean_ms"],
            c=[_STAGE_COLORS.get(s, "#aaa") for s in df["stage"]],
            s=sizes,
            alpha=0.75,
            edgecolors="gray",
            linewidths=0.5,
        )
        ax4.set_title("Parameter vs. Latenz\n(Blasengröße = hidden_dim)", fontsize=10, pad=8)
        ax4.set_xlabel("Parameter (k)")
        ax4.set_ylabel("Latenz (ms)")
        ax4.grid(linestyle=":", alpha=0.6)

        legend_handles = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=_STAGE_COLORS.get(s, "#aaa"),
                markersize=8,
                label=f"S{s}: {STAGE_DEFS[s]['name']}",
            )
            for s in present_stages
        ]
        ax4.legend(handles=legend_handles, fontsize=7, loc="upper left")

        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot → {plot_path}")
    except Exception as exc:
        print(f"[WARNING] Plot generation failed: {exc}")


# ── MLflow helper ─────────────────────────────────────────────────────────────

def _log_mlflow(
    dataset_name: str,
    device: torch.device,
    warmup: int,
    runs: int,
    df: pd.DataFrame,
    csv_path: Path,
    plot_path: Path,
) -> None:
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(f"GNN_Inference_Benchmark_{dataset_name}")
        with mlflow.start_run(run_name=f"sweep_{device.type}"):
            mlflow.log_params({
                "device": device.type,
                "warmup": warmup,
                "runs": runs,
                "num_configurations": len(df),
            })
            for _, row in df.iterrows():
                key = f"S{int(row['stage'])}_{row['architecture']}_L{int(row['num_layers'])}_H{int(row['hidden_dim'])}"
                mlflow.log_metric(f"{key}_mean_ms", row["mean_ms"])
                mlflow.log_metric(f"{key}_throughput", row["throughput_gps"])
            mlflow.log_artifact(str(csv_path))
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path))
            print("[MLflow] Sweep logged.")
    except Exception as exc:
        print(f"[MLflow] Logging failed or server offline: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN inference latency benchmark across the 6 supervised learning stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Single stage:
  python benchmark_inference.py --stages 4 --hidden-dims 128 --num-layers 3

  # Full grid:
  python benchmark_inference.py --sweep

  # Custom grid:
  python benchmark_inference.py --stages 1 2 3 4 5 6 \\
      --hidden-dims 64 128 256 --num-layers 2 3 4
""",
    )
    parser.add_argument(
        "--dataset", default="graphs",
        help="Dataset name passed to GraphDataLoader (default: graphs)",
    )
    parser.add_argument(
        "--stages", nargs="+", type=int, default=list(range(1, 7)),
        choices=list(range(1, 7)), metavar="N",
        help="Stages to benchmark, 1-6 (default: all)",
    )
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[64, 128, 256],
        metavar="D",
        help="Hidden dims to sweep (default: 64 128 256)",
    )
    parser.add_argument(
        "--num-layers", nargs="+", type=int, default=[2, 3, 4],
        metavar="L",
        help="Layer counts to sweep (default: 2 3 4)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["cpu", "cuda", "auto"],
        help="Device (auto selects CUDA if available)",
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Warmup iterations before timing (default: 50)",
    )
    parser.add_argument(
        "--runs", type=int, default=200,
        help="Timing iterations per graph (default: 200)",
    )
    parser.add_argument(
        "--edge-direction", default="top_down",
        choices=["top_down", "bottom_up", "bidirectional"],
        help="Edge direction for tree-based stages (default: top_down)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory for CSV and plot output (default: <script_dir>/benchmark_results/)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run the full grid — all selected stages × dims × layers",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow tracking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "benchmark_results"
    )

    is_grid = args.sweep or len(args.stages) > 1 or len(args.hidden_dims) > 1 or len(args.num_layers) > 1
    if is_grid:
        run_sweep(
            dataset_name=args.dataset,
            stage_ids=args.stages,
            hidden_dims=args.hidden_dims,
            num_layers_list=args.num_layers,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
            out_dir=out_dir,
            use_mlflow=not args.no_mlflow,
            edge_direction=args.edge_direction,
        )
    else:
        # Single configuration
        stage_id  = args.stages[0]
        hidden_dim = args.hidden_dims[0]
        num_layers = args.num_layers[0]
        result = benchmark_config(
            dataset_name=args.dataset,
            stage_id=stage_id,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
            edge_direction=args.edge_direction,
        )
        if result:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"benchmark_s{stage_id}_H{hidden_dim}_L{num_layers}.csv"
            pd.DataFrame([result]).to_csv(csv_path, index=False)
            print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
