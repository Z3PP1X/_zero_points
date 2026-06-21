#!/usr/bin/env python3
"""
benchmark_inference.py

Measures GNN inference latency across 4 supervised learning stages using the
unified ExpressionGNN backbone. Stages 1-3 represent a fixed progression of
feature complexity; stage 4 is open for free experimentation.

Stage overview
--------------
  1  Pure AST          — node identity only  (14 features: node_type + label)
  2  AST + Roots       — adds function identity (19 features: + root_color)
  3  Full graph        — all structural features (28 features: + topology + PE)
  4  Experiment        — configurable; default: all 28 features, GIN

Usage
-----
  # Single stage:
  python benchmark_inference.py --stages 3 --hidden-dims 128 --num-layers 3

  # Full grid sweep (all 4 stages):
  python benchmark_inference.py --sweep

  # Custom grid:
  python benchmark_inference.py --stages 1 2 3 4 \\
      --hidden-dims 64 128 256 --num-layers 2 3 4 \\
      --out-dir results/bench
"""

import sys
import time
import argparse
import itertools
import numpy as np
import torch
import pandas as pd
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── path setup ────────────────────────────────────────────────────────────────
_gnn_root = Path(__file__).resolve().parents[1]
_src_root  = Path(__file__).resolve().parents[2]
for _p in (_gnn_root, _src_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.models.gnn_backbones import ExpressionGNN
from gnn.shared.utils.graph_vocab import NODE_FEATURE_SCHEMA
from gnn.shared.utils.feature_extraction import slice_active_features


# ── Feature subsets per stage ─────────────────────────────────────────────────

# Stage 1: node type + label only — the bare AST without any topology signal.
_FEATURES_STAGE1: list[str] = (
    [f for f in NODE_FEATURE_SCHEMA if f.startswith("node_type_")]
    + [f for f in NODE_FEATURE_SCHEMA if f.startswith("label_")]
)

# Stage 2: adds root_color to distinguish f / f' / f'' root nodes.
_FEATURES_STAGE2: list[str] = (
    [f for f in NODE_FEATURE_SCHEMA if f.startswith("node_type_")]
    + [f for f in NODE_FEATURE_SCHEMA if f.startswith("root_color_")]
    + [f for f in NODE_FEATURE_SCHEMA if f.startswith("label_")]
)

# Stage 3 / Stage 4: full 28-feature schema.
_FEATURES_ALL: list[str] = list(NODE_FEATURE_SCHEMA)


# ── Stage registry ────────────────────────────────────────────────────────────

STAGE_DEFS: dict[int, dict] = {
    1: dict(
        name="Pure AST",
        description="Node type + label only (14 features)",
        active_features=_FEATURES_STAGE1,
    ),
    2: dict(
        name="AST + Root identity",
        description="Adds root_color: distinguishes f / f' / f'' (19 features)",
        active_features=_FEATURES_STAGE2,
    ),
    3: dict(
        name="Full graph",
        description="All 28 features: topology, histogram, anchor PE",
        active_features=_FEATURES_ALL,
    ),
    4: dict(
        name="Experiment",
        description="Free configuration — edit active_features as needed",
        active_features=_FEATURES_ALL,
    ),
}


# ── Timing primitives ─────────────────────────────────────────────────────────

def _time_model(
    model: torch.nn.Module,
    data,
    active_features: list[str],
    device: torch.device,
    warmup: int,
    runs: int,
    global_dim: int,
) -> tuple[float, float, float]:
    """Time one forward pass. Returns (mean_ms, median_ms, std_ms)."""
    model.eval()
    data = data.to(device)

    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    x = slice_active_features(data.x, active_features)
    gf = getattr(data, "global_features", None)
    if gf is None:
        gf = torch.zeros((1, global_dim), dtype=torch.float, device=device)
    gf = gf.to(device)

    def _fwd():
        return model(x, data.edge_index, data.batch, gf)

    with torch.no_grad():
        for _ in range(warmup):
            _fwd()

    times_ms: list[float] = []
    with torch.no_grad():
        if device.type == "cuda":
            for _ in range(runs):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); _fwd(); e.record()
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
    hidden_dim: int = 128,
    num_layers: int = 3,
    device: torch.device,
    warmup: int = 50,
    runs: int = 200,
    edge_direction: str = "top_down",
) -> dict | None:
    stage_def = STAGE_DEFS[stage_id]
    active_features = stage_def["active_features"]
    input_dim = len(active_features)
    label = f"Stage {stage_id} | GINConv | L={num_layers} | H={hidden_dim} | F={input_dim}"
    print(f"  {label}")

    try:
        loader = GraphDataLoader(name=dataset_name, edge_direction=edge_direction)
        graphs = loader.load_all()
    except Exception as exc:
        print(f"    [SKIP] Data load failed: {exc}")
        return None

    if not graphs:
        print(f"    [SKIP] No graphs found.")
        return None

    graph_list = list(graphs.values())
    sample = graph_list[0]
    gf = getattr(sample, "global_features", None)
    global_dim = int(gf.shape[-1]) if gf is not None else 5

    try:
        model = ExpressionGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            global_dim=global_dim,
            num_layers=num_layers,
            classify=True,
        ).to(device)

        latencies: list[float] = []
        for g in graph_list:
            mean_t, _, _ = _time_model(model, g, active_features, device, warmup, runs, global_dim)
            latencies.append(mean_t)

    except Exception as exc:
        print(f"    [FAIL] {exc}")
        return None

    total_params = sum(p.numel() for p in model.parameters())
    mean_lat     = float(np.mean(latencies))
    median_lat   = float(np.median(latencies))
    std_lat      = float(np.std(latencies))
    throughput   = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"    → {mean_lat:.4f} ms (median {median_lat:.4f}) | {total_params:,} params | {len(graph_list)} graphs")
    return {
        "stage":           stage_id,
        "stage_name":      stage_def["name"],
        "num_features":    input_dim,
        "num_layers":      num_layers,
        "hidden_dim":      hidden_dim,
        "params":          total_params,
        "num_graphs":      len(graph_list),
        "mean_ms":         mean_lat,
        "median_ms":       median_lat,
        "std_ms":          std_lat,
        "throughput_gps":  throughput,
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
    print(f"\n{'='*70}")
    print(f"INFERENCE BENCHMARK SWEEP  |  Device: {device.type.upper()}")
    print(f"  Stages:      {stage_ids}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Num layers:  {num_layers_list}")
    print(f"  Total configurations: {len(stage_ids) * len(hidden_dims) * len(num_layers_list)}")
    print(f"{'='*70}\n")

    records: list[dict] = []
    for stage_id in stage_ids:
        stage_def = STAGE_DEFS[stage_id]
        print(f"\n--- Stage {stage_id}: {stage_def['name']} | {stage_def['description']} ---")
        for hidden_dim, num_layers in itertools.product(hidden_dims, num_layers_list):
            result = benchmark_config(
                dataset_name=dataset_name,
                stage_id=stage_id,
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
        print("[WARNING] No results collected.")
        return df

    print(f"\n\n{'='*90}")
    print("SWEEP RESULTS")
    print(f"{'='*90}")
    print(df.to_string(
        columns=["stage", "stage_name", "num_features",
                 "num_layers", "hidden_dim", "params", "mean_ms", "median_ms", "throughput_gps"],
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))
    print(f"{'='*90}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "benchmark_sweep.csv"
    plot_path = out_dir / "benchmark_sweep.png"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV  → {csv_path}")
    plot_sweep(df, plot_path)

    if use_mlflow and MLFLOW_AVAILABLE:
        _log_mlflow(dataset_name, device, warmup, runs, df, csv_path, plot_path)

    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

_STAGE_COLORS = {
    1: "#4e79a7",
    2: "#f28e2b",
    3: "#e15759",
    4: "#76b7b2",
}


def plot_sweep(df: pd.DataFrame, plot_path: Path) -> None:
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

        ax1 = fig.add_subplot(gs[0, 0])
        stage_means = df.groupby("stage")["mean_ms"].mean()
        bars = ax1.bar(
            [str(s) for s in stage_means.index],
            stage_means.values,
            color=[_STAGE_COLORS.get(s, "#aaa") for s in stage_means.index],
            edgecolor="gray", linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax1.set_title("Avg Latency per Stage (mean over layers & dims)", fontsize=10, pad=8)
        ax1.set_xlabel("Stage"); ax1.set_ylabel("Latency (ms)")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        ax2 = fig.add_subplot(gs[0, 1])
        for stage_id, group in df.groupby("stage"):
            layer_means = group.groupby("num_layers")["mean_ms"].mean()
            ax2.plot(layer_means.index, layer_means.values, marker="o",
                     label=f"S{stage_id}", color=_STAGE_COLORS.get(stage_id, "#aaa"))
        ax2.set_title("Latency vs. Number of Layers (per Stage)", fontsize=10, pad=8)
        ax2.set_xlabel("num_layers"); ax2.set_ylabel("Latency (ms)")
        ax2.legend(fontsize=8); ax2.grid(linestyle=":", alpha=0.6)

        ax3 = fig.add_subplot(gs[1, 0])
        for stage_id, group in df.groupby("stage"):
            dim_means = group.groupby("hidden_dim")["mean_ms"].mean()
            ax3.plot(dim_means.index, dim_means.values, marker="s",
                     label=f"S{stage_id}", color=_STAGE_COLORS.get(stage_id, "#aaa"))
        ax3.set_title("Latency vs. Hidden Dimension (per Stage)", fontsize=10, pad=8)
        ax3.set_xlabel("hidden_dim"); ax3.set_ylabel("Latency (ms)")
        ax3.legend(fontsize=8); ax3.grid(linestyle=":", alpha=0.6)

        ax4 = fig.add_subplot(gs[1, 1])
        max_dim = df["hidden_dim"].max()
        sizes = ((df["hidden_dim"] / max_dim) * 200 + 30).values
        ax4.scatter(df["params"] / 1e3, df["mean_ms"],
                    c=[_STAGE_COLORS.get(s, "#aaa") for s in df["stage"]],
                    s=sizes, alpha=0.75, edgecolors="gray", linewidths=0.5)
        ax4.set_title("Parameters vs. Latency\n(bubble size = hidden_dim)", fontsize=10, pad=8)
        ax4.set_xlabel("Parameters (k)"); ax4.set_ylabel("Latency (ms)")
        ax4.grid(linestyle=":", alpha=0.6)
        ax4.legend(handles=[
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_STAGE_COLORS.get(s, "#aaa"), markersize=8,
                   label=f"S{s}: {STAGE_DEFS[s]['name']}")
            for s in present_stages
        ], fontsize=7, loc="upper left")

        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot → {plot_path}")
    except Exception as exc:
        print(f"[WARNING] Plot generation failed: {exc}")


# ── MLflow helper ─────────────────────────────────────────────────────────────

def _log_mlflow(dataset_name, device, warmup, runs, df, csv_path, plot_path):
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(f"GNN_Inference_Benchmark_{dataset_name}")
        with mlflow.start_run(run_name=f"sweep_{device.type}"):
            mlflow.log_params({"device": device.type, "warmup": warmup, "runs": runs, "num_configurations": len(df)})
            for _, row in df.iterrows():
                key = f"S{int(row['stage'])}_L{int(row['num_layers'])}_H{int(row['hidden_dim'])}"
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
        description="GNN inference latency benchmark — 4 stages, unified ExpressionGNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stage descriptions:
  1  Pure AST          — 14 features: node_type + label (no topology)
  2  AST + Root ID     — 19 features: adds root_color (f / f\' / f\'\')
  3  Full graph        — 28 features: all features including topology + anchor PE
  4  Experiment        — 28 features, same as 3 but intended for custom variants

Examples:
  python benchmark_inference.py --stages 3 --hidden-dims 128 --num-layers 3
  python benchmark_inference.py --sweep
  python benchmark_inference.py --stages 1 2 3 4 --hidden-dims 64 128 256 --num-layers 2 3 4
""",
    )
    parser.add_argument("--dataset", default="graphs",
                        help="Dataset name passed to GraphDataLoader (default: graphs)")
    parser.add_argument("--stages", nargs="+", type=int, default=list(range(1, 5)),
                        choices=[1, 2, 3, 4], metavar="N",
                        help="Stages to benchmark, 1-4 (default: all)")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 128, 256], metavar="D")
    parser.add_argument("--num-layers", nargs="+", type=int, default=[2, 3, 4], metavar="L")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--edge-direction", default="top_down",
                        choices=["top_down", "bottom_up", "bidirectional"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="Run the full grid — all selected stages × dims × layers")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(__file__).resolve().parent / "benchmark_results")

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
        stage_id   = args.stages[0]
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
