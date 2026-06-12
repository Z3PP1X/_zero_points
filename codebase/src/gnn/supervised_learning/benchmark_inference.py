#!/usr/bin/env python3
"""
benchmark_inference.py

A scientific benchmarking tool for measuring GNN model inference times
under different graph structural and feature configurations.
Supports saving data to CSV, generating performance visualization plots,
and logging parameters, metrics, and artifacts to MLflow.
"""

import sys
import time
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Try to import mlflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Dynamic sys.path resolution to support package imports when run as scripts
gnn_root = Path(__file__).resolve().parents[1]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.models.classifiers import TestGraphNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="GNN Inference Time Benchmarking Tool")
    parser.add_argument(
        "--dataset",
        type=str,
        default="graphs",
        help="Name of the dataset/experiment (e.g. 'graphs')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="graph",
        choices=["graph", "tree", "tree_derivatives"],
        help="GNN mode for representation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run benchmarks on (cpu, cuda, or auto)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations before measuring",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of measurement runs per graph",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a comparative sweep across all combinations of mode and enrichment",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to save results as CSV (auto-generated if None)",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Path to save results visualization diagram (auto-generated if None)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking integration",
    )
    return parser.parse_args()


def benchmark_single_graph(model, data, device, warmup, runs):
    """
    Benchmarks inference time for a single graph with high-precision timing.
    Uses CUDA Events for GPU timing to avoid host-device synchronization lag,
    and perf_counter for CPU.
    """
    model.eval()
    data = data.to(device)

    # Ensure data batch and global_features are set correctly for batch_size=1
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    if not hasattr(data, "global_features") or data.global_features is None:
        data.global_features = torch.zeros((1, 2), dtype=torch.float, device=device)

    # Warmup runs to compile JIT graph and load kernels
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(data.x, data.edge_index, data.batch, data.global_features, getattr(data, "edge_attr", None))

    # Run actual measurements
    times_ms = []
    with torch.no_grad():
        if device.type == "cuda":
            for _ in range(runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(data.x, data.edge_index, data.batch, data.global_features, getattr(data, "edge_attr", None))
                end_event.record()
                torch.cuda.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(runs):
                t_start = time.perf_counter()
                _ = model(data.x, data.edge_index, data.batch, data.global_features, getattr(data, "edge_attr", None))
                t_end = time.perf_counter()
                times_ms.append((t_end - t_start) * 1000.0)

    return np.mean(times_ms), np.median(times_ms), np.std(times_ms)


def plot_single_config(latencies, nodes, graph_ids, plot_path):
    """Generates a scatter plot of Latency vs Node Count with a regression fit line."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 6))
        
        # Sort nodes and latencies to draw a correct fit line
        nodes_arr = np.array(nodes)
        latencies_arr = np.array(latencies)
        sort_idx = np.argsort(nodes_arr)
        nodes_sorted = nodes_arr[sort_idx]
        latencies_sorted = latencies_arr[sort_idx]
        
        # Scatter points
        plt.scatter(nodes_arr, latencies_arr, color="#4f81bd", s=100, zorder=3, label="Graphen")
        
        # Fit line
        if len(nodes_arr) > 1:
            coef = np.polyfit(nodes_arr, latencies_arr, 1)
            poly1d_fn = np.poly1d(coef)
            plt.plot(nodes_sorted, poly1d_fn(nodes_sorted), color="#c0504d", linestyle="--", linewidth=2, label=f"Fit (Steigung={coef[0]:.4f} ms/Knoten)")
            
        # Annotate points
        for i, txt in enumerate(graph_ids):
            plt.annotate(
                txt, 
                (nodes[i], latencies[i]), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="gray", lw=0.5)
            )
            
        plt.title("GNN Inferenzzeit vs. Graph-Größe", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Anzahl der Knoten", fontsize=12)
        plt.ylabel("Inferenzzeit (ms)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved Scatter-Plot to: {plot_path}")
    except Exception as e:
        print(f"Error generating plot: {e}")


def plot_sweep(results, plot_path):
    """Generates a grouped bar chart comparing latencies of all GNN configurations."""
    try:
        import matplotlib.pyplot as plt
        modes = ["graph", "tree", "tree_derivatives"]
        
        # Extract latency values per mode (single feature schema)
        latencies = []
        for m in modes:
            latencies.append(next((r["mean"] for r in results if r["mode"] == m), 0.0))

        x = np.arange(len(modes))
        width = 0.5

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x, latencies, width, label="Node-Features", color="#4f81bd", edgecolor="gray")

        ax.set_ylabel("Inferenzzeit (ms)", fontsize=12)
        ax.set_title("Vergleich der GNN-Inferenzzeiten nach Modus und Features", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        
        # Add labels on top of the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f"{height:.2f} ms",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=9, fontweight="semibold")

        autolabel(rects1)

        fig.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved Sweep Bar Chart to: {plot_path}")
    except Exception as e:
        print(f"Error generating sweep plot: {e}")


def run_single_config(dataset_name, mode, device, warmup, runs, csv_path=None, plot_path=None, use_mlflow=True):
    """Loads dataset and runs inference benchmarks on all graphs."""
    print(f"\n=== Configuration: dataset={dataset_name}, mode={mode} ===")

    loader = GraphDataLoader(name=dataset_name, mode=mode, heterogeneous=False)
    graphs = loader.load_all()
    
    if not graphs:
        print("No graphs found in dataset. Aborting.")
        return None

    # Get sample to initialize model
    sample_g = next(iter(graphs.values()))
    input_dim = sample_g.x.shape[1]
    
    model = TestGraphNetwork(input_dim=input_dim, global_dim=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized GNN model on {device.type.upper()} with input_dim={input_dim} | Total Parameters: {total_params}")

    latencies = []
    nodes = []
    edges = []
    depths = []
    widths = []
    graph_ids = []
    
    print(f"Benchmarking {len(graphs)} graphs (warmup={warmup}, runs={runs})...")
    for gid, graph in graphs.items():
        mean_t, median_t, std_t = benchmark_single_graph(model, graph, device, warmup, runs)
        latencies.append(mean_t)
        nodes.append(graph.num_nodes)
        edges.append(graph.num_edges)
        depths.append(getattr(graph, "tree_depth", 0))
        widths.append(getattr(graph, "tree_width", 0))
        graph_ids.append(gid)

    # General Statistics
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    std_lat = np.std(latencies)
    
    print("\n--- Latency Results (ms) ---")
    print(f"  Mean:   {mean_lat:.4f} ms")
    print(f"  Median: {median_lat:.4f} ms")
    print(f"  StdDev: {std_lat:.4f} ms")
    print(f"  Min:    {np.min(latencies):.4f} ms")
    print(f"  Max:    {np.max(latencies):.4f} ms")
    throughput = 1000.0 / mean_lat
    print(f"  Throughput: {throughput:.2f} graphs/sec")

    # Pearson Correlation Coefficient Calculation
    def pearson_corr(x, y):
        x_arr, y_arr = np.array(x), np.array(y)
        if len(x_arr) <= 1 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return 0.0
        return np.corrcoef(x_arr, y_arr)[0, 1]

    corr_nodes = pearson_corr(nodes, latencies)
    corr_edges = pearson_corr(edges, latencies)
    corr_depths = pearson_corr(depths, latencies)
    corr_widths = pearson_corr(widths, latencies)

    print("\n--- Structural Correlations (Pearson r) ---")
    print(f"  Nodes vs Latency:  {corr_nodes:+.4f}")
    print(f"  Edges vs Latency:  {corr_edges:+.4f}")
    print(f"  Depth vs Latency:  {corr_depths:+.4f}")
    print(f"  Width vs Latency:  {corr_widths:+.4f}")
    
    # Save CSV
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / f"benchmark_graphs_{mode}.csv"
    df = pd.DataFrame({
        "graph_id": graph_ids,
        "nodes": nodes,
        "edges": edges,
        "depth": depths,
        "width": widths,
        "latency_mean_ms": latencies
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved graph statistics to CSV: {csv_path}")

    # Generate Plot
    if plot_path is None:
        plot_path = Path(__file__).resolve().parent / f"benchmark_graphs_{mode}.png"
    plot_single_config(latencies, nodes, graph_ids, plot_path)
    
    # MLflow Tracking Integration
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(f"GNN_Inference_Benchmark_{dataset_name}")
            with mlflow.start_run(run_name=f"{mode}_{device.type}"):
                mlflow.log_params({
                    "mode": mode,
                    "device": device.type,
                    "warmup": warmup,
                    "runs": runs,
                    "input_dim": input_dim,
                    "parameters": total_params
                })
                mlflow.log_metrics({
                    "latency_mean_ms": mean_lat,
                    "latency_median_ms": median_lat,
                    "latency_std_ms": std_lat,
                    "latency_min_ms": np.min(latencies),
                    "latency_max_ms": np.max(latencies),
                    "throughput_gps": throughput,
                    "pearson_r_nodes": corr_nodes,
                    "pearson_r_edges": corr_edges,
                    "pearson_r_depth": corr_depths,
                    "pearson_r_width": corr_widths
                })
                mlflow.log_artifact(str(csv_path))
                mlflow.log_artifact(str(plot_path))
                print("[MLflow] Run successfully logged parameters, metrics, and artifacts.")
        except Exception as e:
            print(f"[MLflow] Logging failed or skipped (is server offline?): {e}")

    return {
        "mode": mode,
        "mean": mean_lat,
        "median": median_lat,
        "std": std_lat,
        "throughput": throughput,
        "params": total_params
    }


def run_sweep(dataset_name, device, warmup, runs, csv_path=None, plot_path=None, use_mlflow=True):
    """Runs a sweep over all GNN modes (single feature schema)."""
    print(f"\n================ Running Configuration Sweep (Device: {device.type.upper()}) ================")

    configs = ["graph", "tree", "tree_derivatives"]

    results = []
    for mode in configs:
        try:
            # We don't save single CSV/plot files inside sweep mode to avoid clutter
            res_loader = GraphDataLoader(name=dataset_name, mode=mode, heterogeneous=False)
            graphs = res_loader.load_all()
            if not graphs:
                continue
            sample_g = next(iter(graphs.values()))
            input_dim = sample_g.x.shape[1]
            model = TestGraphNetwork(input_dim=input_dim, global_dim=2).to(device)

            latencies = []
            for gid, graph in graphs.items():
                mean_t, _, _ = benchmark_single_graph(model, graph, device, warmup, runs)
                latencies.append(mean_t)

            mean_lat = np.mean(latencies)
            results.append({
                "mode": mode,
                "input_dim": input_dim,
                "mean": mean_lat,
                "throughput": 1000.0 / mean_lat,
                "params": sum(p.numel() for p in model.parameters())
            })
            print(f"Finished Mode={mode:<16} | Mean Latency={mean_lat:.4f} ms")
        except Exception as e:
            print(f"Failed to benchmark configuration Mode={mode}: {e}")

    print("\n\n======================= COMPARATIVE SWEEP RESULTS =======================")
    print("| GNN Mode | Input Dim | Parameters | Mean Latency (ms) | Throughput (g/s) |")
    print("| --- | --- | --- | --- | --- |")
    for r in results:
        print(f"| {r['mode']:<16} | {r['input_dim']:<9} | {r['params']:<10} | {r['mean']:<17.4f} | {r['throughput']:<16.2f} |")
    print("=========================================================================\n")

    # Save CSV
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "benchmark_sweep.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved comparative sweep results to CSV: {csv_path}")

    # Generate Bar Chart
    if plot_path is None:
        plot_path = Path(__file__).resolve().parent / "benchmark_sweep.png"
    plot_sweep(results, plot_path)
    
    # MLflow Tracking Integration
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(f"GNN_Inference_Benchmark_{dataset_name}")
            with mlflow.start_run(run_name=f"sweep_{device.type}"):
                mlflow.log_params({
                    "benchmark_type": "sweep",
                    "device": device.type,
                    "warmup": warmup,
                    "runs": runs,
                    "num_configurations": len(results)
                })
                # Log comparative metrics for quick comparison
                for r in results:
                    cfg_label = f"{r['mode']}"
                    mlflow.log_metric(f"{cfg_label}_latency_mean_ms", r["mean"])
                    mlflow.log_metric(f"{cfg_label}_throughput_gps", r["throughput"])
                    
                mlflow.log_artifact(str(csv_path))
                mlflow.log_artifact(str(plot_path))
                print("[MLflow] Sweep successfully logged parameters, metrics, and artifacts.")
        except Exception as e:
            print(f"[MLflow] Logging failed or skipped (is server offline?): {e}")


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = args.device
    device = torch.device(device_type)
    
    use_mlflow = not args.no_mlflow

    if args.sweep:
        run_sweep(args.dataset, device, args.warmup, args.runs, args.csv_path, args.plot_path, use_mlflow)
    else:
        run_single_config(args.dataset, args.mode, device, args.warmup, args.runs, args.csv_path, args.plot_path, use_mlflow)


if __name__ == "__main__":
    main()
