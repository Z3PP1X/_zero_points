#!/usr/bin/env python3
import sys
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Resolve paths
script_dir = Path(__file__).resolve().parent
gnn_root = script_dir.parent
src_root = gnn_root.parent

# Import configs generator
sys.path.insert(0, str(script_dir))
from configs_gen import generate_configs


def plot_results(df, output_dir):
    try:
        # Create output directory for plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Bar Chart: Latency Comparison by Layer Type and Compile Flag
        plt.figure(figsize=(10, 6))
        # Group by layer_type and compile
        grouped = df.groupby(["layer_type", "compile"])["latency_ms"].mean().unstack()
        
        ax = grouped.plot(kind="bar", color=["#4f81bd", "#c0504d"], edgecolor="gray", figsize=(10, 6))
        plt.title("GNN Inference Latency: Compiled vs. Uncompiled (CPU)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Layer Type", fontsize=12)
        plt.ylabel("Mean Latency (ms)", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(["compile=False", "compile=True"], fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "compile_comparison.png", dpi=150)
        plt.close()
        
        # 2. Line Plot: Latency vs dim_inner by Layer Type (compile=False)
        plt.figure(figsize=(10, 6))
        uncompiled_df = df[df["compile"] == False]
        layer_types = uncompiled_df["layer_type"].unique()
        colors = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2"]
        
        for idx, l_type in enumerate(layer_types):
            subset = uncompiled_df[uncompiled_df["layer_type"] == l_type]
            dim_grouped = subset.groupby("dim_inner")["latency_ms"].mean().reset_index()
            dim_grouped = dim_grouped.sort_values(by="dim_inner")
            plt.plot(
                dim_grouped["dim_inner"], 
                dim_grouped["latency_ms"], 
                label=l_type, 
                color=colors[idx % len(colors)],
                marker='o',
                linewidth=2.5,
                markersize=8
            )
            
        plt.title("Impact of Inner Dimension (dim_inner) on GNN Latency (compile=False)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Inner Dimension Size", fontsize=12)
        plt.ylabel("Mean Latency (ms)", fontsize=12)
        plt.xticks([64, 128, 256, 512])
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "dim_inner_comparison.png", dpi=150)
        plt.close()

        # 3. Bar Chart: Throughput Comparison (graphs/second)
        plt.figure(figsize=(10, 6))
        throughput_grouped = df.groupby(["layer_type", "compile"])["throughput_gps"].mean().unstack()
        ax = throughput_grouped.plot(kind="bar", color=["#4f81bd", "#c0504d"], edgecolor="gray", figsize=(10, 6))
        plt.title("GNN Inference Throughput (graphs/sec) on CPU", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Layer Type", fontsize=12)
        plt.ylabel("Throughput (graphs/s)", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(["compile=False", "compile=True"], fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "throughput_comparison.png", dpi=150)
        plt.close()

        print(f"Successfully generated comparative plots in: {plots_dir}")
    except Exception as e:
        print(f"Warning: Failed to generate visualization plots: {e}")


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "results" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_base = script_dir / "config_base.yaml"
    grid_path = script_dir / "grid.yaml"
    configs_dir = script_dir / "configs"
    
    # 1. Generate all config files from grid
    print("[Orchestrator] Generating configuration files...")
    config_files = generate_configs(config_base, grid_path, configs_dir)
    print(f"[Orchestrator] Generated {len(config_files)} configuration files.")
    
    # 2. Iterate through configs and run benchmark.py in a separate process
    all_results = []
    
    python_exe = sys.executable
    benchmark_script = script_dir / "benchmark.py"
    
    for idx, cfg_file in enumerate(config_files):
        print(f"\n[Orchestrator] [{idx+1}/{len(config_files)}] Running benchmark for {cfg_file.name}...")
        
        run_csv = output_dir / "runs" / f"res_{cfg_file.stem}.csv"
        
        # Execute benchmark.py in isolated process to ensure clean PyTorch cache/GraphGym cfg
        cmd = [
            python_exe,
            str(benchmark_script),
            "--cfg", str(cfg_file),
            "--runs", "1000",
            "--warmup", "20",
            "--out_csv", str(run_csv)
        ]
        
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(res.stdout)
            
            # Load the results CSV of this run
            if run_csv.exists():
                run_df = pd.read_csv(run_csv)
                all_results.append(run_df)
            else:
                print(f"[Error] Run CSV not found at: {run_csv}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed to benchmark {cfg_file.name} (exit code {e.returncode})")
            print(e.stderr)
            
    # 3. Aggregate all results
    if all_results:
        aggregated_df = pd.concat(all_results, ignore_index=True)
        agg_csv = output_dir / "aggregated_results.csv"
        aggregated_df.to_csv(agg_csv, index=False)
        print(f"\n[Orchestrator] Successfully aggregated all runs to: {agg_csv}")
        
        # 4. Generate plots
        plot_results(aggregated_df, output_dir)
        
        # Save a summary markdown file in the run folder for quick review
        summary_file = output_dir / "summary.md"
        with open(summary_file, "w", encoding="utf-8") as sf:
            sf.write(f"# GNN Inference Benchmark Run Summary ({timestamp})\n\n")
            sf.write("## Comparative Architecture Mean Latencies (ms)\n\n")
            
            summary_df = aggregated_df.groupby(["layer_type", "compile", "layers_mp", "dim_inner"])["latency_ms"].mean().reset_index()
            # Sort by latency
            summary_df = summary_df.sort_values(by="latency_ms")
            
            sf.write("| Layer Type | Compile | MP Layers | Dim Inner | Mean Latency (ms) |\n")
            sf.write("| --- | --- | --- | --- | --- |\n")
            for _, r in summary_df.iterrows():
                sf.write(f"| {r['layer_type']} | {r['compile']} | {r['layers_mp']} | {r['dim_inner']} | {r['latency_ms']:.4f} |\n")
                
        print(f"[Orchestrator] Summary report generated at: {summary_file}")
    else:
        print("[Error] No benchmark results to aggregate.")


if __name__ == "__main__":
    main()
