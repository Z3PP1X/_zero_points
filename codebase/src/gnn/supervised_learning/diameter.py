import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

# Add project root to sys.path
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.supervised_config import (
    bootstrap_graphgym_cfg,
    apply_expression_graph_overrides,
    validate_layer_type,
)
from gnn.supervised_learning.preprocessing import GraphPipeline
from gnn.shared.utils.unified_loader import UnifiedDataLoader

def main():
    parser = argparse.ArgumentParser(description="Analyze GNN dataset graph diameters.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_supervised.yaml",
        help="Base configuration file"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args.config
    
    print(f"Loading configuration from: {config_path}")
    cfg = bootstrap_graphgym_cfg(config_path)
    
    dataset_name = cfg.dataset.name
    layer_type = cfg.gnn.layer_type
    mode = cfg.expression_graph.mode
    add_kappa = getattr(cfg.expression_graph, "add_kappa", False)

    print(f"Dataset: {dataset_name}")
    print(f"Add Kappa: {add_kappa}")

    # Load dataset pipeline
    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=dataset_name,
        mode=mode,
        add_kappa=add_kappa,
    )

    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        seed=cfg.seed,
        mode=mode,
        active_features=None if not cfg.expression_graph.active_features else [
            f.strip() for f in str(cfg.expression_graph.active_features).split(",") if f.strip()
        ],
        unified_loader=unified_loader,
        synthetic=cfg.expression_graph.synthetic,
        synthetic_dataset_name=cfg.expression_graph.synthetic_dataset,
        layer_type=layer_type,
        add_kappa=add_kappa,
    )
    
    print("Loading datasets through pipeline...")
    # This splits and processes the datasets
    _, _, _ = pipeline.pipe(
        test_size=0.2,
        batch_size=cfg.train.batch_size,
        stratify=True,
        num_workers=0,
    )
    
    train_ds = pipeline.train_dataset
    test_ds = pipeline.test_dataset
    curated_ds = getattr(pipeline, "curated_dataset", None)

    records = []
    
    def process_dataset(ds, split_name):
        if ds is None:
            return
        print(f"Calculating diameters for {split_name} split ({len(ds)} graphs)...")
        for i, data in enumerate(ds):
            # Convert PyG Data to NetworkX (undirected)
            g = to_networkx(data, to_undirected=True)
            if g.number_of_nodes() <= 1:
                diameter = 0
            else:
                if nx.is_connected(g):
                    diameter = nx.diameter(g)
                else:
                    # For disconnected graphs, take the maximum diameter of its connected components
                    components = list(nx.connected_components(g))
                    diameters = [nx.diameter(g.subgraph(c)) for c in components if len(c) > 1]
                    diameter = max(diameters) if diameters else 0
            records.append({
                "split": split_name,
                "graph_index": i,
                "num_nodes": data.num_nodes,
                "num_edges": data.num_edges,
                "diameter": diameter
            })
            
    process_dataset(train_ds, "Train (Synthetic)")
    process_dataset(test_ds, "Val (Synthetic)")
    if curated_ds is not None:
        process_dataset(curated_ds, "Curated (Real)")
        
    df = pd.DataFrame(records)
    
    # Save results to CSV
    csv_out = script_dir / "graph_diameters.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved diameter results to: {csv_out}")
    
    # Plot results
    splits = df["split"].unique()
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Color palette matching other plots
    colors = {
        "Train (Synthetic)": "#2A9D8F",
        "Val (Synthetic)": "#E76F51",
        "Test (Synthetic)": "#F4A261",
        "Curated (Real)": "#264653"
    }
    
    split_to_x = {split: i for i, split in enumerate(splits)}
    
    print("\n--- Summary Statistics ---")
    for split in splits:
        split_df = df[df["split"] == split]
        x_val = split_to_x[split]
        
        # Calculate stats
        mean_val = split_df["diameter"].mean()
        std_val = split_df["diameter"].std()
        
        # Jitter points to show distribution density (point cloud)
        jitter = np.random.uniform(-0.15, 0.15, size=len(split_df))
        xs = x_val + jitter
        
        # Plot point cloud
        color = colors.get(split, "#457B9D")
        plt.scatter(
            xs, 
            split_df["diameter"], 
            alpha=0.35, 
            color=color, 
            edgecolors="none", 
            s=25, 
            label=f"{split} (n={len(split_df)})"
        )
        
        # Plot mean value horizontal bar and diamond marker
        plt.hlines(mean_val, x_val - 0.25, x_val + 0.25, colors="#D90429", linewidth=2.5, zorder=5)
        plt.scatter(x_val, mean_val, color="#D90429", marker="D", s=60, zorder=6)
        
        print(f"Split: {split:20s} | Mean: {mean_val:6.2f} | Std: {std_val:6.2f} | Max: {split_df['diameter'].max():3d}")
        
    plt.title("Graph Diameter Distribution (Point Cloud with Means)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Graph Diameter (Longest Shortest Path)", fontsize=11)
    
    # Styling and ticks
    plt.xticks(range(len(splits)), splits, fontsize=10, fontweight="semibold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    # Custom legend elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#D90429", lw=2.5, label="Mean Diameter Marker"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D90429", markersize=8, label="Mean Value")
    ]
    for split in splits:
        color = colors.get(split, "#457B9D")
        legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, alpha=0.7, label=f"{split} Graph"))
        
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)
    plt.ylim(-0.5, df["diameter"].max() + 1.5)
    
    plot_out = script_dir / "graph_diameters.png"
    plt.savefig(plot_out, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization plot to: {plot_out}")

if __name__ == "__main__":
    main()
