import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.supervised_config import (
    bootstrap_graphgym_cfg,
    create_graphgym_model,
    apply_expression_graph_overrides,
    validate_layer_type,
)
from gnn.supervised_learning.preprocessing import GraphPipeline
from gnn.shared.utils.unified_loader import UnifiedDataLoader
from gnn.shared.utils.graph_utils import compute_normalized_dirichlet_energy

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

def evaluate_dirichlet_for_depth(model, val_loader):
    model.eval()
    
    mp_embeddings = []
    def hook_fn(module, inputs, outputs):
        mp_embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu(), getattr(outputs, 'edge_attr', None)))
        
    hook_handle = model.mp.register_forward_hook(hook_fn)
    
    # Run a single evaluation epoch (one pass over the val_loader)
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            batch.split = "val"
            # Forward pass: GNN runs and hook_fn captures embeddings
            _ = model(batch)
            
    hook_handle.remove()
    
    # Compute average dirichlet energy
    energies = []
    for x, edge_index, edge_attr in mp_embeddings:
        energy = compute_normalized_dirichlet_energy(x, edge_index)
        energies.append(energy)
        
    avg_energy = sum(energies) / len(energies) if energies else 0.0
    return avg_energy

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dirichlet Energy vs GNN Depth.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_supervised.yaml",
        help="Base configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name override"
    )
    parser.add_argument(
        "--layer-type",
        type=str,
        default=None,
        help="Layer type override (e.g., gatv2conv, gcnconv)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args.config
    
    print(f"Loading configuration from: {config_path}")
    cfg = bootstrap_graphgym_cfg(config_path)
    
    dataset_name = args.dataset or cfg.dataset.name
    layer_type = args.layer_type or cfg.gnn.layer_type
    mode = cfg.expression_graph.mode
    edge_direction = cfg.expression_graph.edge_direction

    print(f"Dataset: {dataset_name}")
    print(f"Layer Type: {layer_type}")
    print(f"Edge Direction: {edge_direction}")

    # Load dataset pipeline
    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=dataset_name,
        mode=mode,
        edge_direction=edge_direction,
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
    )
    
    _, val_loader, _ = pipeline.pipe(
        test_size=0.2,
        batch_size=cfg.train.batch_size,
        stratify=True,
        num_workers=0,
    )
    
    sample = pipeline.train_dataset[0]
    dim_in = sample.x.shape[1]
    
    depths = [2, 4, 6, 8, 12, 16, 24]
    results = []
    
    print("\n--- Evaluating Dirichlet Energy vs. GNN Depth ---")
    for depth in depths:
        # Override layers_mp in config
        cfg.gnn.layers_mp = depth
        cfg.gnn.layer_type = validate_layer_type(layer_type)
        
        # Instantiate GNN model
        model = create_graphgym_model(cfg, dim_in=dim_in, device=DEVICE)
        
        # Evaluate Dirichlet Energy
        avg_energy = evaluate_dirichlet_for_depth(model, val_loader)
        print(f"Message Passing Layers: {depth:2d} | Normalized Dirichlet Energy: {avg_energy:.6f}")
        results.append({"layers_mp": depth, "dirichlet_energy": avg_energy})
        
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_out = script_dir / "dirichlet_energy_by_depth.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved CSV results to: {csv_out}")
    
    # Plot results
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(df["layers_mp"], df["dirichlet_energy"], marker="o", color="#2A9D8F", linewidth=2, markersize=6)
    plt.title(f"Normalized Dirichlet Energy vs. GNN MP Layers\n(Dataset: {dataset_name}, Layer: {layer_type})", fontsize=12, fontweight="bold")
    plt.xlabel("Message Passing Layers (Depth)", fontsize=10)
    plt.ylabel("Normalized Dirichlet Energy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, max(1.1 * df["dirichlet_energy"].max(), 0.5))
    
    plot_out = script_dir / "dirichlet_energy_vs_depth.png"
    plt.savefig(plot_out, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization plot to: {plot_out}")

if __name__ == "__main__":
    main()
