#!/usr/bin/env python3
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Resolve paths
gnn_root = Path(__file__).resolve().parents[1]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
import gnn.supervised_learning.loader_graphgym  # Registers custom config/loader
from gnn.supervised_learning.preprocessing import GraphPipeline, ProblemRunDataset
from gnn.shared.utils.graph_loader import GraphDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="GNN Inference Benchmarking")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the GraphGym config file (YAML)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of inference steps to measure",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Path to save the output CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Reset and load configuration
    set_cfg(cfg)
    
    class GymArgs:
        cfg_file = args.cfg
        opts = []
    
    load_cfg(cfg, GymArgs())
    
    # Force CPU accelerator
    cfg.accelerator = "cpu"
    device = torch.device("cpu")
    
    print(f"\n========================================================")
    print(f"Benchmarking Config: {Path(args.cfg).name}")
    print(f"Architecture: Type={cfg.gnn.layer_type}, MP_Layers={cfg.gnn.layers_mp}, Dim_Inner={cfg.gnn.dim_inner}, Pooling={cfg.model.graph_pooling}")
    print(f"========================================================")
    
    # 2. Load curated dataset (always run_20260603_123013/parallel_benchmark_results)
    edge_direction = getattr(cfg.expression_graph, "edge_direction", "top_down")
    loader = GraphDataLoader(
        name="run_20260603_123013/parallel_benchmark_results",
        mode=cfg.expression_graph.mode,
        heterogeneous=False,
        edge_direction=edge_direction,
    )
    pipeline = GraphPipeline(
        dataset_name="run_20260603_123013/parallel_benchmark_results",
        seed=cfg.seed,
        mode=cfg.expression_graph.mode,
        graph_loader=loader,
        synthetic=False,
    )
    pipeline.pipe(test_size=0.2, batch_size=cfg.train.batch_size, stratify=False)
    df = pipeline.loader.data
    graphs = pipeline.graphs
    df_matched = df[df["problem_id"].isin(graphs.keys())].copy()
    df_unique = df_matched.drop_duplicates(subset=["problem_id"]).copy()
    
    curated_dataset = ProblemRunDataset(
        df_unique,
        graphs,
        mode=cfg.expression_graph.mode,
        active_features=None
    )
    
    print(f"Discovered {len(curated_dataset)} graphs in curated dataset.")
    
    # 3. Setup GNN dimensions in cfg
    cfg.share.dim_in = curated_dataset[0].x.shape[1]
    cfg.share.dim_out = 1
    
    # 4. Construct GNN model (untrained/empty network)
    model = create_model(to_device=True)
    model.eval()
    
    results = []
    
    # 5. Sweep through compile flag False and True
    for compile_flag in [False, True]:
        print(f"\n--- Condition: compile={compile_flag} ---")
        
        if compile_flag:
            if hasattr(torch, "compile"):
                try:
                    print("Compiling model with torch.compile(dynamic=True)...")
                    compiled_model = torch.compile(model, dynamic=True)
                except Exception as e:
                    print(f"Compilation error: {e}. Falling back to uncompiled.")
                    compiled_model = model
            else:
                print("torch.compile not supported. Falling back.")
                compiled_model = model
        else:
            compiled_model = model
            
    # 5. Collate all graphs into a single Batch
    from torch_geometric.data import Batch
    collatable_list = []
    for i in range(len(curated_dataset)):
        data = curated_dataset[i]
        collatable_list.append(data)
    batch = Batch.from_data_list(collatable_list).to(device)
    
    total_nodes = batch.num_nodes
    total_edges = batch.num_edges
    print(f"Collated {len(curated_dataset)} graphs into a single Batch (Nodes: {total_nodes}, Edges: {total_edges})")
    
    results = []
    
    # 6. Sweep through compile flag False and True
    for compile_flag in [False, True]:
        print(f"\n--- Condition: compile={compile_flag} ---")
        
        if compile_flag:
            if hasattr(torch, "compile"):
                try:
                    print("Compiling model with torch.compile()...")
                    compiled_model = torch.compile(model)
                except Exception as e:
                    print(f"Compilation error: {e}. Falling back to uncompiled.")
                    compiled_model = model
            else:
                print("torch.compile not supported. Falling back.")
                compiled_model = model
        else:
            compiled_model = model
            
        # Warmup
        print("Running warmup steps...")
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = compiled_model(batch.clone())
        
        # Benchmark timing runs
        print(f"Measuring timing for {args.runs} runs...")
        with torch.no_grad():
            t_start = time.perf_counter()
            for _ in range(args.runs):
                _ = compiled_model(batch.clone())
            t_end = time.perf_counter()
            
        total_time_ms = (t_end - t_start) * 1000.0
        avg_time_ms = total_time_ms / args.runs
        throughput = (len(curated_dataset) * 1000.0) / avg_time_ms if avg_time_ms > 0 else 0.0
        
        print(f"  Compile: {compile_flag:<5} | Latency: {avg_time_ms:.4f} ms | Throughput: {throughput:.1f} graphs/s")
        
        results.append({
            "config_name": Path(args.cfg).stem,
            "layer_type": cfg.gnn.layer_type,
            "layers_mp": cfg.gnn.layers_mp,
            "dim_inner": cfg.gnn.dim_inner,
            "pooling": cfg.model.graph_pooling,
            "nodes": total_nodes,
            "edges": total_edges,
            "compile": compile_flag,
            "latency_ms": avg_time_ms,
            "throughput_gps": throughput,
        })
            
    # 7. Save to CSV if specified
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_path, index=False)
        print(f"\nSaved benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
