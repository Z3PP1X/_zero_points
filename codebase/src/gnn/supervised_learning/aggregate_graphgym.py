import os
import re
from pathlib import Path
from torch_geometric.graphgym.utils.agg_runs import agg_runs, agg_batch

def main():
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"Error: results directory not found at {results_dir}")
        return

    # 1. Rename directories to the PyG/GraphGym expected format if necessary
    print("Checking and renaming directories to PyG GraphGym convention...")
    for run in os.listdir(results_dir):
        if run == "agg":
            continue
        
        # Match current underscore-based format (flexible: captures all key=value pairs)
        match = re.match(
            r"grid_(.+)",
            run
        )
        if match:
            # Convert underscore-separated key_value pairs to hyphen-separated key=value pairs
            parts = match.group(1)
            # Split into key-value pairs: key1_val1_key2_val2_...
            # The pattern is: word_word_word_value where value can be numeric
            # Use a more robust approach: find known parameter names
            known_params = ['layer_type', 'layers_mp', 'dim_inner', 'dropout', 'graph_pooling', 'act']
            
            remaining = parts
            extracted = {}
            for param in known_params:
                pattern = rf"{param}_([a-zA-Z0-9\.]+)"
                m = re.search(pattern, remaining)
                if m:
                    extracted[param] = m.group(1)
            
            if extracted:
                new_name = "grid-" + "-".join(f"{k}={v}" for k, v in extracted.items())
                old_path = results_dir / run
                new_path = results_dir / new_name
                if old_path != new_path and not new_path.exists():
                    os.rename(old_path, new_path)
                    print(f"  Renamed: {run} -> {new_name}")

    print("Directories normalized. Running PyG internal aggregation...")
    
    # 2. Run seed-level aggregation on all run directories
    for run in os.listdir(results_dir):
        run_dir = results_dir / run
        if run_dir.is_dir() and run != "agg":
            try:
                agg_runs(str(run_dir), metric_best='accuracy')
            except Exception as e:
                print(f"Warning: Failed to aggregate runs for {run}: {e}")

    # 3. Run batch-level aggregation across all configurations using PyG built-in
    try:
        agg_batch(str(results_dir), metric_best='accuracy')
        print(f"Successfully aggregated batch results inside: {results_dir / 'agg'}")
    except Exception as e:
        print(f"Error running batch aggregation: {e}")
    
    # 4. Report which CSV files were generated
    agg_dir = results_dir / "agg"
    if agg_dir.exists():
        csvs = sorted(agg_dir.glob("*.csv"))
        print(f"\nGenerated {len(csvs)} aggregated CSV files:")
        for csv in csvs:
            print(f"  - {csv.name}")

if __name__ == "__main__":
    main()
