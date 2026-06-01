import sys
from pathlib import Path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader, register_config
from yacs.config import CfgNode as CN

# Dynamic sys.path resolution
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.preprocessing import GraphPipeline


def set_custom_cfg(cfg):
    """
    Registers custom expression graph config parameters inside GraphGym.
    Allows specifying experimental variables inside YAML configuration.
    """
    cfg.expression_graph = CN()
    cfg.expression_graph.mode = "graph"            # "graph" or "tree"
    cfg.expression_graph.enrich = False             # True or False
    cfg.expression_graph.active_features = ""       # Comma-separated list or empty for all


register_config("expression_graph", set_custom_cfg)


def load_custom_expression_graphs():
    """
    GraphGym Loader for custom expression graphs.
    Uses Dependency Injection by reading dataset properties dynamically from GraphGym's global config:
    - cfg.dataset.name: e.g., "run_20260408_160456/dataset_4" (injects dataset selection)
    - cfg.train.batch_size: injects batch size
    - cfg.seed: injects seed (defaults to 42001 if not set)
    - cfg.expression_graph.mode: injects GNN mode ("graph" or "tree")
    - cfg.expression_graph.enrich: injects GNN feature enrichment (True or False)
    - cfg.expression_graph.active_features: injects list of active features
    """
    dataset_name = cfg.dataset.name
    batch_size = cfg.train.batch_size
    seed = getattr(cfg, "seed", 42001)

    # Read custom injected expression graph properties
    mode = getattr(cfg.expression_graph, "mode", "graph")
    enrich = getattr(cfg.expression_graph, "enrich", False)
    active_features_str = getattr(cfg.expression_graph, "active_features", "")

    active_features = None
    if active_features_str:
        active_features = [f.strip() for f in active_features_str.split(",") if f.strip()]

    # Resolve run_key prefix to find the matching datasets folder
    if "/" in dataset_name:
        run_key, _ = dataset_name.split("/", 1)
    else:
        run_key = dataset_name

    repo_root = Path(__file__).resolve().parents[4]
    experiments_dir = repo_root / "_datasets" / run_key / "graphs"

    print(f"\n--- [GraphGym Dependency Injection] ---")
    print(f"  Injecting Dataset Name:  {dataset_name}")
    print(f"  Resolved Directory:      {experiments_dir}")
    print(f"  Injected Batch Size:     {batch_size}")
    print(f"  Injected Random Seed:    {seed}")
    print(f"  Injected Mode:           {mode}")
    print(f"  Injected Enrich:         {enrich}")
    if active_features is not None:
        print(f"  Injected Features ({len(active_features)}): {active_features}")
    else:
        print(f"  Injected Features:       ALL")
    print(f"----------------------------------------\n")

    # Instantiate the GraphPipeline using the injected dependencies
    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        experiments_dir=str(experiments_dir),
        seed=seed,
        mode=mode,
        enrich=enrich,
        active_features=active_features,
    )

    # Use pipeline loaders as requested
    train_loader, test_loader, _ = pipeline.pipe(
        test_size=0.2,
        batch_size=batch_size,
    )

    # GraphGym expects a dict with train, val, and test loaders
    return {
        "train": train_loader,
        "val": test_loader,
        "test": test_loader
    }


# Register the dependency-injected loader in the GraphGym global registry
register_loader("custom_expression_graphs", load_custom_expression_graphs)
