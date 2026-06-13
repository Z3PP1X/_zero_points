import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Dynamic sys.path resolution to support package imports when run as scripts
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.dataset import DatasetLoader
from gnn.shared.utils.graph_utils import (
    EDGE_FEATURE_SCHEMA,
    NODE_FEATURE_SCHEMA,
    slice_active_features,
)
from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.utils.unified_loader import UnifiedDataLoader
from gnn.supervised_learning.supervised_config import (
    architecture_from_layer_type,
    validate_layer_type,
)


class FeatureEngineering:
    """Contains offline feature tagging and calculation logic for supervised learning."""
    def __init__(self, loader: DatasetLoader):
        self._loader = loader

    def _tag_faster_algorithm(self):
        """Set binary labels for the faster algorithm: 1: Newton, 0: gMGF"""
        boundaries = [
            self._loader.data["Newton_absTime"] < self._loader.data["GMGF_absTime"],
            self._loader.data["Newton_absTime"] > self._loader.data["GMGF_absTime"],
        ]
        values = [1, 0]
        self._loader.add_column("faster_algorithm", np.select(boundaries, values))

    def _conserve_relationships(self):
        """Conserve relationships between absolute times"""
        self._loader.add_column(
            "conserved_step_rel",
            self._loader.data["Newton_iterSteps"] / self._loader.data["GMGF_iterSteps"],
        )


class GraphPipeline:
    def __init__(
        self,
        dataset_name: str,
        experiments_dir: str = "",
        seed: int = 42,
        mode: str = "graph",
        active_features: list[str] | None = None,
        graph_loader: GraphDataLoader | None = None,
        unified_loader: UnifiedDataLoader | None = None,
        synthetic: bool = False,
        synthetic_dataset_name: str | None = None,
        layer_type: str = "gatv2conv",
        heterogeneous: bool = False,
        add_kappa: bool = False,
        add_virtual_supernode: bool = False,
    ):
        self.seed = seed
        self.mode = mode
        self.active_features = active_features
        self.synthetic = synthetic
        self.synthetic_dataset_name = synthetic_dataset_name if synthetic_dataset_name else None
        self.layer_type = validate_layer_type(layer_type)
        self.heterogeneous = heterogeneous
        self.add_kappa = add_kappa
        self.add_virtual_supernode = add_virtual_supernode

        # Use unified_loader or get/create singleton instance
        if unified_loader is not None:
            self.unified_loader = unified_loader
        else:
            self.unified_loader = UnifiedDataLoader.get_instance(
                dataset_name=dataset_name,
                mode=mode,
                heterogeneous=heterogeneous,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
            )

        if self.synthetic and self.synthetic_dataset_name is not None:
            self.synthetic_unified_loader = UnifiedDataLoader.get_instance(
                dataset_name=self.synthetic_dataset_name,
                mode=mode,
                heterogeneous=heterogeneous,
                is_synthetic=True,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
            )
        else:
            self.synthetic_unified_loader = None

        # Backward compatibility aliases
        self.loader = self.unified_loader.dataset_loader
        self.graph_loader = self.unified_loader.graph_loader
        
        fe = FeatureEngineering(self.loader)
        fe._tag_faster_algorithm()

        if self.synthetic_unified_loader is not None:
            fe_synth = FeatureEngineering(self.synthetic_unified_loader.dataset_loader)
            fe_synth._tag_faster_algorithm()
        
        # Override graph_loader if explicitly passed (for legacy call sites/tests).
        # Build the kappa_map either way so each graph merges only its active
        # h-function; without it load_all() falls back to merging ALL kappas,
        # inflating every graph ~18x and blocking the pipeline.
        kappa_map = self.unified_loader.build_kappa_map() if self.add_kappa else None
        if graph_loader is not None:
            self.graph_loader = graph_loader
            self.graphs = self.graph_loader.load_all(kappa_map=kappa_map)
        else:
            self.graphs = self.unified_loader.load_all(kappa_map=kappa_map)
            
        self.graph_pipeline = self
        
        self.train_loader = None
        self.test_loader = None
        self.curated_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.curated_dataset = None
        self.class_weights = None
        self.Y_train = None
        self.Y_test = None
        self.train_pids = None
        self.test_pids = None

    def pipe(
        self, test_size=0.2, batch_size=32, stratify: bool = False, num_workers: int = 0
    ):
        self._validate_edge_features()
        if self.synthetic:
            # --- Synthetic Mode ---
            if self.synthetic_unified_loader is None:
                raise ValueError("Synthetic mode is active, but synthetic_dataset_name is not provided.")
            
            # 1. Curated dataset as validation (test) data
            df_curated = self.unified_loader.dataset_loader.data
            kappa_map_curated = self.unified_loader.build_kappa_map() if self.add_kappa else None
            graphs_curated = self.unified_loader.load_all(kappa_map=kappa_map_curated)
            graph_ids_curated = set(graphs_curated.keys())
            test_df = df_curated[df_curated["problem_id"].isin(graph_ids_curated)].copy()

            # 2. Synthetic dataset as training data
            df_synth = self.synthetic_unified_loader.dataset_loader.data
            kappa_map_synth = self.synthetic_unified_loader.build_kappa_map() if self.add_kappa else None
            graphs_synth = self.synthetic_unified_loader.load_all(kappa_map=kappa_map_synth)
            graph_ids_synth = set(graphs_synth.keys())
            train_df = df_synth[df_synth["problem_id"].isin(graph_ids_synth)].copy()
            
            # Ensure "faster_algorithm" labels are tagged
            fe_curated = FeatureEngineering(self.unified_loader.dataset_loader)
            fe_curated._tag_faster_algorithm()
            
            fe_synth = FeatureEngineering(self.synthetic_unified_loader.dataset_loader)
            fe_synth._tag_faster_algorithm()
            
            # Refresh data frames with newly tagged labels
            test_df = df_curated[df_curated["problem_id"].isin(graph_ids_curated)].copy()
            train_df = df_synth[df_synth["problem_id"].isin(graph_ids_synth)].copy()
            
            # Split synthetic dataset based on problem_id
            unique_synth_pids = train_df["problem_id"].unique()
            unique_synth_labels = [
                train_df[train_df["problem_id"] == p]["faster_algorithm"].iloc[0]
                for p in unique_synth_pids
            ]
            
            from collections import Counter
            synth_class_counts = Counter(unique_synth_labels)
            can_stratify_synth = stratify and all(count >= 2 for count in synth_class_counts.values())
            
            train_synth_pids, test_synth_pids = train_test_split(
                unique_synth_pids,
                test_size=test_size,
                random_state=self.seed,
                stratify=unique_synth_labels if can_stratify_synth else None,
            )
            
            train_synth_pids_set = set(train_synth_pids)
            test_synth_pids_set = set(test_synth_pids)
            
            synthetic_train_df = train_df[train_df["problem_id"].isin(train_synth_pids_set)]
            synthetic_test_df = train_df[train_df["problem_id"].isin(test_synth_pids_set)]
            
            # Calculate class weights for synthetic training dataset
            class_counts = synthetic_train_df["faster_algorithm"].value_counts()
            total_train = len(synthetic_train_df)
            num_classes = len(class_counts)
            
            weight_0 = total_train / (num_classes * class_counts.get(0, 1))
            weight_1 = total_train / (num_classes * class_counts.get(1, 1))
            self.class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)
            
            # Assign datasets
            self.train_dataset = ProblemRunDataset(synthetic_train_df, graphs_synth, mode=self.mode, active_features=self.active_features)
            self.test_dataset = ProblemRunDataset(synthetic_test_df, graphs_synth, mode=self.mode, active_features=self.active_features)
            self.curated_dataset = ProblemRunDataset(test_df, graphs_curated, mode=self.mode, active_features=self.active_features)
            
            from torch_geometric.loader import DataLoader
            
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )
            self.curated_loader = DataLoader(
                self.curated_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )
            
            print("-" * 40)
            print(f"[Synthetic Mode] Train-IDs (synthetic): {len(train_synth_pids)}, Test-IDs (synthetic): {len(test_synth_pids)}, Curated-IDs (real): {len(test_df['problem_id'].unique())}")
            print(f"[Synthetic Mode] Train-runs: {len(self.train_dataset)}, Test-runs: {len(self.test_dataset)}, Curated-runs: {len(self.curated_dataset)}")
            print(f"[Synthetic Mode] Train class distribution: 0: {class_counts.get(0, 0)}, 1: {class_counts.get(1, 0)}")
            print(f"[Synthetic Mode] Computed Weights: 0: {weight_0:.4f}, 1: {weight_1:.4f}")
            print("-" * 40)
            
            return self.train_loader, self.test_loader, self.class_weights
        else:
            # --- Standard Mode ---
            df = self.loader.data
            graph_ids = set(self.graphs.keys())
            df_matched = df[df["problem_id"].isin(graph_ids)].copy()

            unique_problem_ids = df_matched["problem_id"].unique()

            unique_problem_labels = [
                df_matched[df_matched["problem_id"] == p]["faster_algorithm"].iloc[0]
                for p in unique_problem_ids
            ]

            # Ensure stratification is only applied if every class has at least 2 problem instances
            from collections import Counter
            class_counts = Counter(unique_problem_labels)
            can_stratify = stratify and all(count >= 2 for count in class_counts.values())
            if stratify and not can_stratify:
                print("[Warning] Cannot stratify because one of the classes has fewer than 2 members in unique_problem_labels. Disabling stratification.")

            train_pids, test_pids = train_test_split(
                unique_problem_ids,
                test_size=test_size,
                random_state=self.seed,
                stratify=unique_problem_labels if can_stratify else None,
            )

            train_pids_set = set(train_pids)
            test_pids_set = set(test_pids)

            train_df = df_matched[df_matched["problem_id"].isin(train_pids_set)]
            test_df = df_matched[df_matched["problem_id"].isin(test_pids_set)]

            class_counts = train_df["faster_algorithm"].value_counts()
            total_train = len(train_df)
            num_classes = len(class_counts)

            weight_0 = total_train / (num_classes * class_counts.get(0, 1))
            weight_1 = total_train / (num_classes * class_counts.get(1, 1))
            self.class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

            self.train_dataset = ProblemRunDataset(train_df, self.graphs, mode=self.mode, active_features=self.active_features)
            self.test_dataset = ProblemRunDataset(test_df, self.graphs, mode=self.mode, active_features=self.active_features)

            from torch_geometric.loader import DataLoader

            self.train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )

            print("-" * 40)
            print(
                f"IDs gesamt: {len(unique_problem_ids)} "
                f"(Train-IDs: {len(train_pids)}, Test-IDs: {len(test_pids)})"
            )
            print(
                f"Runs gesamt: {len(df_matched)} "
                f"(Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)})"
            )
            print(
                f"Trainings-Verteilung: 0: {class_counts.get(0, 0)}, "
                f"1: {class_counts.get(1, 0)}"
            )
            print(f"Berechnete Gewichte: 0: {weight_0:.4f}, 1: {weight_1:.4f}")
            print("-" * 40)

            return self.train_loader, self.test_loader, self.class_weights

    @property
    def architecture(self) -> str:
        return architecture_from_layer_type(self.layer_type)

    @property
    def input_dim(self) -> int:
        if self.active_features is not None:
            return len(self.active_features)
        return len(NODE_FEATURE_SCHEMA)

    @property
    def global_dim(self) -> int:
        return 2

    @property
    def edge_dim(self) -> int:
        return len(EDGE_FEATURE_SCHEMA)

    def _validate_edge_features(self):
        if not self.graphs:
            return
        sample = next(iter(self.graphs.values()))
        edge_attr = getattr(sample, "edge_attr", None)
        if edge_attr is None:
            return
        expected_dim = self.edge_dim
        if edge_attr.ndim != 2 or edge_attr.shape[1] != expected_dim:
            raise ValueError(
                f"expected edge_attr with {expected_dim} features, "
                f"got shape {tuple(edge_attr.shape)}"
            )


def parse_float(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        try:
            if "/" in val:
                parts = val.split("/")
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


class ProblemRunDataset(Dataset):
    def __init__(self, df, base_graphs, mode: str = "graph", active_features: list[str] | None = None):
        self.df = df.reset_index(drop=True)
        self.base_graphs = base_graphs
        self.mode = mode
        self.active_features = active_features
        
        self._node_id_indices = {}
        for pid, graph in self.base_graphs.items():
            if hasattr(graph, 'node_types') and 'virtual' in graph.node_types:
                if hasattr(graph['virtual'], 'node_ids') and graph['virtual'].node_ids is not None:
                    self._node_id_indices[pid] = {nid: i for i, nid in enumerate(graph['virtual'].node_ids)}
            elif hasattr(graph, 'node_ids') and graph.node_ids is not None:
                self._node_id_indices[pid] = {nid: i for i, nid in enumerate(graph.node_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["problem_id"]

        data = self.base_graphs[pid].clone()

        data.y = torch.tensor([row["faster_algorithm"]], dtype=torch.long)
        
        # Parse inputs safely to support fraction strings (e.g. from Mathematica)
        cx_val = parse_float(row.get("x0", 0.0))
        yt_val = parse_float(row.get("y_target", 0.0))
        fx_val = parse_float(row.get("fx", 0.0))
        # Local derivative state at x0. These are the physically decisive signals
        # for a Newton (uses f') vs gMGF/Halley (uses f'') decision. They are
        # injected onto the d1_root / d2_root aggregator nodes so the message
        # passing in graph mode can use them; when absent in the dataset they
        # default to 0.0 (previous behaviour).
        d1x_val = parse_float(row.get("d1x", 0.0))
        d2x_val = parse_float(row.get("d2x", 0.0))

        data.global_features = torch.tensor(
            [cx_val, yt_val], dtype=torch.float
        )
        data.pid = pid

        # Slice active features if selection is active
        if self.active_features is not None and data.x is not None:
            data.x = slice_active_features(data.x, self.active_features)

        # Remove laplacian if present to prevent PyG collation mismatch errors
        if hasattr(data, "laplacian"):
            del data.laplacian

        return data
