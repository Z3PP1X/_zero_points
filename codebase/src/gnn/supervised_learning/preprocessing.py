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
from gnn.shared.utils.graph_utils import GraphConversionPipeline
from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.utils.unified_loader import UnifiedDataLoader


class FeatureEngineering:
    """Contains offline feature tagging and calculation logic for supervised learning."""
    def __init__(self, loader: DatasetLoader):
        self._loader = loader

    def _tag_faster_algorithm(self):
        """Set binary labels for the faster algorithm: 0: Newton, 1: gMGF"""
        boundaries = [
            self._loader.data["Newton_absTime"] < self._loader.data["GMGF_absTime"],
            self._loader.data["Newton_absTime"] > self._loader.data["GMGF_absTime"],
        ]
        values = [0, 1]
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
        enrich: bool = False,
        active_features: list[str] | None = None,
        graph_loader: GraphDataLoader | None = None,
        unified_loader: UnifiedDataLoader | None = None,
    ):
        self.seed = seed
        self.mode = mode
        self.enrich = enrich
        self.active_features = active_features

        # Use unified_loader or get/create singleton instance
        if unified_loader is not None:
            self.unified_loader = unified_loader
        else:
            self.unified_loader = UnifiedDataLoader.get_instance(
                dataset_name=dataset_name,
                mode=mode,
                enrich=enrich,
            )

        # Backward compatibility aliases
        self.loader = self.unified_loader.dataset_loader
        self.graph_loader = self.unified_loader.graph_loader
        
        fe = FeatureEngineering(self.loader)
        fe._tag_faster_algorithm()
        
        # Override graph_loader if explicitly passed (for legacy call sites/tests)
        if graph_loader is not None:
            self.graph_loader = graph_loader
            self.graphs = self.graph_loader.load_all()
        else:
            self.graphs = self.unified_loader.load_all()
            
        self.graph_pipeline = self
        
        self.train_loader = None
        self.test_loader = None
        self.class_weights = None
        self.Y_train = None
        self.Y_test = None
        self.train_pids = None
        self.test_pids = None

    def pipe(
        self, test_size=0.2, batch_size=32, stratify: bool = True, num_workers: int = 0
    ):
        df = self.loader.data
        graph_ids = set(self.graphs.keys())
        df_matched = df[df["problem_id"].isin(graph_ids)].copy()

        unique_problem_ids = df_matched["problem_id"].unique()

        unique_problem_labels = [
            df_matched[df_matched["problem_id"] == p]["faster_algorithm"].iloc[0]
            for p in unique_problem_ids
        ]

        train_pids, test_pids = train_test_split(
            unique_problem_ids,
            test_size=test_size,
            random_state=self.seed,
            stratify=unique_problem_labels if stratify else None,
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

        train_dataset = ProblemRunDataset(train_df, self.graphs, mode=self.mode, enrich=self.enrich, active_features=self.active_features)
        test_dataset = ProblemRunDataset(test_df, self.graphs, mode=self.mode, enrich=self.enrich, active_features=self.active_features)

        from torch_geometric.loader import DataLoader

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )

        print("-" * 40)
        print(
            f"IDs gesamt: {len(unique_problem_ids)} "
            f"(Train-IDs: {len(train_pids)}, Test-IDs: {len(test_pids)})"
        )
        print(
            f"Runs gesamt: {len(df_matched)} "
            f"(Train: {len(train_dataset)}, Test: {len(test_dataset)})"
        )
        print(
            f"Trainings-Verteilung: 0: {class_counts.get(0, 0)}, "
            f"1: {class_counts.get(1, 0)}"
        )
        print(f"Berechnete Gewichte: 0: {weight_0:.4f}, 1: {weight_1:.4f}")
        print("-" * 40)

        return self.train_loader, self.test_loader, self.class_weights

    @property
    def input_dim(self) -> int:
        if self.active_features is not None:
            return len(self.active_features)
        return 19 if self.enrich else 8

    @property
    def global_dim(self) -> int:
        return 2


class ProblemRunDataset(Dataset):
    def __init__(self, df, base_graphs, mode: str = "graph", enrich: bool = False, active_features: list[str] | None = None):
        self.df = df.reset_index(drop=True)
        self.base_graphs = base_graphs
        self.mode = mode
        self.enrich = enrich
        self.active_features = active_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["problem_id"]

        data = self.base_graphs[pid].clone()

        data.y = torch.tensor([row["faster_algorithm"]], dtype=torch.long)
        data.global_features = torch.tensor(
            [row["startwert"], row["zielwert"]], dtype=torch.float
        )
        data.pid = pid

        # Initialize virtual nodes once from the dataset (without any Taylor series fallback)
        if hasattr(data, "node_ids") and data.node_ids is not None:
            try:
                cx_val = row.get("startwert", row.get("x0", 0.0))
                yt_val = row.get("zielwert", row.get("y_target", 0.0))
                fx_val = row.get("fx", 0.0)  # Default to 0.0 to completely avoid Taylor series fallback
                
                if self.mode == "graph":
                    idx_cx = data.node_ids.index("virtual_current_x")
                    idx_fx = data.node_ids.index("virtual_f_x")
                    idx_yt = data.node_ids.index("virtual_y_target")
                    
                    if data.x is not None and len(data.x.shape) == 2:
                        num_features = data.x.shape[1]
                        if num_features == 19:  # enrich=True
                            data.x[idx_cx, 7] = float(cx_val)
                            data.x[idx_fx, 7] = float(fx_val)
                            data.x[idx_yt, 7] = float(yt_val)
                        elif num_features == 8:  # enrich=False
                            data.x[idx_cx, 2] = float(cx_val)
                            data.x[idx_fx, 2] = float(fx_val)
                            data.x[idx_yt, 2] = float(yt_val)
                            
                            data.x[idx_cx, 3] = 1.0
                            data.x[idx_fx, 3] = 1.0
                            data.x[idx_yt, 3] = 1.0
                elif self.mode in ["tree", "tree_derivatives"]:
                    # Populate slots on the global node directly
                    idx_global = data.node_ids.index("global")
                    if data.x is not None and len(data.x.shape) == 2:
                        num_features = data.x.shape[1]
                        if num_features == 19:  # enrich=True
                            data.x[idx_global, 16] = float(cx_val)
                            data.x[idx_global, 17] = float(fx_val)
                            data.x[idx_global, 18] = float(yt_val)
                        elif num_features == 8:  # enrich=False
                            data.x[idx_global, 5] = float(cx_val)
                            data.x[idx_global, 6] = float(fx_val)
                            data.x[idx_global, 7] = float(yt_val)
            except ValueError:
                pass

        # Slice active features if selection is active
        if self.active_features is not None and data.x is not None:
            from gnn.shared.utils.graph_utils import slice_active_features
            data.x = slice_active_features(data.x, self.active_features, enrich=self.enrich)

        return data
