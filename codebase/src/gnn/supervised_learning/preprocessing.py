import numpy as np
import torch
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from gnn.supervised_learning.dataset import DatasetLoader
from gnn.shared.utils.graph_utils import (
    NODE_FEATURE_SCHEMA,
    slice_active_features,
)
from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.utils.unified_loader import UnifiedDataLoader
from gnn.supervised_learning.supervised_config import validate_layer_type


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


def _pid_split(df, test_size, seed, stratify=False):
    """Split a DataFrame by problem_id, stratifying on faster_algorithm when possible."""
    pids = df["problem_id"].unique()
    labels = [df[df["problem_id"] == p]["faster_algorithm"].iloc[0] for p in pids]
    can_strat = stratify and all(c >= 2 for c in Counter(labels).values())
    train_ids, test_ids = train_test_split(
        pids, test_size=test_size, random_state=seed,
        stratify=labels if can_strat else None,
    )
    train_set, test_set = set(train_ids), set(test_ids)
    return df[df["problem_id"].isin(train_set)], df[df["problem_id"].isin(test_set)], can_strat


def _compute_class_weights(df, label_col="faster_algorithm"):
    """Inverse-frequency class weights as a float32 tensor [w0, w1]."""
    counts = df[label_col].value_counts()
    total, n = len(df), len(counts)
    return torch.tensor(
        [total / (n * counts.get(0, 1)), total / (n * counts.get(1, 1))],
        dtype=torch.float,
    ), counts


class GraphPipeline:
    def __init__(
        self,
        dataset_name: str,
        seed: int = 42,
        mode: str = "graph",
        active_features: list[str] | None = None,
        scalar_features: list[str] | None = None,
        graph_loader: GraphDataLoader | None = None,
        unified_loader: UnifiedDataLoader | None = None,
        synthetic: bool = False,
        synthetic_dataset_name: str | None = None,
        layer_type: str = "ginconv",
        add_kappa: bool = False,
        add_virtual_supernode: bool = False,
        curated_csv_path: "Path | str | None" = None,
        synthetic_csv_path: "Path | str | None" = None,
        curated_graphs_path: "Path | str | None" = None,
        synthetic_graphs_path: "Path | str | None" = None,
    ):
        self.seed = seed
        self.mode = mode
        self.active_features = active_features
        self.scalar_features = scalar_features
        self.synthetic = synthetic
        self.synthetic_dataset_name = synthetic_dataset_name if synthetic_dataset_name else None
        self.layer_type = validate_layer_type(layer_type)
        self.add_kappa = add_kappa
        self.add_virtual_supernode = add_virtual_supernode

        if unified_loader is not None:
            self.unified_loader = unified_loader
        else:
            self.unified_loader = UnifiedDataLoader.get_instance(
                dataset_name=dataset_name,
                mode=mode,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
                csv_path=curated_csv_path,
                graphs_path=curated_graphs_path,
            )

        if self.synthetic and (self.synthetic_dataset_name is not None or synthetic_csv_path is not None):
            self.synthetic_unified_loader = UnifiedDataLoader.get_instance(
                dataset_name=self.synthetic_dataset_name or "synthetic",
                mode=mode,
                is_synthetic=True,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
                csv_path=synthetic_csv_path,
                graphs_path=synthetic_graphs_path,
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
            
        self.train_loader = None
        self.test_loader = None
        self.curated_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.curated_dataset = None
        self.class_weights = None

    def pipe(
        self, test_size=0.2, batch_size=32, stratify: bool = False, num_workers: int = 0
    ):
        if self.synthetic:
            if self.synthetic_unified_loader is None:
                raise ValueError("Synthetic mode is active, but synthetic_dataset_name is not provided.")

            df_curated = self.unified_loader.dataset_loader.data
            kappa_map_curated = self.unified_loader.build_kappa_map() if self.add_kappa else None
            graphs_curated = self.unified_loader.load_all(kappa_map=kappa_map_curated)
            graph_ids_curated = set(graphs_curated.keys())
            test_df = df_curated[df_curated["problem_id"].isin(graph_ids_curated)].copy()

            df_synth = self.synthetic_unified_loader.dataset_loader.data
            kappa_map_synth = self.synthetic_unified_loader.build_kappa_map() if self.add_kappa else None
            graphs_synth = self.synthetic_unified_loader.load_all(kappa_map=kappa_map_synth)
            graph_ids_synth = set(graphs_synth.keys())
            train_df = df_synth[df_synth["problem_id"].isin(graph_ids_synth)].copy()
            
            fe_curated = FeatureEngineering(self.unified_loader.dataset_loader)
            fe_curated._tag_faster_algorithm()

            fe_synth = FeatureEngineering(self.synthetic_unified_loader.dataset_loader)
            fe_synth._tag_faster_algorithm()

            test_df = df_curated[df_curated["problem_id"].isin(graph_ids_curated)].copy()
            train_df = df_synth[df_synth["problem_id"].isin(graph_ids_synth)].copy()

            synthetic_train_df, synthetic_test_df, _ = _pid_split(
                train_df, test_size, self.seed, stratify
            )
            self.class_weights, class_counts = _compute_class_weights(synthetic_train_df)

            pin = torch.cuda.is_available()
            self.train_dataset = ProblemRunDataset(synthetic_train_df, graphs_synth, mode=self.mode, active_features=self.active_features, scalar_features=self.scalar_features)
            self.test_dataset = ProblemRunDataset(synthetic_test_df, graphs_synth, mode=self.mode, active_features=self.active_features, scalar_features=self.scalar_features)
            self.curated_dataset = ProblemRunDataset(test_df, graphs_curated, mode=self.mode, active_features=self.active_features, scalar_features=self.scalar_features)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
            self.curated_loader = DataLoader(self.curated_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)

            w0, w1 = self.class_weights.tolist()
            print("-" * 40)
            print(f"[Synthetic Mode] Train-IDs (synthetic): {len(synthetic_train_df['problem_id'].unique())}, Test-IDs (synthetic): {len(synthetic_test_df['problem_id'].unique())}, Curated-IDs (real): {len(test_df['problem_id'].unique())}")
            print(f"[Synthetic Mode] Train-runs: {len(self.train_dataset)}, Test-runs: {len(self.test_dataset)}, Curated-runs: {len(self.curated_dataset)}")
            print(f"[Synthetic Mode] Train class distribution: 0: {class_counts.get(0, 0)}, 1: {class_counts.get(1, 0)}")
            print(f"[Synthetic Mode] Computed Weights: 0: {w0:.4f}, 1: {w1:.4f}")
            print("-" * 40)
            
            return self.train_loader, self.test_loader, self.class_weights
        else:
            df = self.loader.data
            graph_ids = set(self.graphs.keys())
            df_matched = df[df["problem_id"].isin(graph_ids)].copy()

            train_df, test_df, can_stratify = _pid_split(df_matched, test_size, self.seed, stratify)
            if stratify and not can_stratify:
                print("[Warning] Cannot stratify: a class has fewer than 2 members. Disabling stratification.")
            self.class_weights, class_counts = _compute_class_weights(train_df)

            pin = torch.cuda.is_available()
            self.train_dataset = ProblemRunDataset(train_df, self.graphs, mode=self.mode, active_features=self.active_features, scalar_features=self.scalar_features)
            self.test_dataset = ProblemRunDataset(test_df, self.graphs, mode=self.mode, active_features=self.active_features, scalar_features=self.scalar_features)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)

            w0, w1 = self.class_weights.tolist()
            n_pids = len(df_matched["problem_id"].unique())
            print("-" * 40)
            print(f"Total IDs: {n_pids} (Train: {len(train_df['problem_id'].unique())}, Test: {len(test_df['problem_id'].unique())})")
            print(f"Total runs: {len(df_matched)} (Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)})")
            print(f"Train class distribution: 0: {class_counts.get(0, 0)}, 1: {class_counts.get(1, 0)}")
            print(f"Class weights: 0: {w0:.4f}, 1: {w1:.4f}")
            print("-" * 40)

            return self.train_loader, self.test_loader, self.class_weights

    @property
    def input_dim(self) -> int:
        if self.active_features is not None:
            return len(self.active_features)
        return len(NODE_FEATURE_SCHEMA)


def _parse_scalar(value) -> float:
    """Best-effort float for a per-problem scalar cell.

    Handles plain numbers, Mathematica fraction strings (e.g. "1/3"), and
    missing/non-finite cells (-> 0.0). Mirrors the tolerance the AST label
    encoder applies to numeric constants (see graph_vocab._is_numeric_label).
    """
    if value is None:
        return 0.0
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0.0
        try:
            return float(s)
        except ValueError:
            if "/" in s:
                num, _, den = s.partition("/")
                try:
                    d = float(den)
                    return float(num) / d if d != 0.0 else 0.0
                except ValueError:
                    return 0.0
            return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


class ProblemRunDataset(Dataset):
    def __init__(self, df, base_graphs, mode: str = "graph", active_features: list[str] | None = None, scalar_features: list[str] | None = None):
        self.df = df.reset_index(drop=True)
        self.base_graphs = base_graphs
        self.mode = mode
        self.active_features = active_features
        self.scalar_features = scalar_features

        if self.scalar_features:
            missing = [c for c in self.scalar_features if c not in self.df.columns]
            if missing:
                raise RuntimeError(
                    f"scalar_features requested {self.scalar_features} but column(s) {missing} "
                    f"are absent from the dataset (available columns: {list(self.df.columns)}). "
                    f"Provide them in the CSV or disable cfg.expression_graph.use_scalar_features."
                )

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
        data.pid = pid

        if self.scalar_features:
            # Per-graph scalar vector stored as [1, k] (leading batch dim) so PyG collate
            # stacks it to [num_graphs, k] — same convention as the RL preprocessor. The
            # global encoder in ExpressionGNN then fuses it with the pooled embedding.
            data.global_features = torch.tensor(
                [_parse_scalar(row.get(c)) for c in self.scalar_features],
                dtype=torch.float,
            ).unsqueeze(0)

        if self.active_features is not None and data.x is not None:
            data.x = slice_active_features(data.x, self.active_features)

        return data
