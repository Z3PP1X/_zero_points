from __future__ import annotations
import logging
from pathlib import Path
from typing import Union, Optional, Any
import pandas as pd

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.supervised_learning.dataset import DatasetLoader

logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """
    Unified data loader that wraps both DatasetLoader (tabular CSVs) and
    GraphDataLoader (PyG graph objects), operating under a singleton/multiton pattern.
    """
    _instances = {}

    @classmethod
    def get_instance(
        cls,
        dataset_name: str,
        run_key: Optional[str] = None,
        mode: str = "graph",
        add_traces: bool = False,
        base_dir: Union[Path, str, None] = None,
        is_synthetic: bool = False,
        edge_direction: str = "top_down",
        add_kappa: bool = False,
        add_virtual_supernode: bool = False,
        csv_path: Union[Path, str, None] = None,
        graphs_path: Union[Path, str, None] = None,
    ) -> "UnifiedDataLoader":
        """
        Retrieves a cached singleton/multiton instance matching the parameter configuration.
        """
        if "/" in dataset_name and run_key is None:
            r_key, d_name = dataset_name.split("/", 1)
        else:
            r_key = run_key if run_key is not None else dataset_name
            d_name = dataset_name

        key = (
            d_name,
            r_key,
            mode,
            add_traces,
            str(base_dir) if base_dir else None,
            is_synthetic,
            edge_direction,
            add_kappa,
            add_virtual_supernode,
            str(csv_path) if csv_path else None,
            str(graphs_path) if graphs_path else None,
        )
        if key not in cls._instances:
            cls._instances[key] = cls(
                dataset_name=d_name,
                run_key=r_key,
                mode=mode,
                add_traces=add_traces,
                base_dir=base_dir,
                is_synthetic=is_synthetic,
                edge_direction=edge_direction,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
                csv_path=csv_path,
                graphs_path=graphs_path,
            )
        return cls._instances[key]

    @classmethod
    def clear_instances(cls):
        """Clears all cached instances (useful for testing)."""
        cls._instances.clear()

    def __init__(
        self,
        dataset_name: str,
        run_key: str,
        mode: str,
        add_traces: bool,
        base_dir: Union[Path, str, None] = None,
        is_synthetic: bool = False,
        edge_direction: str = "top_down",
        add_kappa: bool = False,
        add_virtual_supernode: bool = False,
        csv_path: Union[Path, str, None] = None,
        graphs_path: Union[Path, str, None] = None,
    ):
        self.dataset_name = dataset_name
        self.run_key = run_key
        self.mode = mode
        self.add_traces = add_traces
        self.base_dir = base_dir
        self.is_synthetic = is_synthetic
        self.edge_direction = edge_direction
        self.add_kappa = add_kappa
        self.add_virtual_supernode = add_virtual_supernode

        graph_loader_name = f"{self.run_key}/{self.dataset_name}" if self.run_key != self.dataset_name else self.dataset_name

        self.dataset_loader = DatasetLoader(
            dataset_name=self.dataset_name,
            run_key=self.run_key,
            addTraces=self.add_traces,
            base_dir=self.base_dir,
            csv_path=csv_path,
        )
        kappa_map: dict[str, float] = {}
        if self.add_kappa:
            # F-08: never silently fall back to kappa-less graphs when add_kappa=True.
            # A swallowed failure (or an empty map) would turn a "kappa" experiment into a
            # secret baseline run while configs, logs and reports still claim Kappa=True —
            # invalidating any kappa ablation. Fail loudly instead.
            try:
                kappa_map = self.build_kappa_map()
            except Exception as e:
                raise RuntimeError(
                    f"add_kappa=True but kappa-map construction failed for dataset "
                    f"{self.dataset_name!r}: {e}. Refusing to load graphs without the kappa "
                    f"subgraph. Fix the kappa source or set add_kappa=False."
                ) from e
            if not kappa_map:
                raise RuntimeError(
                    f"add_kappa=True but the kappa map is EMPTY for dataset "
                    f"{self.dataset_name!r} (no 'kappa' column / values found in the tabular "
                    f"data). Refusing to silently train/evaluate without kappa; set "
                    f"add_kappa=False if this dataset genuinely has no kappa."
                )

        self.graph_loader = GraphDataLoader(
            name=graph_loader_name,
            mode=self.mode,
            base_dir=graphs_path if graphs_path is not None else self.base_dir,
            is_synthetic=self.is_synthetic,
            edge_direction=self.edge_direction,
            add_kappa=self.add_kappa,
            add_virtual_supernode=self.add_virtual_supernode,
            kappa_map=kappa_map,
        )

        # Automatically enrich missing x0/startwert values from graph data
        try:
            self.enrich_x0_if_missing(self.dataset_loader.data)
        except Exception as e:
            logger.warning(f"Could not perform x0 dataset enrichment: {e}")

    def enrich_x0_if_missing(self, df: pd.DataFrame) -> None:
        """
        Scans the DataFrame for the 'x0' key. If missing or null,
        fetches the fitting value from the graph data for that problem ID.
        """
        target_col = "x0"
        if "x0" not in df.columns:
            df["x0"] = None

        if df[target_col].isnull().any():
            for idx, row in df.iterrows():
                pid = row.get("problem_id", row.get("problemID"))
                if pid is None:
                    continue
                pid_str = str(pid)

                if pd.isnull(row[target_col]):
                    try:
                        if self.graph_loader.has_graph(pid_str):
                            raw_val = self.graph_loader._raw_sources.get(pid_str)
                            if raw_val is not None:
                                if isinstance(raw_val, Path):
                                    import json
                                    with open(raw_val, "r", encoding="utf-8") as f:
                                        raw_dict = json.load(f)
                                else:
                                    raw_dict = raw_val

                                x0_val = raw_dict.get("x0", raw_dict.get("startwert"))
                                if x0_val is not None:
                                    df.at[idx, target_col] = float(x0_val)
                    except Exception as e:
                        logger.warning(f"Failed to enrich x0 for problem ID '{pid_str}': {e}")

    @property
    def data(self) -> pd.DataFrame:
        """Returns the tabular pandas DataFrame from DatasetLoader."""
        return self.dataset_loader.data

    def add_column(self, name: str, values) -> None:
        """Forwards adding a column to the tabular data DataFrame."""
        self.dataset_loader.add_column(name, values)

    def get_graph(self, graph_id: Any):
        """Forwards retrieving a PyG Graph object."""
        return self.graph_loader.get_graph(graph_id)

    def list_graph_ids(self) -> set[str]:
        """Forwards listing all discovered graph IDs."""
        return self.graph_loader.list_graph_ids()

    def has_graph(self, graph_id: Any) -> bool:
        """Forwards checking if the graph exists."""
        return self.graph_loader.has_graph(graph_id)

    def build_kappa_map(self) -> dict[str, float]:
        """Build a {graph_id: kappa_value} mapping from the tabular dataset.

        Each graph ID is expected to map to exactly one kappa value. Returns an
        empty dict if the dataset has no 'kappa' column.
        """
        df = self.dataset_loader.data
        # DatasetLoader normalises headers, so the id column arrives as the
        # lowercase "problem_id"; accept the raw "Problem_ID" too for safety.
        id_col = next((c for c in ("problem_id", "Problem_ID") if c in df.columns), None)
        if "kappa" not in df.columns or id_col is None:
            return {}
        return (
            df.dropna(subset=["kappa"])
            .groupby(id_col)["kappa"]
            .first()
            .apply(float)
            .to_dict()
        )

    def load_all(self, kappa_map: dict = None) -> dict[str, Any]:
        """Forwards to GraphDataLoader.load_all, merging only the active kappa per graph."""
        return self.graph_loader.load_all(kappa_map=kappa_map)
