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
        enrich: bool = True,
        heterogeneous: bool = False,
        add_traces: bool = False,
    ) -> UnifiedDataLoader:
        """
        Retrieves a cached singleton/multiton instance matching the parameter configuration.
        """
        # Parse run_key if slash is present in dataset_name
        if "/" in dataset_name and run_key is None:
            r_key, d_name = dataset_name.split("/", 1)
        else:
            r_key = run_key if run_key is not None else dataset_name
            d_name = dataset_name

        key = (d_name, r_key, mode, enrich, heterogeneous, add_traces)
        if key not in cls._instances:
            cls._instances[key] = cls(
                dataset_name=d_name,
                run_key=r_key,
                mode=mode,
                enrich=enrich,
                heterogeneous=heterogeneous,
                add_traces=add_traces,
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
        enrich: bool,
        heterogeneous: bool,
        add_traces: bool,
    ):
        self.dataset_name = dataset_name
        self.run_key = run_key
        self.mode = mode
        self.enrich = enrich
        self.heterogeneous = heterogeneous
        self.add_traces = add_traces

        # Unified lookup name for GraphDataLoader
        # For compatibility with GraphDataLoader's parsing, if run_key differs from dataset_name, pass "run_key/dataset_name"
        graph_loader_name = f"{self.run_key}/{self.dataset_name}" if self.run_key != self.dataset_name else self.dataset_name

        # Instantiate underlying loaders
        self.dataset_loader = DatasetLoader(
            dataset_name=self.dataset_name,
            run_key=self.run_key,
            addTraces=self.add_traces,
        )
        self.graph_loader = GraphDataLoader(
            name=graph_loader_name,
            mode=self.mode,
            enrich=self.enrich,
            heterogeneous=self.heterogeneous,
        )

        # Automatically enrich missing x0/startwert values from graph data
        try:
            self.enrich_x0_if_missing(self.dataset_loader.data)
        except Exception as e:
            logger.warning(f"Could not perform x0 dataset enrichment: {e}")

    def enrich_x0_if_missing(self, df: pd.DataFrame) -> None:
        """
        Scans the DataFrame for the 'x0' or 'startwert' key. If missing or null,
        fetches the fitting value from the graph data for that problem ID.
        """
        # Determine target column name
        target_col = "x0"
        if "startwert" in df.columns:
            target_col = "startwert"
        elif "x0" in df.columns:
            target_col = "x0"
        else:
            df[target_col] = None

        # Check if column has any missing values
        if df[target_col].isnull().any():
            for idx, row in df.iterrows():
                pid = row.get("problem_id", row.get("problemID"))
                if pid is None:
                    continue
                pid_str = str(pid)
                
                # If current row value is null/NaN, fetch from graph data
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
                                
                                # Try 'x0' first, then 'startwert'
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

    def load_all(self) -> dict[str, Any]:
        """Forwards preloading all graphs into memory."""
        return self.graph_loader.load_all()
