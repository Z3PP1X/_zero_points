import json
import logging
from pathlib import Path
from typing import Union, Any, Dict, Set
from torch_geometric.data import Data, HeteroData
from gnn.shared.utils.graph_utils import ExpressionGraphConverter

logger = logging.getLogger(__name__)


class GraphDataLoader:
    """
    Unified graph data loader supporting:
    - Checking 'graphs/' in project root first (both <name>.json and <name>/ subfolders).
    - Legacy path fallbacks (both supervised and reinforcement learning paths).
    - Loading from a single JSON file (as a dictionary, a list of graphs, or a single graph).
    - Directory walks with prioritized loading of *_meta.json files.
    - RAM caching of converted PyG Data/HeteroData objects to eliminate training disk I/O.
    """

    def __init__(
        self,
        name: str,
        mode: str = "graph",
        enrich: bool = True,
        heterogeneous: bool = False,
        base_dir: Union[Path, str, None] = None,
    ):
        self.name = name
        self.mode = mode
        self.enrich = enrich
        self.heterogeneous = heterogeneous
        self.converter = ExpressionGraphConverter()

        # Resolve the source path (root graphs vs legacy fallbacks)
        self.source_path = self._resolve_source(name, base_dir)
        print(f"[GraphDataLoader] Resolved graph source for '{name}' to: {self.source_path}")

        self._raw_sources: dict[str, Union[dict, Path]] = {}
        self._converted_cache: dict[str, Union[Data, HeteroData]] = {}

        self._discover_graphs()
        print(f"[GraphDataLoader] Discovered {len(self._raw_sources)} graph IDs.")

    def _resolve_source(self, name: str, base_dir: Union[Path, str, None]) -> Path:
        # Walk up to find the repository root containing '.git' or name '_zero_points'
        current = Path(__file__).resolve()
        repo_root = None
        for parent in current.parents:
            if (parent / ".git").exists() or parent.name == "_zero_points":
                repo_root = parent
                break
        if repo_root is None:
            repo_root = current.parents[5]

        # Use base_dir directly if provided and exists
        if base_dir is not None:
            p = Path(base_dir)
            if p.exists():
                return p

        # Parse run_key if slash is present (e.g. run_key/dataset_name)
        if "/" in name:
            run_key, _ = name.split("/", 1)
        else:
            run_key = name

        # Candidates search order
        candidates = [
            repo_root / "datasets" / f"{name}.json",
            repo_root / "datasets" / f"{run_key}.json",
            repo_root / "datasets" / name,
            repo_root / "datasets" / run_key,
            repo_root / "datasets" / run_key / "graphs",
            repo_root / "graphs" / f"{name}.json",
            repo_root / "graphs" / f"{run_key}.json",
            repo_root / "graphs" / name,
            repo_root / "graphs" / run_key,
            repo_root / "codebase" / "src" / "gnn" / "graphs" / name,
            repo_root / "_datasets" / run_key / "graphs",
        ]

        for cand in candidates:
            if cand.exists():
                if cand.is_dir():
                    # Check if there is a unified JSON file inside the directory matching its name or 'graphs.json'
                    file_match = cand / f"{cand.name}.json"
                    if file_match.exists() and file_match.is_file():
                        return file_match
                    graphs_file = cand / "graphs.json"
                    if graphs_file.exists() and graphs_file.is_file():
                        return graphs_file
                return cand

        # Default fallback target if none exists yet
        return repo_root / "datasets" / f"{name}.json"

    def _discover_graphs(self):
        if not self.source_path.exists():
            print(f"[GraphDataLoader] Source path does not exist: {self.source_path}")
            return

        if self.source_path.is_file():
            # Load all graphs from a single JSON file into memory raw cache
            try:
                with open(self.source_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                parsed = self._parse_single_json_content(raw_data, self.source_path.stem)
                self._raw_sources = parsed
            except Exception as e:
                print(f"[GraphDataLoader] Error reading single JSON file {self.source_path}: {e}")
        elif self.source_path.is_dir():
            # Discover file paths in directory for lazy load
            json_files = list(self.source_path.glob("**/*.json"))
            # Prioritize meta files
            for path in json_files:
                if path.stem.endswith("_meta"):
                    graph_id = path.stem.removesuffix("_meta")
                    self._raw_sources[graph_id] = path
            # Fallback to standard json files if meta is missing
            for path in json_files:
                if not path.stem.endswith("_meta"):
                    graph_id = path.stem
                    if graph_id not in self._raw_sources:
                        self._raw_sources[graph_id] = path

    def _parse_single_json_content(self, raw_data: Any, name: str) -> dict[str, Any]:
        if isinstance(raw_data, list):
            parsed = {}
            for i, item in enumerate(raw_data):
                if isinstance(item, dict):
                    graph_id = item.get("id", f"G_{i}")
                    parsed[str(graph_id)] = item
            return parsed
        elif isinstance(raw_data, dict):
            # Check if it represents a single graph
            if any(k in raw_data for k in ["nodes", "edges", "graphml_f"]):
                graph_id = raw_data.get("id", name)
                return {str(graph_id): raw_data}
            else:
                # Dict of graphs
                parsed = {}
                for k, v in raw_data.items():
                    if isinstance(v, dict):
                        parsed[str(k)] = v
                return parsed
        return {}

    def list_graph_ids(self) -> Set[str]:
        return set(self._raw_sources.keys())

    def has_graph(self, graph_id: Any) -> bool:
        return str(graph_id) in self._raw_sources

    def get_graph(self, graph_id: Any) -> Union[Data, HeteroData]:
        gid_str = str(graph_id)
        if gid_str not in self._raw_sources:
            raise KeyError(f"Graph ID '{gid_str}' not found in loaded graphs.")

        # Return clone from memory cache if already converted
        if gid_str in self._converted_cache:
            return self._converted_cache[gid_str].clone()

        # Retrieve and parse raw data
        raw_val = self._raw_sources[gid_str]
        if isinstance(raw_val, Path):
            with open(raw_val, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)
        else:
            raw_dict = raw_val

        # Convert to PyG object
        converted = self.converter.convert(
            raw_dict,
            heterogeneous=self.heterogeneous,
            enrich=self.enrich,
            mode=self.mode
        )
        self._converted_cache[gid_str] = converted
        return converted.clone()

    def load_all(self) -> dict[str, Union[Data, HeteroData]]:
        """Preloads and converts all discovered graphs into memory."""
        result = {}
        for gid in self._raw_sources.keys():
            result[gid] = self.get_graph(gid)
        return result
