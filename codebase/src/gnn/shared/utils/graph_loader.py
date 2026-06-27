import hashlib
import json
import logging
from pathlib import Path
from typing import Union, Any, Set
from torch_geometric.data import Data, HeteroData
from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
)

logger = logging.getLogger(__name__)

# Short fingerprint of the active node-feature schema. Embedded in every disk
# cache key so that a schema change (adding/removing/reordering a feature)
# yields a fresh key instead of silently reusing a stale tensor of the wrong
# feature dimension. Without this, a grown schema (e.g. 28 -> 32 columns) would
# load old 28-column .pt files and crash at batch collation.
SCHEMA_TAG = hashlib.md5(",".join(NODE_FEATURE_SCHEMA).encode("utf-8")).hexdigest()[:8]


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
        mode: str = "tree_derivatives",
        base_dir: Union[Path, str, None] = None,
        is_synthetic: bool = False,
        kappas_dir: Union[Path, str, None] = None,
        add_kappa: bool = False,
        add_virtual_supernode: bool = False,
        kappa_map: Union[dict, None] = None,
    ):
        self.name = name
        self.mode = mode
        self.is_synthetic = is_synthetic
        self.add_kappa = add_kappa
        self.add_virtual_supernode = add_virtual_supernode
        self.kappa_map: dict[str, float] = {str(k): float(v) for k, v in (kappa_map or {}).items()}
        self.converter = ExpressionGraphConverter()

        self.source_path = self._resolve_source(name, base_dir)
        logger.info("Resolved graph source for '%s' to: %s", name, self.source_path)

        current = Path(__file__).resolve()
        repo_root = None
        for parent in current.parents:
            if (parent / ".git").exists() or parent.name == "_zero_points":
                repo_root = parent
                break
        if repo_root is None:
            repo_root = current.parents[5]

        if kappas_dir is not None:
            self.kappas_dir = Path(kappas_dir)
            self.kappas_dir_explicit = True
        else:
            self.kappas_dir = repo_root / "datasets" / "kappas"
            self.kappas_dir_explicit = False

        if self.source_path.is_file():
            self.cache_dir = self.source_path.parent / ".pt_cache" / self.source_path.stem
        else:
            self.cache_dir = self.source_path / ".pt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using disk cache at: %s", self.cache_dir)

        self._raw_sources: dict[str, Union[dict, Path]] = {}
        self._converted_cache: dict[str, Union[Data, HeteroData]] = {}

        self._discover_graphs()
        logger.info("Discovered %d graph IDs.", len(self._raw_sources))

    def _resolve_source(self, name: str, base_dir: Union[Path, str, None]) -> Path:
        current = Path(__file__).resolve()
        repo_root = None
        for parent in current.parents:
            if (parent / ".git").exists() or parent.name == "_zero_points":
                repo_root = parent
                break
        if repo_root is None:
            repo_root = current.parents[5]

        if base_dir is not None:
            p = Path(base_dir)
            if p.exists():
                return p

        if "/" in name:
            run_key, _ = name.split("/", 1)
        else:
            run_key = name

        candidates = []
        if self.is_synthetic:
            candidates.extend([
                repo_root / "datasets" / "graphs" / "synthetic_graphs.json",
                repo_root / "datasets" / "synthetic_graphs.json",
                repo_root / "graphs" / "synthetic_graphs.json",
            ])
        candidates.extend([
            repo_root / "datasets" / f"{name}.json",
            repo_root / "datasets" / f"{run_key}.json",
            repo_root / "datasets" / name,
            repo_root / "datasets" / run_key,
            repo_root / "datasets" / run_key / "graphs",
            repo_root / "datasets" / "graphs",
            repo_root / "graphs" / f"{name}.json",
            repo_root / "graphs" / f"{run_key}.json",
            repo_root / "graphs" / name,
            repo_root / "graphs" / run_key,
            repo_root / "codebase" / "src" / "gnn" / "graphs" / name,
            repo_root / "_datasets" / run_key / "graphs",
        ])

        for cand in candidates:
            if cand.exists():
                if cand.is_dir():
                    file_match = cand / f"{cand.name}.json"
                    if file_match.exists() and file_match.is_file():
                        return file_match
                    graphs_file = cand / "graphs.json"
                    if graphs_file.exists() and graphs_file.is_file():
                        return graphs_file
                    if not any(cand.glob("**/*.json")):
                        continue
                return cand

        return repo_root / "datasets" / f"{name}.json"

    def _discover_graphs(self):
        if not self.source_path.exists():
            print(f"[GraphDataLoader] Source path does not exist: {self.source_path}")
            return

        if self.source_path.is_file():
            try:
                with open(self.source_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                parsed = self._parse_single_json_content(raw_data, self.source_path.stem)
                self._raw_sources = parsed
            except Exception as e:
                print(f"[GraphDataLoader] Error reading single JSON file {self.source_path}: {e}")
        elif self.source_path.is_dir():
            json_files = list(self.source_path.glob("**/*.json"))
            for path in json_files:
                if path.stem.endswith("_meta"):
                    graph_id = path.stem.removesuffix("_meta")
                    self._raw_sources[graph_id] = path
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
            if any(k in raw_data for k in ["nodes", "edges", "graphml_f"]):
                graph_id = raw_data.get("id", name)
                return {str(graph_id): raw_data}
            else:
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

    def get_graph(self, graph_id: Any, kappa_value: Union[float, None] = None) -> Union[Data, HeteroData]:
        import torch
        gid_str = str(graph_id)
        if gid_str not in self._raw_sources:
            raise KeyError(f"Graph ID '{gid_str}' not found in loaded graphs.")

        # Resolve kappa_value: explicit arg wins; fall back to kappa_map lookup.
        if self.add_kappa and kappa_value is None:
            kappa_value = self.kappa_map.get(gid_str)

        mem_key = f"{gid_str}_k{kappa_value}" if kappa_value is not None else gid_str
        if mem_key in self._converted_cache:
            return self._converted_cache[mem_key].clone()

        # An explicit non-None kappa_value (the RL live-kappa path) enables augmentation
        # regardless of self.add_kappa — it is the dynamic per-step mechanism, distinct
        # from the static add_kappa ablation knob. Supervised never passes kappa_value, so
        # `kappa_value is not None` is False there and the add_kappa-gated behaviour is
        # unchanged. The is_temp/kappas_dir_explicit guard below is intentionally kept.
        use_augmented = False
        if (self.add_kappa or kappa_value is not None) and self.kappas_dir.exists() and any(self.kappas_dir.glob("**/*.json")):
            source_str = str(self.source_path)
            is_temp = (
                "/tmp" in source_str
                or "pytest" in source_str
                or "temp" in source_str
            )
            if self.kappas_dir_explicit or not is_temp:
                use_augmented = True

        clean_gid = "".join(c for c in gid_str if c.isalnum() or c in ('_', '-'))
        sn_marker = "_sn" if self.add_virtual_supernode else ""
        if use_augmented:
            kappa_marker = f"_k{int(kappa_value)}" if kappa_value is not None else ""
            suffix = f"{sn_marker}{kappa_marker}_augmented.pt"
        else:
            suffix = f"{sn_marker}.pt"
        cache_file = self.cache_dir / (
            f"{clean_gid}_{self.mode}_{SCHEMA_TAG}{suffix}"
        )

        if cache_file.exists():
            try:
                converted = torch.load(cache_file, weights_only=False)
                self._converted_cache[mem_key] = converted
                return converted.clone()
            except Exception as e:
                logger.warning(f"Failed to load cached graph {cache_file}: {e}")

        if use_augmented:
            from gnn.shared.utils.graph_utils import LoadAugmentedFunctionGraph
            main_graph = LoadAugmentedFunctionGraph(
                graphId=gid_str,
                graphsFolder=self.source_path,
                kappasFolder=self.kappas_dir,
                kappa_value=kappa_value,
            )
            converted = self.converter.convert(
                main_graph,
                mode=self.mode,
                add_virtual_supernode=self.add_virtual_supernode,
            )
        else:
            raw_val = self._raw_sources[gid_str]
            if isinstance(raw_val, Path):
                with open(raw_val, "r", encoding="utf-8") as f:
                    raw_dict = json.load(f)
            else:
                raw_dict = raw_val

            converted = self.converter.convert(
                raw_dict,
                mode=self.mode,
                add_virtual_supernode=self.add_virtual_supernode,
            )

        try:
            torch.save(converted, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cached graph {cache_file}: {e}")

        self._converted_cache[mem_key] = converted
        return converted.clone()

    def load_all(self, kappa_map: Union[dict, None] = None) -> dict[str, Union[Data, HeteroData]]:
        """Preload and convert all discovered graphs into memory.

        When kappa_map is provided, each graph merges only its matching kappa
        subgraph instead of all subgraphs — produces smaller graphs and correct filtering.
        """
        result = {}
        for gid in self._raw_sources.keys():
            kv = kappa_map.get(gid, 0.0) if kappa_map else None
            result[gid] = self.get_graph(gid, kappa_value=kv)
        return result
