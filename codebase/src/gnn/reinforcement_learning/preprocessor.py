import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, FrozenSet

import torch
from torch_geometric.data import Data

from gnn.shared.utils.graph_utils import ExpressionGraphConverter
from gnn.reinforcement_learning.observation_sanitize import finite_float, sanitize_torch_features

STATE_GLOBAL_FEATURE_KEYS = (
    "currentX",
    "yTarget",
    "lastStepError",
    "solver",
    "fx",
    "dfx",
    "ddfx",
    "kappa",
    "lastKappa",
)


class Preprocessor:
    def __init__(self, graphs_dir: str, graph_cache_max: int = 128):
        """
        Initialisiert den Preprocessor.
        :param graphs_dir: Pfad zum Verzeichnis, in dem die Graph-JSONs liegen
                           (z. B. 'experiments/exp1/graphs').
        :param graph_cache_max: Max. Anzahl unterschiedlicher Graph-IDs, für die ein
            unveränderliches PyG-Template (Topologie + statische Knotenmerkmale) im RAM
            gehalten wird (LRU). Pro Schritt wird nur noch ``global_features`` und
            Metadaten gesetzt; ``convert`` läuft pro ID nur beim ersten Mal.
        """
        self.graphs_dir = Path(graphs_dir)
        self.converter = ExpressionGraphConverter()
        self._graph_cache_max = graph_cache_max
        self._known_problem_ids = self._discover_problem_ids()
        self._pyg_template_cache: OrderedDict[str, Data] = OrderedDict()

    @property
    def known_problem_ids(self) -> FrozenSet[str]:
        return self._known_problem_ids

    @property
    def cached_problem_ids(self) -> FrozenSet[str]:
        return frozenset(self._pyg_template_cache)

    def _discover_problem_ids(self) -> FrozenSet[str]:
        if not self.graphs_dir.is_dir():
            return frozenset()

        problem_ids: set[str] = set()
        for path in sorted(self.graphs_dir.glob("*_meta.json")):
            problem_ids.add(path.stem.removesuffix("_meta"))

        for path in sorted(self.graphs_dir.glob("*.json")):
            stem = path.stem
            if stem.endswith("_meta"):
                continue
            if (self.graphs_dir / f"{stem}_meta.json").is_file():
                continue
            problem_ids.add(stem)

        return frozenset(problem_ids)

    def _resolve_graph_path(self, graph_id: Any) -> Path:
        graph_id_str = str(graph_id)
        meta_path = self.graphs_dir / f"{graph_id_str}_meta.json"
        if meta_path.is_file():
            return meta_path

        json_path = self.graphs_dir / f"{graph_id_str}.json"
        if json_path.is_file():
            return json_path

        raise FileNotFoundError(
            f"Graph-Datei nicht gefunden: {graph_id_str}_meta.json (oder .json)"
        )

    def _store_template(self, cache_key: str, data: Data) -> None:
        self._pyg_template_cache[cache_key] = data.clone()
        self._pyg_template_cache.move_to_end(cache_key)
        while len(self._pyg_template_cache) > self._graph_cache_max:
            self._pyg_template_cache.popitem(last=False)

    def _graph_template_for_problem_id(self, graph_id: Any) -> Data:
        cache_key = str(graph_id)
        template = self._pyg_template_cache.get(cache_key)
        if template is not None:
            self._pyg_template_cache.move_to_end(cache_key)
            return template.clone()

        graph_path = self._resolve_graph_path(graph_id)
        with graph_path.open("r", encoding="utf-8") as graph_file:
            raw_graph = json.load(graph_file)

        data = self.converter.convert(raw_graph, heterogeneous=False)
        self._store_template(cache_key, data)
        return data

    def process(self, message: Dict[str, Any], dataloader: Any = None):
        """
        Verarbeitet eine vom network_gateway empfangene Nachricht:
        - Extrahiert relevante Status-Keys
        - Lädt den passenden Graphen anhand der 'id' (beim ersten Mal) bzw. nutzt
          ein gecachtes PyG-Template mit fester Struktur
        - Setzt pro Schritt die dynamischen ``global_features`` aus der Nachricht
        - Sendet die aufbereiteten Daten optional an den Dataloader
        """
        graph_id = message.get("id")
        if graph_id is None:
            raise ValueError("Nachricht enthält keine 'id', Graph kann nicht geladen werden.")

        extracted_features = {
            key: finite_float(message.get(key))
            for key in STATE_GLOBAL_FEATURE_KEYS
        }

        data = self._graph_template_for_problem_id(graph_id)

        feat_list = [extracted_features[key] for key in STATE_GLOBAL_FEATURE_KEYS]
        raw_tensor = torch.tensor(feat_list, dtype=torch.float).unsqueeze(0)
        data.global_features = sanitize_torch_features(
            torch.sign(raw_tensor) * torch.log1p(torch.abs(raw_tensor))
        )

        data.uuid = message.get("uuid")
        data.state_id = message.get("stateId")
        data.network_job_id = message.get("networkJobId")

        if dataloader is not None:
            pass

        return data, extracted_features
