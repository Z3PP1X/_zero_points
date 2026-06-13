import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, FrozenSet

import torch
from torch_geometric.data import Data

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.reinforcement_learning.feature_layout import (
    NATIVE_NODE_FEATURE_COUNT,
)
from gnn.shared.utils.graph_utils import populate_task_virtual_values
from gnn.reinforcement_learning.observation_sanitize import finite_float, sanitize_torch_features

# NOTE: `solver` is intentionally NOT a global feature — the chosen solver is the
# network's action/decision, not an observation. Removing it makes the global vector 8-wide.
STATE_GLOBAL_FEATURE_KEYS = (
    "currentX",
    "yTarget",
    "lastStepError",
    "fx",
    "dfx",
    "ddfx",
    "kappa",
    "lastKappa",
)


class Preprocessor:
    def __init__(
        self,
        loader: GraphDataLoader | None = None,
        graphs_dir: str | None = None,
        graph_cache_max: int = 128,
        mode: str = "graph",
        active_features: list[str] | None = None,
        add_kappa: bool = False,
    ):
        """
        Initialisiert den Preprocessor.
        :param loader: Eine GraphDataLoader Instanz (Dependency Injection).
        :param graphs_dir: Pfad zum Verzeichnis, in dem die Graph-JSONs liegen (Legacy Fallback).
        :param graph_cache_max: Max. Anzahl unterschiedlicher Graph-IDs, für die ein
            unveränderliches PyG-Template (Topologie + statische Knotenmerkmale) im RAM
            gehalten wird (LRU). Pro Schritt wird nur noch ``global_features`` und
            Metadaten gesetzt; ``convert`` läuft pro ID nur beim ersten Mal.
        """
        self.mode = mode
        self.active_features = active_features
        self._graph_cache_max = graph_cache_max
        self._pyg_template_cache: OrderedDict[str, Data] = OrderedDict()

        if loader is not None:
            self.loader = loader
            self.graphs_dir = loader.source_path
            self.converter = loader.converter
        elif graphs_dir is not None:
            self.loader = GraphDataLoader(
                name=Path(graphs_dir).name,
                mode=mode,
                heterogeneous=False,
                base_dir=graphs_dir,
                add_kappa=add_kappa,
            )
            self.graphs_dir = Path(graphs_dir)
            self.converter = self.loader.converter
        else:
            raise ValueError("Entweder 'loader' oder 'graphs_dir' muss an den Preprocessor uebergeben werden.")

        self._known_problem_ids = frozenset(self.loader.list_graph_ids())

    @property
    def padded_node_feature_count(self) -> int:
        if self.active_features is not None:
            return len(self.active_features)
        return NATIVE_NODE_FEATURE_COUNT

    @property
    def known_problem_ids(self) -> FrozenSet[str]:
        return self._known_problem_ids

    @property
    def cached_problem_ids(self) -> FrozenSet[str]:
        return frozenset(self._pyg_template_cache)

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

        data = self.loader.get_graph(graph_id)
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
        # Raw (un-normalized) global features; the GlobalEncoder's learnable
        # LayerNorm + Linear handle scaling. No hand-crafted sign-log transform.
        raw_tensor = torch.tensor(feat_list, dtype=torch.float).unsqueeze(0)
        data.global_features = sanitize_torch_features(raw_tensor)

        populate_task_virtual_values(
            data,
            cx_val=extracted_features.get("currentX", 0.0),
            fx_val=extracted_features.get("fx", 0.0),
            yt_val=extracted_features.get("yTarget", 0.0),
            d1x_val=extracted_features.get("dfx", 0.0),
            d2x_val=extracted_features.get("ddfx", 0.0),
            mode=self.mode,
        )

        data.uuid = message.get("uuid")
        data.state_id = message.get("stateId")
        data.network_job_id = message.get("networkJobId")

        # Slice active features if selection is active
        if self.active_features is not None and data.x is not None:
            from gnn.shared.utils.graph_utils import slice_active_features
            data.x = slice_active_features(data.x, self.active_features)

        if dataloader is not None:
            pass

        return data, extracted_features
