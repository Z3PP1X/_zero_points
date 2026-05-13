import os
import json
from collections import OrderedDict
from typing import Dict, Any

import torch
from torch_geometric.data import Data

from graph_utils import ExpressionGraphConverter


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
        self.graphs_dir = graphs_dir
        self.converter = ExpressionGraphConverter()
        self._graph_cache_max = graph_cache_max
        self._pyg_template_cache: OrderedDict[str, Data] = OrderedDict()

    def _graph_data_from_message(self, cache_key: str, graph_id: Any) -> Data:
        """
        Liefert ein frisches ``Data``-Objekt: entweder ``clone()`` eines gecachten
        Templates (gleiche Graph-Struktur) oder einmalig JSON laden + ``convert``.
        """
        template = self._pyg_template_cache.get(cache_key)
        if template is not None:
            self._pyg_template_cache.move_to_end(cache_key)
            return template.clone()

        graph_path = os.path.join(self.graphs_dir, f"{graph_id}_meta.json")
        if not os.path.exists(graph_path):
            graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
            if not os.path.exists(graph_path):
                raise FileNotFoundError(
                    f"Graph-Datei nicht gefunden: {graph_id}_meta.json (oder .json)"
                )

        with open(graph_path, "r", encoding="utf-8") as f:
            raw_graph = json.load(f)

        data = self.converter.convert(raw_graph, heterogeneous=False)
        self._pyg_template_cache[cache_key] = data.clone()
        self._pyg_template_cache.move_to_end(cache_key)
        while len(self._pyg_template_cache) > self._graph_cache_max:
            self._pyg_template_cache.popitem(last=False)

        # Kein extra clone: frisches ``convert``-Objekt, Template ist separate Kopie.
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
        cache_key = str(graph_id)

        # 1. Wichtige Parameter aus der Nachricht extrahieren
        keys_to_extract = [
            "currentX", "yTarget", "lastStepError", "solver",
            "fx", "dfx", "ddfx", "kappa", "lastKappa",
        ]

        extracted_features = {}
        for key in keys_to_extract:
            val = message.get(key)
            extracted_features[key] = float(val) if val is not None else 0.0

        data = self._graph_data_from_message(cache_key, graph_id)

        # 4. Globale Features als Tensor an das Data-Objekt hängen
        feat_list = [extracted_features[k] for k in keys_to_extract]
        raw_tensor = torch.tensor(feat_list, dtype=torch.float).unsqueeze(0)

        # Symmetrische logarithmische Normalisierung: y = sign(x) * ln(1 + |x|)
        data.global_features = torch.sign(raw_tensor) * torch.log1p(torch.abs(raw_tensor))

        # Metadaten anhängen
        data.uuid = message.get("uuid")
        data.state_id = message.get("stateId")
        data.network_job_id = message.get("networkJobId")

        # 5. Dataloader aufrufen
        if dataloader is not None:
            pass

        return data, extracted_features
