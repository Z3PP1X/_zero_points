import os
import json
from typing import Dict, Any

import torch
from torch_geometric.data import Data

from graph_utils import ExpressionGraphConverter


class Preprocessor:
    def __init__(self, graphs_dir: str):
        """
        Initialisiert den Preprocessor.
        :param graphs_dir: Pfad zum Verzeichnis, in dem die Graph-JSONs liegen
                           (z. B. 'experiments/exp1/graphs').
        """
        self.graphs_dir = graphs_dir
        self.converter = ExpressionGraphConverter()
        # Static AST / topology per graph id; per-step scalars go into global_features only.
        self._static_template_cache: Dict[str, Data] = {}

    def process(self, message: Dict[str, Any], dataloader: Any = None):
        """
        Verarbeitet eine vom network_gateway empfangene Nachricht:
        - Extrahiert relevante Status-Keys
        - Lädt den passenden Graphen anhand der 'id' (einmalig pro id, dann gecacht)
        - Baut PyG ``Data``; Zeitreihen-Skalare liegen in ``global_features`` (pro Nachricht)
        - Sendet die aufbereiteten Daten optional an den Dataloader
        """
        graph_id = message.get("id")
        if graph_id is None:
            raise ValueError("Nachricht enthält keine 'id', Graph kann nicht geladen werden.")

        cache_key = str(graph_id)

        # 1. Wichtige Parameter aus der Nachricht extrahieren
        keys_to_extract = [
            "currentX", "yTarget", "lastStepError", "solver", 
            "fx", "dfx", "ddfx", "kappa", "lastKappa"
        ]
        
        extracted_features = {}
        for key in keys_to_extract:
            val = message.get(key)
            extracted_features[key] = float(val) if val is not None else 0.0

        if cache_key not in self._static_template_cache:
            graph_path = os.path.join(self.graphs_dir, f"{graph_id}_meta.json")
            if not os.path.exists(graph_path):
                graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
                if not os.path.exists(graph_path):
                    raise FileNotFoundError(
                        f"Graph-Datei nicht gefunden: {graph_id}_meta.json (oder .json)"
                    )

            with open(graph_path, "r", encoding="utf-8") as f:
                raw_graph = json.load(f)

            template = self.converter.convert(raw_graph, heterogeneous=False)
            self._static_template_cache[cache_key] = template

        template = self._static_template_cache[cache_key]
        # Share static topology tensors (no clone): only global_features vary per message.
        # Downstream must not mutate template.x / edge_index in-place.
        feat_list = [extracted_features[k] for k in keys_to_extract]
        raw_tensor = torch.tensor(feat_list, dtype=torch.float).unsqueeze(0)  # Shape [1, 9]
        
        # Symmetrische logarithmische Normalisierung: y = sign(x) * ln(1 + |x|)
        # Verhindert, dass extrem große Werte das neuronale Netz übersättigen.
        global_features = torch.sign(raw_tensor) * torch.log1p(torch.abs(raw_tensor))
        data = Data(
            x=template.x,
            edge_index=template.edge_index,
            global_features=global_features,
        )

        # Metadaten anhängen
        data.uuid = message.get("uuid")
        data.state_id = message.get("stateId")
        data.network_job_id = message.get("networkJobId")

        # 5. Dataloader aufrufen
        if dataloader is not None:
            pass

        return data, extracted_features
