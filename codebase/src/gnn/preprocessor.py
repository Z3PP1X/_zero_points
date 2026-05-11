import os
import json
import networkx as nx
from typing import Dict, Any
from graph_construction import ExpressionGraphConverter

class Preprocessor:
    def __init__(self, graphs_dir: str):
        """
        Initialisiert den Preprocessor.
        :param graphs_dir: Pfad zum Verzeichnis, in dem die Graph-JSONs liegen
                           (z. B. 'experiments/exp1/graphs').
        """
        self.graphs_dir = graphs_dir
        self.converter = ExpressionGraphConverter()

    def process(self, message: Dict[str, Any], dataloader: Any = None):
        """
        Verarbeitet eine vom network_gateway empfangene Nachricht:
        - Extrahiert relevante Status-Keys
        - Lädt den passenden Graphen anhand der 'id'
        - Baut ein NetworkX-Objekt und injiziert die Keys als Node-Features beim globalen Knoten
        - Sendet die aufbereiteten Daten optional an den Dataloader
        """
        graph_id = message.get("id")
        if graph_id is None:
            raise ValueError("Nachricht enthält keine 'id', Graph kann nicht geladen werden.")

        # 1. Wichtige Parameter aus der Nachricht extrahieren
        keys_to_extract = [
            "currentX", "yTarget", "lastStepError", "solver", 
            "fx", "dfx", "ddfx", "kappa", "lastKappa"
        ]
        
        extracted_features = {}
        for key in keys_to_extract:
            val = message.get(key)
            extracted_features[key] = float(val) if val is not None else 0.0

        # Die Datei wird anhand der extrahierten ID aus dem festgelegten Verzeichnis gelesen.
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}_meta.json")
        if not os.path.exists(graph_path):
            # Fallback für alte Dateinamen
            graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph-Datei nicht gefunden: {graph_id}_meta.json (oder .json)")
            
        with open(graph_path, "r", encoding="utf-8") as f:
            raw_graph = json.load(f)

        # 3. Graph in PyTorch Geometric Data konvertieren (nutzt intern _build_networkx und from_networkx)
        # Dadurch wird data.x mit den korrekten Dimensionen (input_dim=5) erstellt.
        data = self.converter.convert(raw_graph, heterogeneous=False)

        # 4. Globale Features als Tensor an das Data-Objekt hängen 
        # (wird im Batch automatisch korrekt zusammengefasst)
        feat_list = [extracted_features[k] for k in keys_to_extract]
        import torch
        raw_tensor = torch.tensor(feat_list, dtype=torch.float).unsqueeze(0) # Shape [1, 9]
        
        # Symmetrische logarithmische Normalisierung: y = sign(x) * ln(1 + |x|)
        # Verhindert, dass extrem große Werte das neuronale Netz übersättigen.
        data.global_features = torch.sign(raw_tensor) * torch.log1p(torch.abs(raw_tensor))

        # Metadaten anhängen
        data.uuid = message.get("uuid")
        data.state_id = message.get("stateId")
        data.network_job_id = message.get("networkJobId")

        # 5. Dataloader aufrufen
        if dataloader is not None:
            pass

        return data, extracted_features
