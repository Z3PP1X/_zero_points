import json
import torch
import networkx as nx
from pathlib import Path
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
from dataset import DatasetLoader
import numpy as np


class TopologicalFeatureExtractor:
    """Extrahiert topologische Features aus einem NetworkX Graphen."""

    @staticmethod
    def extract_and_annotate(G: nx.DiGraph) -> dict[str, int]:
        """
        Berechnet degree_centrality und fügt sie als Knoten-Attribut hinzu.
        Berechnet und liefert tree_depth und tree_width zurück.
        """
        deg_cent = nx.degree_centrality(G)
        nx.set_node_attributes(G, deg_cent, "degree_centrality")

        roots = [n for n, d in G.in_degree() if d == 0]

        if not roots:
            return {"tree_depth": 0, "tree_width": 0}

        levels = {}
        for root in roots:
            lengths = nx.single_source_shortest_path_length(G, root)
            for node, length in lengths.items():
                if node not in levels or length > levels[node]:
                    levels[node] = length

        if not levels:
            return {"tree_depth": 0, "tree_width": 0}
        tree_depth = max(levels.values())
        level_counts = {}
        for lvl in levels.values():
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        tree_width = max(level_counts.values())

        return {"tree_depth": tree_depth, "tree_width": tree_width}


class ExpressionGraphConverter:
    NODE_TYPES = {
        "global": 0,
        "operator": 1,
        "constant": 2,
        "variable": 3,
        "function": 4,
    }

    def __init__(self):
        self.label_vocab: dict[str, int] = {}
        self.edge_type_vocab: dict[str, int] = {}

    def convert(
        self, source: Union[str, Path, dict], heterogeneous: bool = False
    ) -> Union[Data, HeteroData]:
        raw = self._load(source)
        G = self._build_networkx(raw)
        graph_features = TopologicalFeatureExtractor.extract_and_annotate(G)

        if heterogeneous:
            data = self._to_hetero(G, raw)
        else:
            data = self._to_homogeneous(G, raw)

        data.tree_depth = graph_features["tree_depth"]
        data.tree_width = graph_features["tree_width"]
        return data

    @staticmethod
    def _load(source: Union[str, Path, dict]) -> dict:
        if isinstance(source, dict):
            return source
        with open(Path(source)) as f:
            return json.load(f)

    def _encode_label(self, label: str) -> int:
        if label not in self.label_vocab:
            self.label_vocab[label] = len(self.label_vocab)
        return self.label_vocab[label]

    def _encode_edge_type(self, etype: str) -> int:
        if etype not in self.edge_type_vocab:
            self.edge_type_vocab[etype] = len(self.edge_type_vocab)
        return self.edge_type_vocab[etype]

    def _build_networkx(self, raw: dict) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in raw["nodes"]:
            G.add_node(
                node["id"],
                node_type=self.NODE_TYPES[node["type"]],
                label_id=self._encode_label(node["label"]),
                value=node["value"] if node["value"] is not None else 0.0,
                has_value=1.0 if node["value"] is not None else 0.0,
            )
        for edge in raw["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                edge_type=self._encode_edge_type(edge["type"]),
            )
        return G

    @staticmethod
    def _parse_coeffs(raw_list) -> Union[torch.Tensor, None]:
        if raw_list is None:
            return None
        if isinstance(raw_list, str):
            raw_list = json.loads(raw_list)
        numeric_coeffs = []
        for c in raw_list:
            if isinstance(c, (int, float)):
                numeric_coeffs.append(float(c))
            elif isinstance(c, str):
                try:
                    numeric_coeffs.append(float(c))
                except ValueError:
                    numeric_coeffs.append(0.0)
        if not numeric_coeffs:
            return None
        return torch.tensor(numeric_coeffs, dtype=torch.float)

    def _to_homogeneous(self, G: nx.DiGraph, raw: dict) -> Data:
        data = from_networkx(
            G,
            group_node_attrs=[
                "node_type",
                "label_id",
                "value",
                "has_value",
                "degree_centrality",
            ],
            group_edge_attrs=["edge_type"],
        )
        taylor = self._parse_coeffs(raw.get("taylorCoeffs"))
        inv_taylor = self._parse_coeffs(raw.get("inverseTaylorCoeffs"))

        default_len = 11
        data.taylor_coeffs = taylor if taylor is not None else torch.zeros(default_len)
        data.inv_taylor_coeffs = (
            inv_taylor if inv_taylor is not None else torch.zeros(default_len)
        )

        return data

    def _to_hetero(self, G: nx.DiGraph, raw: dict) -> HeteroData:
        node_ids = list(G.nodes)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        node_type = torch.tensor(
            [G.nodes[n]["node_type"] for n in node_ids], dtype=torch.long
        )
        label_id = torch.tensor(
            [G.nodes[n]["label_id"] for n in node_ids], dtype=torch.long
        )
        value = torch.tensor([G.nodes[n]["value"] for n in node_ids], dtype=torch.float)
        has_value = torch.tensor(
            [G.nodes[n]["has_value"] for n in node_ids], dtype=torch.float
        )
        deg_cent = torch.tensor(
            [G.nodes[n]["degree_centrality"] for n in node_ids], dtype=torch.float
        )

        x = torch.stack(
            [node_type.float(), label_id.float(), value, has_value, deg_cent], dim=1
        )

        edge_buckets: dict[str, list[tuple[int, int]]] = {}
        for edge in raw["edges"]:
            etype = edge["type"]
            src = id_to_idx[edge["source"]]
            tgt = id_to_idx[edge["target"]]
            edge_buckets.setdefault(etype, []).append((src, tgt))

        hetero = HeteroData()
        hetero["node"].x = x
        hetero["node"].node_type = node_type
        hetero["node"].label_id = label_id

        for etype, pairs in edge_buckets.items():
            src_ids, tgt_ids = zip(*pairs)
            hetero["node", etype, "node"].edge_index = torch.tensor(
                [list(src_ids), list(tgt_ids)], dtype=torch.long
            )

        taylor = self._parse_coeffs(raw.get("taylorCoeffs"))
        inv_taylor = self._parse_coeffs(raw.get("inverseTaylorCoeffs"))
        if taylor is not None:
            hetero.taylor_coeffs = taylor
        if inv_taylor is not None:
            hetero.inv_taylor_coeffs = inv_taylor

        return hetero


class GraphConversionPipeline:
    """Loads all JSON graph files from a directory and converts them to PyG objects."""

    def __init__(self, experiments_dir: Union[str, Path], heterogeneous: bool = False):
        self.experiments_dir = Path(experiments_dir)
        self.heterogeneous = heterogeneous
        self.converter = ExpressionGraphConverter()
        self.graphs: dict[str, Union[Data, HeteroData]] = {}
        self._convert_all()

    def _discover_json_files(self) -> list[Path]:
        return sorted(self.experiments_dir.glob("**/*.json"))

    def _convert_all(self):
        for json_path in self._discover_json_files():
            raw = self.converter._load(json_path)
            if "nodes" not in raw or "edges" not in raw:
                print(f"Übersprungen: {json_path} — Keys: {list(raw.keys())}")
                continue
            graph_id = raw.get("id", json_path.stem)
            self.graphs[graph_id] = self.converter.convert(
                raw, heterogeneous=self.heterogeneous
            )
        print(f"Geladen: {len(self.graphs)} Graphen")

    def get_data(self) -> dict[str, Union[Data, HeteroData]]:
        return self.graphs

    @property
    def label_vocab(self) -> dict[str, int]:
        return self.converter.label_vocab

    @property
    def edge_type_vocab(self) -> dict[str, int]:
        return self.converter.edge_type_vocab

    @property
    def input_dim(self) -> int:
        """Gibt die Anzahl der Node-Features (input_dim) zurück."""
        if not self.graphs:
            return 0

        sample_graph = next(iter(self.graphs.values()))

        if self.heterogeneous:
            return sample_graph["node"].x.shape[1]
        else:
            return sample_graph.x.shape[1]

    def get_feature_schema(self) -> list[str]:
        """Hilfsfunktion: Welche Features stecken in welcher Spalte?"""
        return ["node_type", "label_id", "value", "has_value", "degree_centrality"]


class FeatureEngineering:
    def __init__(self, loader: DatasetLoader):
        self._loader = loader

    def _tag_faster_algorithm(self):
        """Set binary labels for the faster algorithm: 0: Newton, 1: gMGF"""
        boundaries = [
            self._loader.data["avg_abs_time_newton"]
            < self._loader.data["avg_abs_time_gmgf"],
            self._loader.data["avg_abs_time_newton"]
            > self._loader.data["avg_abs_time_gmgf"],
        ]
        values = [0, 1]

        self._loader.add_column("faster_algorithm", np.select(boundaries, values))

    def _conserve_relationships(self):
        """Conserve relationships between absolute times"""
        """self._data["conserved_time_rel"] = (
            self._data["avg_abs_time_newton"] / self._data["avg_abs_time_gmgf"]
        )"""

        self._loader.add_column(
            "conserved_step_rel",
            self._loader.data["schritte_newton"] / self._loader.data["schritte_gmgf"],
        )
