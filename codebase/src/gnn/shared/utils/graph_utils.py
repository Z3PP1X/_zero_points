import json
import torch
import networkx as nx
from pathlib import Path
from typing import Union, Any, Dict
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
import numpy as np


class TopologicalFeatureExtractor:
    """Extrahiert topologische Features aus einem NetworkX Graphen."""

    @staticmethod
    def extract_and_annotate(G: nx.DiGraph, enrich: bool = True) -> dict:
        deg_cent = nx.degree_centrality(G)
        nx.set_node_attributes(G, deg_cent, "degree_centrality")

        roots = [n for n, d in G.in_degree() if d == 0]

        # Depths (Depth)
        levels = {}
        if roots:
            for root in roots:
                lengths = nx.single_source_shortest_path_length(G, root)
                for node, length in lengths.items():
                    if node not in levels or length > levels[node]:
                        levels[node] = length
        for node in G.nodes:
            if node not in levels:
                levels[node] = 0

        # Global tree depth
        tree_depth = max(levels.values()) if levels else 0

        # Global tree width
        level_counts = {}
        for lvl in levels.values():
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        tree_width = max(level_counts.values()) if level_counts else 0

        # Basic results
        results = {
            "tree_depth": tree_depth,
            "tree_width": tree_width,
            "depths": levels,
        }

        # Return early if enrichment is not requested (supervised learning legacy mode)
        if not enrich:
            return results

        # Heights (Height)
        heights = {}
        visiting = set()
        def get_height(node):
            if node in heights:
                return heights[node]
            if node in visiting:
                return 0
            visiting.add(node)
            children = list(G.successors(node))
            if not children:
                visiting.remove(node)
                heights[node] = 0
                return 0
            h = 1 + max(get_height(child) for child in children)
            visiting.remove(node)
            heights[node] = h
            return h
        for node in G.nodes:
            get_height(node)

        # Subtree Sizes (SubtreeSize)
        subtree_sizes = {}
        for node in G.nodes:
            subtree_sizes[node] = len(nx.descendants(G, node)) + 1

        # Out-Degrees (Out-Degree)
        out_degrees = {node: G.out_degree(node) for node in G.nodes}

        # Undirected graph representation for Laplacians, LPE, RWPE, and Undirected Centralities
        G_und = G.to_undirected()
        num_nodes = G_und.number_of_nodes()

        # Betweenness Centrality (computed on undirected graph)
        betweenness = nx.betweenness_centrality(G_und)

        # Edge Betweenness Centrality (computed on undirected graph)
        edge_betweenness = nx.edge_betweenness_centrality(G_und)
        eb_lookup = {}
        for (u, v), val in edge_betweenness.items():
            eb_lookup[(u, v)] = val
            eb_lookup[(v, u)] = val

        # Laplace-Matrix (Graph)
        if num_nodes > 0:
            laplace_matrix = nx.laplacian_matrix(G_und).toarray()
        else:
            laplace_matrix = np.zeros((0, 0))

        # Laplacian Positional Encodings (LPE) (dimension 4)
        lpe_features = np.zeros((num_nodes, 4))
        if num_nodes > 1:
            try:
                A = nx.to_numpy_array(G_und)
                d = A.sum(axis=1)
                d_inv_sqrt = np.zeros_like(d)
                d_inv_sqrt[d > 0] = np.power(d[d > 0], -0.5)
                D_inv_sqrt = np.diag(d_inv_sqrt)
                L_norm = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
                
                evals, evecs = np.linalg.eigh(L_norm)
                idx = np.argsort(evals)
                evals = evals[idx]
                evecs = evecs[:, idx]
                
                lpe_list = []
                for i in range(1, 5):
                    if i < num_nodes:
                        lpe_list.append(evecs[:, i])
                    else:
                        lpe_list.append(np.zeros(num_nodes))
                lpe_features = np.stack(lpe_list, axis=1)
            except Exception:
                lpe_features = np.zeros((num_nodes, 4))

        # Random Walk Positional Encodings (RWPE) (4 steps)
        rwpe_features = np.zeros((num_nodes, 4))
        if num_nodes > 0:
            try:
                A = nx.to_numpy_array(G_und)
                d = A.sum(axis=1)
                d_inv = np.zeros_like(d)
                d_inv[d > 0] = 1.0 / d[d > 0]
                D_inv = np.diag(d_inv)
                P = D_inv @ A
                
                Pk = np.eye(num_nodes)
                for step in range(4):
                    Pk = Pk @ P
                    rwpe_features[:, step] = np.diag(Pk)
            except Exception:
                rwpe_features = np.zeros((num_nodes, 4))

        results.update({
            "heights": heights,
            "subtree_sizes": subtree_sizes,
            "out_degrees": out_degrees,
            "betweenness": betweenness,
            "edge_betweenness": eb_lookup,
            "laplace_matrix": laplace_matrix,
            "lpe": lpe_features,
            "rwpe": rwpe_features,
        })
        return results


class ExpressionGraphConverter:
    NODE_TYPES = {
        "global": 0,
        "operator": 1,
        "constant": 2,
        "variable": 3,
        "function": 4,
        "virtual_current_x": 5,
        "virtual_f_x": 6,
        "virtual_y_target": 7,
    }

    def __init__(self):
        self.label_vocab: dict[str, int] = {}
        self.edge_type_vocab: dict[str, int] = {}

    def convert(
        self, source: Union[str, Path, dict], heterogeneous: bool = False, enrich: bool = True, mode: str = "graph"
    ) -> Union[Data, HeteroData]:
        raw = self._load(source)
        
        # Make a copy of raw to avoid modifying the original dict in-place if passed as object
        raw = dict(raw)
        raw["nodes"] = list(raw.get("nodes", []))
        raw["edges"] = list(raw.get("edges", []))
        
        if mode == "graph":
            # Find variable nodes and global node
            variable_node_ids = []
            global_node_id = None
            for node in raw["nodes"]:
                if node.get("type") == "variable":
                    variable_node_ids.append(node["id"])
                elif node.get("type") == "global":
                    global_node_id = node["id"]
            
            # Add the three virtual nodes
            raw["nodes"].append({
                "id": "virtual_current_x",
                "label": "virtual_current_x",
                "type": "virtual_current_x",
                "value": None
            })
            raw["nodes"].append({
                "id": "virtual_f_x",
                "label": "virtual_f_x",
                "type": "virtual_f_x",
                "value": None
            })
            raw["nodes"].append({
                "id": "virtual_y_target",
                "label": "virtual_y_target",
                "type": "virtual_y_target",
                "value": None
            })
            
            # Add virtual edges:
            # virtual_current_x -> all variables
            for var_id in variable_node_ids:
                raw["edges"].append({
                    "source": "virtual_current_x",
                    "target": var_id,
                    "type": "virtual"
                })
            # virtual_current_x -> virtual_f_x
            raw["edges"].append({
                "source": "virtual_current_x",
                "target": "virtual_f_x",
                "type": "virtual"
            })
            # virtual_f_x -> virtual_current_x
            raw["edges"].append({
                "source": "virtual_f_x",
                "target": "virtual_current_x",
                "type": "virtual"
            })
            # virtual_f_x -> virtual_y_target
            raw["edges"].append({
                "source": "virtual_f_x",
                "target": "virtual_y_target",
                "type": "virtual"
            })
            # virtual_y_target -> virtual_f_x
            raw["edges"].append({
                "source": "virtual_y_target",
                "target": "virtual_f_x",
                "type": "virtual"
            })
            # virtual_y_target -> global node
            if global_node_id is not None:
                raw["edges"].append({
                    "source": "virtual_y_target",
                    "target": global_node_id,
                    "type": "virtual"
                })
        
        # 1. Build directed graph first (forward edges only)
        G_directed = self._build_networkx(raw)
        topo = TopologicalFeatureExtractor.extract_and_annotate(G_directed, enrich=enrich)

        # 2. Enrich attributes based on mode
        G_enriched = nx.DiGraph()
        node_ids = list(G_directed.nodes)
        
        if enrich:
            # Full 16-feature set (RL mode)
            for i, node in enumerate(node_ids):
                attrs = G_directed.nodes[node]
                enriched_attrs = attrs.copy()
                enriched_attrs["depth"] = float(topo["depths"].get(node, 0.0))
                enriched_attrs["height"] = float(topo["heights"].get(node, 0.0))
                enriched_attrs["subtree_size"] = float(topo["subtree_sizes"].get(node, 1.0))
                enriched_attrs["out_degree"] = float(topo["out_degrees"].get(node, 0.0))
                enriched_attrs["betweenness_centrality"] = float(topo["betweenness"].get(node, 0.0))
                
                # Laplacian Positional Encodings
                enriched_attrs["lpe_1"] = float(topo["lpe"][i, 0])
                enriched_attrs["lpe_2"] = float(topo["lpe"][i, 1])
                enriched_attrs["lpe_3"] = float(topo["lpe"][i, 2])
                enriched_attrs["lpe_4"] = float(topo["lpe"][i, 3])
                
                # Random Walk Positional Encodings
                enriched_attrs["rwpe_1"] = float(topo["rwpe"][i, 0])
                enriched_attrs["rwpe_2"] = float(topo["rwpe"][i, 1])
                enriched_attrs["rwpe_3"] = float(topo["rwpe"][i, 2])
                enriched_attrs["rwpe_4"] = float(topo["rwpe"][i, 3])
                
                G_enriched.add_node(node, **enriched_attrs)

            # Build bidirectional edges for rich representation
            child_counters = {}
            for edge in raw.get("edges", []):
                parent = edge["source"]
                child = edge["target"]
                etype = edge["type"]

                child_idx = child_counters.get(parent, 0)
                child_counters[parent] = child_idx + 1

                # Fetch edge betweenness centrality
                eb_val = float(topo["edge_betweenness"].get((parent, child), 0.0))

                # Forward Edge
                G_enriched.add_edge(
                    parent,
                    child,
                    child_index=float(child_idx),
                    direction=0.0,
                    relation_type=float(self._encode_edge_type(etype)),
                    edge_betweenness_centrality=eb_val,
                )

                # Backward Edge
                G_enriched.add_edge(
                    child,
                    parent,
                    child_index=float(child_idx),
                    direction=1.0,
                    relation_type=float(self._encode_edge_type(etype + "_reverse")),
                    edge_betweenness_centrality=eb_val,
                )
        else:
            # Basic 5-feature set (Supervised learning mode)
            for node in node_ids:
                attrs = G_directed.nodes[node]
                enriched_attrs = attrs.copy()
                enriched_attrs["degree_centrality"] = float(topo["depths"].get(node, 0.0))  # wait, using degree centrality as required
                # Wait, degree centrality was already set on G_directed inside TopologicalFeatureExtractor
                G_enriched.add_node(node, **enriched_attrs)

            for edge in raw.get("edges", []):
                G_enriched.add_edge(
                    edge["source"],
                    edge["target"],
                    edge_type=self._encode_edge_type(edge["type"]),
                )

        if heterogeneous:
            data = self._to_hetero(G_enriched, raw, topo, enrich)
            data["node"].node_ids = node_ids
        else:
            data = self._to_homogeneous(G_enriched, raw, enrich)
            data.node_ids = node_ids

        # Add global graph features
        data.tree_depth = topo["tree_depth"]
        data.tree_width = topo["tree_width"]
        if enrich:
            data.treewidth = topo["tree_width"]
            data.nodes = G_directed.number_of_nodes()
            data.num_nodes = G_directed.number_of_nodes()
            data.edges = G_directed.number_of_edges()
            data.num_edges = G_directed.number_of_edges()
            data.laplacian = torch.tensor(topo["laplace_matrix"], dtype=torch.float)

        return data

    @staticmethod
    def _load(source: Union[str, Path, dict]) -> dict:
        if isinstance(source, dict):
            return source
        with open(Path(source), encoding="utf-8") as f:
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
            val_dict = node.get("value")
            if isinstance(val_dict, dict) and val_dict.get("mantissa") is not None:
                mantissa = val_dict["mantissa"]
                exponent = val_dict.get("exponent", 0)
                actual_value = float(mantissa * (10 ** exponent))
                has_val = 1.0
            else:
                if isinstance(val_dict, (int, float)):
                    actual_value = float(val_dict)
                    has_val = 1.0
                else:
                    actual_value = 0.0
                    has_val = 0.0
                    
            G.add_node(
                node["id"],
                node_type=self.NODE_TYPES[node["type"]],
                label_id=self._encode_label(node["label"]),
                value=actual_value,
                has_value=has_val,
                virtual_current_x_val=0.0,
                virtual_f_x_val=0.0,
                virtual_y_target_val=0.0,
            )
        for edge in raw["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                edge_type=self._encode_edge_type(edge["type"]),
            )
        return G

    def _to_homogeneous(self, G: nx.DiGraph, raw: dict, enrich: bool) -> Data:
        if enrich:
            if G.number_of_edges() == 0:
                data = from_networkx(
                    G,
                    group_node_attrs=[
                        "node_type",
                        "depth",
                        "height",
                        "subtree_size",
                        "out_degree",
                        "betweenness_centrality",
                        "label_id",
                        "value",
                        "lpe_1", "lpe_2", "lpe_3", "lpe_4",
                        "rwpe_1", "rwpe_2", "rwpe_3", "rwpe_4",
                        "virtual_current_x_val",
                        "virtual_f_x_val",
                        "virtual_y_target_val",
                    ],
                )
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
                data.edge_attr = torch.empty((0, 4), dtype=torch.float)
                return data

            return from_networkx(
                G,
                group_node_attrs=[
                    "node_type",
                    "depth",
                    "height",
                    "subtree_size",
                    "out_degree",
                    "betweenness_centrality",
                    "label_id",
                    "value",
                    "lpe_1", "lpe_2", "lpe_3", "lpe_4",
                    "rwpe_1", "rwpe_2", "rwpe_3", "rwpe_4",
                    "virtual_current_x_val",
                    "virtual_f_x_val",
                    "virtual_y_target_val",
                ],
                group_edge_attrs=["child_index", "direction", "relation_type", "edge_betweenness_centrality"],
            )
        else:
            if G.number_of_edges() == 0:
                data = from_networkx(
                    G,
                    group_node_attrs=[
                        "node_type",
                        "label_id",
                        "value",
                        "has_value",
                        "degree_centrality",
                        "virtual_current_x_val",
                        "virtual_f_x_val",
                        "virtual_y_target_val",
                    ],
                )
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
                data.edge_attr = torch.empty((0, 1), dtype=torch.float)
                return data

            return from_networkx(
                G,
                group_node_attrs=[
                    "node_type",
                    "label_id",
                    "value",
                    "has_value",
                    "degree_centrality",
                    "virtual_current_x_val",
                    "virtual_f_x_val",
                    "virtual_y_target_val",
                ],
                group_edge_attrs=["edge_type"],
            )

    def _to_hetero(self, G: nx.DiGraph, raw: dict, topo: dict, enrich: bool) -> HeteroData:
        node_ids = list(G.nodes)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        node_type = torch.tensor([G.nodes[n]["node_type"] for n in node_ids], dtype=torch.long)
        label_id = torch.tensor([G.nodes[n]["label_id"] for n in node_ids], dtype=torch.long)
        value = torch.tensor([G.nodes[n]["value"] for n in node_ids], dtype=torch.float)
        
        v_cx = torch.tensor([G.nodes[n]["virtual_current_x_val"] for n in node_ids], dtype=torch.float)
        v_fx = torch.tensor([G.nodes[n]["virtual_f_x_val"] for n in node_ids], dtype=torch.float)
        v_yt = torch.tensor([G.nodes[n]["virtual_y_target_val"] for n in node_ids], dtype=torch.float)

        if enrich:
            depth = torch.tensor([G.nodes[n]["depth"] for n in node_ids], dtype=torch.float)
            height = torch.tensor([G.nodes[n]["height"] for n in node_ids], dtype=torch.float)
            subtree_size = torch.tensor([G.nodes[n]["subtree_size"] for n in node_ids], dtype=torch.float)
            out_degree = torch.tensor([G.nodes[n]["out_degree"] for n in node_ids], dtype=torch.float)
            betweenness = torch.tensor([G.nodes[n]["betweenness_centrality"] for n in node_ids], dtype=torch.float)
            lpe = torch.tensor(topo["lpe"], dtype=torch.float)
            rwpe = torch.tensor(topo["rwpe"], dtype=torch.float)

            x = torch.stack(
                [
                    node_type.float(), depth, height, subtree_size, out_degree, betweenness, label_id.float(), value,
                    lpe[:, 0], lpe[:, 1], lpe[:, 2], lpe[:, 3],
                    rwpe[:, 0], rwpe[:, 1], rwpe[:, 2], rwpe[:, 3],
                    v_cx, v_fx, v_yt
                ],
                dim=1
            )
        else:
            has_value = torch.tensor([G.nodes[n]["has_value"] for n in node_ids], dtype=torch.float)
            deg_cent = torch.tensor([G.nodes[n]["degree_centrality"] for n in node_ids], dtype=torch.float)
            x = torch.stack(
                [node_type.float(), label_id.float(), value, has_value, deg_cent, v_cx, v_fx, v_yt], dim=1
            )

        edge_buckets: dict[str, list[tuple[int, int]]] = {}
        for edge in raw["edges"]:
            etype = edge["type"]
            src = id_to_idx[edge["source"]]
            tgt = id_to_idx[edge["target"]]
            edge_buckets.setdefault(etype, []).append((src, tgt))
            if enrich:
                edge_buckets.setdefault(etype + "_reverse", []).append((tgt, src))

        hetero = HeteroData()
        hetero["node"].x = x
        hetero["node"].node_type = node_type
        hetero["node"].label_id = label_id

        for etype, pairs in edge_buckets.items():
            src_ids, tgt_ids = zip(*pairs)
            hetero["node", etype, "node"].edge_index = torch.tensor(
                [list(src_ids), list(tgt_ids)], dtype=torch.long
            )

        return hetero


class GraphConversionPipeline:
    """Loads all JSON graph files from a directory and converts them to PyG objects."""

    def __init__(self, experiments_dir: Union[str, Path], heterogeneous: bool = False, enrich: bool = True, mode: str = "graph"):
        self.experiments_dir = Path(experiments_dir)
        self.heterogeneous = heterogeneous
        self.enrich = enrich
        self.mode = mode
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
                raw, heterogeneous=self.heterogeneous, enrich=self.enrich, mode=self.mode
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
        if not self.graphs:
            return 0

        sample_graph = next(iter(self.graphs.values()))

        if self.heterogeneous:
            return sample_graph["node"].x.shape[1]
        else:
            return sample_graph.x.shape[1]

    def get_feature_schema(self) -> list[str]:
        if self.enrich:
            return [
                "node_type", "depth", "height", "subtree_size", "out_degree", "betweenness_centrality", "label_id", "value",
                "lpe_1", "lpe_2", "lpe_3", "lpe_4", "rwpe_1", "rwpe_2", "rwpe_3", "rwpe_4",
                "virtual_current_x_val", "virtual_f_x_val", "virtual_y_target_val"
            ]
        else:
            return [
                "node_type", "label_id", "value", "has_value", "degree_centrality",
                "virtual_current_x_val", "virtual_f_x_val", "virtual_y_target_val"
            ]


def slice_active_features(x: torch.Tensor, active_features: list[str] | None, enrich: bool) -> torch.Tensor:
    if active_features is None:
        return x
    full_schema = [
        "node_type", "depth", "height", "subtree_size", "out_degree", "betweenness_centrality", "label_id", "value",
        "lpe_1", "lpe_2", "lpe_3", "lpe_4", "rwpe_1", "rwpe_2", "rwpe_3", "rwpe_4",
        "virtual_current_x_val", "virtual_f_x_val", "virtual_y_target_val"
    ] if enrich else [
        "node_type", "label_id", "value", "has_value", "degree_centrality",
        "virtual_current_x_val", "virtual_f_x_val", "virtual_y_target_val"
    ]
    indices = []
    for f in active_features:
        if f in full_schema:
            indices.append(full_schema.index(f))
        else:
            raise ValueError(f"Feature '{f}' is not in the schema (enrich={enrich}). Available: {full_schema}")
    return x[:, indices]
