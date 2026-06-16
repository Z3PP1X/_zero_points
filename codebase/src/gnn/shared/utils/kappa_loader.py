import json
import logging
import math
from pathlib import Path
from typing import Union
import torch
import networkx as nx
from torch_geometric.data import Data, HeteroData
from gnn.shared.utils.graph_vocab import (
    ROOT_COLOR_VOCAB, encode_label, encode_edge_type,
)
from gnn.shared.utils.graph_converter import (
    ExpressionGraphConverter,
    parse_graphml_node_name, _determine_node_type_from_label, _parse_constant_value,
    create_virtual_global_node, _mark_function_roots,
)

logger = logging.getLogger(__name__)


# Module-level cache: resolved kappas folder path -> {kappa_value: (root_id, normalized_graph)}.
# Populated once on the first call to _load_normalized_kappas(); all subsequent graph loads reuse it.
_kappa_graph_cache: dict[str, dict[float, tuple]] = {}


def _parse_kappa_raw(raw) -> nx.DiGraph:
    """Parse a raw kappa subgraph (GraphML str, dict, or DiGraph) into a bare nx.DiGraph."""
    if isinstance(raw, str):
        content = raw.replace("attr.type='String'", "attr.type='string'")
        content = content.replace('attr.type="String"', 'attr.type="string"')
        return nx.parse_graphml(content)
    if isinstance(raw, nx.DiGraph):
        return raw
    if isinstance(raw, dict):
        g = nx.DiGraph()
        for node in raw.get("nodes", []):
            g.add_node(node["id"], **node)
        for edge in raw.get("edges", []):
            g.add_edge(edge["source"], edge["target"], **edge)
        return g
    raise TypeError(f"Unsupported kappa subgraph type: {type(raw)}")


def _normalize_kappa_graph(g_raw: nx.DiGraph) -> tuple:
    """Normalize a raw kappa DiGraph, identify its root, and pre-mark root attributes.

    Returns:
        (original_root_id, normalized_nx.DiGraph) or (None, empty DiGraph) if the
        subgraph is empty.  The root node already has node_type/root_color/type set
        to the kappa-root values so the per-graph copy loop needs no special case.
    """
    normalized = nx.DiGraph()
    for nid, attrs in g_raw.nodes(data=True):
        name_val = attrs.get("Name") or attrs.get("nodeKey1") or attrs.get("label") or str(nid)
        label = parse_graphml_node_name(name_val) if isinstance(name_val, str) else str(name_val)
        type_str = attrs.get("type") or _determine_node_type_from_label(label)
        ntype_code = ExpressionGraphConverter.NODE_TYPES.get(type_str, 1)
        normalized.add_node(
            nid,
            node_type=ntype_code,
            root_color=float(ROOT_COLOR_VOCAB["none"]),
            label_id=encode_label(label),
            label=label,
            type=type_str,
        )
    for u, v, attrs in g_raw.edges(data=True):
        etype = attrs.get("type") or attrs.get("etype") or "child_of"
        normalized.add_edge(u, v, edge_type=encode_edge_type(etype), etype=etype)

    roots = [n for n, d in normalized.in_degree() if d == 0]
    original_root = roots[0] if roots else (list(normalized.nodes)[0] if normalized.nodes else None)

    if original_root is not None:
        normalized.nodes[original_root]["node_type"] = ExpressionGraphConverter.NODE_TYPES["root"]
        normalized.nodes[original_root]["root_color"] = float(ROOT_COLOR_VOCAB["kappa"])
        normalized.nodes[original_root]["type"] = "root"

    return original_root, normalized


def _load_normalized_kappas(kappas_path: Path) -> dict:
    """Return {kappa_value: (root_id, normalized_graph)} for all entries in *kappas_path*.

    The result is computed once per unique folder and then stored in
    ``_kappa_graph_cache`` so every subsequent call is an O(1) dict lookup.
    The dict key enables a single O(1) lookup when only one kappa is needed.
    """
    key = str(kappas_path.resolve())
    if key in _kappa_graph_cache:
        return _kappa_graph_cache[key]

    entries: dict[float, tuple] = {}
    for file_path in sorted(kappas_path.glob("**/*.json")):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            containers = data if isinstance(data, list) else [data]
            for kc in containers:
                if not isinstance(kc, dict) or kc.get("id") != "kappa":
                    continue
                try:
                    kappa_val = float(kc.get("value", 0.0))
                except (ValueError, TypeError):
                    kappa_val = 0.0
                raw_subgraph = kc.get("graphStructure") or kc.get("graphml_h")
                if not raw_subgraph:
                    continue
                g_raw = _parse_kappa_raw(raw_subgraph)
                original_root, normalized = _normalize_kappa_graph(g_raw)
                if original_root is None:
                    continue
                entries[kappa_val] = (original_root, normalized)
        except Exception as e:
            logger.warning(f"Error loading kappa file {file_path}: {e}")

    _kappa_graph_cache[key] = entries
    return entries


def _tag_and_connect_kappa(
    graph: "AugmentedFunctionGraph", global_node: str, kappa_root_id: str, kappa_val: float
) -> None:
    """Tag merged kappa nodes with their value and wire global↔kappa edges."""
    prefix_parts = kappa_root_id.split("_")
    if len(prefix_parts) >= 2:
        prefix_str = f"{prefix_parts[0]}_{prefix_parts[1]}"
        for node in graph.nodes:
            if str(node).startswith(prefix_str + "_"):
                graph.nodes[node]["kappa_value"] = kappa_val

    fwd = KappaEdge(source=global_node, target=kappa_root_id, type="GlobalToKappa")
    fwd.features["weight"] = kappa_val
    graph.AddEdge(fwd)

    bwd = KappaEdge(source=kappa_root_id, target=global_node, type="KappaToGlobal")
    bwd.features["weight"] = kappa_val
    graph.AddEdge(bwd)


class KappaEdge:
    """Represents a connection between the global node and a kappa root node.

    Attributes:
        source (str): The ID of the source node.
        target (str): The ID of the target node.
        type (str): The type of the edge ("GlobalToKappa" or "KappaToGlobal").
        features (dict[str, float]): A dictionary containing edge features (e.g., "weight").
    """

    def __init__(self, source: str, target: str, type: str):
        """Initializes a new instance of KappaEdge.

        Arguments:
            source: The ID of the source node.
            target: The ID of the target node.
            type: The type of the edge.

        Returns:
            None

        Raises:
            None
        """
        self.source = source
        self.target = target
        self.type = type
        self.features: dict[str, float] = {"weight": 0.0}


class AugmentedFunctionGraph(nx.DiGraph):
    """NetworkX DiGraph wrapper that supports operations required by the LoadAugmentedFunctionGraph algorithm.

    Attributes:
        subgraph_counter (int): Counter to generate unique node IDs when merging subgraphs.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """Initializes the AugmentedFunctionGraph with optional incoming graph data.

        Arguments:
            incoming_graph_data: Graph data to initialize the NetworkX DiGraph.
            attr: Additional graph attributes.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(incoming_graph_data, **attr)
        self.subgraph_counter = 0

    def HasGlobalNode(self) -> bool:
        """Checks if a node of type 'global' exists in the graph.

        Returns:
            bool: True if a global node is present, False otherwise.

        Raises:
            None
        """
        for node, attrs in self.nodes(data=True):
            if attrs.get("type") == "global" or node == "global":
                return True
        return False

    def GetGlobalNode(self) -> str:
        """Retrieves the ID of the global node.

        Returns:
            str: The ID of the global node.

        Raises:
            KeyError: If no global node exists in the graph.
        """
        for node, attrs in self.nodes(data=True):
            if attrs.get("type") == "global" or node == "global":
                return str(node)
        raise KeyError("Global node not found in graph.")

    def CreateVirtualGlobalNode(self, nodeType: str = "GlobalContext") -> str:
        """Creates a virtual global node in the graph and returns its ID.

        Arguments:
            nodeType: The type attribute to assign to the new node.

        Returns:
            str: The ID of the created global node (always 'global').

        Raises:
            None
        """
        global_id = "global"
        self.add_node(
            global_id,
            node_type=ExpressionGraphConverter.NODE_TYPES.get("global", 0),
            root_color=float(ROOT_COLOR_VOCAB["none"]),
            label_id=encode_label("GLOBAL"),
            label="GLOBAL",
            type="global",
            context_type=nodeType,
        )
        return global_id

    def MergeDisjointSubgraph(self, kappa_subgraph: Union[nx.DiGraph, str, dict]) -> str:
        """Merges a disjoint kappa subgraph into the main graph, avoiding ID collisions.

        Arguments:
            kappa_subgraph: The kappa subgraph to merge. Can be an nx.DiGraph, a GraphML string, or a dict.

        Returns:
            str: The shifted node ID of the root node of the merged kappa subgraph.

        Raises:
            TypeError: If the kappa_subgraph has an unsupported type.
            ValueError: If the kappa_subgraph is empty or invalid.
        """
        self.subgraph_counter += 1
        prefix = f"kappa_{self.subgraph_counter}"

        # 1. Parse/normalize the input subgraph into an nx.DiGraph
        if isinstance(kappa_subgraph, str):
            content = kappa_subgraph.replace("attr.type='String'", "attr.type='string'")
            content = content.replace('attr.type="String"', 'attr.type="string"')
            g_kappa = nx.parse_graphml(content)
        elif isinstance(kappa_subgraph, nx.DiGraph):
            g_kappa = kappa_subgraph
        elif isinstance(kappa_subgraph, dict):
            g_kappa = nx.DiGraph()
            for node in kappa_subgraph.get("nodes", []):
                g_kappa.add_node(node["id"], **node)
            for edge in kappa_subgraph.get("edges", []):
                g_kappa.add_edge(edge["source"], edge["target"], **edge)
        else:
            raise TypeError(f"Unsupported subgraph type: {type(kappa_subgraph)}")

        # Normalize nodes in g_kappa so they have all standard attributes
        normalized_g_kappa = nx.DiGraph()
        for nid, attrs in g_kappa.nodes(data=True):
            name_val = attrs.get("Name") or attrs.get("nodeKey1") or attrs.get("label") or str(nid)
            label = parse_graphml_node_name(name_val) if isinstance(name_val, str) else str(name_val)

            type_str = attrs.get("type") or _determine_node_type_from_label(label)
            val = 0.0
            has_val = 0.0
            if type_str == "constant":
                val_attr = attrs.get("value")
                if isinstance(val_attr, (int, float)):
                    val = float(val_attr)
                else:
                    val = _parse_constant_value(label)
                has_val = 1.0

            ntype_code = ExpressionGraphConverter.NODE_TYPES.get(type_str, 1)

            normalized_g_kappa.add_node(
                nid,
                node_type=ntype_code,
                root_color=float(ROOT_COLOR_VOCAB["none"]),
                label_id=encode_label(label),
                label=label,
                type=type_str,
            )

        for u, v, attrs in g_kappa.edges(data=True):
            etype = attrs.get("type") or attrs.get("etype") or "child_of"
            normalized_g_kappa.add_edge(
                u, v,
                edge_type=encode_edge_type(etype),
                etype=etype
            )

        # 2. Identify the root node in the normalized subgraph
        roots = [n for n, d in normalized_g_kappa.in_degree() if d == 0]
        if roots:
            original_root = roots[0]
        else:
            original_root = list(normalized_g_kappa.nodes)[0] if normalized_g_kappa.nodes else None

        if original_root is None:
            raise ValueError("Cannot merge an empty kappa subgraph.")

        # 3. Add nodes and edges to self, shifting the node IDs
        shifted_root_id = f"{prefix}_{original_root}"

        for nid, attrs in normalized_g_kappa.nodes(data=True):
            shifted_id = f"{prefix}_{nid}"
            node_attrs = dict(attrs)
            if nid == original_root:
                node_attrs["node_type"] = ExpressionGraphConverter.NODE_TYPES["root"]
                node_attrs["root_color"] = float(ROOT_COLOR_VOCAB["kappa"])
                node_attrs["type"] = "root"
            self.add_node(shifted_id, **node_attrs)

        for u, v, attrs in normalized_g_kappa.edges(data=True):
            self.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **attrs)

        return shifted_root_id

    def MergePrenormalizedSubgraph(self, original_root: str, normalized: nx.DiGraph) -> str:
        """Fast merge path for pre-normalized kappa subgraphs.

        Skips GraphML parsing and node normalization — only does the cheap
        copy+prefix step.  Call this with results from _load_normalized_kappas()
        instead of MergeDisjointSubgraph() when the kappa graphs are static.

        Arguments:
            original_root: The root node ID in *normalized* (no prefix yet).
            normalized: A pre-normalized nx.DiGraph whose root node already has
                node_type/root_color/type set to kappa-root values.

        Returns:
            str: The prefixed root node ID in the merged graph.
        """
        self.subgraph_counter += 1
        prefix = f"kappa_{self.subgraph_counter}"
        for nid, attrs in normalized.nodes(data=True):
            self.add_node(f"{prefix}_{nid}", **dict(attrs))
        for u, v, attrs in normalized.edges(data=True):
            self.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **attrs)
        return f"{prefix}_{original_root}"

    def AddEdge(self, edge: KappaEdge) -> None:
        """Adds a KappaEdge connection between the global node and a kappa root node.

        Arguments:
            edge: The KappaEdge to add to the graph.

        Returns:
            None

        Raises:
            None
        """
        weight = edge.features.get("weight", 0.0)
        edge_type_code = encode_edge_type(edge.type)
        direction_val = 0.0 if edge.type == "GlobalToKappa" else 1.0

        self.add_edge(
            edge.source,
            edge.target,
            edge_type=edge_type_code,
            etype=edge.type,
            kappa_weight=weight,
            child_index=0.0,
            direction=direction_val,
            relation_type=float(edge_type_code),
            edge_betweenness_centrality=0.0,
        )


def LoadGraphFromLocalStructure(folder: Union[Path, str], id: str) -> AugmentedFunctionGraph:
    """Loads a mathematical basis graph by ID from local graphs folder and returns it as an AugmentedFunctionGraph.

    Arguments:
        folder: The folder (directory or file) containing the graph data.
        id: The unique ID of the graph to load.

    Returns:
        AugmentedFunctionGraph containing the loaded mathematical graph.

    Raises:
        FileNotFoundError: If the folder does not exist.
        KeyError: If the graph ID is not found.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder or file not found: {folder}")

    raw_data = None
    try:
        if folder_path.is_file():
            with open(folder_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("id") == id:
                        raw_data = item
                        break
            elif isinstance(data, dict):
                if data.get("id") == id:
                    raw_data = data
                elif id in data:
                    raw_data = data[id]
        else:
            direct_files = [
                folder_path / f"{id}.json",
                folder_path / f"{id}_meta.json"
            ]
            for df in direct_files:
                if df.exists() and df.is_file():
                    with open(df, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    break

            if raw_data is None:
                for filepath in folder_path.glob("**/*.json"):
                    if ".pt_cache" in filepath.parts:
                        continue
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and item.get("id") == id:
                                    raw_data = item
                                    break
                        elif isinstance(data, dict):
                            if data.get("id") == id:
                                raw_data = data
                                break
                            elif id in data:
                                raw_data = data[id]
                                break
                    except Exception:
                        continue
                    if raw_data is not None:
                        break
    except Exception as e:
        logger.error(f"Error reading local graph structure for ID {id}: {e}")
        raise

    if raw_data is None:
        raise KeyError(f"Graph ID '{id}' not found in folder '{folder}'")

    if "graphml_f" in raw_data:
        g_f = raw_data.get("graphml_f", "")
        g_d1 = raw_data.get("graphml_derivative1", "")
        g_d2 = raw_data.get("graphml_derivative2", "")
        nx_graph = create_virtual_global_node(g_f, g_d1, g_d2)
    else:
        nx_graph = nx.DiGraph()
        nodes = raw_data.get("nodes", [])
        edges = raw_data.get("edges", [])

        _mark_function_roots(raw_data)

        for node in nodes:
            orig_type = node.get("type", "operator")
            ntype_code = ExpressionGraphConverter.NODE_TYPES.get(orig_type, 1)
            nx_graph.add_node(
                node["id"],
                node_type=ntype_code,
                root_color=float(node.get("root_color", ROOT_COLOR_VOCAB["none"])),
                label_id=encode_label(node["label"]),
                type=orig_type,
                label=node["label"],
            )
        for edge in edges:
            nx_graph.add_edge(
                edge["source"],
                edge["target"],
                edge_type=encode_edge_type(edge["type"]),
                etype=edge["type"]
            )

    return AugmentedFunctionGraph(nx_graph)


def LoadAugmentedFunctionGraph(
    graphId: str,
    graphsFolder: Union[str, Path],
    kappasFolder: Union[str, Path],
    kappa_value: Union[float, None] = None,
) -> AugmentedFunctionGraph:
    """Enhances a main function graph by merging kappa h-function subgraphs.

    When *kappa_value* is given, only the matching kappa subgraph is merged
    (O(1) hash-table lookup).  When it is None all subgraphs are merged for
    backward compatibility.

    Arguments:
        graphId: The unique ID of the main function graph to load.
        graphsFolder: Path to the folder containing mathematical basis graphs.
        kappasFolder: Path to the folder containing kappa h-functions.
        kappa_value: If provided, merge only this kappa; otherwise merge all.

    Returns:
        An AugmentedFunctionGraph that is compatible with PyTorch-Geometric/NetworkX.

    Raises:
        FileNotFoundError: If the folders do not exist.
        KeyError: If the graphId is not found.
    """
    mainGraph = LoadGraphFromLocalStructure(folder=graphsFolder, id=graphId)

    if not mainGraph.HasGlobalNode():
        globalNode = mainGraph.CreateVirtualGlobalNode(nodeType="GlobalContext")
    else:
        globalNode = mainGraph.GetGlobalNode()

    kappas_path = Path(kappasFolder)
    if not kappas_path.exists():
        raise FileNotFoundError(f"Kappas folder not found: {kappasFolder}")

    # Kappa subgraphs are static — parse and normalize once, reuse via O(1) dict lookup.
    kappa_lookup = _load_normalized_kappas(kappas_path)

    if kappa_value is None:
        logger.debug("LoadAugmentedFunctionGraph called without kappa_value — no kappa merged.")
        return mainGraph

    kv = float(kappa_value)
    entry = kappa_lookup.get(kv)
    if entry is None:
        # kappa == 0 intentionally has no h-function.  Any other unknown value is a
        # misconfiguration worth flagging, but the graph is still returned unchanged.
        if kv != 0.0:
            logger.warning(
                f"Kappa value {kv} not found for graph '{graphId}'; "
                f"available: {sorted(kappa_lookup)}. No kappa merged."
            )
    else:
        original_root, normalized = entry
        kappa_root_id = mainGraph.MergePrenormalizedSubgraph(original_root, normalized)
        _tag_and_connect_kappa(mainGraph, globalNode, kappa_root_id, kv)

    return mainGraph


def filter_active_kappa(
    data: Union[Data, HeteroData], active_kappa: Union[float, int, None]
) -> Union[Data, HeteroData]:
    """Filters the PyG Data object to only keep nodes and edges of the active kappa subgraph.

    All base graph nodes, global nodes, and aggregator nodes are kept. Inactive kappa subgraph nodes
    and their associated edges are removed. If active_kappa is None, 0, or NaN, all kappa subgraphs
    are deactivated and removed.

    Arguments:
        data: The PyG Data or HeteroData object containing node_kappas.
        active_kappa: The kappa value to keep active.

    Returns:
        The filtered PyG Data or HeteroData object.

    Raises:
        None
    """
    if not isinstance(data, Data) or not hasattr(data, "node_kappas") or data.node_kappas is None:
        return data

    node_kappas = data.node_kappas
    num_nodes = data.x.size(0) if data.x is not None else len(node_kappas)

    # 1. Determine active_kappa validity
    is_active_kappa_valid = False
    if active_kappa is not None:
        try:
            act_k = float(active_kappa)
            if not math.isnan(act_k) and act_k != 0.0:
                is_active_kappa_valid = True
        except (ValueError, TypeError):
            pass

    # 2. Identify nodes to keep
    keep_node_indices = []
    for i in range(num_nodes):
        kappa_val = node_kappas[i]
        if kappa_val is None:
            # Base node / global node / aggregator node
            keep_node_indices.append(i)
        elif is_active_kappa_valid:
            try:
                if abs(float(kappa_val) - float(active_kappa)) < 1e-3:
                    keep_node_indices.append(i)
            except (ValueError, TypeError):
                pass

    # 3. Create node mask
    keep_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    keep_node_mask[keep_node_indices] = True

    # If all nodes are kept, return unmodified
    if keep_node_mask.all():
        return data

    # 4. Filter node-level attributes
    if hasattr(data, "node_ids") and data.node_ids is not None:
        data.node_ids = [data.node_ids[i] for i in keep_node_indices]

    data.node_kappas = [node_kappas[i] for i in keep_node_indices]

    # Map old node indices to new indices
    map_tensor = torch.empty(num_nodes, dtype=torch.long)
    map_tensor[keep_node_mask] = torch.arange(len(keep_node_indices))

    # 5. Filter edge-level attributes
    num_edges = 0
    keep_edge_mask = None
    if hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
        src, dst = data.edge_index[0], data.edge_index[1]
        keep_edge_mask = keep_node_mask[src] & keep_node_mask[dst]
        data.edge_index = map_tensor[data.edge_index[:, keep_edge_mask]]
        num_edges = keep_edge_mask.size(0)

    # 6. Apply masks to all tensors in the data object
    for key, value in list(data.items()):
        if isinstance(value, torch.Tensor):
            if key == "edge_index":
                continue
            # Check known key lists first to prevent size collision
            if key in ("x", "node_type", "label_id", "belongs_to_f", "belongs_to_d1", "belongs_to_d2"):
                data[key] = value[keep_node_mask]
            elif key in ("edge_attr", "edge_type"):
                if keep_edge_mask is not None:
                    data[key] = value[keep_edge_mask]
            elif value.dim() > 0 and value.size(0) == num_nodes:
                data[key] = value[keep_node_mask]
            elif value.dim() > 0 and value.size(0) == num_edges:
                if keep_edge_mask is not None:
                    data[key] = value[keep_edge_mask]

    # Update counts if present
    if hasattr(data, "nodes"):
        data.nodes = len(keep_node_indices)
    if hasattr(data, "num_nodes"):
        data.num_nodes = len(keep_node_indices)
    if hasattr(data, "edges") and keep_edge_mask is not None:
        data.edges = int(keep_edge_mask.sum().item())
    if hasattr(data, "num_edges") and keep_edge_mask is not None:
        data.num_edges = int(keep_edge_mask.sum().item())

    return data
