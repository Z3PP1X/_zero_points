import json
import pytest
import networkx as nx

from graph_utils import (
    LoadGraphFromLocalStructure,
    LoadAugmentedFunctionGraph,
    AugmentedFunctionGraph,
)


@pytest.fixture
def sample_main_graph_dict():
    """Returns a mock mathematical basis graph dictionary in the older format."""
    return {
        "id": "P-mock-1",
        "nodes": [
            {"id": "n1", "label": "Plus", "type": "operator", "value": None},
            {"id": "n2", "label": "x", "type": "variable", "value": None},
            {"id": "n3", "label": "5.0", "type": "constant", "value": 5.0}
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "child_of"},
            {"source": "n1", "target": "n3", "type": "child_of"}
        ]
    }


@pytest.fixture
def sample_kappa_dict():
    """Returns a mock kappa container dictionary."""
    return {
        "id": "kappa",
        "value": "-15.5",
        "graphStructure": {
            "nodes": [
                {"id": "k_root", "label": "Log", "type": "function", "value": None},
                {"id": "k_var", "label": "x", "type": "variable", "value": None}
            ],
            "edges": [
                {"source": "k_root", "target": "k_var", "type": "child_of"}
            ]
        }
    }


@pytest.fixture
def sample_kappa_graphml():
    """Returns a mock kappa container dictionary using the GraphML string format."""
    graphml_str = """<?xml version='1.0' encoding='UTF-8'?>
<graphml>
 <key id='nodeKey1' for='node' attr.name='Name' attr.type='String' />
 <graph id='Graph1' edgedefault='directed'>
  <node id='1'>
   <data key='nodeKey1'>Log</data>
  </node>
  <node id='2'>
   <data key='nodeKey1'>x</data>
  </node>
  <edge id='e1' source='1' target='2' />
 </graph>
</graphml>"""
    return {
        "id": "kappa",
        "value": "-25.0",
        "graphml_h": graphml_str
    }


def test_augmented_function_graph_global_node():
    """Tests the global node operations in AugmentedFunctionGraph."""
    G = nx.DiGraph()
    aug = AugmentedFunctionGraph(G)

    # Check on empty graph
    assert aug.HasGlobalNode() is False
    with pytest.raises(KeyError):
        aug.GetGlobalNode()

    # Create virtual global node
    global_id = aug.CreateVirtualGlobalNode(nodeType="TestGlobalContext")
    assert global_id == "global"
    assert aug.HasGlobalNode() is True
    assert aug.GetGlobalNode() == "global"

    # Check attributes of created global node
    attrs = aug.nodes["global"]
    assert attrs["type"] == "global"
    assert attrs["label"] == "GLOBAL"
    assert attrs["context_type"] == "TestGlobalContext"


def test_augmented_function_graph_merge_disjoint(sample_kappa_dict):
    """Tests that merging a kappa subgraph shifts IDs correctly and normalizes nodes."""
    aug = AugmentedFunctionGraph()
    aug.CreateVirtualGlobalNode()

    # Merge first kappa
    root_id_1 = aug.MergeDisjointSubgraph(sample_kappa_dict["graphStructure"])
    assert root_id_1 == "kappa_1_k_root"
    assert "kappa_1_k_root" in aug.nodes
    assert "kappa_1_k_var" in aug.nodes

    # Merge second kappa (collision check)
    root_id_2 = aug.MergeDisjointSubgraph(sample_kappa_dict["graphStructure"])
    assert root_id_2 == "kappa_2_k_root"
    assert "kappa_2_k_root" in aug.nodes
    assert "kappa_2_k_var" in aug.nodes

    # Check that attributes on merged nodes are normalized
    root_attrs = aug.nodes["kappa_1_k_root"]
    # Kappa root is marked as "root" type with root_color=4 (kappa)
    assert root_attrs["type"] == "function"
    assert root_attrs["label"] == "Log"
    assert root_attrs["root_color"] == 4.0   # ROOT_COLOR_VOCAB["kappa"] = 4

    # Check that edges were correctly added with shifted IDs
    assert aug.has_edge("kappa_1_k_root", "kappa_1_k_var")
    assert aug.has_edge("kappa_2_k_root", "kappa_2_k_var")


def test_augmented_function_graph_merge_graphml(sample_kappa_graphml):
    """Tests that merging a kappa subgraph in GraphML format parses/normalizes."""
    aug = AugmentedFunctionGraph()
    aug.CreateVirtualGlobalNode()

    root_id = aug.MergeDisjointSubgraph(sample_kappa_graphml["graphml_h"])
    assert root_id == "kappa_1_1"
    assert aug.has_node("kappa_1_1")
    assert aug.has_node("kappa_1_2")
    assert aug.has_edge("kappa_1_1", "kappa_1_2")

    root_attrs = aug.nodes["kappa_1_1"]
    assert root_attrs["type"] == "function"
    assert root_attrs["label"] == "Log"


def test_load_graph_from_local_structure(tmp_path, sample_main_graph_dict):
    """Tests loading a main graph from local directory using JSON formats."""
    # Write sample json to tmp path
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    # Single file list format
    list_file = graphs_dir / "graphs_list.json"
    list_file.write_text(json.dumps([sample_main_graph_dict]), encoding="utf-8")

    g_loaded = LoadGraphFromLocalStructure(list_file, "P-mock-1")
    assert isinstance(g_loaded, AugmentedFunctionGraph)
    assert g_loaded.number_of_nodes() == 3
    assert g_loaded.has_node("n1")

    # Directory scanning format
    direct_file = graphs_dir / "P-mock-1.json"
    direct_file.write_text(json.dumps(sample_main_graph_dict), encoding="utf-8")

    g_dir_loaded = LoadGraphFromLocalStructure(graphs_dir, "P-mock-1")
    assert g_dir_loaded.number_of_nodes() == 3


def test_load_augmented_function_graph(
    tmp_path, sample_main_graph_dict, sample_kappa_dict, sample_kappa_graphml
):
    """Only the requested kappa is merged; no kappa_value → base graph returned unchanged."""
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    kappas_dir = tmp_path / "kappas"
    kappas_dir.mkdir()

    (graphs_dir / "P-mock-1.json").write_text(json.dumps(sample_main_graph_dict), encoding="utf-8")
    (kappas_dir / "kappa_1.json").write_text(json.dumps(sample_kappa_dict), encoding="utf-8")
    (kappas_dir / "kappa_2.json").write_text(json.dumps(sample_kappa_graphml), encoding="utf-8")

    # ── kappa_value=None → no kappa merged ──────────────────────────────────
    base = LoadAugmentedFunctionGraph(
        graphId="P-mock-1", graphsFolder=graphs_dir, kappasFolder=kappas_dir
    )
    assert base.has_node("n1") and base.has_node("n2") and base.has_node("n3")
    assert not any(n.startswith("kappa_") for n in base.nodes)

    # ── kappa_value=-15.5 → only kappa_1 merged ─────────────────────────────
    g1 = LoadAugmentedFunctionGraph(
        graphId="P-mock-1", graphsFolder=graphs_dir, kappasFolder=kappas_dir,
        kappa_value=-15.5,
    )
    assert g1.HasGlobalNode()
    globalNode = g1.GetGlobalNode()
    assert g1.has_node("kappa_1_k_root")
    assert not any(n.startswith("kappa_2_") for n in g1.nodes)

    edge_fwd = g1.edges[globalNode, "kappa_1_k_root"]
    assert edge_fwd["kappa_weight"] == -15.5
    assert edge_fwd["etype"] == "GlobalToKappa"
    assert edge_fwd["edge_type"] == 0

    edge_bwd = g1.edges["kappa_1_k_root", globalNode]
    assert edge_bwd["kappa_weight"] == -15.5
    assert edge_bwd["etype"] == "KappaToGlobal"
    assert edge_bwd["edge_type"] == 0

    # ── kappa_value=-25.0 → only kappa_2 merged ─────────────────────────────
    # Counter resets per call, so the merged node gets prefix "kappa_1_" regardless.
    g2 = LoadAugmentedFunctionGraph(
        graphId="P-mock-1", graphsFolder=graphs_dir, kappasFolder=kappas_dir,
        kappa_value=-25.0,
    )
    globalNode2 = g2.GetGlobalNode()
    kappa_nodes_g2 = [n for n in g2.nodes if str(n).startswith("kappa_")]
    assert len(kappa_nodes_g2) > 0
    # All kappa edges must carry weight -25.0
    for n in kappa_nodes_g2:
        if g2.has_edge(globalNode2, n):
            assert g2.edges[globalNode2, n]["kappa_weight"] == -25.0
        if g2.has_edge(n, globalNode2):
            assert g2.edges[n, globalNode2]["kappa_weight"] == -25.0


def test_filter_active_kappa_nodes_edges(
    tmp_path, sample_main_graph_dict, sample_kappa_dict, sample_kappa_graphml
):
    """filter_active_kappa strips all-but-active kappa nodes from a multi-kappa PyG graph."""
    from kappa_loader import _load_normalized_kappas, _tag_and_connect_kappa

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    kappas_dir = tmp_path / "kappas"
    kappas_dir.mkdir()

    (graphs_dir / "P-mock-1.json").write_text(json.dumps(sample_main_graph_dict), encoding="utf-8")
    (kappas_dir / "kappa_1.json").write_text(json.dumps(sample_kappa_dict), encoding="utf-8")
    (kappas_dir / "kappa_2.json").write_text(json.dumps(sample_kappa_graphml), encoding="utf-8")

    # Build a multi-kappa graph manually (merge both kappas) to exercise filter_active_kappa.
    mainGraph = LoadGraphFromLocalStructure(folder=graphs_dir, id="P-mock-1")
    if not mainGraph.HasGlobalNode():
        mainGraph.CreateVirtualGlobalNode(nodeType="GlobalContext")
    globalNode = mainGraph.GetGlobalNode()

    kappa_lookup = _load_normalized_kappas(kappas_dir)
    for kv, (orig_root, normalized) in kappa_lookup.items():
        kappa_root_id = mainGraph.MergePrenormalizedSubgraph(orig_root, normalized)
        _tag_and_connect_kappa(mainGraph, globalNode, kappa_root_id, kv)

    # Convert to PyG homogeneous Data
    from graph_utils import ExpressionGraphConverter, filter_active_kappa
    converter = ExpressionGraphConverter()
    data = converter.convert(
        mainGraph,
        mode="tree_derivatives",
        edge_direction="top_down",
    )

    # Initial checks:
    # 3 math nodes + 1 global + 0 aggregators + 2 kappa_1 nodes + 2 kappa_2 nodes = 8 nodes total
    assert len(data.node_ids) == 8
    assert data.x.size(0) == 8
    assert len(data.node_kappas) == 8
    
    # We have two kappas: -15.5 and -25.0
    # Let's filter to activate -15.5
    data_15 = filter_active_kappa(data.clone(), -15.5)
    # Filtered graph should keep 3 math + 1 global + 2 nodes from kappa_1 = 6 nodes
    assert len(data_15.node_ids) == 6
    assert data_15.x.size(0) == 6
    assert len(data_15.node_kappas) == 6
    assert all(k is None or abs(k - (-15.5)) < 1e-3 for k in data_15.node_kappas)
    
    # Nodes in data_15 should not contain kappa_2 nodes
    for nid in data_15.node_ids:
        assert not nid.startswith("kappa_2_")

    # Let's filter to activate -25.0
    data_25 = filter_active_kappa(data.clone(), -25.0)
    assert len(data_25.node_ids) == 6
    assert data_25.x.size(0) == 6
    assert len(data_25.node_kappas) == 6
    for nid in data_25.node_ids:
        assert not nid.startswith("kappa_1_")

    # Let's filter to activate None / 0.0 (deactivate all)
    data_none = filter_active_kappa(data.clone(), None)
    assert len(data_none.node_ids) == 4
    assert data_none.x.size(0) == 4
    for nid in data_none.node_ids:
        assert not nid.startswith("kappa_")

