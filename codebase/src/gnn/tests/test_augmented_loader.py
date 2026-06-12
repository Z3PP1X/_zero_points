import json
import pytest
import networkx as nx

from graph_utils import (
    LoadGraphFromLocalStructure,
    LoadAugmentedFunctionGraph,
    AugmentedFunctionGraph,
    CANONICAL_EDGE_TYPE_VOCAB,
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
    assert root_attrs["type"] == "function"
    assert root_attrs["label"] == "Log"
    assert "virtual_current_x_val" in root_attrs

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
    """Tests the full orchestration of loading and merging multiple kappas."""
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    kappas_dir = tmp_path / "kappas"
    kappas_dir.mkdir()

    # Write basis graph
    (graphs_dir / "P-mock-1.json").write_text(
        json.dumps(sample_main_graph_dict), encoding="utf-8"
    )

    # Write kappa h-functions
    (kappas_dir / "kappa_1.json").write_text(
        json.dumps(sample_kappa_dict), encoding="utf-8"
    )
    (kappas_dir / "kappa_2.json").write_text(
        json.dumps(sample_kappa_graphml), encoding="utf-8"
    )

    # Load augmented graph
    mainGraph = LoadAugmentedFunctionGraph(
        graphId="P-mock-1",
        graphsFolder=graphs_dir,
        kappasFolder=kappas_dir
    )

    # Check that base nodes are loaded
    assert mainGraph.has_node("n1")
    assert mainGraph.has_node("n2")
    assert mainGraph.has_node("n3")

    # Check that a virtual global node exists (or created)
    assert mainGraph.HasGlobalNode()
    globalNode = mainGraph.GetGlobalNode()

    # Check that both kappas were merged
    assert mainGraph.has_node("kappa_1_k_root")
    assert mainGraph.has_node("kappa_2_1")

    # Check that forward and backward edges exist
    assert mainGraph.has_edge(globalNode, "kappa_1_k_root")
    assert mainGraph.has_edge("kappa_1_k_root", globalNode)
    assert mainGraph.has_edge(globalNode, "kappa_2_1")
    assert mainGraph.has_edge("kappa_2_1", globalNode)

    # Verify weights on edges
    edge_1_fwd = mainGraph.edges[globalNode, "kappa_1_k_root"]
    assert edge_1_fwd["weight"] == -15.5
    assert edge_1_fwd["etype"] == "GlobalToKappa"
    assert edge_1_fwd["edge_type"] == CANONICAL_EDGE_TYPE_VOCAB["GlobalToKappa"]

    edge_1_bwd = mainGraph.edges["kappa_1_k_root", globalNode]
    assert edge_1_bwd["weight"] == -15.5
    assert edge_1_bwd["etype"] == "KappaToGlobal"
    assert edge_1_bwd["edge_type"] == CANONICAL_EDGE_TYPE_VOCAB["KappaToGlobal"]

    edge_2_fwd = mainGraph.edges[globalNode, "kappa_2_1"]
    assert edge_2_fwd["weight"] == -25.0

    edge_2_bwd = mainGraph.edges["kappa_2_1", globalNode]
    assert edge_2_bwd["weight"] == -25.0
