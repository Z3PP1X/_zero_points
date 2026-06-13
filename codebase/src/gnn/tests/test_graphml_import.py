import pytest
import networkx as nx
import torch
from graph_utils import (
    parse_graphml_node_name,
    _determine_node_type_from_label,
    _parse_constant_value,
    find_roots,
    create_virtual_global_node,
    ExpressionGraphConverter,
)

GRAPHML_F = """<?xml version='1.0' encoding='UTF-8'?>
<graphml>
 <key id='nodeKey1' for='node' attr.name='Name' attr.type='String' />
 <key id='nodeKey2' for='node' attr.name='VertexCoordinates' attr.type='String' />
 <graph id='Graph1' edgedefault='directed'>
  <node id='1'>
   <data key='nodeKey1'>{Plus, {}}</data>
   <data key='nodeKey2'>List[0.0, 0.0]</data>
  </node>
  <node id='2'>
   <data key='nodeKey1'>{x, {1}}</data>
   <data key='nodeKey2'>List[0.0, 0.0]</data>
  </node>
  <node id='3'>
   <data key='nodeKey1'>{5, {2}}</data>
   <data key='nodeKey2'>List[0.0, 0.0]</data>
  </node>
  <edge id='e1' source='1' target='2' />
  <edge id='e2' source='1' target='3' />
 </graph>
</graphml>"""

GRAPHML_D1 = """<?xml version='1.0' encoding='UTF-8'?>
<graphml>
 <key id='nodeKey1' for='node' attr.name='Name' attr.type='String' />
 <graph id='Graph1' edgedefault='directed'>
  <node id='1'>
   <data key='nodeKey1'>{1, {}}</data>
  </node>
 </graph>
</graphml>"""

GRAPHML_D2 = """<?xml version='1.0' encoding='UTF-8'?>
<graphml>
 <key id='nodeKey1' for='node' attr.name='Name' attr.type='String' />
 <graph id='Graph1' edgedefault='directed'>
  <node id='1'>
   <data key='nodeKey1'>{0, {}}</data>
  </node>
 </graph>
</graphml>"""


def test_helper_functions():
    # 1. Test parse_graphml_node_name
    assert parse_graphml_node_name("{Plus, {}}") == "Plus"
    assert parse_graphml_node_name("{5, {1}}") == "5"
    assert parse_graphml_node_name("{-1/5, {4, 1}}") == "-1/5"
    assert parse_graphml_node_name("x") == "x"

    # 2. Test _determine_node_type_from_label
    assert _determine_node_type_from_label("Plus") == "operator"
    assert _determine_node_type_from_label("x") == "variable"
    assert _determine_node_type_from_label("E") == "variable"
    assert _determine_node_type_from_label("Cos") == "function"
    assert _determine_node_type_from_label("5") == "constant"
    assert _determine_node_type_from_label("-1/5") == "constant"

    # 3. Test _parse_constant_value
    assert _parse_constant_value("5") == 5.0
    assert _parse_constant_value("1/5") == 0.2
    assert _parse_constant_value("-2/3") == pytest.approx(-0.666666666)

    # 4. Test find_roots
    nodes = [{"id": "f_1"}, {"id": "f_2"}]
    edges = [{"source": "f_1", "target": "f_2"}]
    assert find_roots(nodes, edges) == ["f_1"]


def test_create_virtual_global_node():
    # Test creation using string input
    G_comb = create_virtual_global_node(GRAPHML_F, GRAPHML_D1, GRAPHML_D2)

    assert "global" in G_comb.nodes
    assert "f_1" in G_comb.nodes
    assert "d1_1" in G_comb.nodes
    assert "d2_1" in G_comb.nodes

    # Check node attributes
    assert G_comb.nodes["global"]["type"] == "global"
    assert G_comb.nodes["f_1"]["type"] == "operator"
    assert G_comb.nodes["f_2"]["type"] == "variable"
    assert G_comb.nodes["f_3"]["type"] == "constant"
    assert G_comb.nodes["f_3"]["value"] == pytest.approx(5.0)

    assert "f_root" in G_comb.nodes
    assert "d1_root" in G_comb.nodes
    assert "d2_root" in G_comb.nodes
    assert G_comb.has_edge("global", "f_root")
    assert G_comb.has_edge("f_root", "f_1")
    assert G_comb.has_edge("global", "d1_root")
    assert G_comb.has_edge("d1_root", "d1_1")
    assert G_comb.has_edge("global", "d2_root")
    assert G_comb.has_edge("d2_root", "d2_1")


def test_expression_graph_converter_with_container_format():
    raw_container = {
        "id": "P-container-test",
        "x0": 0.0,
        "yRange": [-10.0, 8.0],
        "graphml_f": GRAPHML_F,
        "graphml_derivative1": GRAPHML_D1,
        "graphml_derivative2": GRAPHML_D2
    }

    converter = ExpressionGraphConverter()
    
    # 1. Test "graph" mode (compiles f, d1, d2 + aggregators, no task virtual nodes)
    # Nodes: 3 + 1 + 1 + 1 + 3 aggregators = 9
    data_graph = converter.convert(raw_container, heterogeneous=False, mode="graph")
    assert data_graph.num_nodes == 9
    assert "global" in data_graph.node_ids
    assert "f_1" in data_graph.node_ids
    assert "d1_1" in data_graph.node_ids
    assert "d2_1" in data_graph.node_ids

    # 2. Test "tree_derivatives" mode (compiles f, d1, d2 + aggregators, no task virtual nodes)
    data_tree_deriv = converter.convert(raw_container, heterogeneous=False, mode="tree_derivatives")
    assert data_tree_deriv.num_nodes == 9
    assert "global" in data_tree_deriv.node_ids
    assert "f_1" in data_tree_deriv.node_ids
    assert "d1_1" in data_tree_deriv.node_ids
    assert "d2_1" in data_tree_deriv.node_ids
    assert "virtual_current_x" not in data_tree_deriv.node_ids

    # 3. Test "tree" mode (compiles only f + f_root aggregator, without task virtual nodes)
    data_tree = converter.convert(raw_container, heterogeneous=False, mode="tree")
    assert data_tree.num_nodes == 5
    assert "global" in data_tree.node_ids
    assert "f_1" in data_tree.node_ids
    assert "d1_1" not in data_tree.node_ids
    assert "d2_1" not in data_tree.node_ids
    assert "virtual_current_x" not in data_tree.node_ids

