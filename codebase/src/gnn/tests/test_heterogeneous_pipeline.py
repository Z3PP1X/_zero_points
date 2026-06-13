import pytest
import torch
import math
from torch_geometric.data import HeteroData
from graph_utils import ExpressionGraphConverter

GRAPHML_F = """<?xml version='1.0' encoding='UTF-8'?>
<graphml>
 <key id='nodeKey1' for='node' attr.name='Name' attr.type='String' />
 <graph id='Graph1' edgedefault='directed'>
  <node id='1'>
   <data key='nodeKey1'>{Plus, {}}</data>
  </node>
  <node id='2'>
   <data key='nodeKey1'>{x, {1}}</data>
  </node>
  <node id='3'>
   <data key='nodeKey1'>{5, {2}}</data>
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


def test_heterogeneous_converter_shapes_and_types():
    raw_container = {
        "id": "P-hetero-test",
        "x0": 0.0,
        "yRange": [-10.0, 8.0],
        "graphml_f": GRAPHML_F,
        "graphml_derivative1": GRAPHML_D1,
        "graphml_derivative2": GRAPHML_D2
    }

    converter = ExpressionGraphConverter()
    
    # Convert using heterogeneous=True
    data = converter.convert(raw_container, heterogeneous=True, mode="graph")
    
    # 1. Integrity Check
    assert isinstance(data, HeteroData)
    # New node types: global (1), root (f_1, d1_1, d2_1 = 3), operator (f_2=x, f_3=5 = 2)
    assert set(data.node_types) == {"global", "operator", "root"}

    # 2. Shape & Dimension Verification — all types share the same 16-feature schema
    n_feats = 16  # len(NODE_FEATURE_SCHEMA)
    assert data["global"].x.shape == (1, n_feats)
    assert data["root"].x.shape == (3, n_feats)    # f_1, d1_1, d2_1
    assert data["operator"].x.shape == (2, n_feats)  # f_2 (x), f_3 (5)

    # Root nodes have root_color > 0 (encoded at feature index 1)
    root_colors = data["root"].x[:, 1].tolist()
    assert all(c > 0 for c in root_colors), "All root nodes must have non-zero root_color"


def test_heterogeneous_local_index_boundaries():
    raw_container = {
        "id": "P-hetero-test",
        "x0": 0.0,
        "yRange": [-10.0, 8.0],
        "graphml_f": GRAPHML_F,
        "graphml_derivative1": GRAPHML_D1,
        "graphml_derivative2": GRAPHML_D2
    }

    converter = ExpressionGraphConverter()
    data = converter.convert(raw_container, heterogeneous=True, mode="graph")
    
    # Verify index boundaries for all edge index tables
    for triplet in data.edge_types:
        src_type, rel_type, dst_type = triplet
        edge_index = data[triplet].edge_index
        num_src = data[src_type].num_nodes
        num_dst = data[dst_type].num_nodes
        
        if edge_index.numel() > 0:
            assert torch.all(edge_index[0] >= 0)
            assert torch.all(edge_index[0] < num_src)
            assert torch.all(edge_index[1] >= 0)
            assert torch.all(edge_index[1] < num_dst)



def test_regression_homogeneous_mode():
    raw_container = {
        "id": "P-hetero-test",
        "x0": 0.0,
        "yRange": [-10.0, 8.0],
        "graphml_f": GRAPHML_F,
        "graphml_derivative1": GRAPHML_D1,
        "graphml_derivative2": GRAPHML_D2
    }

    converter = ExpressionGraphConverter()
    data = converter.convert(raw_container, heterogeneous=False, mode="graph")

    # 6 nodes: global + f_1/f_2/f_3 + d1_1 + d2_1 (no aggregator nodes)
    assert data.num_nodes == 6
    assert not isinstance(data, HeteroData)
    assert data.x.shape[1] == 16  # 16 node features (new position-aware schema)
    assert hasattr(data, "node_ids")
