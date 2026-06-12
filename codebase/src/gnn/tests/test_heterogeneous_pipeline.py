import pytest
import torch
import math
from torch_geometric.data import HeteroData
from graph_utils import (
    ExpressionGraphConverter,
    populate_task_virtual_values,
    signed_log_value,
    fourier_frequency_encoding,
)

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
    assert set(data.node_types) == {"operator", "variable", "constant", "virtual"}
    
    # 2. Shape & Dimension Verification
    # Operators: Plus (from f), 1 (d1 is constant, 0 is constant). Wait, Plus is the only operator!
    # Let's count variable: x (1)
    # Constants: 5 (from f), 1 (from d1), 0 (from d2).
    # Virtuals: global, f_root, d1_root, d2_root (4 nodes; task virtual nodes removed)
    # Operator/variable feature dim = len(CANONICAL_LABELS) one-hot + 5 topology + 8 (lpe/rwpe).
    assert data["operator"].x.shape[1] == 42
    assert data["variable"].x.shape[1] == 42
    assert data["constant"].x.shape[1] == 9
    assert data["virtual"].x.shape[1] == 7
    
    # Verify constant feature matches: signed log value + fourier frequency encoding
    c_val = signed_log_value(5.0)
    expected_fourier = torch.tensor(fourier_frequency_encoding(c_val), dtype=torch.float)
    
    # Search for constant with value c_val
    found = False
    for idx in range(data["constant"].x.shape[0]):
        val = data["constant"].x[idx, 0].item()
        if abs(val - c_val) < 1e-5:
            found = True
            # Check Sinusoidal features
            assert torch.allclose(data["constant"].x[idx, 1:], expected_fourier, atol=1e-5)
    assert found, "Constant with value 5.0 not found or mismatch in feature encoding"


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


def test_heterogeneous_state_injection():
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
    
    # Keep copy of variable, constant, and operator feature matrix
    op_x_orig = data["operator"].x.clone()
    var_x_orig = data["variable"].x.clone()
    const_x_orig = data["constant"].x.clone()
    
    # Inject values
    cx = 2.5
    fx = 4.7
    yt = 1.2
    d1x = 0.5
    d2x = 9.9
    
    populate_task_virtual_values(
        data,
        cx_val=cx,
        fx_val=fx,
        yt_val=yt,
        d1x_val=d1x,
        d2x_val=d2x,
        mode="graph",
    )
    
    # Verify that virtual node feature values are injected correctly in data['virtual'].x
    v_ids = data["virtual"].node_ids
    
    idx_f_root = v_ids.index("f_root")
    idx_d1_root = v_ids.index("d1_root")
    idx_d2_root = v_ids.index("d2_root")
    
    delta_val = yt - fx
    assert data["virtual"].x[idx_f_root, 0].item() == pytest.approx(signed_log_value(cx))
    assert data["virtual"].x[idx_f_root, 1].item() == pytest.approx(signed_log_value(delta_val))
    assert data["virtual"].x[idx_d1_root, 2].item() == pytest.approx(signed_log_value(d1x))
    assert data["virtual"].x[idx_d2_root, 3].item() == pytest.approx(signed_log_value(d2x))
    
    # Ensure static AST nodes are NOT mutated
    assert torch.allclose(data["operator"].x, op_x_orig)
    assert torch.allclose(data["variable"].x, var_x_orig)
    assert torch.allclose(data["constant"].x, const_x_orig)


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
    
    # Ensure legacy homogeneous structure is correct and contains 9 nodes (no task virtual nodes)
    assert data.num_nodes == 9
    assert not isinstance(data, HeteroData)
    assert data.x.shape[1] == 24
    assert hasattr(data, "node_ids")
