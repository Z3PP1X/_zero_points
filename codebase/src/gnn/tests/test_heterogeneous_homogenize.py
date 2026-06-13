"""heterogeneous:true graphs are homogenized via PyG's to_homogeneous before collation.

Reproduces and fixes the IndexError from InMemoryDataset.collate/separate over HeteroData
with type-dependent stores, and confirms the homogenized graph feeds the (homogeneous)
expression_classifier backbone. Package-style imports throughout (the repo's dual import
hazard).
"""

import torch
from torch_geometric.data import Data, HeteroData

from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    NODE_FEATURE_SCHEMA,
    EDGE_FEATURE_SCHEMA,
)
from gnn.shared.models.classifiers import TestGraphNetwork
from gnn.supervised_learning.loader_graphgym import (
    ExpressionGraphDataset,
    homogenize_for_classifier,
)


def _raw_small():
    return {
        "id": "P-small",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Plus", "type": "operator", "value": None},
            {"id": "f2", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [
            {"source": "global", "target": "f1", "type": "belongs_to_f"},
            {"source": "f1", "target": "f2", "type": "child_of"},
        ],
    }


def _raw_large():
    return {
        "id": "P-large",
        "nodes": [
            {"id": "global", "label": "GLOBAL", "type": "global", "value": None},
            {"id": "f1", "label": "Times", "type": "operator", "value": None},
            {"id": "f2", "label": "Sin", "type": "function", "value": None},
            {"id": "f3", "label": "x", "type": "variable", "value": None},
            {"id": "f4", "label": "Cos", "type": "function", "value": None},
            {"id": "f5", "label": "x", "type": "variable", "value": None},
        ],
        "edges": [
            {"source": "global", "target": "f1", "type": "belongs_to_f"},
            {"source": "f1", "target": "f2", "type": "child_of"},
            {"source": "f2", "target": "f3", "type": "child_of"},
            {"source": "f1", "target": "f4", "type": "child_of"},
            {"source": "f4", "target": "f5", "type": "child_of"},
        ],
    }


def _hetero(raw, label):
    data = ExpressionGraphConverter().convert(raw, heterogeneous=True, mode="graph")
    if hasattr(data, "laplacian"):
        del data.laplacian
    data.y = torch.tensor([label], dtype=torch.long)
    data.global_features = torch.zeros((1, 2), dtype=torch.float)
    return data


def test_homogenize_produces_classifier_ready_data():
    hetero = _hetero(_raw_large(), 1)
    assert isinstance(hetero, HeteroData)

    homo = homogenize_for_classifier(hetero)
    assert isinstance(homo, Data)
    # Full node schema preserved as columns of x (node_type stays a column).
    assert homo.x.size(1) == len(NODE_FEATURE_SCHEMA)
    # Graph label survives the conversion.
    assert homo.y.item() == 1
    # 2-D hetero edge_attr is dropped; the backbone zero-fills to EDGE_FEATURE_SCHEMA dim.
    assert getattr(homo, "edge_attr", None) is None
    # PyG records the type indices separately; they must not be inside x.
    assert hasattr(homo, "node_type")


def test_homogeneous_data_passes_through_unchanged():
    plain = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]]))
    assert homogenize_for_classifier(plain) is plain


def test_homogenized_graph_runs_through_classifier():
    homo = homogenize_for_classifier(_hetero(_raw_large(), 0))
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader([homo], batch_size=1)))
    model = TestGraphNetwork(
        input_dim=len(NODE_FEATURE_SCHEMA),
        hidden_dim=16,
        edge_dim=len(EDGE_FEATURE_SCHEMA),
        architecture="gine_stack",
        variant="legacy",
    )
    model.eval()
    with torch.no_grad():
        out = model(
            batch.x, batch.edge_index, batch.batch, batch.global_features,
            edge_attr=getattr(batch, "edge_attr", None),
        )
    assert out.shape == (1, 2)


def test_dataset_collates_structurally_varied_hetero_graphs():
    """The original crash: HeteroData of differing structure broke collate/separate.

    ExpressionGraphDataset now homogenizes internally, so a mixed-size batch builds and
    every item is indexable without an IndexError.
    """
    data_list = [
        _hetero(_raw_small(), 0),
        _hetero(_raw_large(), 1),
        _hetero(_raw_small(), 1),
    ]
    ds = ExpressionGraphDataset(
        data_list, train_idx=[0], val_idx=[1], test_idx=[2]
    )
    assert len(ds) == 3
    for i in range(len(ds)):
        item = ds[i]  # would raise IndexError before the fix
        assert item.x.size(1) == len(NODE_FEATURE_SCHEMA)
