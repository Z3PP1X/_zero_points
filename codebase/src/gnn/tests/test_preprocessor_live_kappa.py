"""Tests for per-step live-kappa graph augmentation in the RL preprocessor, plus a
verification that RL already exposes the full shared node-feature schema (deliverable B).

The Mathematica state carries an integer ``kappa`` (the optimal h-function index at
the current iterate x) that changes every step. ``Preprocessor`` clamps it to the
kappa-library bounds ``[-25, 25]`` (0 => no subgraph), threads it into the graph loader
so the matching h-function subgraph is merged, and keys its template cache per-kappa so
a step's augmentation is not frozen for the whole episode. No Mathematica/GPU needed.
"""

import json

import pytest

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.utils.graph_utils import NODE_FEATURE_SCHEMA
from gnn.reinforcement_learning.preprocessor import (
    STATE_GLOBAL_FEATURE_KEYS,
    Preprocessor,
)
from gnn.reinforcement_learning.feature_layout import (
    NATIVE_GLOBAL_FEATURE_COUNT,
    NATIVE_NODE_FEATURE_COUNT,
)
from gnn.reinforcement_learning.rl_config import (
    RL_STAGE_DEFAULT,
    resolve_rl_features,
    resolve_stage_dataset,
)

# Library kappa indices used by the fixtures below (all within [-25, 25] \ {0}).
_LIBRARY_KAPPAS = (5, 12, 13, 25, -25)


def _kappa_entry(value: int) -> dict:
    """A minimal h-function container (id must be exactly 'kappa' for the loader)."""
    return {
        "id": "kappa",
        "value": str(value),
        "graphStructure": {
            "nodes": [
                {"id": "k_root", "label": "Log", "type": "function", "value": None},
                {"id": "k_var", "label": "x", "type": "variable", "value": None},
            ],
            "edges": [{"source": "k_root", "target": "k_var", "type": "child_of"}],
        },
    }


def _make_preprocessor(tmp_path) -> Preprocessor:
    """A Preprocessor over a single tmp expression graph + a small kappa library.

    The loader is built with ``add_kappa=False`` on purpose: live-kappa augmentation is
    always-on and must fire from an explicit ``kappa_value`` alone, independent of the
    stage's static ``add_kappa`` flag. An explicit ``kappas_dir`` is required so the
    /tmp temp-dir guard in GraphDataLoader does not silently disable augmentation.
    """
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    kappas_dir = tmp_path / "kappas"
    kappas_dir.mkdir()

    main_graph = {
        "id": "P-mock-1",
        "nodes": [
            {"id": "n1", "label": "Plus", "type": "operator", "value": None},
            {"id": "n2", "label": "x", "type": "variable", "value": None},
            {"id": "n3", "label": "5.0", "type": "constant", "value": 5.0},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "child_of"},
            {"source": "n1", "target": "n3", "type": "child_of"},
        ],
    }
    (graphs_dir / "P-mock-1.json").write_text(json.dumps(main_graph), encoding="utf-8")
    (kappas_dir / "kappas.json").write_text(
        json.dumps([_kappa_entry(v) for v in _LIBRARY_KAPPAS]), encoding="utf-8"
    )

    loader = GraphDataLoader(
        name="graphs",
        mode="tree_derivatives",
        base_dir=str(graphs_dir),
        add_kappa=False,
        kappas_dir=str(kappas_dir),
    )
    return Preprocessor(loader=loader, mode="tree_derivatives")


def _message(kappa, graph_id: str = "P-mock-1") -> dict:
    return {"id": graph_id, "currentX": 1.0, "fx": 0.5, "kappa": kappa, "uuid": "u"}


def _kappa_tags(data) -> set:
    """The distinct non-None kappa values tagged on the graph's nodes."""
    tags = getattr(data, "node_kappas", None) or []
    return {round(float(k), 3) for k in tags if k is not None}


# ── live-kappa augmentation (deliverable A) ────────────────────────────────────


def test_live_kappa_merges_matching_subgraph(tmp_path):
    pre = _make_preprocessor(tmp_path)
    base, _ = pre.process(_message(0))
    aug, _ = pre.process(_message(13))
    assert aug.num_nodes > base.num_nodes      # the h-function subgraph was merged
    assert 13.0 in _kappa_tags(aug)            # tagged with the live kappa
    assert _kappa_tags(base) == set()          # kappa == 0 => no subgraph


def test_live_kappa_zero_is_base_graph(tmp_path):
    pre = _make_preprocessor(tmp_path)
    d0, _ = pre.process(_message(0))
    assert _kappa_tags(d0) == set()
    # The base must still carry the FULL node-feature schema (no silent drop), so it
    # collates against augmented graphs of the same width.
    assert d0.x.shape[1] == NATIVE_NODE_FEATURE_COUNT == len(NODE_FEATURE_SCHEMA)


def test_live_kappa_clamps_out_of_range(tmp_path):
    pre = _make_preprocessor(tmp_path)
    hi, _ = pre.process(_message(40))      # clamp -> 25
    hi25, _ = pre.process(_message(25))
    assert 25.0 in _kappa_tags(hi)
    assert hi.num_nodes == hi25.num_nodes
    lo, _ = pre.process(_message(-99))     # clamp -> -25
    assert -25.0 in _kappa_tags(lo)


def test_live_kappa_cache_is_kappa_aware(tmp_path):
    # Regression for the #1 trap: two consecutive steps on the SAME graph_id with
    # different kappa MUST return different graphs. A graph_id-only cache key would
    # freeze the first step's kappa subgraph for the whole episode.
    pre = _make_preprocessor(tmp_path)
    d5, _ = pre.process(_message(5))
    d12, _ = pre.process(_message(12))
    assert 5.0 in _kappa_tags(d5) and 12.0 not in _kappa_tags(d5)
    assert 12.0 in _kappa_tags(d12) and 5.0 not in _kappa_tags(d12)


def test_missing_kappa_fails_loudly(tmp_path):
    # A missing/null kappa is a protocol bug, NOT a silent "no subgraph" (0.0).
    pre = _make_preprocessor(tmp_path)
    with pytest.raises(KeyError):
        pre.process({"id": "P-mock-1", "currentX": 1.0})   # key absent
    with pytest.raises(KeyError):
        pre.process({"id": "P-mock-1", "kappa": None})     # key present but null


def test_global_feature_vector_dropped_kappa(tmp_path):
    assert len(STATE_GLOBAL_FEATURE_KEYS) == NATIVE_GLOBAL_FEATURE_COUNT == 7
    assert "kappa" not in STATE_GLOBAL_FEATURE_KEYS      # now drives topology, not a scalar
    assert "lastKappa" in STATE_GLOBAL_FEATURE_KEYS      # kept
    pre = _make_preprocessor(tmp_path)
    data, _ = pre.process(_message(0))
    assert data.global_features.shape[1] == 7


# ── feature flexibility already present in RL (deliverable B verification) ──────


def test_rl_default_stage_exposes_full_feature_schema():
    # RL routes node features through the SAME shared active_features mechanism as
    # supervised. On the default (full-graph) stage nothing is sliced, so every schema
    # column — including topological anchors and structural features — reaches the policy.
    ds = resolve_stage_dataset(RL_STAGE_DEFAULT)
    _selection, active = resolve_rl_features(ds.expression_graph)
    effective = set(active) if active is not None else set(NODE_FEATURE_SCHEMA)

    structural = {
        "subtree_size", "subtree_depth",
        "hist_trigonometric", "hist_exponential", "hist_variables", "hist_constants",
    }
    anchors = {"anchor_trigonometric", "anchor_exponential", "anchor_variable"}
    assert structural <= effective
    assert anchors <= effective


def test_rl_stage1_slices_out_structural_and_anchor_features():
    # Proof the selection is live: stage 1 (pure AST) drops topology + anchors.
    ds = resolve_stage_dataset("stage1_pure_ast")
    _selection, active = resolve_rl_features(ds.expression_graph)
    assert active is not None
    assert 0 < len(active) < len(NODE_FEATURE_SCHEMA)
    assert "anchor_trigonometric" not in active
    assert "subtree_size" not in active
