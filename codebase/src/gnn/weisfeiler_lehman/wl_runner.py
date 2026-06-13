"""Core 1-Weisfeiler-Lehman (1-WL) machinery for the distinguishability study.

This module wraps PyTorch Geometric's :class:`~torch_geometric.nn.WLConv` to run
the classic 1-WL colour-refinement test over a *whole dataset* of expression
graphs at once. Because a single ``WLConv`` instance keeps a colour hashmap that
is shared across every graph it processes, applying the same conv to all graphs
yields globally comparable colours — which is exactly what we need to decide
whether two graphs are 1-WL distinguishable.

Distinguishability criterion (WL subtree kernel):
    Two graphs are 1-WL *indistinguishable* iff, at **every** refinement
    iteration, their colour histograms (the multiset of node colours) are
    identical. We therefore fingerprint each graph by the concatenation of its
    per-iteration sorted ``(global_colour_id, count)`` pairs and group graphs by
    that fingerprint. Graphs sharing a fingerprint form an equivalence class the
    test cannot tell apart.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import WLConv

# Initial node-colouring schemes. ``label`` keys off the semantic operator/operand
# label (CANONICAL_LABEL_VOCAB code) and is the most meaningful for our expression
# graphs; the others isolate pure structure.
COLORING_SCHEMES: tuple[str, ...] = ("label", "node_type", "degree", "constant")


def build_initial_coloring(data: Any, scheme: str) -> Tensor:
    """Return a 1-D long tensor of initial node colours for one PyG ``Data``.

    The returned colour ids must be consistent across graphs, so we key off
    globally shared vocabularies (``label_id`` / ``node_type``) or off the node
    degree, never off a per-graph relabelling.
    """
    num_nodes = int(data.num_nodes)
    if scheme == "label":
        colors = getattr(data, "label_id", None)
        if colors is None:
            raise ValueError("Graph is missing 'label_id'; cannot colour by label.")
        return colors.view(-1).long()
    if scheme == "node_type":
        colors = getattr(data, "node_type", None)
        if colors is None:
            raise ValueError("Graph is missing 'node_type'; cannot colour by type.")
        return colors.view(-1).long()
    if scheme == "degree":
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, dtype=torch.long)
        deg = torch.bincount(edge_index.view(-1), minlength=num_nodes)
        return deg.long()
    if scheme == "constant":
        return torch.zeros(num_nodes, dtype=torch.long)
    raise ValueError(f"Unknown coloring scheme {scheme!r}; expected {COLORING_SCHEMES}")


def prepare_edge_index(data: Any, symmetrize: bool) -> Tensor:
    """Return the edge_index used for refinement.

    Classic 1-WL is defined on undirected graphs, so by default we symmetrize the
    (possibly directed) AST edges. ``symmetrize=False`` keeps the stored direction
    for a direction-sensitive variant.
    """
    edge_index = data.edge_index.long()
    if edge_index.numel() == 0:
        return edge_index
    if symmetrize:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


@dataclass
class WLRunResult:
    """Outcome of running 1-WL over a dataset for a single configuration."""

    graph_ids: list[str]
    iterations: int  # number of refinement rounds actually performed
    # history[i] maps graph_id -> 1-D long tensor of global colour ids at round i.
    # Round 0 is the initial colouring; rounds 1..iterations are refinements.
    history: list[dict[str, Tensor]]
    fingerprints: dict[str, str]
    equivalence_classes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def num_graphs(self) -> int:
        return len(self.graph_ids)

    @property
    def num_classes(self) -> int:
        return len(self.equivalence_classes)

    @property
    def distinguishability_rate(self) -> float:
        if not self.graph_ids:
            return 0.0
        return self.num_classes / self.num_graphs

    def colliding_classes(self) -> dict[str, list[str]]:
        """Equivalence classes with more than one member (mutually indistinct)."""
        return {fp: g for fp, g in self.equivalence_classes.items() if len(g) > 1}

    def num_colliding_graphs(self) -> int:
        return sum(len(g) for g in self.colliding_classes().values())


def _histogram_counts(colors: Tensor) -> dict[int, int]:
    values, counts = torch.unique(colors, return_counts=True)
    return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}


def run_wl(
    graphs: dict[str, Any],
    *,
    coloring: str = "label",
    symmetrize: bool = True,
    max_iterations: int = 10,
) -> WLRunResult:
    """Run dataset-wide 1-WL colour refinement and compute equivalence classes.

    Parameters
    ----------
    graphs:
        Mapping ``graph_id -> PyG Data`` (homogeneous). All graphs are coloured in
        the same shared colour space so their histograms are comparable.
    coloring:
        Initial colouring scheme (see :data:`COLORING_SCHEMES`).
    symmetrize:
        Treat edges as undirected (classic 1-WL). See :func:`prepare_edge_index`.
    max_iterations:
        Hard cap on refinement rounds. Refinement stops early once the global
        colour partition is stable (no further node colours split).
    """
    graph_ids = sorted(graphs.keys())
    edge_index = {gid: prepare_edge_index(graphs[gid], symmetrize) for gid in graph_ids}
    initial = {gid: build_initial_coloring(graphs[gid], coloring) for gid in graph_ids}

    history: list[dict[str, Tensor]] = [initial]
    prev_distinct = len({int(c) for col in initial.values() for c in col.tolist()})

    current = initial
    performed = 0
    for _ in range(max_iterations):
        conv = WLConv()
        nxt = {gid: conv(current[gid], edge_index[gid]) for gid in graph_ids}
        distinct = len(conv.hashmap)
        history.append(nxt)
        performed += 1
        # WL refinement is monotone: colours only ever split. Once the number of
        # distinct colours stops growing the partition is stable -> converged.
        if distinct == prev_distinct:
            break
        prev_distinct = distinct
        current = nxt

    fingerprints = _compute_fingerprints(graph_ids, history)
    classes = _group_by_fingerprint(graph_ids, fingerprints)
    return WLRunResult(
        graph_ids=graph_ids,
        iterations=performed,
        history=history,
        fingerprints=fingerprints,
        equivalence_classes=classes,
    )


def _compute_fingerprints(
    graph_ids: list[str], history: list[dict[str, Tensor]]
) -> dict[str, str]:
    """Hash each graph's per-iteration colour histograms into a fingerprint."""
    fingerprints: dict[str, str] = {}
    for gid in graph_ids:
        rounds = []
        for round_colors in history:
            counts = _histogram_counts(round_colors[gid])
            rounds.append(sorted(counts.items()))
        payload = json.dumps(rounds, sort_keys=True, separators=(",", ":"))
        fingerprints[gid] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return fingerprints


def _group_by_fingerprint(
    graph_ids: list[str], fingerprints: dict[str, str]
) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = defaultdict(list)
    for gid in graph_ids:
        classes[fingerprints[gid]].append(gid)
    # Stable, deterministic ordering: largest classes first, then by first member.
    ordered = sorted(classes.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
    return {fp: members for fp, members in ordered}


def global_color_histogram(
    result: WLRunResult, round_index: int = -1
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Per-graph colour-count vectors over the shared colour space of one round.

    Returns the sorted list of global colour ids that appear in that round and a
    mapping ``graph_id -> count vector`` aligned to that colour list.
    """
    round_colors = result.history[round_index]
    all_colors = sorted(
        {int(c) for col in round_colors.values() for c in col.tolist()}
    )
    index = {color: i for i, color in enumerate(all_colors)}
    matrix: dict[str, np.ndarray] = {}
    for gid in result.graph_ids:
        vec = np.zeros(len(all_colors), dtype=np.int64)
        for color, count in _histogram_counts(round_colors[gid]).items():
            vec[index[color]] = count
        matrix[gid] = vec
    return all_colors, matrix


def distinguishability_matrix(
    result: WLRunResult, ordered_ids: Iterable[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Boolean matrix ``M[i, j] = graphs i and j are 1-WL distinguishable``.

    Graphs are ordered so that members of the same equivalence class are adjacent,
    making collision blocks visible on the diagonal.
    """
    if ordered_ids is None:
        ordered_ids = [gid for members in result.equivalence_classes.values()
                       for gid in members]
    ordered_ids = list(ordered_ids)
    fps = [result.fingerprints[gid] for gid in ordered_ids]
    n = len(ordered_ids)
    matrix = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if fps[i] == fps[j]:
                matrix[i, j] = False
    return ordered_ids, matrix
