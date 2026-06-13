"""Matplotlib/networkx diagrams for the 1-WL distinguishability study.

Every function takes a target path and writes a PNG; nothing is shown
interactively (the Agg backend is forced by the caller).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# The package __init__ forces the headless 'Agg' backend before pyplot loads.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gnn.weisfeiler_lehman.wl_runner import (
    WLRunResult,
    distinguishability_matrix,
    global_color_histogram,
)

# Hard caps so figures stay sane even when refinement yields thousands of distinct
# colours on large datasets (otherwise figsize * dpi can blow up memory).
_MAX_FIG_IN = 40.0
_MAX_TICK_LABELS = 80


def _bounded(size_in: float) -> float:
    return float(min(max(size_in, 3.0), _MAX_FIG_IN))


def plot_per_graph_histogram(
    gid: str,
    color_ids: list[int],
    counts: np.ndarray,
    path: Path,
    *,
    round_label: str,
) -> None:
    """One bar chart: WL colour id -> node count for a single graph.

    Only the colours actually present in this graph are drawn, sorted by count.
    """
    nonzero = np.nonzero(counts)[0]
    order = nonzero[np.argsort(counts[nonzero])[::-1]]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(_bounded(max(4, len(order) * 0.3)), 3.2))
    ax.bar(x, counts[order], color="#3b7dd8")
    if len(order) <= _MAX_TICK_LABELS:
        ax.set_xticks(x)
        ax.set_xticklabels([str(color_ids[i]) for i in order], rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    ax.set_xlabel(f"WL colour id ({round_label}), {len(order)} distinct")
    ax.set_ylabel("node count")
    ax.set_title(f"Graph {gid} — WL colour histogram")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_color_heatmap(
    result: WLRunResult,
    path: Path,
    *,
    round_index: int = -1,
) -> None:
    """Heatmap of every graph's colour histogram (graphs x colours)."""
    color_ids, matrix = global_color_histogram(result, round_index)
    gids = result.graph_ids
    data = np.vstack([matrix[g] for g in gids]) if gids else np.zeros((0, 0))

    fig, ax = plt.subplots(
        figsize=(
            _bounded(max(6, len(color_ids) * 0.28)),
            _bounded(max(3, len(gids) * 0.28)),
        )
    )
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    if len(gids) <= _MAX_TICK_LABELS:
        ax.set_yticks(np.arange(len(gids)))
        ax.set_yticklabels(gids, fontsize=7)
    if len(color_ids) <= _MAX_TICK_LABELS:
        ax.set_xticks(np.arange(len(color_ids)))
        ax.set_xticklabels(color_ids, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    ax.set_xlabel(f"global WL colour id, final round ({len(color_ids)} distinct)")
    ax.set_ylabel("graph id")
    ax.set_title("Per-graph WL colour histograms")
    fig.colorbar(im, ax=ax, label="node count")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distinguishability_matrix(result: WLRunResult, path: Path) -> None:
    """N x N heatmap: which graph pairs 1-WL can / cannot tell apart."""
    ordered_ids, matrix = distinguishability_matrix(result)
    n = len(ordered_ids)
    side = _bounded(max(4, n * 0.3))
    fig, ax = plt.subplots(figsize=(side, side))
    # True (distinguishable) -> light, False (indistinguishable) -> dark.
    ax.imshow(matrix, cmap="Greys_r", vmin=0, vmax=1)
    if n <= _MAX_TICK_LABELS:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(ordered_ids, rotation=90, fontsize=7)
        ax.set_yticklabels(ordered_ids, fontsize=7)
    ax.set_title(
        "1-WL distinguishability\n(dark = indistinguishable pair)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_class_size_distribution(result: WLRunResult, path: Path) -> None:
    """Bar chart of equivalence-class sizes (sorted descending)."""
    sizes = [len(m) for m in result.equivalence_classes.values()]
    fig, ax = plt.subplots(figsize=(_bounded(max(4, len(sizes) * 0.3)), 3.4))
    x = np.arange(len(sizes))
    colors = ["#d8643b" if s > 1 else "#3b7dd8" for s in sizes]
    ax.bar(x, sizes, color=colors)
    ax.set_xlabel("equivalence class (sorted by size)")
    ax.set_ylabel("number of graphs")
    ax.set_title(
        f"WL equivalence classes: {result.num_classes} classes / "
        f"{result.num_graphs} graphs"
    )
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sample_graph(
    data: Any,
    colors: np.ndarray,
    gid: str,
    path: Path,
) -> None:
    """Draw a single expression graph, nodes tinted by their final WL colour."""
    edge_index = data.edge_index.cpu().numpy()
    g = nx.DiGraph()
    g.add_nodes_from(range(int(data.num_nodes)))
    for s, t in edge_index.T:
        g.add_edge(int(s), int(t))

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = nx.spring_layout(g, seed=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, alpha=0.4, edge_color="grey")
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=colors, cmap="tab20", node_size=240
    )
    labels = {i: str(int(c)) for i, c in enumerate(colors)}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title(f"Graph {gid} — nodes coloured by final WL colour")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mode_summary(summary_rows: list[dict], path: Path) -> None:
    """Cross-mode bar chart of distinguishability rates (the study payoff)."""
    labels = [r["label"] for r in summary_rows]
    rates = [r["distinguishability_rate"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.1), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color="#3b7dd8")
    for bar, row in zip(bars, summary_rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{row['num_classes']}/{row['num_graphs']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("distinguishability rate (classes / graphs)")
    ax.set_title("1-WL distinguishability across graph modes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
