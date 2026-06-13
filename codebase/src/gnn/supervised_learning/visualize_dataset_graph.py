#!/usr/bin/env python3
"""
visualize_dataset_graph.py

A tool to visualize expression graphs from the dataset, mirroring the *current*
graph-generation pipeline in ``gnn.shared.utils.graph_utils``.

The graph for a single problem is assembled from up to three abstract syntax
trees — the function ``f`` and its derivatives ``f'`` and ``f''`` — joined through
a ``global`` node. On top of that base structure the pipeline offers a number of
independent, toggleable augmentations, all of which this tool can render:

Modes (``--mode``)
    tree                only f
    tree_derivatives    f, f', f'' (no augmented edges)
    graph               f, f', f'' (augmented edges available)

Augmentations
    --supernode         inject a fully-connected ``virtual_supernode``
    --kappa             merge the kappa (h-function) subgraph(s) via
                        LoadAugmentedFunctionGraph
    --func-var-edges    add the augmented NextUse (variable reuse) and
                        OuterToInner/InnerToOuter (function nesting) edges
    --edge-direction    top_down | bottom_up | bidirectional (AST edges only;
                        virtual / kappa edges stay bidirectional)

When *no* augmentation/mode/direction flag is supplied, every sensible
combination is rendered for the chosen problem graph, written into a nested
sub-directory tree so the output stays organised. Function/variable edges are
only enumerated for ``graph`` mode (the only mode in which the real pipeline
adds them); for other modes they are off unless explicitly requested.

Supports rendering to PDF, SVG, PNG and GEXF (for Gephi).
"""

import sys
import json
import math
import argparse
import itertools
from collections import deque
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add the codebase paths
script_dir = Path(__file__).resolve().parent
gnn_root = script_dir.parent
src_root = gnn_root.parent
for p in [str(gnn_root), str(src_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.shared.utils.graph_utils import (
    ExpressionGraphConverter,
    parse_graphml_to_nodes_and_edges,
    find_roots,
    build_augmented_math_graph,
    inject_virtual_supernode,
    validate_edge_direction,
    LoadAugmentedFunctionGraph,
    _mark_function_roots,
    ROOT_COLOR_VOCAB,
    SUPERNODE_NODE_ID,
)

# Reverse map: root_color code -> name ("f"/"d1"/"d2"/"kappa"/"none").
_COLOR_TO_NAME = {int(v): k for k, v in ROOT_COLOR_VOCAB.items()}

# Structural AST relation types (forward direction) produced by _add_ast_edges.
_STRUCTURAL_FORWARD = {"child_of", "left_operand", "right_operand"}

# Canonical graph modes accepted by the converter.
ALL_MODES = ["tree", "tree_derivatives", "graph"]
ALL_DIRECTIONS = ["top_down", "bottom_up", "bidirectional"]

# Colour palette (group -> fill colour).
GROUP_COLORS = {
    "f": "#2ecc71",          # green
    "d1": "#3498db",         # blue
    "d2": "#e67e22",         # orange
    "kappa": "#e74c3c",      # red
    "global": "#9b59b6",     # purple
    "supernode": "#8e44ad",  # darker purple
    "other": "#bdc3c7",      # silver
}


# --------------------------------------------------------------------------- #
# Graph construction (mirrors ExpressionGraphConverter.convert, sans tensors)  #
# --------------------------------------------------------------------------- #
def _normalize_mode(mode: str) -> str:
    """Accept user-friendly spellings (e.g. 'tree-derivative')."""
    if mode is None:
        return None
    m = mode.strip().lower().replace("-", "_")
    if m in ("tree_derivative", "tree_derivatives", "derivatives"):
        return "tree_derivatives"
    if m in ("tree", "f"):
        return "tree"
    if m in ("graph", "full"):
        return "graph"
    raise ValueError(f"Unknown mode {mode!r}; expected one of {ALL_MODES}")


def _build_raw_for_mode(raw_dict: dict, mode: str) -> dict:
    """Normalize a raw graph dict the same way convert() does for the dict path.

    Parses the GraphML container (or legacy node/edge lists), marks AST roots with
    ``root_color`` and wires ``global -> root`` child_of edges. ``tree`` mode keeps
    only ``f``; the derivative subtrees are parsed for the other two modes.
    """
    raw = dict(raw_dict)

    if "graphml_f" in raw:
        nodes_f, edges_f = parse_graphml_to_nodes_and_edges(raw.get("graphml_f", ""), "f")

        if mode in ("tree_derivatives", "graph"):
            nodes_d1, edges_d1 = parse_graphml_to_nodes_and_edges(raw.get("graphml_derivative1", ""), "d1")
            nodes_d2, edges_d2 = parse_graphml_to_nodes_and_edges(raw.get("graphml_derivative2", ""), "d2")
        else:
            nodes_d1, edges_d1 = [], []
            nodes_d2, edges_d2 = [], []

        combined_nodes = nodes_f + nodes_d1 + nodes_d2
        combined_nodes.insert(0, {"id": "global", "label": "GLOBAL", "type": "global", "value": None})

        roots_f = find_roots(nodes_f, edges_f)
        roots_d1 = find_roots(nodes_d1, edges_d1)
        roots_d2 = find_roots(nodes_d2, edges_d2)

        for color, roots, nodes_list in [
            ("f", roots_f, nodes_f),
            ("d1", roots_d1, nodes_d1),
            ("d2", roots_d2, nodes_d2),
        ]:
            root_set = set(roots)
            for node in nodes_list:
                if node["id"] in root_set:
                    node["type"] = "root"
                    node["root_color"] = ROOT_COLOR_VOCAB[color]

        combined_edges = edges_f + edges_d1 + edges_d2
        for color, roots in [("f", roots_f), ("d1", roots_d1), ("d2", roots_d2)]:
            for root in roots:
                combined_edges.append({"source": "global", "target": root, "type": "child_of"})

        raw["nodes"] = combined_nodes
        raw["edges"] = combined_edges
    else:
        # Legacy node/edge list format: provenance edges -> root_color marking.
        raw["nodes"] = list(raw.get("nodes", []))
        raw["edges"] = list(raw.get("edges", []))
        _mark_function_roots(raw)

    return raw


def build_visual_graph(source, mode, edge_direction, func_var_edges, add_supernode):
    """Build an enriched NetworkX graph for visualization.

    Mirrors the relevant parts of ``ExpressionGraphConverter.convert`` but stops
    before tensorization, keeping human-readable node/edge attributes (``label``,
    ``type``, ``root_color``, ``etype``) for drawing.

    ``source`` is either a raw graph dict (no-kappa path) or an ``nx.DiGraph``
    already merged with kappa subgraphs (kappa path).

    Returns ``(G_enriched, children_dict)`` where ``children_dict`` maps each
    parent to its child_of children (independent of the drawn edge direction).
    """
    converter = ExpressionGraphConverter()
    edge_direction = validate_edge_direction(edge_direction)

    G_enriched = nx.DiGraph()
    children_dict: dict[str, list] = {}

    if isinstance(source, nx.DiGraph):
        # --- Kappa-augmented path (source is an AugmentedFunctionGraph) ---------
        node_ids = list(source.nodes)
        for node in node_ids:
            attrs = dict(source.nodes[node])
            attrs.setdefault("root_color", float(ROOT_COLOR_VOCAB["none"]))
            G_enriched.add_node(node, **attrs)

        child_counters: dict[str, int] = {}
        for u, v, attrs in source.edges(data=True):
            etype = attrs.get("type") or attrs.get("etype") or "child_of"
            if etype == "child_of":
                children_dict.setdefault(u, []).append(v)

            # Kappa / supernode edges already carry their relation attributes; copy
            # them verbatim so we draw the real connection (GlobalToKappa, ...).
            if "relation_type" in attrs or "child_index" in attrs or "direction" in attrs:
                G_enriched.add_edge(u, v, **attrs)
                continue

            child_idx = child_counters.get(u, 0)
            child_counters[u] = child_idx + 1
            converter._add_ast_edges(G_enriched, u, v, child_idx, etype, edge_direction)
    else:
        # --- Plain (no-kappa) path (source is a raw dict) ----------------------
        raw = _build_raw_for_mode(source, mode)
        G_directed = converter._build_networkx(raw)
        node_ids = list(G_directed.nodes)
        for node in node_ids:
            G_enriched.add_node(node, **dict(G_directed.nodes[node]))

        child_counters = {}
        for edge in raw.get("edges", []):
            u, v, etype = edge["source"], edge["target"], edge["type"]
            if etype == "child_of":
                children_dict.setdefault(u, []).append(v)
            child_idx = child_counters.get(u, 0)
            child_counters[u] = child_idx + 1
            converter._add_ast_edges(G_enriched, u, v, child_idx, etype, edge_direction)

    # Augmented NextUse / function-nesting edges (turn the tree into a graph).
    if func_var_edges:
        if "global" in G_enriched:
            build_augmented_math_graph(G_enriched, "global", {}, children_dict, edge_direction)
        else:
            for r in [n for n, d in G_enriched.in_degree() if d == 0]:
                build_augmented_math_graph(G_enriched, r, {}, children_dict, edge_direction)

    # Optional fully-connected virtual supernode.
    if add_supernode:
        # inject_virtual_supernode mutates both graphs + node_ids; a throwaway
        # directed graph is enough since we only need it applied to G_enriched.
        inject_virtual_supernode(G_enriched, nx.DiGraph(), node_ids)

    # tree mode: keep only f (drop the derivative subtrees that the kappa loader
    # always pulls in) so every mode is consistent across the kappa toggle.
    groups = compute_node_groups(G_enriched, children_dict)
    if mode == "tree":
        keep = [n for n in G_enriched.nodes if groups.get(n) not in ("d1", "d2")]
        G_enriched = G_enriched.subgraph(keep).copy()

    return G_enriched, children_dict


def compute_node_groups(G: nx.DiGraph, children_dict: dict) -> dict:
    """Assign every node a colour group: f / d1 / d2 / kappa / global / supernode / other.

    Function membership is propagated from the coloured AST roots down the
    child_of adjacency (every AST node is a descendant of exactly one root).
    """
    group: dict[str, str] = {}
    queue = deque()

    for n in G.nodes:
        sn = str(n)
        if sn == "global":
            group[n] = "global"
        elif sn == SUPERNODE_NODE_ID:
            group[n] = "supernode"
        elif sn.startswith("kappa_"):
            group[n] = "kappa"

    for n in G.nodes:
        if n in group:
            continue
        rc = G.nodes[n].get("root_color")
        name = _COLOR_TO_NAME.get(int(rc), "none") if rc is not None else "none"
        if name in ("f", "d1", "d2", "kappa"):
            group[n] = name
            queue.append(n)

    while queue:
        parent = queue.popleft()
        for child in children_dict.get(parent, []):
            if child not in group:
                group[child] = group[parent]
                queue.append(child)

    for n in G.nodes:
        group.setdefault(n, "other")
    return group


# --------------------------------------------------------------------------- #
# Layout                                                                       #
# --------------------------------------------------------------------------- #
def _compute_depths(G: nx.DiGraph, children_dict: dict, groups: dict) -> dict:
    """Depth of each node (global=0, roots=1, ...) over the child_of adjacency."""
    depth: dict[str, int] = {}
    queue = deque()

    if "global" in G:
        depth["global"] = 0
    # Seed every AST/kappa root at depth 1, then sweep their subtrees.
    for n in G.nodes:
        rc = G.nodes[n].get("root_color")
        if rc is not None and int(rc) != 0:
            depth[n] = 1
            queue.append(n)

    while queue:
        parent = queue.popleft()
        for child in children_dict.get(parent, []):
            if child not in depth:
                depth[child] = depth.get(parent, 1) + 1
                queue.append(child)

    for n in G.nodes:
        depth.setdefault(n, 0)
    return depth


def compute_hierarchical_layout(G: nx.DiGraph, children_dict: dict, groups: dict) -> dict:
    """Lay out f / f' / f'' (and kappa) as separate vertical columns."""
    depths = _compute_depths(G, children_dict, groups)

    column_x = {"f": -3.0, "d1": 0.0, "d2": 3.0, "kappa": 6.0,
                "global": 0.0, "supernode": -5.5, "other": 0.0}

    # Bucket nodes by (group, depth) so we can spread siblings horizontally.
    buckets: dict[tuple, list] = {}
    for n in G.nodes:
        key = (groups.get(n, "other"), depths.get(n, 0))
        buckets.setdefault(key, []).append(n)

    pos: dict[str, tuple] = {}
    for (group, depth), nodes in buckets.items():
        center = column_x.get(group, 0.0)
        if len(nodes) == 1:
            pos[nodes[0]] = (center, -float(depth))
        else:
            width = 2.2
            step = width / (len(nodes) - 1)
            for i, node in enumerate(sorted(nodes, key=str)):
                pos[node] = (center - width / 2 + i * step, -float(depth))

    # Pin the structural anchors.
    if "global" in pos:
        pos["global"] = (1.5, 1.4)
    if SUPERNODE_NODE_ID in pos:
        pos[SUPERNODE_NODE_ID] = (-5.5, 0.5)

    for n in G.nodes:
        pos.setdefault(n, (0.0, 0.0))
    return pos


# --------------------------------------------------------------------------- #
# Edge categorisation + drawing                                                #
# --------------------------------------------------------------------------- #
def _edge_category(etype: str) -> str:
    if etype is None:
        return "structural"
    if etype.startswith("supernode_connection"):
        return "supernode"
    if etype in ("GlobalToKappa", "KappaToGlobal"):
        return "kappa"
    if etype in ("NextUse", "NextUseBackward"):
        return "variable"
    if etype.startswith("OuterToInner") or etype.startswith("InnerToOuter"):
        return "function"
    if etype.endswith("_reverse"):
        base = etype[:-len("_reverse")]
        if base in _STRUCTURAL_FORWARD:
            return "structural_reverse"
        return "structural_reverse"
    if etype in _STRUCTURAL_FORWARD:
        return "structural"
    return "structural"


# (color, alpha, width, style, arrowsize) per edge category.
_EDGE_STYLE = {
    "structural":         ("#7f8c8d", 0.85, 1.2, "solid",  10),
    "structural_reverse": ("#d5dbdb", 0.40, 0.8, "dotted",  6),
    "supernode":          ("#9b59b6", 0.15, 0.5, "dashed",  5),
    "variable":           ("#16a085", 0.55, 1.0, "dashed",  8),
    "function":           ("#2980b9", 0.55, 1.0, "dashed",  8),
    "kappa":              ("#e74c3c", 0.70, 1.4, "solid",  10),
}

_EDGE_LEGEND = {
    "structural": "AST edge (child_of)",
    "structural_reverse": "AST edge (reverse)",
    "supernode": "Supernode link",
    "variable": "Variable reuse (NextUse)",
    "function": "Function nesting",
    "kappa": "Global <-> Kappa",
}


def visualize_graph(G, children_dict, groups, output_path, fmt, meta, layout_name="hierarchical"):
    """Render the NetworkX graph and save it as PDF / SVG / PNG."""
    fig, ax = plt.subplots(figsize=(13, 9), dpi=200)
    ax.axis("off")

    if layout_name == "hierarchical":
        pos = compute_hierarchical_layout(G, children_dict, groups)
    elif layout_name == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    for node in G.nodes:
        pos.setdefault(node, (0.0, 0.0))

    # Node colours / sizes / labels.
    node_colors, node_sizes, labels = [], [], {}
    for node in G.nodes:
        grp = groups.get(node, "other")
        node_colors.append(GROUP_COLORS.get(grp, GROUP_COLORS["other"]))
        if grp in ("global", "supernode"):
            node_sizes.append(620)
        elif G.nodes[node].get("type") == "root":
            node_sizes.append(440)
        else:
            node_sizes.append(300)
        labels[node] = str(G.nodes[node].get("label") or node)

    # Bucket edges by category.
    edges_by_cat: dict[str, list] = {}
    for u, v in G.edges:
        cat = _edge_category(G.edges[u, v].get("etype"))
        edges_by_cat.setdefault(cat, []).append((u, v))

    present_edge_cats = []
    for cat in ("structural", "structural_reverse", "variable", "function", "kappa", "supernode"):
        edgelist = edges_by_cat.get(cat)
        if not edgelist:
            continue
        present_edge_cats.append(cat)
        color, alpha, width, style, arrowsize = _EDGE_STYLE[cat]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, ax=ax, edge_color=color, alpha=alpha,
            width=width, style=style, arrows=True, arrowsize=arrowsize,
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors="#2c3e50", linewidths=0.7,
    )

    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_size=6,
        font_family="sans-serif", font_color="#2c3e50",
    )

    # Legend: node groups + edge categories actually present.
    present_groups = sorted({groups.get(n, "other") for n in G.nodes})
    group_label = {
        "f": "Function f(x)", "d1": "1st Derivative f'(x)", "d2": "2nd Derivative f''(x)",
        "kappa": "Kappa (h-function)", "global": "Global", "supernode": "Supernode",
        "other": "Other",
    }
    legend_elements = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="#2c3e50", label=group_label.get(g, g))
        for g in present_groups
    ]
    from matplotlib.lines import Line2D
    for cat in present_edge_cats:
        color, alpha, width, style, _ = _EDGE_STYLE[cat]
        legend_elements.append(
            Line2D([0], [0], color=color, lw=1.6, linestyle=style, label=_EDGE_LEGEND[cat])
        )
    leg = ax.legend(
        handles=legend_elements, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, -0.02), frameon=True, facecolor="white",
        edgecolor="#bdc3c7", fontsize=8,
    )

    title = (
        f"Graph {meta['graph_id']} | mode={meta['mode']} | dir={meta['edge_direction']}\n"
        f"supernode={meta['supernode']}  kappa={meta['kappa']}  "
        f"func/var-edges={meta['func_var_edges']}  "
        f"(nodes={G.number_of_nodes()}, edges={G.number_of_edges()})"
    )
    plt.title(title, fontsize=11, fontweight="bold", pad=14)

    plt.savefig(output_path, format=fmt, bbox_inches="tight",
                bbox_extra_artists=(leg,), dpi=200)
    plt.close(fig)
    print(f"[Visualizer] Saved: {output_path}")


def export_gexf(G, output_path):
    """Export to Gephi GEXF, sanitising None/unsupported attribute values."""
    G_export = G.copy()
    for _, attrs in G_export.nodes(data=True):
        for k, val in list(attrs.items()):
            if val is None:
                attrs[k] = ""
    for _, _, attrs in G_export.edges(data=True):
        for k, val in list(attrs.items()):
            if val is None:
                attrs[k] = ""
    nx.write_gexf(G_export, str(output_path))
    print(f"[Visualizer] Gephi GEXF exported: {output_path}")


# --------------------------------------------------------------------------- #
# Source loading                                                               #
# --------------------------------------------------------------------------- #
def load_raw_dict(loader: GraphDataLoader, graph_id: str) -> dict:
    raw_val = loader._raw_sources[graph_id]
    if isinstance(raw_val, Path):
        with open(raw_val, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return raw_val


def load_kappa_source(loader: GraphDataLoader, graph_id: str, kappa_value):
    """Load the kappa-augmented nx.DiGraph (or None if no kappas are available)."""
    kappas_dir = loader.kappas_dir
    if not kappas_dir.exists() or not any(kappas_dir.glob("**/*.json")):
        return None
    return LoadAugmentedFunctionGraph(
        graphId=str(graph_id),
        graphsFolder=loader.source_path,
        kappasFolder=kappas_dir,
        kappa_value=kappa_value,
    )


def available_kappa_values(kappas_dir: Path) -> list:
    """All kappa 'value' entries found under the kappas directory, as floats."""
    values = []
    for path in kappas_dir.glob("**/*.json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and "value" in item:
                try:
                    values.append(float(item["value"]))
                except (TypeError, ValueError):
                    pass
    return sorted(set(values))


def resolve_kappa_value(args, loader, dataset_name, graph_id, is_synthetic):
    """Pick which single kappa to merge for the visualization.

    The current pipeline merges exactly one kappa per problem (selective merge),
    so the representative view shows a single h-function rather than all of them.
    Resolution order: explicit ``--kappa-value`` -> the problem's own kappa (from
    the tabular dataset) -> the first available kappa as a stand-in example. When
    ``--kappa-value`` is the sentinel ``inf`` we merge *all* kappas.
    """
    if args.kappa_value is not None:
        return args.kappa_value  # honour explicit choice (incl. inf == merge all)

    available = available_kappa_values(loader.kappas_dir)
    if not available:
        return None

    try:
        from gnn.shared.utils.unified_loader import UnifiedDataLoader
        unified = UnifiedDataLoader.get_instance(
            dataset_name=dataset_name, is_synthetic=is_synthetic,
        )
        kappa_map = unified.build_kappa_map()
        own = kappa_map.get(str(graph_id))
        if own is not None and float(own) in available:
            return float(own)
    except Exception as exc:
        print(f"[Visualizer] Could not resolve problem kappa from dataset ({exc}); "
              f"using a representative kappa instead.")

    chosen = available[0]
    print(f"[Visualizer] No specific kappa for graph '{graph_id}'; "
          f"showing representative kappa value {chosen}.")
    return chosen


# --------------------------------------------------------------------------- #
# Rendering one combination                                                    #
# --------------------------------------------------------------------------- #
def render_combination(loader, graph_id, mode, edge_direction, supernode, kappa,
                       func_var_edges, formats, out_root, layout, kappa_value):
    """Build and render one full configuration. Returns True on success."""
    # inf is the "merge all kappas" sentinel; otherwise merge the single value.
    merge_value = None if (kappa_value is not None and math.isinf(kappa_value)) else kappa_value
    if kappa:
        source = load_kappa_source(loader, graph_id, merge_value)
        if source is None:
            print(f"[Visualizer] Skipping kappa=True for graph {graph_id}: "
                  f"no kappas found in {loader.kappas_dir}")
            return False
    else:
        source = load_raw_dict(loader, graph_id)

    G, children_dict = build_visual_graph(
        source, mode=mode, edge_direction=edge_direction,
        func_var_edges=func_var_edges, add_supernode=supernode,
    )
    groups = compute_node_groups(G, children_dict)

    if not kappa:
        kappa_label = "False"
    elif merge_value is None:
        kappa_label = "all"
    else:
        kappa_label = f"{merge_value:g}"
    meta = {
        "graph_id": graph_id, "mode": mode, "edge_direction": edge_direction,
        "supernode": supernode, "kappa": kappa_label, "func_var_edges": func_var_edges,
    }

    # Nested sub-directories keep the (potentially dozens of) outputs organised.
    out_dir = Path(out_root) / f"graph_{graph_id}" / mode / edge_direction
    out_dir.mkdir(parents=True, exist_ok=True)
    base = (f"{mode}_dir-{edge_direction}_sn-{int(supernode)}"
            f"_kappa-{int(kappa)}_fve-{int(func_var_edges)}")

    for fmt in formats:
        filepath = out_dir / f"{base}.{fmt}"
        if fmt == "gexf":
            export_gexf(G, filepath)
        else:
            visualize_graph(G, children_dict, groups, str(filepath), fmt, meta, layout)
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize expression graphs from the dataset under the current pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", "-d", type=str, default="graphs",
                        help="Dataset name (e.g. 'graphs', 'synthetic_graphs', 'run_XXXX/graphs')")
    parser.add_argument("--graph-id", "-i", type=str, default="0",
                        help="ID of the problem graph to visualize")

    parser.add_argument("--mode", "-m", type=str, default=None,
                        help="Graph mode: tree | tree_derivatives (tree-derivative) | graph. "
                             "Omit to render all modes.")
    parser.add_argument("--edge-direction", "-e", type=str, default=None,
                        choices=ALL_DIRECTIONS,
                        help="AST edge direction. Omit to render all directions.")

    parser.add_argument("--supernode", action=argparse.BooleanOptionalAction, default=None,
                        help="Inject a fully-connected virtual supernode. "
                             "Omit (no --supernode/--no-supernode) to render both.")
    parser.add_argument("--kappa", action=argparse.BooleanOptionalAction, default=None,
                        help="Merge the kappa (h-function) subgraph(s). "
                             "Omit to render both with and without.")
    parser.add_argument("--func-var-edges", action=argparse.BooleanOptionalAction, default=None,
                        help="Add augmented function-nesting + variable-reuse edges. "
                             "Omit to enumerate (graph mode only).")
    parser.add_argument("--kappa-value", type=float, default=None,
                        help="Merge only this kappa value. Default: the problem's own "
                             "kappa (selective merge, as in training). Pass 'inf' to "
                             "merge all 50 kappas at once.")

    parser.add_argument("--format", "-f", type=str, default="pdf",
                        choices=["pdf", "svg", "gexf", "png", "all"],
                        help="Output format (or 'all').")
    parser.add_argument("--output-dir", "-o", type=str, default="visualizations",
                        help="Base directory for the (sub-foldered) results.")
    parser.add_argument("--layout", type=str, default="hierarchical",
                        choices=["hierarchical", "spring", "kamada_kawai"],
                        help="Graph layout algorithm.")
    parser.add_argument("--is-synthetic", action="store_true",
                        help="Set when loading a synthetic dataset file.")
    args = parser.parse_args()

    mode_arg = _normalize_mode(args.mode) if args.mode else None

    # A single loader instance is enough: it only provides raw sources, the source
    # path and the kappas directory. Conversion happens here in the visualizer.
    loader = GraphDataLoader(
        name=args.dataset_name,
        mode=mode_arg or "graph",
        is_synthetic=args.is_synthetic or "synthetic" in args.dataset_name,
    )

    if not loader.has_graph(args.graph_id):
        print(f"Error: Graph ID '{args.graph_id}' not found in dataset '{args.dataset_name}'.")
        available = sorted(loader.list_graph_ids())[:20]
        print(f"Available Graph IDs (first 20): {available}")
        sys.exit(1)

    # Expand each axis: a fixed value if supplied, else every option.
    modes = [mode_arg] if mode_arg else list(ALL_MODES)
    directions = [args.edge_direction] if args.edge_direction else list(ALL_DIRECTIONS)
    supernodes = [args.supernode] if args.supernode is not None else [False, True]
    kappas = [args.kappa] if args.kappa is not None else [False, True]
    formats = [args.format] if args.format != "all" else ["pdf", "svg", "png", "gexf"]

    # Resolve the single kappa to merge once (shared across every kappa combo).
    resolved_kappa = None
    if True in kappas:
        resolved_kappa = resolve_kappa_value(
            args, loader, args.dataset_name, args.graph_id,
            args.is_synthetic or "synthetic" in args.dataset_name,
        )

    combos = []
    for mode, direction, supernode, kappa in itertools.product(modes, directions, supernodes, kappas):
        if args.func_var_edges is not None:
            fve_values = [args.func_var_edges]
        else:
            # The real pipeline only adds these edges in graph mode; enumerate
            # both there, and keep them off for tree / tree_derivatives.
            fve_values = [False, True] if mode == "graph" else [False]
        for fve in fve_values:
            combos.append((mode, direction, supernode, kappa, fve))

    print(f"[Visualizer] Rendering {len(combos)} configuration(s) for graph "
          f"'{args.graph_id}' into '{args.output_dir}/graph_{args.graph_id}/' ...")

    rendered = skipped = 0
    for mode, direction, supernode, kappa, fve in combos:
        try:
            ok = render_combination(
                loader, args.graph_id, mode, direction, supernode, kappa, fve,
                formats, args.output_dir, args.layout, resolved_kappa,
            )
            rendered += int(ok)
            skipped += int(not ok)
        except Exception as exc:  # keep going through the rest of the matrix
            skipped += 1
            print(f"[Visualizer] FAILED mode={mode} dir={direction} sn={supernode} "
                  f"kappa={kappa} fve={fve}: {exc}")

    print(f"[Visualizer] Done. Rendered {rendered}, skipped {skipped}.")


if __name__ == "__main__":
    main()
