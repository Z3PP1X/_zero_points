#!/usr/bin/env python3
"""
visualize_graphs.py

Scientific visualization of AST expression graphs from graphs.json.
Supports all graph modes (tree, tree_derivatives, graph), kappa augmentation,
virtual supernode, and edge-type-aware coloring.

Usage:
    python visualize_graphs.py --graphs-file datasets/graphs/graphs.json --graph-id P1
    python visualize_graphs.py --mode graph --kappa-value -5.0 --graph-id P1 P2
    python visualize_graphs.py --mode tree --graph-id P1 --format svg
"""

import os
import sys
import json
import argparse
import logging
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visual design constants
# ---------------------------------------------------------------------------

NODE_TYPE_COLORS: dict[str, str] = {
    "global":    "#4c72b0",   # deep blue
    "root_f":    "#dd8452",   # orange  (f root)
    "root_d1":   "#55a868",   # green   (f' root)
    "root_d2":   "#c44e52",   # red     (f'' root)
    "root_kappa":"#8172b2",   # purple  (kappa root)
    "root":      "#dd8452",   # fallback root (tree mode)
    "operator":  "#937860",   # brown
    "function":  "#ccb974",   # golden
    "constant":  "#da8bc3",   # pink
    "variable":  "#9e9e9e",   # grey
    "supernode": "#64b5cd",   # cyan
    "unknown":   "#eeeeee",
}

EDGE_TYPE_COLORS: dict[str, str] = {
    "child_of":                     "#888888",
    "child_of_reverse":             "#aaaaaa",
    "GlobalToKappa":                "#d62728",
    "KappaToGlobal":                "#ff9896",
    "supernode_connection":         "#9467bd",
    "supernode_connection_reverse": "#c5b0d5",
    "virtual":                      "#bcbd22",
    "virtual_reverse":              "#dbdb8d",
    "left_operand":                 "#17becf",
    "right_operand":                "#9edae5",
    "unknown":                      "#cccccc",
}

EDGE_TYPE_WIDTHS: dict[str, float] = {
    "child_of": 0.9,
    "GlobalToKappa": 1.5,
    "KappaToGlobal": 1.5,
    "supernode_connection": 0.5,
    "supernode_connection_reverse": 0.5,
}

EDGE_TYPE_ALPHAS: dict[str, float] = {
    "child_of": 0.55,
    "child_of_reverse": 0.35,
    "GlobalToKappa": 0.85,
    "KappaToGlobal": 0.7,
    "supernode_connection": 0.25,
    "supernode_connection_reverse": 0.2,
}


def _edge_family(etype: str) -> str:
    """Map a full edge type name to its color/width family key."""
    if etype.startswith("left_operand"):
        return "left_operand"
    if etype.startswith("right_operand"):
        return "right_operand"
    return etype


def _resolve_edge_type(attrs: dict) -> str:
    """Return a string edge type from raw edge attribute dict."""
    etype = attrs.get("etype")
    if etype:
        return str(etype)
    code = attrs.get("edge_type")
    if code is not None:
        try:
            from gnn.shared.utils.graph_vocab import CANONICAL_EDGE_TYPES
            return CANONICAL_EDGE_TYPES[int(code)]
        except (IndexError, ImportError, TypeError):
            pass
    return "child_of"


def _node_color_key(node: str, attrs: dict) -> str:
    """Return a NODE_TYPE_COLORS key for this node."""
    ntype = attrs.get("type", "")
    if ntype == "global":
        return "global"
    if ntype == "supernode":
        return "supernode"
    if ntype == "root":
        rc = attrs.get("root_color", 0.0)
        # root_color codes from ROOT_COLOR_VOCAB
        _code_map = {0.0: "root", 1.0: "root_f", 2.0: "root_d1", 3.0: "root_d2",
                     4.0: "root_kappa", 5.0: "root_kappa"}
        return _code_map.get(float(rc), "root")
    if ntype in NODE_TYPE_COLORS:
        return ntype
    # Fallback: infer from node id prefix used in tree_derivatives/graph modes
    sid = str(node)
    if sid == "global":
        return "global"
    if sid == "virtual_supernode":
        return "supernode"
    for prefix, key in (("f_", "root_f"), ("d1_", "root_d1"), ("d2_", "root_d2"), ("kappa_", "root_kappa")):
        if sid.startswith(prefix):
            if "root" in ntype or attrs.get("node_type") == 2:
                return key
    return "unknown"


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _parse_graphml(content: str) -> nx.DiGraph:
    content = content.replace("attr.type='String'", "attr.type='string'")
    content = content.replace('attr.type="String"', 'attr.type="string"')
    return nx.parse_graphml(content)


def _load_entry(graphs_file: str, graph_id: str) -> dict:
    with open(graphs_file, encoding="utf-8") as f:
        data = json.load(f)
    for entry in (data if isinstance(data, list) else [data]):
        if entry.get("id") == graph_id:
            return entry
    raise KeyError(f"Graph ID '{graph_id}' not found in {graphs_file}")


def _build_tree(entry: dict) -> nx.DiGraph:
    """Only the function AST (graphml_f), no global node."""
    try:
        from gnn.shared.utils.graph_converter import parse_graphml_node_name, _determine_node_type_from_label
        _have_gnn = True
    except ImportError:
        _have_gnn = False

    G = _parse_graphml(entry["graphml_f"])
    # Annotate nodes with type and label
    for nid, attrs in list(G.nodes(data=True)):
        name_val = attrs.get("Name") or attrs.get("nodeKey1") or str(nid)
        if _have_gnn and isinstance(name_val, str):
            label = parse_graphml_node_name(name_val)
            is_root = G.in_degree(nid) == 0
            ntype = "root" if is_root else _determine_node_type_from_label(label)
        else:
            label = str(name_val)
            is_root = G.in_degree(nid) == 0
            ntype = "root" if is_root else "operator"
        G.nodes[nid]["type"] = ntype
        G.nodes[nid]["label"] = label
        G.nodes[nid]["root_color"] = 1.0 if is_root else 0.0
    for u, v, attrs in G.edges(data=True):
        if "etype" not in attrs:
            G.edges[u, v]["etype"] = "child_of"
    return G


def _build_tree_derivatives(graphs_file: str, graph_id: str) -> nx.DiGraph:
    """f + f' + f'' merged via global node."""
    from gnn.shared.utils.kappa_loader import LoadGraphFromLocalStructure
    return LoadGraphFromLocalStructure(graphs_file, graph_id)




def _add_kappa(G: nx.DiGraph, graphs_file: str, graph_id: str,
               kappas_dir: str, kappa_value: float) -> nx.DiGraph:
    """Load kappa-augmented graph (base tree_derivatives + kappa merge)."""
    from gnn.shared.utils.kappa_loader import LoadAugmentedFunctionGraph
    return LoadAugmentedFunctionGraph(graph_id, graphs_file, kappas_dir, kappa_value=kappa_value)


def _add_virtual_supernode(G: nx.DiGraph) -> None:
    """Add a virtual_supernode connected bidirectionally to all nodes."""
    sn = "virtual_supernode"
    if sn in G:
        return
    existing = list(G.nodes())
    G.add_node(sn, type="supernode", label="SUPERNODE", root_color=0.0)
    for nid in existing:
        G.add_edge(sn, nid, etype="supernode_connection")
        G.add_edge(nid, sn, etype="supernode_connection_reverse")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _compute_layout(G: nx.DiGraph, layout_name: str, seed: int) -> dict:
    ug = G.to_undirected()
    if layout_name == "spring":
        k = 1.5 / (len(G) ** 0.5) if len(G) > 1 else 1.0
        return nx.spring_layout(ug, seed=seed, k=k)
    if layout_name == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(ug)
        except Exception as e:
            logger.warning(f"Kamada-Kawai failed ({e}); falling back to spring.")
            return nx.spring_layout(ug, seed=seed)
    if layout_name == "spectral":
        try:
            return nx.spectral_layout(ug)
        except Exception:
            return nx.spring_layout(ug, seed=seed)
    if layout_name == "shell":
        return nx.shell_layout(ug)
    return nx.spring_layout(ug, seed=seed)


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def _compute_centrality(G: nx.DiGraph, metric: str) -> dict:
    ug = G.to_undirected()
    if metric == "degree":
        return nx.degree_centrality(G)
    if metric == "betweenness":
        return nx.betweenness_centrality(ug)
    if metric == "closeness":
        return nx.closeness_centrality(ug)
    if metric == "eigenvector":
        try:
            return nx.eigenvector_centrality(ug, max_iter=1000)
        except Exception:
            logger.warning("Eigenvector centrality failed; using degree.")
            return nx.degree_centrality(G)
    raise ValueError(f"Unknown centrality metric: {metric}")


# ---------------------------------------------------------------------------
# Core visualization
# ---------------------------------------------------------------------------

def visualize_graph(
    G: nx.DiGraph,
    graph_id: str,
    output_dir: str,
    mode: str,
    output_format: str = "pdf",
    layout_name: str = "spring",
    color_by: str = "type",
    colormap_name: str = "viridis",
    node_scale: float = 2500.0,
    min_node_size: float = 80.0,
    seed: int = 42,
    show_labels: bool = True,
) -> None:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        logger.warning(f"Graph {graph_id} is empty — skipping.")
        return

    pos = _compute_layout(G, layout_name, seed)

    # --- node colors / sizes ------------------------------------------------
    if color_by == "type":
        node_colors = [NODE_TYPE_COLORS.get(_node_color_key(nd, G.nodes[nd]), NODE_TYPE_COLORS["unknown"]) for nd in nodes]
        node_sizes = [node_scale * 0.6 for _ in nodes]  # uniform for type mode
        # Emphasise global & supernode
        node_sizes = [
            node_scale * 0.9 if G.nodes[nd].get("type") in ("global", "supernode")
            else node_scale * 0.35 if G.nodes[nd].get("root_color", 0.0) == 0.0 and G.nodes[nd].get("type") not in ("root",)
            else node_scale * 0.55
            for nd in nodes
        ]
        node_sizes = [max(s, min_node_size) for s in node_sizes]
    else:
        centrality = _compute_centrality(G, color_by)
        vals = [centrality.get(nd, 0.0) for nd in nodes]
        cmap = plt.get_cmap(colormap_name)
        norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
        node_colors = [cmap(norm(v)) for v in vals]
        node_sizes = [max(v * node_scale, min_node_size) for v in vals]

    # --- group edges by type for layered drawing ----------------------------
    edge_groups: dict[str, list[tuple]] = {}
    for u, v, attrs in G.edges(data=True):
        etype = _resolve_edge_type(attrs)
        edge_groups.setdefault(etype, []).append((u, v))

    # --- figure setup -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)
    ax.axis("off")
    ax.set_title(
        f"Graph {graph_id}  |  mode: {mode}  |  {n} nodes, {G.number_of_edges()} edges",
        fontsize=10, family="sans-serif", pad=8,
    )

    # --- draw edges by type (so each group can have its own color/alpha) ----
    for etype, edgelist in edge_groups.items():
        family = _edge_family(etype)
        color = EDGE_TYPE_COLORS.get(family, EDGE_TYPE_COLORS.get(etype, EDGE_TYPE_COLORS["unknown"]))
        alpha = EDGE_TYPE_ALPHAS.get(family, EDGE_TYPE_ALPHAS.get(etype, 0.4))
        width = EDGE_TYPE_WIDTHS.get(family, EDGE_TYPE_WIDTHS.get(etype, 0.8))
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edgelist,
            edge_color=color,
            alpha=alpha,
            width=width,
            arrows=True,
            arrowsize=7,
            connectionstyle="arc3,rad=0.08",
        )

    # --- draw nodes ---------------------------------------------------------
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nodes,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#1a1a1a",
        linewidths=0.5,
    )

    # --- labels -------------------------------------------------------------
    if show_labels:
        label_dict: dict = {}
        if n <= 30:
            for nd in nodes:
                label_dict[nd] = G.nodes[nd].get("label", str(nd))
        else:
            # Top-10 by degree
            top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
            for nd, _ in top:
                label_dict[nd] = G.nodes[nd].get("label", str(nd))

        nx.draw_networkx_labels(
            G, pos, labels=label_dict, ax=ax,
            font_size=7, font_family="sans-serif", font_weight="bold", alpha=0.9,
        )

    # --- legend (node types) ------------------------------------------------
    legend_handles: list = []

    if color_by == "type":
        seen_types: set[str] = set()
        for nd in nodes:
            key = _node_color_key(nd, G.nodes[nd])
            if key not in seen_types:
                seen_types.add(key)
                color = NODE_TYPE_COLORS.get(key, NODE_TYPE_COLORS["unknown"])
                label = key.replace("_", " ").replace("root f", "root (f)").replace("root d1", "root (f')").replace("root d2", "root (f'')").replace("root kappa", "root (kappa)")
                legend_handles.append(mpatches.Patch(color=color, label=label))
    else:
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=min(vals), vmax=max(vals)), cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.6, aspect=35)
        cbar.set_label(f"{color_by.capitalize()} Centrality", fontsize=8, family="sans-serif", weight="bold")
        cbar.ax.tick_params(labelsize=7)

    # Edge type legend (only types actually present)
    _edge_legend_shown: set[str] = set()
    edge_legend_labels: dict[str, str] = {
        "child_of":                     "AST edge (child_of)",
        "child_of_reverse":             "AST edge (reverse)",
        "GlobalToKappa":                "Global → Kappa",
        "KappaToGlobal":                "Kappa → Global",
        "supernode_connection":         "Supernode connection",
        "supernode_connection_reverse": "Supernode connection (rev)",
        "virtual":                      "Virtual edge",
        "left_operand":                 "Left operand",
        "right_operand":                "Right operand",
    }
    for etype in edge_groups:
        family = _edge_family(etype)
        if family not in _edge_legend_shown:
            _edge_legend_shown.add(family)
            color = EDGE_TYPE_COLORS.get(family, EDGE_TYPE_COLORS["unknown"])
            label = edge_legend_labels.get(family, family)
            legend_handles.append(mlines.Line2D([], [], color=color, linewidth=1.5, label=label))

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=7,
            framealpha=0.85,
            ncol=2 if len(legend_handles) > 6 else 1,
        )

    # --- save ---------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{mode}"
    formats_to_save: list[str] = []
    if output_format.lower() in ("pdf", "both"):
        formats_to_save.append("pdf")
    if output_format.lower() in ("svg", "both"):
        formats_to_save.append("svg")

    for fmt in formats_to_save:
        dest = os.path.join(output_dir, f"{graph_id}{suffix}_plot.{fmt}")
        plt.savefig(dest, format=fmt, bbox_inches="tight", transparent=True, dpi=200)
        logger.info(f"Saved: {dest}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific visualization of AST expression graphs from graphs.json"
    )
    parser.add_argument(
        "--graphs-file", default="datasets/graphs/graphs.json",
        help="Path to graphs.json (default: datasets/graphs/graphs.json).",
    )
    parser.add_argument(
        "--graph-id", nargs="*", default=None,
        help="One or more graph IDs to visualize (e.g. P1 P2). Default: all.",
    )
    parser.add_argument(
        "--mode", default="graph",
        choices=["tree", "tree_derivatives", "graph"],
        help=(
            "Graph structure mode. "
            "'tree': f only (no global). "
            "'tree_derivatives'/'graph': f+f'+f'' merged via global."
        ),
    )
    parser.add_argument(
        "--kappas-dir", default="datasets/kappas",
        help="Path to kappas directory (default: datasets/kappas).",
    )
    parser.add_argument(
        "--kappa-value", type=float, default=None,
        help="Merge this kappa h-function into the graph (requires --kappas-dir). "
             "E.g. --kappa-value -5.0",
    )
    parser.add_argument(
        "--virtual-supernode", action="store_true",
        help="Add a virtual supernode connected to all other nodes.",
    )
    parser.add_argument(
        "--output-dir", default="output_plots",
        help="Directory for output plots (default: output_plots).",
    )
    parser.add_argument(
        "--format", default="pdf", choices=["pdf", "svg", "both"],
        help="Output vector format (default: pdf).",
    )
    parser.add_argument(
        "--layout", default="spring",
        choices=["spring", "kamada_kawai", "spectral", "shell"],
        help="Layout algorithm (default: spring).",
    )
    parser.add_argument(
        "--color-by", default="type",
        choices=["type", "degree", "betweenness", "closeness", "eigenvector"],
        help=(
            "Node color strategy. 'type': semantic node-type palette. "
            "Others: centrality-based colormap (use --colormap to choose)."
        ),
    )
    parser.add_argument(
        "--colormap", default="viridis",
        help="Matplotlib colormap for centrality coloring (default: viridis).",
    )
    parser.add_argument(
        "--node-scale", type=float, default=2500.0,
        help="Base scaling factor for node size (default: 2500).",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Suppress node labels.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for layout reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Resolve graphs file path
    graphs_file = args.graphs_file
    if not os.path.isabs(graphs_file):
        # Try relative to CWD first, then relative to this script's repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidate = os.path.join(repo_root, graphs_file)
        if not os.path.exists(graphs_file) and os.path.exists(candidate):
            graphs_file = candidate
    if not os.path.exists(graphs_file):
        logger.error(f"graphs.json not found: {graphs_file}")
        sys.exit(1)

    # Enumerate available IDs
    with open(graphs_file, encoding="utf-8") as f:
        all_data = json.load(f)
    all_ids = [e["id"] for e in (all_data if isinstance(all_data, list) else [all_data])]

    graph_ids = args.graph_id if args.graph_id else all_ids
    unknown = [gid for gid in graph_ids if gid not in all_ids]
    if unknown:
        logger.error(f"Unknown graph IDs: {unknown}. Available: {all_ids}")
        sys.exit(1)

    kappas_dir = args.kappas_dir
    if not os.path.isabs(kappas_dir):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidate = os.path.join(repo_root, kappas_dir)
        if not os.path.exists(kappas_dir) and os.path.exists(candidate):
            kappas_dir = candidate

    logger.info(
        f"Visualizing {len(graph_ids)} graph(s): {graph_ids} "
        f"| mode={args.mode} | kappa={args.kappa_value} "
        f"| supernode={args.virtual_supernode} | color_by={args.color_by}"
    )

    for gid in graph_ids:
        try:
            entry = _load_entry(graphs_file, gid)

            if args.kappa_value is not None and args.mode != "tree":
                G = _add_kappa(None, graphs_file, gid, kappas_dir, args.kappa_value)
            elif args.mode == "tree":
                G = _build_tree(entry)
            else:
                G = _build_tree_derivatives(graphs_file, gid)

            if args.virtual_supernode:
                _add_virtual_supernode(G)

            mode_label = args.mode
            if args.kappa_value is not None:
                mode_label += f"_kappa{args.kappa_value:g}"
            if args.virtual_supernode:
                mode_label += "_supernode"

            visualize_graph(
                G=G,
                graph_id=gid,
                output_dir=args.output_dir,
                mode=mode_label,
                output_format=args.format,
                layout_name=args.layout,
                color_by=args.color_by,
                colormap_name=args.colormap,
                node_scale=args.node_scale,
                seed=args.seed,
                show_labels=not args.no_labels,
            )
        except Exception as e:
            logger.error(f"Failed to visualize {gid}: {e}", exc_info=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()
