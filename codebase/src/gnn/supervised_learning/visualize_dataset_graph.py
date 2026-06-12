#!/usr/bin/env python3
"""
visualize_dataset_graph.py

A tool to visualize expression graphs from the dataset under different modes:
- graph: Full directed graph with ASTs for f, f', f'' and virtual aggregator nodes/connections.
- tree: AST representation of function f only.
- tree_derivatives: AST representations of f, f', and f'' connected via their roots.

Supports rendering to PDF, SVG, and GEXF (for Gephi).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

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
    compute_belongs_to_f,
    compute_belongs_to_d1,
    compute_belongs_to_d2,
    TopologicalFeatureExtractor,
    get_hetero_node_type
)


def build_networkx_graph(raw_dict, mode):
    """
    Constructs a NetworkX graph from raw graph dictionary matching the dataset loader's modes.
    """
    converter = ExpressionGraphConverter()
    raw = dict(raw_dict)
    
    # Check GraphML container vs legacy format
    if "graphml_f" in raw:
        nodes_f, edges_f = parse_graphml_to_nodes_and_edges(raw.get("graphml_f", ""), "f")
        
        if mode in ["tree_derivatives", "graph"]:
            nodes_d1, edges_d1 = parse_graphml_to_nodes_and_edges(raw.get("graphml_derivative1", ""), "d1")
            nodes_d2, edges_d2 = parse_graphml_to_nodes_and_edges(raw.get("graphml_derivative2", ""), "d2")
        else:
            nodes_d1, edges_d1 = [], []
            nodes_d2, edges_d2 = [], []
        
        combined_nodes = nodes_f + nodes_d1 + nodes_d2
        combined_nodes.insert(0, {
            "id": "global",
            "label": "GLOBAL",
            "type": "global",
            "value": None
        })
        
        roots_f = find_roots(nodes_f, edges_f)
        roots_d1 = find_roots(nodes_d1, edges_d1)
        roots_d2 = find_roots(nodes_d2, edges_d2)

        combined_edges = edges_f + edges_d1 + edges_d2
        if roots_f:
            combined_nodes.append({
                "id": "f_root", "label": "f_root", "type": "f_root", "value": None
            })
            combined_edges.append({"source": "f_root", "target": "global", "type": "belongs_to_f"})
            for root in roots_f:
                combined_edges.append({"source": root, "target": "f_root", "type": "child_of"})
        if roots_d1:
            combined_nodes.append({
                "id": "d1_root", "label": "d1_root", "type": "d1_root", "value": None
            })
            combined_edges.append({"source": "d1_root", "target": "global", "type": "belongs_to_d1"})
            for root in roots_d1:
                combined_edges.append({"source": root, "target": "d1_root", "type": "child_of"})
        if roots_d2:
            combined_nodes.append({
                "id": "d2_root", "label": "d2_root", "type": "d2_root", "value": None
            })
            combined_edges.append({"source": "d2_root", "target": "global", "type": "belongs_to_d2"})
            for root in roots_d2:
                combined_edges.append({"source": root, "target": "d2_root", "type": "child_of"})

        raw["nodes"] = combined_nodes
        raw["edges"] = combined_edges
    else:
        raw["nodes"] = list(raw.get("nodes", []))
        raw["edges"] = list(raw.get("edges", []))
        from gnn.shared.utils.graph_utils import _insert_function_aggregators
        _insert_function_aggregators(raw)
        
    belongs_to_f_map = compute_belongs_to_f(raw)
    belongs_to_d1_map = compute_belongs_to_d1(raw)
    belongs_to_d2_map = compute_belongs_to_d2(raw)

    # Build Directed Graph with node attributes
    G = converter._build_networkx(
        raw,
        belongs_to_f_map=belongs_to_f_map,
        belongs_to_d1_map=belongs_to_d1_map,
        belongs_to_d2_map=belongs_to_d2_map,
    )
    
    # Store clean attributes for exporters/drawers
    for u, v in list(G.edges):
        # Resolve edge labels
        edge_data = raw_dict.get("edges", [])
        etype = "child_of"
        for e in edge_data:
            if e["source"] == u and e["target"] == v:
                etype = e.get("type", "child_of")
                break
        G.edges[u, v]["etype"] = etype

        # Reverse edge (single schema always has bidirectional AST edges)
        G.add_edge(v, u, etype=etype + "_reverse")

    return G


def compute_hierarchical_layout(G):
    """
    Computes a clean hierarchical layout for expression trees:
    - Vertical position (y) is node depth (inverted so root is at top).
    - Horizontal position (x) separates f, f', and f'' subtrees.
    """
    # Exclude global and virtual nodes to extract pure AST topology
    ast_nodes = [
        node for node in G.nodes 
        if node not in ["global", "f_root", "d1_root", "d2_root", "virtual_current_x", "virtual_y_target", "virtual_supernode"]
    ]
    
    # Build clean directed graph for depth extraction (filtering out reverse edges and reversing directions to get top-down depth)
    G_ast_clean = nx.DiGraph()
    G_ast_clean.add_nodes_from(ast_nodes)
    for u, v, d in G.edges(data=True):
        if u in ast_nodes and v in ast_nodes:
            if "reverse" not in d.get("etype", ""):
                G_ast_clean.add_edge(v, u)
    
    if len(G_ast_clean) > 0:
        topo = TopologicalFeatureExtractor.extract_and_annotate(G_ast_clean)
        depths = topo["depths"]
    else:
        depths = {}
        
    pos = {}
    
    # Group nodes by function category and depth level
    group_depth_nodes = {}
    for node in G.nodes:
        attrs = G.nodes[node]
        if node == "global":
            group = "global"
        elif node in ["f_root", "d1_root", "d2_root"]:
            group = node
        elif node in ["virtual_current_x", "virtual_y_target", "virtual_supernode"]:
            group = "virtual"
        elif attrs.get("belongs_to_f", 0.0) == 1.0:
            group = "f"
        elif attrs.get("belongs_to_d1", 0.0) == 1.0:
            group = "d1"
        elif attrs.get("belongs_to_d2", 0.0) == 1.0:
            group = "d2"
        else:
            group = "other"
            
        depth = depths.get(node, 0) if node in depths else 0
        key = (group, depth)
        if key not in group_depth_nodes:
            group_depth_nodes[key] = []
        group_depth_nodes[key].append(node)
        
    # Horizontal center of groups
    group_x_centers = {
        "global": 0.0,
        "f_root": -1.5,
        "d1_root": 0.0,
        "d2_root": 1.5,
        "f": -1.5,
        "d1": 0.0,
        "d2": 1.5,
        "virtual": 0.0,
        "other": 0.0
    }
    
    # Lay out nodes level by level
    for (group, depth), nodes in group_depth_nodes.items():
        center_x = group_x_centers.get(group, 0.0)
        num_nodes = len(nodes)
        
        if num_nodes == 1:
            pos[nodes[0]] = (center_x, -depth)
        else:
            # Spread nodes evenly horizontally
            width = 0.8
            xs = [center_x - width/2 + i * (width / (num_nodes - 1)) for i in range(num_nodes)]
            # Sort nodes by ID to keep layouts deterministic
            for node, x in zip(sorted(nodes), xs):
                pos[node] = (x, -depth)
                
    # Fine-tune vertical position of structural/virtual nodes
    if "global" in pos:
        pos["global"] = (0.0, 1.2)
    if "f_root" in pos:
        pos["f_root"] = (-1.5, 0.6)
    if "d1_root" in pos:
        pos["d1_root"] = (0.0, 0.6)
    if "d2_root" in pos:
        pos["d2_root"] = (1.5, 0.6)
    if "virtual_current_x" in pos:
        pos["virtual_current_x"] = (-2.8, -2.0)
    if "virtual_y_target" in pos:
        pos["virtual_y_target"] = (0.0, 2.0)
    if "virtual_supernode" in pos:
        pos["virtual_supernode"] = (2.8, -2.0)
        
    return pos


def visualize_graph(G, output_path, fmt, layout_name="hierarchical"):
    """
    Renders the NetworkX graph and saves it as PDF or SVG.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.axis("off")
    
    # 1. Compute Node Layout
    if layout_name == "hierarchical":
        pos = compute_hierarchical_layout(G)
    elif layout_name == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
        
    # Fill in any missing positions
    for node in G.nodes:
        if node not in pos:
            pos[node] = (0.0, 0.0)
            
    # 2. Extract Colors & Sizes
    node_colors = []
    node_sizes = []
    labels = {}
    is_hetero = (G.graph.get("mode") == "graph_hetero")
    
    for node in G.nodes:
        attrs = G.nodes[node]
        ntype = attrs.get("type", "")
        labels[node] = attrs.get("label") or str(node)
        
        # Determine node size based on its structural role
        if node == "global":
            size = 650
        elif node in ["f_root", "d1_root", "d2_root"]:
            size = 500
        elif ntype in ["virtual_current_x", "virtual_y_target", "virtual_supernode"]:
            size = 450
        else:
            size = 300
        node_sizes.append(size)
        
        if is_hetero:
            h_type = get_hetero_node_type(ntype)
            if h_type == "operator":
                node_colors.append("#3498db")  # Vibrant Blue
            elif h_type == "variable":
                node_colors.append("#2ecc71")  # Vibrant Green
            elif h_type == "constant":
                node_colors.append("#e67e22")  # Vibrant Orange
            else:  # virtual
                node_colors.append("#9b59b6")  # Vibrant Purple
        else:
            if node == "global" or node in ["f_root", "d1_root", "d2_root"] or ntype in ["virtual_current_x", "virtual_y_target", "virtual_supernode"]:
                node_colors.append("#9b59b6")  # Uniform Purple for all Virtual/Structural Nodes
            elif attrs.get("belongs_to_f", 0.0) == 1.0:
                node_colors.append("#2ecc71")  # Vibrant Green (f)
            elif attrs.get("belongs_to_d1", 0.0) == 1.0:
                node_colors.append("#3498db")  # Vibrant Blue (f')
            elif attrs.get("belongs_to_d2", 0.0) == 1.0:
                node_colors.append("#e67e22")  # Vibrant Orange (f'')
            else:
                node_colors.append("#bdc3c7")  # Silver

    # 3. Categorize and color edges
    ast_edges = []
    virtual_edges = []
    reverse_edges = []
    
    for u, v in G.edges:
        etype = G.edges[u, v].get("etype", "child_of")
        # Any edge connected to a virtual node is categorized as a virtual edge
        if u in ["virtual_current_x", "virtual_y_target", "virtual_supernode"] or \
           v in ["virtual_current_x", "virtual_y_target", "virtual_supernode"]:
            virtual_edges.append((u, v))
        elif "reverse" in etype:
            reverse_edges.append((u, v))
        else:
            ast_edges.append((u, v))
            
    # Draw AST edges (solid, soft dark grey)
    nx.draw_networkx_edges(
        G, pos, edgelist=ast_edges, ax=ax, edge_color="#7f8c8d",
        alpha=0.8, width=1.2, arrows=True, arrowsize=10
    )
    
    # Draw Virtual coupling edges (very thin, dashed, transparent purple)
    nx.draw_networkx_edges(
        G, pos, edgelist=virtual_edges, ax=ax, edge_color="#9b59b6",
        alpha=0.15, width=0.5, style="dashed", arrows=True, arrowsize=5
    )
    
    # Draw Reverse message-passing edges (dotted, very light grey)
    nx.draw_networkx_edges(
        G, pos, edgelist=reverse_edges, ax=ax, edge_color="#d5dbdb",
        alpha=0.4, width=0.8, style="dotted", arrows=True, arrowsize=6
    )
    
    # 4. Draw Nodes with thin border
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors="#2c3e50", linewidths=0.8
    )
    
    # 5. Draw Labels only for virtual/structural nodes, offset slightly above
    virtual_labels = {}
    for node in G.nodes:
        attrs = G.nodes[node]
        ntype = attrs.get("type", "")
        if node == "global" or node in ["f_root", "d1_root", "d2_root"] or ntype in ["virtual_current_x", "virtual_y_target", "virtual_supernode"]:
            virtual_labels[node] = attrs.get("label") or str(node)
    label_pos = {node: (x, y + 0.12) for node, (x, y) in pos.items() if node in virtual_labels}
    
    nx.draw_networkx_labels(
        G, label_pos, labels=virtual_labels, ax=ax, font_size=8,
        font_family="sans-serif", font_color="black", font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
    )
    
    # 6. Add Color-Coordinated Legend
    from matplotlib.patches import Patch
    if is_hetero:
        legend_elements = [
            Patch(facecolor="#3498db", edgecolor="#2980b9", label="Operator"),
            Patch(facecolor="#2ecc71", edgecolor="#27ae60", label="Variable"),
            Patch(facecolor="#e67e22", edgecolor="#d35400", label="Constant"),
            Patch(facecolor="#9b59b6", edgecolor="#8e44ad", label="Virtual"),
        ]
    else:
        legend_elements = [
            Patch(facecolor="#2ecc71", edgecolor="#27ae60", label="Function f(x)"),
            Patch(facecolor="#3498db", edgecolor="#2980b9", label="1st Derivative f'(x)"),
            Patch(facecolor="#e67e22", edgecolor="#d35400", label="2nd Derivative f''(x)"),
            Patch(facecolor="#9b59b6", edgecolor="#8e44ad", label="Virtual / Structural"),
        ]
    leg = ax.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#bdc3c7",
        fontsize=8
    )
    
    # Add title
    plt.title(f"Expression Graph (Mode: {G.graph.get('mode', 'N/A')}, ID: {G.graph.get('id', 'N/A')})",
              fontsize=12, fontweight="bold", pad=15)
    
    # Save the output
    plt.savefig(
        output_path,
        format=fmt,
        bbox_inches="tight",
        bbox_extra_artists=(leg,),
        transparent=True,
        dpi=300
    )
    plt.close(fig)
    print(f"[Visualizer] Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize expression graphs from dataset.")
    parser.add_argument("--dataset-name", "-d", type=str, default="graphs",
                        help="Dataset name (e.g. 'graphs', 'synthetic_graphs', or 'run_XXXX_XXXX/graphs')")
    parser.add_argument("--graph-id", "-i", type=str, default="0",
                        help="ID of the graph to visualize (e.g. '0', '1')")
    parser.add_argument("--mode", "-m", type=str, default="graph",
                        choices=["graph", "tree", "tree_derivatives", "graph_bidirectional", "graph_hetero"],
                        help="Graph mode layout representation")
    parser.add_argument("--format", "-f", type=str, default="pdf",
                        choices=["pdf", "svg", "gexf", "png", "all"],
                        help="Output format (pdf, svg, gexf for Gephi, png, or all)")
    parser.add_argument("--output-dir", "-o", type=str, default="visualizations",
                        help="Subdirectory to save the visualization results")
    parser.add_argument("--layout", type=str, default="hierarchical",
                        choices=["hierarchical", "spring", "kamada_kawai"],
                        help="Graph layout algorithm (hierarchical matches f/f'/f'')")
    parser.add_argument("--is-synthetic", action="store_true",
                        help="Set flag if loading a synthetic dataset file")
    args = parser.parse_args()
    
    # Map visualizer modes to loader configurations
    if args.mode in ("graph", "graph_bidirectional", "graph_hetero"):
        loader_mode = "graph"
    elif args.mode in ("tree", "tree_derivatives"):
        loader_mode = args.mode
    else:
        loader_mode = args.mode

    # Initialize unified graph loader
    loader = GraphDataLoader(
        name=args.dataset_name,
        mode=loader_mode,
        is_synthetic=args.is_synthetic or "synthetic" in args.dataset_name
    )
    
    if not loader.has_graph(args.graph_id):
        print(f"Error: Graph ID '{args.graph_id}' not found in dataset '{args.dataset_name}'.")
        print(f"Available Graph IDs: {sorted(list(loader.list_graph_ids()))[:20]}...")
        sys.exit(1)
        
    # Retrieve raw graph JSON
    raw_val = loader._raw_sources[args.graph_id]
    if isinstance(raw_val, Path):
        with open(raw_val, "r", encoding="utf-8") as file:
            raw_dict = json.load(file)
    else:
        raw_dict = raw_val
        
    # Build the NetworkX graph with mode layout
    G = build_networkx_graph(raw_dict, loader_mode)
    G.graph["mode"] = args.mode
    G.graph["id"] = args.graph_id
    
    # Resolve output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"graph_{args.graph_id}_{args.mode}"
    
    # Save selected formats
    formats = [args.format] if args.format != "all" else ["pdf", "svg", "gexf", "png"]
    
    for fmt in formats:
        filepath = out_dir / f"{base_filename}.{fmt}"
        if fmt == "gexf":
            # Strip networkx unsupported edge attributes or convert to string for GEXF export
            G_export = G.copy()
            for u, v, data in G_export.edges(data=True):
                if "relation_type" in data:
                    data["relation_type"] = str(data["relation_type"])
                if "child_index" in data:
                    data["child_index"] = str(data["child_index"])
                    
            # Export to Gephi GEXF format
            nx.write_gexf(G_export, str(filepath))
            print(f"[Visualizer] Gephi GEXF exported: {filepath}")
        else:
            visualize_graph(G, str(filepath), fmt, args.layout)


if __name__ == "__main__":
    main()
