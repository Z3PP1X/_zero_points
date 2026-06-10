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
    FUNCTION_AGGREGATOR_IDS,
    TopologicalFeatureExtractor
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
            combined_edges.append({"source": "global", "target": "f_root", "type": "belongs_to_f"})
            for root in roots_f:
                combined_edges.append({"source": "f_root", "target": root, "type": "child_of"})
        if roots_d1:
            combined_nodes.append({
                "id": "d1_root", "label": "d1_root", "type": "d1_root", "value": None
            })
            combined_edges.append({"source": "global", "target": "d1_root", "type": "belongs_to_d1"})
            for root in roots_d1:
                combined_edges.append({"source": "d1_root", "target": root, "type": "child_of"})
        if roots_d2:
            combined_nodes.append({
                "id": "d2_root", "label": "d2_root", "type": "d2_root", "value": None
            })
            combined_edges.append({"source": "global", "target": "d2_root", "type": "belongs_to_d2"})
            for root in roots_d2:
                combined_edges.append({"source": "d2_root", "target": root, "type": "child_of"})

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
    
    if mode == "graph":
        # Find variable nodes and global node
        variable_node_ids = []
        global_node_id = None
        for node in raw["nodes"]:
            if node.get("type") == "variable":
                variable_node_ids.append(node["id"])
            elif node.get("type") == "global":
                global_node_id = node["id"]
        
        # Add task virtual nodes
        raw["nodes"].append({
            "id": "virtual_current_x",
            "label": "virtual_current_x",
            "type": "virtual_current_x",
            "value": None
        })
        raw["nodes"].append({
            "id": "virtual_y_target",
            "label": "virtual_y_target",
            "type": "virtual_y_target",
            "value": None
        })
        raw["nodes"].append({
            "id": "virtual_supernode",
            "label": "virtual_supernode",
            "type": "virtual_supernode",
            "value": None
        })

        existing_aggregators = [
            agg_id for agg_id in FUNCTION_AGGREGATOR_IDS
            if any(node["id"] == agg_id for node in raw["nodes"])
        ]

        # virtual_current_x -> all variables
        for var_id in variable_node_ids:
            raw["edges"].append({
                "source": "virtual_current_x",
                "target": var_id,
                "type": "virtual"
            })
        # virtual_current_x <-> per-function aggregators
        for agg_id in existing_aggregators:
            raw["edges"].append({
                "source": "virtual_current_x",
                "target": agg_id,
                "type": "virtual"
            })
            raw["edges"].append({
                "source": agg_id,
                "target": "virtual_current_x",
                "type": "virtual"
            })
        # f_root <-> virtual_y_target
        if "f_root" in existing_aggregators:
            raw["edges"].append({
                "source": "f_root",
                "target": "virtual_y_target",
                "type": "virtual"
            })
            raw["edges"].append({
                "source": "virtual_y_target",
                "target": "f_root",
                "type": "virtual"
            })
        # Newton/Halley coupling between derivative aggregators
        for src, tgt in (("f_root", "d1_root"), ("d1_root", "d2_root"), ("f_root", "d2_root")):
            if src in existing_aggregators and tgt in existing_aggregators:
                raw["edges"].append({"source": src, "target": tgt, "type": "virtual"})
                raw["edges"].append({"source": tgt, "target": src, "type": "virtual"})
        # virtual_y_target -> global node
        if global_node_id is not None:
            raw["edges"].append({
                "source": "virtual_y_target",
                "target": global_node_id,
                "type": "virtual"
            })

    # Build Directed Graph with node attributes
    G = converter._build_networkx(
        raw,
        belongs_to_f_map=belongs_to_f_map,
        belongs_to_d1_map=belongs_to_d1_map,
        belongs_to_d2_map=belongs_to_d2_map,
    )
    
    # Store clean attributes for exporters/drawers
    for u, v in G.edges:
        # Resolve edge labels
        edge_data = raw_dict.get("edges", [])
        etype = "child_of"
        for e in edge_data:
            if e["source"] == u and e["target"] == v:
                etype = e.get("type", "child_of")
                break
        G.edges[u, v]["etype"] = etype

    # Add virtual supernode edges if virtual_supernode is present
    if "virtual_supernode" in G.nodes:
        for node in list(G.nodes):
            if node != "virtual_supernode":
                G.add_edge("virtual_supernode", node, etype="supernode_connection")
                G.add_edge(node, "virtual_supernode", etype="supernode_connection_reverse")
        
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
    G_ast = G.subgraph(ast_nodes)
    
    if len(G_ast) > 0:
        topo = TopologicalFeatureExtractor.extract_and_annotate(nx.DiGraph(G_ast), enrich=True)
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
    
    for node in G.nodes:
        attrs = G.nodes[node]
        ntype = attrs.get("type", "")
        labels[node] = attrs.get("label") or str(node)
        
        if node == "global" or node in ["f_root", "d1_root", "d2_root"] or ntype in ["virtual_current_x", "virtual_y_target", "virtual_supernode"]:
            node_colors.append("#9b59b6")  # Uniform Purple for all Virtual/Structural Nodes
            if node == "global":
                node_sizes.append(650)
            elif node in ["f_root", "d1_root", "d2_root"]:
                node_sizes.append(500)
            else:
                node_sizes.append(450)
        elif attrs.get("belongs_to_f", 0.0) == 1.0:
            node_colors.append("#2ecc71")  # Vibrant Green (f)
            node_sizes.append(300)
        elif attrs.get("belongs_to_d1", 0.0) == 1.0:
            node_colors.append("#3498db")  # Vibrant Blue (f')
            node_sizes.append(300)
        elif attrs.get("belongs_to_d2", 0.0) == 1.0:
            node_colors.append("#e67e22")  # Vibrant Orange (f'')
            node_sizes.append(300)
        else:
            node_colors.append("#bdc3c7")  # Silver
            node_sizes.append(300)

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
    
    # 5. Draw Labels
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_size=8,
        font_family="sans-serif", font_color="black"
    )
    
    # Add title and color legend context
    plt.title(f"Expression Graph (Mode: {G.graph.get('mode', 'N/A')}, ID: {G.graph.get('id', 'N/A')})",
              fontsize=12, fontweight="bold", pad=10)
    
    # Save the output
    plt.savefig(
        output_path,
        format=fmt,
        bbox_inches="tight",
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
                        choices=["graph", "tree", "tree_derivatives"],
                        help="Graph mode layout representation")
    parser.add_argument("--format", "-f", type=str, default="pdf",
                        choices=["pdf", "svg", "gexf", "all"],
                        help="Output format (pdf, svg, gexf for Gephi, or all)")
    parser.add_argument("--output-dir", "-o", type=str, default="visualizations",
                        help="Subdirectory to save the visualization results")
    parser.add_argument("--layout", type=str, default="hierarchical",
                        choices=["hierarchical", "spring", "kamada_kawai"],
                        help="Graph layout algorithm (hierarchical matches f/f'/f'')")
    parser.add_argument("--is-synthetic", action="store_true",
                        help="Set flag if loading a synthetic dataset file")
    args = parser.parse_args()
    
    # Initialize unified graph loader
    loader = GraphDataLoader(
        name=args.dataset_name,
        mode=args.mode,
        enrich=True,
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
    G = build_networkx_graph(raw_dict, args.mode)
    G.graph["mode"] = args.mode
    G.graph["id"] = args.graph_id
    
    # Resolve output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"graph_{args.graph_id}_{args.mode}"
    
    # Save selected formats
    formats = [args.format] if args.format != "all" else ["pdf", "svg", "gexf"]
    
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
