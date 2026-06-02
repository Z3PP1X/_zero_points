#!/usr/bin/env python3
"""
visualize_graphs.py

Automated high-quality scientific visualization of AST GraphML datasets.
Reads graph files, applies network metrics (e.g., Centrality),
and generates publication-ready vector graphics (PDF/SVG) adhering to strict
academic design guidelines.

Usage:
    python visualize_graphs.py --input-dir graphs/ --output-dir output_plots/ --format pdf
"""

import os
import sys
import glob
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def compute_centrality(G: nx.Graph, metric: str) -> dict:
    """
    Computes node centrality based on the selected metric.
    
    Args:
        G: The NetworkX graph object.
        metric: Metric choice ('degree', 'betweenness', 'closeness', 'eigenvector').
        
    Returns:
        A dictionary mapping node ID to its centrality score.
    """
    if metric == "degree":
        return nx.degree_centrality(G)
    elif metric == "betweenness":
        # Using a fallback for very large graphs if computational speed is a concern
        return nx.betweenness_centrality(G)
    elif metric == "closeness":
        return nx.closeness_centrality(G)
    elif metric == "eigenvector":
        try:
            # Tolerant solver for eigenvector centrality in directed/undirected graphs
            return nx.eigenvector_centrality_numpy(G) if G.is_directed() else nx.eigenvector_centrality(G, max_iter=1000)
        except Exception as e:
            logger.warning(f"Eigenvector centrality failed to converge ({e}). Falling back to Degree Centrality.")
            return nx.degree_centrality(G)
    else:
        raise ValueError(f"Unknown centrality metric: {metric}")


def visualize_single_graph(
    file_path: str,
    output_dir: str,
    output_format: str = "pdf",
    layout_name: str = "spring",
    colormap_name: str = "viridis",
    metric_name: str = "degree",
    node_scale: float = 3000.0,
    min_node_size: float = 80.0,
    seed: int = 42
) -> None:
    """
    Reads a GraphML file, computes metrics, lays out the graph beautifully, 
    and saves a high-resolution vector image.
    
    Args:
        file_path: Absolute or relative path to the GraphML file.
        output_dir: Directory where the plotted graph will be saved.
        output_format: File format ('pdf', 'svg', or 'both').
        layout_name: Layout algorithm ('spring', 'kamada_kawai', 'spectral', 'shell').
        colormap_name: Matplotlib scientific colormap name ('viridis', 'plasma', 'inferno', etc.).
        metric_name: Centrality metric used for node coloring/scaling.
        node_scale: Scaling factor for node sizes.
        min_node_size: Minimum node size to keep small nodes visible.
        seed: Random seed for layout reproducibility.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    logger.info(f"Processing graph: {base_name} ...")
    
    try:
        # 1. Load Graph (with a hotfix for capitalized 'String' type in GraphML)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Hotfix: NetworkX expects lowercase 'string' type in graphml key definitions
        content = content.replace("attr.type='String'", "attr.type='string'")
        content = content.replace('attr.type="String"', 'attr.type="string"')
        
        G = nx.parse_graphml(content)
        if len(G) == 0:
            logger.warning(f"Graph in {file_path} is empty. Skipping.")
            return
            
        # Standardize node names to ensure clean handling
        # Convert directed to undirected for cleaner layout results if preferred, but keep directed edges.
        is_directed = G.is_directed()
        
        # 2. Compute Layout with fixed seed for reproducibility
        if layout_name == "spring":
            pos = nx.spring_layout(G, seed=seed, k=1.0 / (len(G) ** 0.5) if len(G) > 0 else 0.1)
        elif layout_name == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except Exception as e:
                logger.warning(f"Kamada-Kawai layout failed ({e}). Falling back to Spring layout.")
                pos = nx.spring_layout(G, seed=seed)
        elif layout_name == "spectral":
            pos = nx.spectral_layout(G)
        elif layout_name == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed)
            
        # 3. Compute scientific centrality metric for nodes
        centrality = compute_centrality(G, metric_name)
        
        # Map values to a list in node order
        nodes = list(G.nodes())
        node_colors = [centrality[node] for node in nodes]
        
        # Scale node sizes based on centrality
        node_sizes = [max(centrality[node] * node_scale, min_node_size) for node in nodes]
        
        # 4. Setup Plot & Figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.axis("off")
        
        # Eliminate NetworkX standard bold colors: W3C compliant grey and scientific colormap
        edge_color = "#999999" # Muted, soft grey
        
        # 5. Draw Edges (low alpha to avoid hairballs)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_color,
            alpha=0.25, # De-emphasizes edge overlap, highlights dense clusters
            width=0.8,
            arrows=is_directed,
            arrowsize=6
        )
        
        # 6. Draw Nodes with fine, dark borders
        cmap = plt.get_cmap(colormap_name)
        node_collection = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=nodes,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            edgecolors="#1a1a1a", # Subtle elegant node border
            linewidths=0.6
        )
        
        # 7. Draw Labels dynamically
        # Rule: Label everything if nodes < 20; otherwise, only top-5 central nodes.
        label_dict = {}
        if len(G) < 20:
            for node in G.nodes():
                # Use GraphML 'label' or 'NodeLabel' attribute if available
                node_label = G.nodes[node].get("label") or G.nodes[node].get("NodeLabel") or str(node)
                label_dict[node] = node_label
        else:
            # Sort nodes by centrality descending and choose top 5
            sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
            top_nodes = [node for node, score in sorted_nodes[:5]]
            for node in top_nodes:
                node_label = G.nodes[node].get("label") or G.nodes[node].get("NodeLabel") or str(node)
                label_dict[node] = node_label
                
        # Draw the filtered labels with a clean sans-serif font
        nx.draw_networkx_labels(
            G,
            pos,
            labels=label_dict,
            ax=ax,
            font_size=8,
            font_family="sans-serif",
            font_weight="bold",
            alpha=0.9
        )
        
        # 8. Add Scientific Context (Colorbar)
        # Explains the scientific metric colors directly on the plot
        cbar = plt.colorbar(
            ScalarMappable(norm=mcolors.Normalize(vmin=min(node_colors), vmax=max(node_colors)), cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.03,
            shrink=0.7,
            aspect=40
        )
        metric_label = f"{metric_name.capitalize()} Centrality"
        cbar.set_label(metric_label, fontsize=9, family="sans-serif", weight="bold", labelpad=5)
        cbar.ax.tick_params(labelsize=8)
        
        # 9. Save as high-quality Vector Graphics
        os.makedirs(output_dir, exist_ok=True)
        
        formats_to_save = []
        if output_format.lower() in ("pdf", "both"):
            formats_to_save.append("pdf")
        if output_format.lower() in ("svg", "both"):
            formats_to_save.append("svg")
            
        for fmt in formats_to_save:
            dest_path = os.path.join(output_dir, f"{base_name}_plot.{fmt}")
            plt.savefig(
                dest_path,
                format=fmt,
                bbox_inches="tight", # Strict margin bounding box
                transparent=True,
                dpi=300
            )
            logger.info(f"Saved: {dest_path}")
            
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Failed to visualize graph {file_path}: {e}", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific Vector Visualization of GraphML AST Datasets"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="graphs",
        help="Path to directory containing .graphml files (or subdirectory structure)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_plots",
        help="Path to output directory for storing generated vector graphics."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "svg", "both"],
        help="Vector output format (pdf, svg, or both)."
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "kamada_kawai", "spectral", "shell"],
        help="Layout algorithm choice."
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Scientific colormap name (e.g. viridis, plasma, inferno, magma, cividis)."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="degree",
        choices=["degree", "betweenness", "closeness", "eigenvector"],
        help="Centrality metric used for scaling and node coloring."
    )
    parser.add_argument(
        "--node-scale",
        type=float,
        default=2500.0,
        help="Scaling factor for node size."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for layout reproducibility."
    )
    args = parser.parse_args()

    # Locate all .graphml files recursively under input_dir
    search_pattern = os.path.join(args.input_dir, "**", "*.graphml")
    graph_files = glob.glob(search_pattern, recursive=True)
    
    if not graph_files:
        # Fallback to direct folder search without recursion if input-dir itself is a direct path
        search_pattern = os.path.join(args.input_dir, "*.graphml")
        graph_files = glob.glob(search_pattern)

    if not graph_files:
        logger.error(f"No GraphML files found in directory: {args.input_dir} using pattern {search_pattern}")
        sys.exit(1)

    logger.info(f"Found {len(graph_files)} GraphML file(s) for visualization.")
    logger.info(f"Configuration: layout={args.layout}, metric={args.metric}, colormap={args.colormap}, format={args.format}")

    for file_path in graph_files:
        # Determine the relative subdirectory to maintain folder hierarchy
        relative_path = os.path.relpath(file_path, args.input_dir)
        relative_dir = os.path.dirname(relative_path)
        target_output_dir = os.path.join(args.output_dir, relative_dir)

        visualize_single_graph(
            file_path=file_path,
            output_dir=target_output_dir,
            output_format=args.format,
            layout_name=args.layout,
            colormap_name=args.colormap,
            metric_name=args.metric,
            node_scale=args.node_scale,
            seed=args.seed
        )

    logger.info("Batch graph visualization complete. All plots generated successfully!")


if __name__ == "__main__":
    main()
