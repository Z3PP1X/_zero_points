import os
import sys
import warnings
import logging

# Suppress warnings and matplotlib's internal log messages (like missing fonts)
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

class GNNResultEvaluator:
    """
    Evaluates GNN training, test and validation runs, plotting pivot grid heatmaps
    of performance metrics and comparing layer summaries.
    
    In synthetic mode:
      - train_*: Training on synthetic data
      - test_*:  Test on unseen synthetic data (problem_id-based split)
      - val_*:   Validation on curated real-world problems
    """
    def __init__(self, naming_var: str, base_dir: Path = None):
        """
        Args:
            naming_var: Folder name for the evaluation (e.g., 'res_with_enrich' or 'res_without_enrich')
            base_dir: Base directory where run_results lies (defaults to parent of this script)
        """
        self.naming_var = naming_var
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = Path(base_dir)
            
        self.data_dir = self.base_dir / self.naming_var / "agg"
        self.output_dir = self.base_dir / self.naming_var / "eval_plots"
        
        # Datasets to evaluate, grouped by purpose
        # train: Training performance (synthetic data)
        # test:  Test performance (unseen synthetic data, problem_id-based split)  
        # val:   Validation performance (curated real-world problems)
        self.runs = [
            "train_best", "train_bestepoch", "train",
            "test_best", "test_bestepoch", "test",
            "val_best", "val_bestepoch", "val",
        ]
        
        # Human-readable labels for each run type
        self.run_labels = {
            "train": "Training (Synthetic)",
            "train_best": "Training Best (Synthetic)",
            "train_bestepoch": "Training Best Epoch (Synthetic)",
            "test": "Test (Unseen Synthetic)",
            "test_best": "Test Best (Unseen Synthetic)",
            "test_bestepoch": "Test Best Epoch (Unseen Synthetic)",
            "val": "Validation (Curated Real)",
            "val_best": "Validation Best (Curated Real)",
            "val_bestepoch": "Validation Best Epoch (Curated Real)",
        }
        
        # Heatmap colormap (white to premium emerald green)
        # Using a sleek color gradient
        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "premium_green", ["#FFFFFF", "#D1E7DD", "#0F5132"]
        )
        # Heatmap colormap for loss (white to premium red)
        # High loss = darker red
        self.cmap_loss = mcolors.LinearSegmentedColormap.from_list(
            "premium_red", ["#FFFFFF", "#F8D7DA", "#842029"]
        )
        
    def load_data(self, run_name: str) -> pd.DataFrame:
        """Loads the CSV file for a given run."""
        file_path = self.data_dir / f"{run_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        return pd.read_csv(file_path)

    def generate_plots_for_df(self, df: pd.DataFrame, overall_df: pd.DataFrame, output_path: Path, title: str):
        """
        Generates a unified plot containing a 2x4 grid of pivot heatmaps
        and a layer summary comparison bar chart.
        
        Args:
            df: The subset of the dataframe to plot (e.g. filtered by a hyperparameter)
            overall_df: The full dataframe for computing the layer summary
            output_path: Path to save the PNG
            title: Title of the visualization
        """
        # Create figure and gridspec
        # 2 rows, 7 columns: cols 0-5 for heatmaps, col 6 for layer summary bar chart
        fig = plt.figure(figsize=(32, 11), dpi=150)
        gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1.8], wspace=0.35, hspace=0.35)
        
        metrics = ['auc', 'pr_auc', 'loss', 'recall', 'f1', 'precision']
        aggfuncs = ['mean', 'max']
        
        # Plot heatmaps
        for row_idx, agg in enumerate(aggfuncs):
            for col_idx, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                # Check if we have enough data to pivot
                try:
                    pivot_grid = df.pivot_table(
                        values=metric,
                        index='dim_inner',
                        columns='dropout',
                        aggfunc=agg
                    )
                    
                    # Sort index and columns to ensure ascending order of hyperparameters
                    pivot_grid = pivot_grid.sort_index(ascending=True)
                    pivot_grid = pivot_grid.reindex(sorted(pivot_grid.columns), axis=1)
                    
                    # Compute min/max values for exact scaling bounds (ignoring NaNs)
                    min_val = pivot_grid.values[~np.isnan(pivot_grid.values)].min() if not pivot_grid.empty else 0
                    max_val = pivot_grid.values[~np.isnan(pivot_grid.values)].max() if not pivot_grid.empty else 1
                    
                    # Handle flat values where min == max
                    if min_val == max_val:
                        vmin_val = min_val - 0.05
                        vmax_val = max_val + 0.05
                    else:
                        vmin_val = min_val
                        vmax_val = max_val
                        
                    # Draw heatmap using imshow with explicit data-driven bounds
                    cmap_to_use = self.cmap_loss if metric == 'loss' else self.cmap
                    im = ax.imshow(pivot_grid.values, cmap=cmap_to_use, aspect='auto', origin='lower', vmin=vmin_val, vmax=vmax_val)
                    
                    # Set ticks and labels
                    ax.set_xticks(range(len(pivot_grid.columns)))
                    ax.set_xticklabels(pivot_grid.columns)
                    ax.set_yticks(range(len(pivot_grid.index)))
                    ax.set_yticklabels(pivot_grid.index)
                    
                    ax.set_xlabel('Dropout', fontsize=9, fontweight='bold', labelpad=4)
                    ax.set_ylabel('Dim Inner', fontsize=9, fontweight='bold', labelpad=4)
                    ax.set_title(f"{metric.upper()} ({agg.upper()})", fontsize=11, fontweight='bold', pad=10)
                    
                    # Remove minor tick marks and adjust styling
                    ax.tick_params(left=False, bottom=False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#cccccc')
                    ax.spines['bottom'].set_color('#cccccc')
                    
                    # Annotate cells with values
                    range_val = max(max_val - min_val, 1e-5)
                    for i in range(len(pivot_grid.index)):
                        for j in range(len(pivot_grid.columns)):
                            val = pivot_grid.values[i, j]
                            if not pd.isna(val):
                                # Determine text color for contrast (if it is close to green, text is white)
                                normalized_val = (val - min_val) / range_val
                                text_color = "white" if normalized_val > 0.6 else "#111111"
                                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='semibold')
                                
                    # Add small colorbar next to plot
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.outline.set_visible(False)
                except Exception as e:
                    ax.text(0.5, 0.5, f"No Data\n{str(e)}", ha="center", va="center", fontsize=8)
                    ax.set_title(f"{metric.upper()} ({agg.upper()}) - N/A", fontsize=10, fontweight='bold')
        
        # Plot Layer Summary Bar Chart on the right (spanning both rows)
        ax_summary = fig.add_subplot(gs[:, 6])
        
        # Compute layer summary comparison
        summary_metrics = ['auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1', 'loss']
        present_metrics = [m for m in summary_metrics if m in overall_df.columns]
        
        # Decide grouping column and title dynamically based on the current slice
        group_col = None
        legend_title = ""
        chart_title = ""
        
        if 'layer_type' in overall_df.columns:
            group_col = 'layer_type'
            legend_title = "Model Architecture"
            chart_title = "Architecture Comparison (Overall mean)"
            
        # If the slice is filtered to one layer_type, group by activation function instead if available
        if 'layer_type' in df.columns and df['layer_type'].nunique() == 1 and 'act' in overall_df.columns:
            group_col = 'act'
            legend_title = "Activation Function"
            chart_title = f"Activation Function Comparison for {df['layer_type'].iloc[0]}"
        # If the slice is filtered to one activation function, group by layer_type
        elif 'act' in df.columns and df['act'].nunique() == 1 and 'layer_type' in overall_df.columns:
            group_col = 'layer_type'
            legend_title = "Model Architecture"
            chart_title = f"Architecture Comparison for {df['act'].iloc[0]}"
        # If the slice is filtered to one graph_pooling, keep architecture grouping
        elif 'graph_pooling' in df.columns and df['graph_pooling'].nunique() == 1:
            if group_col is None and 'layer_type' in overall_df.columns:
                group_col = 'layer_type'
                legend_title = "Model Architecture"
                chart_title = f"Architecture Comparison for pooling={df['graph_pooling'].iloc[0]}"
            
        if group_col is not None and len(present_metrics) > 0:
            # Group the current slice's data (or overall_df if it's overall plot) to show comparison
            comparison_df = df if (df[group_col].nunique() > 1) else overall_df
            layer_summary = comparison_df.groupby(group_col)[present_metrics].mean()
            
            # Premium color palette using qualitative maps
            premium_palette = ['#2A9D8F', '#E76F51', '#264653', '#F4A261', '#E9C46A', '#457B9D', '#1D3557']
            num_groups = len(layer_summary)
            colors = [premium_palette[i % len(premium_palette)] for i in range(num_groups)]
            
            # Plot bar chart (transposed so metrics are on X-axis, layer types/activations are bars)
            layer_summary.T.plot(kind='bar', ax=ax_summary, width=0.8, color=colors, edgecolor='none')
            
            ax_summary.set_title(chart_title, fontsize=13, fontweight='bold', pad=12)
            ax_summary.set_xlabel("Performance Metrics", fontsize=10, fontweight='bold', labelpad=8)
            ax_summary.set_ylabel("Metric Score", fontsize=10, fontweight='bold', labelpad=8)
            ax_summary.set_ylim(0, 1.1)
            ax_summary.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Style legend
            ax_summary.legend(
                title=legend_title, 
                frameon=True, 
                facecolor='#f8f9fa', 
                edgecolor='none', 
                fontsize=9, 
                title_fontsize=10
            )
            
            # Style spines
            ax_summary.spines['top'].set_visible(False)
            ax_summary.spines['right'].set_visible(False)
            ax_summary.spines['left'].set_color('#cccccc')
            ax_summary.spines['bottom'].set_color('#cccccc')
            
            # Rotate x labels for better readability
            ax_summary.set_xticklabels([m.upper() for m in present_metrics], rotation=0, fontsize=10)
            
            # Add values above bars
            for p in ax_summary.patches:
                height = p.get_height()
                if height > 0:
                    ax_summary.annotate(f"{height:.3f}",
                                        xy=(p.get_x() + p.get_width() / 2, height),
                                        xytext=(0, 4),  # 4 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=8, rotation=90, fontweight='semibold')
        else:
            ax_summary.text(0.5, 0.5, "No Architecture Data Available", ha="center", va="center", fontsize=10)
                
        # Main Figure Title
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Save and close
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    def generate_summary_comparison(self, output_path: Path):
        """
        Generates a comparison plot across train/test/val splits showing
        best-epoch metrics side-by-side.
        """
        split_files = {
            "Train (Synthetic)": "train_bestepoch",
            "Test (Unseen Synthetic)": "test_bestepoch",
            "Val (Curated Real)": "val_bestepoch",
        }
        
        summary_metrics = ['auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1', 'loss']
        split_data = {}
        
        for label, filename in split_files.items():
            try:
                df = self.load_data(filename)
                present = [m for m in summary_metrics if m in df.columns]
                split_data[label] = df[present].mean()
            except FileNotFoundError:
                continue
        
        if len(split_data) < 2:
            return  # Not enough splits to compare
        
        summary_df = pd.DataFrame(split_data)
        present_metrics = list(summary_df.index)
        
        fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
        
        premium_palette = ['#2A9D8F', '#E76F51', '#264653']
        colors = [premium_palette[i % len(premium_palette)] for i in range(len(summary_df.columns))]
        
        summary_df.plot(kind='bar', ax=ax, width=0.75, color=colors, edgecolor='none')
        
        ax.set_title("Train / Test / Validation Comparison (Best Epoch Avg)", fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel("Performance Metrics", fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel("Metric Score", fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylim(0, max(1.1, summary_df.max().max() * 1.1))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(
            title="Data Split",
            frameon=True,
            facecolor='#f8f9fa',
            edgecolor='none',
            fontsize=10,
            title_fontsize=11
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.set_xticklabels([m.upper() for m in present_metrics], rotation=0, fontsize=10)
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            xy=(p.get_x() + p.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90, fontweight='semibold')
        
        plt.suptitle(f"Split Comparison — {self.naming_var}", fontsize=17, fontweight='bold', y=1.01)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved split comparison plot: {output_path}")

    def run_all(self):
        """Runs the entire evaluation pipeline for all runs and slices."""
        print(f"Starting GNN Evaluation on '{self.naming_var}'...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        for run in self.runs:
            label = self.run_labels.get(run, run)
            print(f"  Evaluating run: {run} ({label})...")
            try:
                df = self.load_data(run)
            except FileNotFoundError as e:
                print(f"    Skipping {run} (File not found)")
                continue
                
            run_out_dir = self.output_dir / run
            run_out_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Overall Diagram
            self.generate_plots_for_df(
                df=df,
                overall_df=df,
                output_path=run_out_dir / "overall.png",
                title=f"{label} - Overall Hyperparameter Grid ({self.naming_var})"
            )
            
            # 2. Diagrams by observed layer_type
            if 'layer_type' in df.columns:
                layer_types = df['layer_type'].dropna().unique()
                layer_dir = run_out_dir / "layer_type"
                for lt in layer_types:
                    sub_df = df[df['layer_type'] == lt]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=df,
                            output_path=layer_dir / f"{lt}.png",
                            title=f"{label} - Layer: {lt} ({self.naming_var})"
                        )
            
            # 3. Diagrams by observed layers_mp
            if 'layers_mp' in df.columns:
                mp_values = df['layers_mp'].dropna().unique()
                mp_dir = run_out_dir / "layers_mp"
                for mp in mp_values:
                    sub_df = df[df['layers_mp'] == mp]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=df,
                            output_path=mp_dir / f"{mp}_layers.png",
                            title=f"{label} - MP Layers: {mp} ({self.naming_var})"
                        )

            # 4. Diagrams by observed act (activation function)
            if 'act' in df.columns:
                act_values = df['act'].dropna().unique()
                act_dir = run_out_dir / "act"
                for act in act_values:
                    sub_df = df[df['act'] == act]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=df,
                            output_path=act_dir / f"{act}.png",
                            title=f"{label} - Activation: {act} ({self.naming_var})"
                        )

            # 5. Diagrams by observed graph_pooling
            if 'graph_pooling' in df.columns:
                pooling_values = df['graph_pooling'].dropna().unique()
                pooling_dir = run_out_dir / "graph_pooling"
                for pooling in pooling_values:
                    sub_df = df[df['graph_pooling'] == pooling]
                    if not sub_df.empty:
                        self.generate_plots_for_df(
                            df=sub_df,
                            overall_df=df,
                            output_path=pooling_dir / f"{pooling}.png",
                            title=f"{label} - Pooling: {pooling} ({self.naming_var})"
                        )
                        
        # 6. Generate cross-split comparison plot
        print("  Generating split comparison plot...")
        self.generate_summary_comparison(
            output_path=self.output_dir / "split_comparison.png"
        )

        print(f"Evaluation complete! Plots saved to {self.output_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        naming_var = sys.argv[1]
    else:
        naming_var = input("Enter naming var (e.g. res_with_enrich): ").strip()
        if not naming_var:
            naming_var = "res_with_enrich"
            
    evaluator = GNNResultEvaluator(naming_var)
    evaluator.run_all()
