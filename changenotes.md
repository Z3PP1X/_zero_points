# Change Notes - GNN Backbone Enhancements and Edge Feature Projection

This document outlines the modifications made to project edge features to the node base and to implement structural virtual node enhancements to prevent oversmoothing in Graph Neural Network (GNN) architectures.

---

## 1. Node-Projected Edge Features

To enable GNN backbones that do not support edge features to leverage topological and connection properties, we projected key edge characteristics into the node base:
- **`node_child_index`**: The sequential index of a node as a child of its parent (defaults to `0.0` for root nodes).
- **`parent_edge_betweenness`**: The edge betweenness centrality of the edge connecting a node to its parent (taking the maximum betweenness in case of multiple parent edges in a DAG, defaults to `0.0` for root nodes).
- **`parent_relation_type`**: The float-encoded edge relation type of the connection to the parent node (defaults to `0.0` for root nodes).

**Integration & Dimension Updates:**
- `NATIVE_NODE_FEATURE_COUNT` in `feature_layout.py` has been increased from **19** to **22**.
- `Preprocessor` in `preprocessor.py` has been updated to expect **22** features in `enrich=True` mode, keeping virtual node state slots at their correct positions.

---

## 2. Virtual Nodes to Prevent Oversmoothing

To prevent node representations from converging to indistinguishable vectors over deep architectures, the following virtual node mechanisms have been embedded:

### Task 1: Split Pooling
- Replaced the single `global_mean_pool` over all nodes with a split pooling strategy.
- Created a boolean mask to separate real nodes and virtual nodes (types 5, 6, and 7).
- Pooled real and virtual nodes independently using `global_mean_pool`, and concatenated the pooled representations.
- Doubled the pooling representation size to `2 * hidden_dim` and updated downstream dimension expectations/linear layer inputs in `GraphPolicyBackbone` and `TestGraphNetwork`.

### Task 2: Dedicated Global Supernode
- Introduced a dedicated global supernode (type 8) connected bidirectionally to every node in the graph.
- Embedded the supernode at the very end of edge/node creation in `graph_utils.py` to prevent modifying underlying topological features (depths, heights, edge/node betweenness, LPE/RWPE).

### Task 3: Content-Aware Virtual Node Seeding
- Initialized virtual nodes at GNN construction time with content-aware priors:
  - **Type 5 (`virtual_current_x`)**: Initialized via a learned linear projection of the scalar `currentX` RL state feature.
  - **Type 6 (`virtual_f_x`)**: Initialized using the sum/mean of all function/operator node embeddings.
  - **Type 7 (`virtual_y_target`)**: Initialized via a learned linear projection of the scalar `yTarget` RL state feature.
  - **Type 8 (`virtual_supernode`)**: Initialized using the mean of all node embeddings in the graph.

### Task 4: Between-Layer MLP Updates & Broadcasting
- Added layer-specific update MLPs for virtual node embeddings between convolutional layers.
- Broadcasted the updated global supernode (type 8) to all real nodes by adding its embedding to theirs.

---

## 3. Test Coverage & Verification

- **Edge-to-Node verification**: Updated `test_graph_utils.py`, `test_virtual_nodes.py`, and `test_trial_switch.py` to verify the correct computation and projection of edge features and features dimension (22).
- **Node Count Adjustments**: Updated expected node count assertions in `test_graphml_import.py` and `test_virtual_nodes.py` (from 9 to 10 and 7 to 8 respectively) to account for the new global supernode (type 8).
- **Full Verification**: Ran the complete test suite (52 tests) and verified that all tests pass successfully.

---

## 4. Enhanced Metrics and Evaluation (PR-AUC & Loss)

To support evaluation on imbalanced datasets, we introduced Precision-Recall AUC (PR-AUC) alongside ROC-AUC and incorporated loss evaluation:
- **GraphGym Logger Modification**: Updated binary classification logger (`logger.py` in PyTorch Geometric site-packages) to calculate `pr_auc` dynamically using `precision_recall_curve` and `auc` from `sklearn.metrics`. This propagates PR-AUC to `stats.json` and aggregated CSVs automatically.
- **Evaluation Script Updates (`eval.py`)**:
  - Expanded `metrics` inside `GNNResultEvaluator` to include `pr_auc` and `loss`, rendering a wider 2x7 grid of heatmaps (at `figsize=(32, 11)`) and placing the summary bar chart in column 6.
  - Implemented a white-to-red colormap (`self.cmap_loss`) for `loss` heatmaps to intuitively highlight high-loss configuration areas in red.
  - Included `pr_auc` and `loss` in the overall layer-type architecture comparison bar charts.
