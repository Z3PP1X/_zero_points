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
- **GraphGym Logger Modification**: Updated binary classification logger to calculate `pr_auc` dynamically using `precision_recall_curve` and `auc` from `sklearn.metrics`. This propagates PR-AUC to `stats.json` and aggregated CSVs automatically.
  - *WSL Environment Patch*: Directly patched `/home/zapp1x/miniconda3/envs/pytorch/lib/python3.12/site-packages/torch_geometric/graphgym/logger.py` to ensure local experiments compute the metric.
  - *Cloud/Multi-Machine Monkey Patch*: Implemented dynamic runtime monkey patching inside `loader_graphgym.py` (which is tracked in git and loaded automatically by `main_graphgym.py`). This intercepts PyG's `Logger.classification_binary` method on-the-fly and overrides it with our PR-AUC implementation. This guarantees `pr_auc` is recorded on **any external environment (such as your cloud GPU)** without needing manual installation adjustments.
- **Evaluation Script Updates (`eval.py`)**:
  - Expanded `metrics` inside `GNNResultEvaluator` to include `pr_auc` and `loss`, rendering a wider 2x7 grid of heatmaps (at `figsize=(32, 11)`) and placing the summary bar chart in column 6.
  - Implemented a white-to-red colormap (`self.cmap_loss`) for `loss` heatmaps to intuitively highlight high-loss configuration areas in red.
  - Included `pr_auc` and `loss` in the overall layer-type architecture comparison bar charts.
  - Added support for activation function sweeps. If an `'act'` column is present in the results, it automatically generates slice plots under `eval_plots/.../act/{activation_function}.png` comparing that activation across layer types.
  - Made the summary bar chart dynamic: if evaluating a single layer type, it automatically switches to compare different activation functions; if evaluating a single activation function, it switches to compare layer types.

---

## 5. Performance, Memory, and Hardware Precision Optimizations

To maximize training and inference speed on compatible accelerator hardware:
- **`pin_memory` Support**: Enabled page-locked (`pin_memory=torch.cuda.is_available()`) memory allocation in standard and synthetic supervised `DataLoader` objects inside `preprocessing.py`. This optimizes host-to-device CUDA data transfer.
- **Float32 Matrix Multiplication Precision**: Added configuration to leverage TensorFloat-32 (TF32) on Ampere or newer GPUs by setting `torch.set_float32_matmul_precision('high')` at the entry points of `supervised_learning/main.py`, `supervised_learning/main_graphgym.py`, and `reinforcement_learning/main.py`.
- **In-Memory Dataset Usage**:
  - In GraphGym supervised experiments, PyG's `InMemoryDataset` (specifically the custom `ExpressionGraphDataset`) is natively leveraged.
  - In standard supervised learning pipelines, graph templates are pre-cached in-memory as a dictionary during initial loading (`UnifiedDataLoader.load_all()`) to prevent redundant disk reads, and cloned on-the-fly inside the dataloader `__getitem__`.

---

## 6. Positive Label Correction for Imbalanced Dataset (Newton = Positive Class)

To ensure that all classification metrics (Precision, Recall, F1, PR-AUC) are correctly computed with respect to the **minority class** (Newton, ~25%), the binary label assignment was corrected:

- **Before**: Newton was assigned label `0` (negative class), gMGF was assigned label `1` (positive class).
- **After**: Newton is now assigned label `1` (positive class), gMGF is assigned label `0` (negative class).

This ensures that scikit-learn and GraphGym/PyTorch-based metric functions treat Newton as the target `pos_label` by default, so that Precision, Recall, F1, and PR-AUC directly measure how well the model identifies Newton as the faster solver.

**Modified Files:**
- `preprocessing.py`: Swapped `values = [0, 1]` to `values = [1, 0]` in `_tag_faster_algorithm()` (docstring updated accordingly).
- `main.py`: Updated `print_dataset_distribution()` to reflect the new class mapping (Newton = Class 1, gMGF = Class 0).

---

## 7. Train / Test / Validation 3-Way Split (Synthetic Mode)

To provide a rigorous evaluation framework with clean separation between model selection and final real-world performance assessment, the supervised learning pipeline was restructured into a **3-way data split** in synthetic mode:

| Split | Data Source | Purpose |
|---|---|---|
| **Train** (80%) | Synthetic problems | GNN weight updates |
| **Test** (20%) | Synthetic problems (unseen) | Epoch-level evaluation, checkpoint selection |
| **Val** | All curated real problems | Final real-world performance — no influence on training |

**Key Properties:**
- The **Train/Test split is problem-id-based**: all runs belonging to the same problem ID are kept together in the same split (prevents topology leakage).
- The **split is stratified**: `stratify=True` is passed to `train_test_split`, ensuring the class ratio (Newton vs. gMGF) is preserved in both train and test splits.
- The **curated dataset is never used during training or checkpoint selection** — it is only evaluated once at the end, against the best saved model.
- The **80/20 split ratio** is applied consistently across `main.py` (`TEST_SIZE = 0.2`) and `loader_graphgym.py` (`test_size=0.2`).

**GraphGym Index Assignment (Synthetic Mode):**
- `train_graph_index` → Synthetic training problems (80%)
- `val_graph_index` → Curated real-world problems (used as validation during GraphGym training epochs)
- `test_graph_index` → Unseen synthetic test problems (20%, used for final epoch test reporting)

**Modified Files:**
- `preprocessing.py`: Added problem-id-based stratified split of synthetic data, added `self.curated_dataset` / `self.curated_loader` attributes, and a 3-way `DataLoader` construction.
- `loader_graphgym.py`: Corrected `val_indices` to map to curated data and `test_indices` to unseen synthetic data. Enabled `stratify=True` and `num_workers` forwarding.
- `main.py`: Added a post-training evaluation block that loads the best saved checkpoint and runs it against `pipeline.curated_loader`, logging `Loss/curated`, `Accuracy/curated`, `F1/curated`, `Precision/curated`, `Recall/curated`, and `AUC/curated` metrics to MLflow.

---

## 8. Evaluation & Aggregation Script Updates

To reflect the new 3-way split in result visualizations and to handle the expanded hyperparameter grid:

**`eval.py` (GNNResultEvaluator):**
- Expanded the list of evaluated run files from 6 (`train_*`, `val_*`) to 9 (`train_*`, `test_*`, `val_*`).
- Added human-readable labels for each split in plot titles:
  - `train_*` → "Training (Synthetic)"
  - `test_*` → "Test (Unseen Synthetic)"
  - `val_*` → "Validation (Curated Real)"
- Added a new **`graph_pooling` slicing dimension**: if the results contain a `graph_pooling` column, dedicated slice plots are generated under `eval_plots/.../graph_pooling/{pooling}.png`.
- Added a new **`generate_summary_comparison()`** method that produces a single `split_comparison.png` chart overlaying Train / Test / Validation best-epoch metrics side-by-side for a direct generalization gap assessment.

**`aggregate_graphgym.py`:**
- Updated the directory rename regex to flexibly capture all known grid parameters (`layer_type`, `layers_mp`, `dim_inner`, `dropout`, `graph_pooling`, `act`) instead of a hardcoded pattern.
- Added a final summary report listing all generated aggregated CSV files after batch aggregation completes.

