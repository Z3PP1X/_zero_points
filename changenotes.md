# Change Notes - GNN Backbone Enhancements and Native Edge Features

This document outlines modifications to GNN graph construction, edge-aware backbones, virtual node mechanisms, and supervised training integration.

---

## 1. Native Edge Features (replacing node-projected edge features)

Edge semantics are now consumed directly by edge-aware conv layers instead of being flattened onto node features.

### Edge feature schema (`enrich=True`)
- **`child_index`**: Operand order among siblings (0, 1, 2, …).
- **`direction`**: `0.0` = parent→child, `1.0` = child→parent (bidirectional AST edges).
- **`relation_type`**: Encoded edge type (`child_of`, `belongs_to_f`, `virtual`, …).
- **`edge_betweenness_centrality`**: Structural importance of the connection.

Basic mode (`enrich=False`) uses a single **`edge_type`** feature.

Schema constants live in `graph_utils.py` as `ENRICHED_EDGE_FEATURE_SCHEMA` and `BASIC_EDGE_FEATURE_SCHEMA`. Node schema (`ENRICHED_NODE_FEATURE_SCHEMA`) was reduced from **27 → 19** by removing all projected edge features (`node_child_index`, `parent_*`, `child_slot_*`, etc.).

### Edge-aware backbones
Two architectures are supported for supervised and RL training:

| Architecture | Layer | How edges are used |
|---|---|---|
| **`gatv2_stack`** | `GATv2Conv(edge_dim=…)` | Edge features modulate attention coefficients |
| **`gine_stack`** | `GINEConv(edge_dim=…)` | Edge features injected into the neighbor aggregation MLP |

Legacy stacks (`gcn_stack`, `sage_stack`, `gin_stack`) remain available but ignore `edge_attr`.

Shared helpers in `gnn_backbones.py`:
- `coalesce_edge_attr()` — zero-fills when `edge_attr` is missing (RL path without padded obs).
- `apply_edge_conv()` — dispatches to GINE vs GATv2 calling conventions.

`GraphPolicyBackbone` adds an `edge_encoder` linear layer and passes encoded edge features through all conv layers.

### Supervised learning integration
- `TestGraphNetwork` accepts `--architecture gatv2_stack|gine_stack` and reads `pipeline.edge_dim`.
- `train()` / `evaluate()` in `main.py` pass `batch.edge_attr` to the model.
- `GraphPipeline` exposes `architecture` and `edge_dim` properties.
- GraphGym custom layers (`gatv2conv`, `gineconv` in `loader_graphgym.py`) forward `batch.edge_attr`.

### Feature layout
- `NATIVE_NODE_FEATURE_COUNT = 19`
- `NATIVE_EDGE_FEATURE_COUNT = 4`
- `FeatureLayout` adds `edge_input_dim` (choices: 4, 8) for projected edge embeddings in RL backbones.

**Retraining required:** node input dim changed (27→19) and models now expect native `edge_attr`.

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

### Task 5: Full Message-Passing Exclusion for Virtual Nodes
- Virtual and aggregator nodes (types 5–10) no longer participate in convolutional message passing.
- `filter_real_subgraph()` in `gnn_backbones.py` strips any edge touching a virtual node and remaps real nodes to a dense index space before each conv layer.
- Real AST nodes (including the structural `global` node, type 0) message-pass only over real→real edges.
- Virtual nodes are updated exclusively through per-layer `virtual_update_mlps`, whose input/output dims track each conv stage (including GAT head expansion).
- Supernode seeding now pools **real nodes only** (not virtual hubs) before layer 0.
- Applied consistently in `GraphPolicyBackbone` (RL) and `TestGraphNetwork` (supervised).
- Real and virtual embeddings are kept in separate tensors during the layer loop (`h_real` / `h_virt`) so conv dims can diverge from virtual MLP dims; `pool_split_embeddings()` merges them only at readout.

---

## 3. Test Coverage & Verification

- **Edge features**: Updated `test_graph_utils.py` to assert native `edge_attr` shape and removed node-projection assertions. Updated `test_virtual_nodes.py` and `test_trial_switch.py` for **19** node features. Added `edge_attr` to `test_gnn_policy_backbone.py`.
- **MP exclusion tests**: `test_gnn_policy_backbone.py` now covers `filter_real_subgraph()` and verifies that perturbing virtual-only edge features does not change the backbone output.
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

---

## 9. Architecture-Based Evaluation Plot Organization

To ensure that hyperparameter slicing plots (e.g. by layer count, activation function, and pooling strategy) are not misleadingly aggregated across different GNN architectures, the evaluation pipeline in `eval.py` has been updated to organize plots under architecture-specific subdirectories:
- **Nested Directory Structure**: Hyperparameter slice plots are now nested under the respective model architecture directory (e.g. `layer_type/gatv2conv/layers_mp/`, `layer_type/gatv2conv/act/`, and `layer_type/gatv2conv/graph_pooling/`).
- **Targeted Slice Filtering**: Each slice plot is computed and plotted specifically on the filtered subset of data belonging to that architecture, with the summary bar chart on the right comparing other configurations within the same architecture.
- **Run-Level Comparison**: The global overall plot comparing architectures and the cross-split comparison charts are kept at the run-level.
- **Flexible Summary Grouping**: Modified `generate_plots_for_df` to accept an explicit `group_col` argument, enabling correct title labeling and legend mapping depending on the active slice variable.

---

## 10. Epoch-by-Epoch Validation of Both Splits (Synthetic vs Curated)

To support tracking generalization during training, we restructured the validation step in GraphGym to evaluate both the unseen synthetic data and the curated real-world data at each epoch without either influencing training weight updates:
- **Multiple Validation Dataloaders**: Overrode `val_dataloader` in `main_graphgym.py` to return `[val_loader, test_loader]` (synthetic test split + curated split) in synthetic mode.
- **Dataloader-Aware LoggerCallback**: Patched `LoggerCallback.on_validation_batch_end` and `on_validation_epoch_end` in `loader_graphgym.py` to route batch statistics to `val_logger` (for synthetic) and `test_logger` (for curated) separately.
- **Metric Isolation**: Modified `ValMetricLogger` in `main_graphgym.py` to accumulate and log metrics per dataloader index, using `val_pr_auc` (on index 0) for ModelCheckpoint selection, and logging `val_pr_auc_curated` (on index 1) for visualization.
- **Epoch-by-Epoch Test Aggregation**: Patched `is_split` and `agg_runs` in `aggregate_graphgym.py` to allow the `test` split (curated validation) to be aggregated epoch-by-epoch just like `val` (synthetic validation), generating `test.csv`, `test_best.csv`, and `test_bestepoch.csv` automatically.
- **Evaluation Labels**: Updated `run_labels` and the summary comparison chart in `eval.py` to distinguish between `Validation Synthetic (Unseen Synthetic)` and `Validation Curated (Curated Real)`.

---

## 11. True Heterogeneous Architecture for Mathematical Expression Graph Pipeline

Refactored the GNN mathematical expression graph pipeline from a single-node homogeneous packing (`torch_geometric.data.Data`) with zero-padding into a heterogeneous architecture (`torch_geometric.data.HeteroData`) using PyTorch Geometric (PyG).

### Key Architectural Adjustments:
- **Node Type Partitioning**: Partitioned raw expression tree nodes into 4 distinct types: `"operator"`, `"variable"`, `"constant"`, and `"virtual"`.
- **Feature Optimization**: Eliminated zero-padding of features by tailoring feature matrices (`x`) exactly to each node type:
  - `operator`: 32-dim one-hot Label ID + 5 topology metrics + 8 structural LPE/RWPE dimensions = 45 features.
  - `variable`: 32-dim one-hot token ID + 5 topology metrics + 8 structural LPE/RWPE dimensions = 45 features.
  - `constant`: 1-dim signed log value + 8-dim sinusoidal Fourier frequency encodings = 9 features.
  - `virtual`: 7-dim dynamic value features (reduced from 8 by merging target and function values).
- **Delta Target Simplification**: Merged separate `yTarget` and `f(x0)` features into a single relative `virtual_delta_target_val` feature ($y_{target} - f(x_0)$), simplifying GNN input and node learning. Feature size counts were updated accordingly from 25 to 24 (enriched) and 13 to 12 (basic).
- **Relational Metapath Triplets**: Defined explicit metapaths for relations rather than relying on flat edge indexing, supporting type-local re-indexing. Also split non-commutative child relations into explicit `left_operand` and `right_operand` edges to capture mathematical non-commutativity.
- **Dynamic State Injection**: Refactored `populate_task_virtual_values` to dynamically detect and mutate type-local virtual features directly in `HeteroData` node attributes.
- **Testing & Verification**: Introduced a new validation suite under `test_heterogeneous_pipeline.py` verifying shape and indexing correctness, and regression-tested homogeneous path configurations.

**Modified Files:**
- `codebase/src/gnn/shared/utils/graph_utils.py`: Added node splitting, relation/metapath mapping, Fourier encoding, and updated conversion and state injection logic.
- `codebase/src/gnn/supervised_learning/preprocessing.py`: Updated input feature dimension metadata.
- `codebase/src/gnn/reinforcement_learning/feature_layout.py`: Updated global node feature dimension sizes.
- `codebase/src/gnn/tests/`: Updated test validation suites.

---

## 12. Methodology Fixes: Feature Encoding, RWPE Correctness, Derivative Injection, Oversmoothing

Targeted fixes addressing mediocre discrimination and the large synthetic→curated generalization gap, following a methodical review of the supervised workflow and `graph_utils.py`.

### 12.1 Categorical features are now embedded (discrimination)
Previously `node_type`, `label_id`, and edge `relation_type` were integer codes consumed as **raw continuous floats** by the conv layers in both supervised pipelines. This imposes a meaningless ordinal scale (e.g. `Log=17` ≈ 17 × `Plus=3`) and corrupts the most discriminative signal — operator / function identity.
- **Standalone (`TestGraphNetwork`)**: now applies the existing `NodeFeatureEncoder` (embeds `node_type` + `label_id`, projects/normalises continuous columns) and `EdgeFeatureEncoder` (embeds `relation_type`) at the input. Gated by a new `use_feature_encoder` flag; automatically disabled when an explicit `active_features` subset is selected (categorical column positions would otherwise be ambiguous). Edge encoder is only applied for the enriched 4-dim edge schema.
- **GraphGym (the live experiment)**: added `ExpressionNodeEncoder` (`register_node_encoder`) implementing the same embedding + continuous-projection logic with `LazyLinear`, enabled in `config_supervised.yaml` (`dataset.node_encoder: True`, `node_encoder_bn: True`).

### 12.2 RWPE made informative on trees (expressivity)
The AST is a tree (bipartite), so the non-lazy random-walk return probability was identically **0** for odd step counts — `rwpe_1` and `rwpe_3` were dead (all-zero) features. Replaced with a **lazy random walk** `P = ½(I + D⁻¹A)`, recording return probabilities for steps `k=2..5`, so all four RWPE dimensions carry structural signal. (Existing trained checkpoints should be retrained.)

### 12.3 Derivative / function-value injection (discrimination)
`f(x0)`, `f'(x0)`, `f''(x0)` are the physically decisive quantities for a Newton (uses `f'`) vs gMGF/Halley (uses `f''`) decision, but were never reaching the model.
- `dataset.py`: added header-normalisation aliases mapping common column names to canonical `fx`, `d1x`, `d2x`.
- `preprocessing.py` (`ProblemRunDataset`): now reads `d1x`/`d2x` and passes them to `populate_task_virtual_values`, injecting `f'(x0)`/`f''(x0)` onto the `d1_root`/`d2_root` aggregator nodes (used by graph-mode message passing). Defaults to `0.0` when columns are absent (previous behaviour). `global_dim` is unchanged so `benchmark_inference.py` and saved-model shapes remain valid.

### 12.4 Oversmoothing mitigation (GraphGym)
The `virtual_supernode` (connected to every node) turns the graph into a diameter-2 star, which accelerates oversmoothing under plain `stack` message passing with mean/add pooling. Switched `gnn.stage_type` to `skipsum` (residual connections) in `config_supervised.yaml`.

**Modified Files:**
- `codebase/src/gnn/shared/utils/graph_utils.py`: Lazy-random-walk RWPE.
- `codebase/src/gnn/shared/models/classifiers.py`: `TestGraphNetwork` node/edge feature encoders + `use_feature_encoder` flag.
- `codebase/src/gnn/supervised_learning/dataset.py`: `fx`/`d1x`/`d2x` header aliases.
- `codebase/src/gnn/supervised_learning/preprocessing.py`: derivative pass-through in `ProblemRunDataset`.
- `codebase/src/gnn/supervised_learning/loader_graphgym.py`: `ExpressionNodeEncoder` registration.
- `codebase/src/gnn/supervised_learning/config_supervised.yaml`: enable node encoder + `skipsum`.
- `codebase/src/gnn/tests/test_graph_utils.py`: updated RWPE assertion.
- `codebase/src/gnn/tests/test_supervised_classifier.py`: new forward-pass / encoder test suite.

---

## 13. Configurable AST Edge Direction & Grouped Feature Toggles

Added experiment controls for AST message-passing direction and structured feature selection across supervised and RL pipelines.

### 13.1 AST edge direction (`edge_direction`)
AST edges (parent→child in the raw tree) can now be compiled with configurable message-passing direction:

| Value | Message flow |
|---|---|
| `top_down` | parent → child (root to leaves) |
| `bottom_up` | child → parent (leaves to root) |
| `bidirectional` | both directions (separate forward/reverse edges) |

**Virtual nodes are always bidirectional** — edges touching `virtual_current_x`, `virtual_y_target`, `virtual_supernode`, or with type `virtual` / `supernode` ignore the setting.

Core logic lives in `graph_utils.py` via `_add_enriched_ast_edges()` / `_add_basic_ast_edges()`. The setting propagates through `GraphDataLoader`, `UnifiedDataLoader`, GraphGym dependency injection, and both training entry points.

CLI: `--edge-direction top_down|bottom_up|bidirectional`  
YAML: `expression_graph.edge_direction` (supervised) / `experiment.edge_direction` (RL)

### 13.2 Feature classes & positional-encoding experiments
Node/edge features are catalogued in `shared/utils/feature_config.py` by class:

| Class | Enriched members |
|---|---|
| **node** | `node_type`, `label_id`, `value`, `has_value`, `virtual_*`, `belongs_to_*` |
| **topology** | `depth`, `height`, `subtree_size`, `out_degree`, `betweenness_centrality` |
| **positional** | `lpe_1..4` (Laplacian PE), `rwpe_1..4` (random-walk PE) |
| **edge** | `child_index`, `direction`, `relation_type`, `edge_betweenness_centrality` |

Grouped toggles in YAML (`expression_graph.features` / `experiment.features`):

```yaml
features:
  node: true
  topology: true
  positional:
    enabled: true
    encodings: [lpe, rwpe]
  edge: true
active_features: ""   # optional explicit override list
```

CLI list arguments (supervised `main.py`, RL `main.py`, RL `train_best.py`):
- `--feature-groups node topology positional` — enable only selected classes
- `--positional-encoding lpe rwpe` — Laplacian and/or random-walk PE
- `--positional-encoding none` — disable all positional encodings
- `--active-features node_type,depth,lpe_1,...` — explicit override (disables group resolution)

Examples:
```bash
# LPE only
python main.py --config config_supervised.yaml --positional-encoding lpe

# No positional encodings
python main.py --config config_rl.yaml --positional-encoding none
```

Node-feature slicing is fully wired; the **edge** class is documented in config for overview (edge dim still follows `enrich` mode).

### 13.3 Central RL YAML config
- `reinforcement_learning/config_rl.yaml` — documents all RL CLI defaults (experiment, features, Optuna, gateway, train_best sections).
- `reinforcement_learning/rl_config.py` — YAML loader + CLI override resolution shared by `main.py` and `train_best.py`.

### 13.4 Tests
- `test_graph_utils.py`: edge-direction tests (top_down, bottom_up, virtual edges stay bidirectional).
- `test_feature_config.py`: group resolution, positional-encoding toggles, explicit overrides.
- `test_rl_config.py`: RL YAML settings including feature selection.

**Modified / added files:**
- `shared/utils/graph_utils.py`, `graph_loader.py`, `unified_loader.py`
- `shared/utils/feature_config.py` (new)
- `supervised_learning/config_supervised.yaml`, `supervised_config.py`, `loader_graphgym.py`, `main.py`
- `reinforcement_learning/config_rl.yaml`, `rl_config.py`, `main.py`, `train_best.py` (new config modules)
- `time_management/config_base.yaml`, `time_management/benchmark.py`
- `tests/test_graph_utils.py`, `tests/test_feature_config.py`, `tests/test_rl_config.py`


