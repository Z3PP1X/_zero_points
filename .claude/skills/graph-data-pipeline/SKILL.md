---
name: graph-data-pipeline
description: Work with the shared expression-graph data layer in codebase/src/gnn/shared/ — converting Mathematica/GraphML expression trees into PyG graphs, the node feature catalog/schema, graph modes (tree/tree_derivatives), the global merge node, anchor positional encodings, the optional virtual supernode, augmented kappa (h-function) graphs, and the UnifiedDataLoader. Use when adding/changing a feature, a converter, a graph mode, the loader, or when a feature-dimension/schema mismatch appears.
---

# Graph data pipeline (shared core)

The `shared/` package is the single source of truth for how raw expression graphs become PyG tensors. Both the supervised and RL workflows depend on it, so changes here ripple everywhere — keep both consumers in mind.

## Where things live

The old monolithic `graph_utils.py` has been split. It now survives only as a **backward-compatible re-export shim** — every old `from gnn.shared.utils.graph_utils import ...` still works, but the real definitions live in the files below.

| Concern | File |
| --- | --- |
| GraphML/JSON → PyG conversion, AST edges, the `global` merge node, per-node one-hots (`_enrich_nodes`) | `shared/utils/graph_converter.py` (`ExpressionGraphConverter`) |
| Topology features (subtree size/depth, anchor PE, subtree histograms), virtual-supernode injection | `shared/utils/feature_extraction.py` (`TopologicalFeatureExtractor`, `inject_virtual_supernode`) |
| Node feature schema + label / root-color / node-type vocab & one-hot helpers | `shared/utils/graph_vocab.py` (`NODE_FEATURE_SCHEMA`, `CANONICAL_LABELS`, `ROOT_COLOR_VOCAB`, …) |
| Back-compat re-export shim (re-exports all of the above under the legacy path) | `shared/utils/graph_utils.py` |
| Feature catalog + group/PE resolution (shared by supervised & RL) | `shared/utils/feature_config.py` |
| Homogeneous `Data` builder | `shared/utils/homogeneous_converter.py` (`to_homogeneous`) |
| Augmented graph + kappa (h-function) merging | `shared/utils/kappa_loader.py` (`AugmentedFunctionGraph`, `LoadAugmentedFunctionGraph`, `filter_active_kappa`) |
| Tabular CSV loader | `supervised_learning/dataset.py` (`DatasetLoader`) |
| Graph (PyG) loader | `shared/utils/graph_loader.py` (`GraphDataLoader`) |
| Unified facade over both | `shared/utils/unified_loader.py` (`UnifiedDataLoader`) |

There is **no heterogeneous converter** — the pipeline emits homogeneous `Data` only.

## The two graph modes (`--mode`)

`MODE_CHOICES = ("tree", "tree_derivatives")`; default `tree_derivatives`. The mode controls which subgraphs load. There are **no virtual task nodes** (`virtual_current_x` / `virtual_y_target` were removed); the only optional virtual node is the message-passing supernode below.

- **`tree_derivatives`** — f, f', f'' merged under one `global` node (`global → root` of each tree). Default.
- **`tree`** — only the function tree f; derivatives ignored. The `global` node is still present as the merge/relay hub.

### The `global` merge node carries **zero** features

The `global` node (node_type code `0`) is a pure relay/aggregator. In `ExpressionGraphConverter._enrich_nodes` its entire `NODE_FEATURE_SCHEMA` row is forced to `0.0` — it carries the full feature dimensionality but injects **no** signal of its own. This matters most for the structural features (anchor PE, `subtree_size`, `subtree_depth`), which would otherwise leak a spurious "root-of-everything" magnitude into message passing. Its identity stays recoverable downstream via `data.node_type` / `data.root_color`, which are built separately from `G_directed` and are **not** zeroed. If you add a feature that should remain zero on `global`, no extra work is needed — the zeroing loop covers every schema column.

## Edges: always top-down, no edge features

AST edges are always **top-down** (parent → child); orientation lives purely in the `edge_index` topology. The configurable `edge_direction` system (`top_down` / `bottom_up` / `bidirectional`, `validate_edge_direction`, `EDGE_DIRECTIONS`) has been **removed** from the shared layer. `EDGE_FEATURE_SCHEMA` is empty (`[]`) — the homogeneous graph emits no `edge_attr`. The only bidirectional edges are the virtual-supernode shortcut edges added by `inject_virtual_supernode`.

## Feature catalog — change features only via `feature_config.py`

`FEATURE_CLASSES = ("node", "topology", "positional")` are resolved centrally so supervised and RL stay identical. Positional encoding is **anchor-based**: 3 semantic groups (`anchor_trigonometric`, `anchor_exponential`, `anchor_variable`), each `1/(1+hops)` to the nearest anchor of that group within the node's own function subgraph (`0.0` if absent). Defined in `graph_vocab.py` (`ANCHOR_GROUP_FEATURES`, `ANCHOR_GROUP_BY_LABEL`) and computed in `feature_extraction.py` (`_compute_anchor_positional_encoding`). **Mutually exclusive with `add_virtual_supernode`** — combining them raises `PositionalSupernodeConflictError` (`validate_positional_supernode_compatibility` in `feature_config.py`), since the supernode collapses the shortest-path distances the encoding relies on.

The node schema is a **single** 32-column tuple `NODE_FEATURE_SCHEMA` (no basic/enriched split): node_type one-hot (3) + root_color one-hot (5) + label one-hot (15, `CANONICAL_LABELS`) + `subtree_size` + `subtree_depth` + subtree histogram (4) + anchor PE (3). Column order is load-bearing.

**To add or remove a node feature, keep all of these in sync:**
1. `NODE_FEATURE_SCHEMA` in `graph_vocab.py` (defines tensor column order) **and** the relevant vocab tuple it draws from (`CANONICAL_LABELS` + `LABEL_ONEHOT_NAMES`, `ROOT_COLOR_VOCAB`, `ANCHOR_GROUP_FEATURES`, `HISTOGRAM_FEATURES`).
2. Where the value is written: one-hots in `ExpressionGraphConverter._enrich_nodes`; topology in `TopologicalFeatureExtractor.extract_and_annotate`; histograms in `_compute_subtree_histograms`. (The `global`-node zero loop in `_enrich_nodes` auto-covers any new column.)
3. The catalog tuples in `feature_config.py` (`NODE_FEATURES`, `TOPOLOGY_FEATURES`, `POSITIONAL_ENCODING_FEATURES`) so group toggles / `--active-features` resolve correctly.
4. Model input dim — derived from `len(NODE_FEATURE_SCHEMA)`, or the active subset via `slice_active_features`; fed to the backbone's `input_dim`. After any schema change, **delete `datasets/graphs/.pt_cache/`** (stale cached tensors carry the old column count).

A wrong column count surfaces downstream as a shape mismatch in the model's node encoder — trace it back to a schema/catalog desync, not the model.

## Augmented kappa (h-function) graphs

`LoadAugmentedFunctionGraph(graphId, graphsFolder, kappasFolder)` (in `kappa_loader.py`) loads a base math graph as an `AugmentedFunctionGraph` (an `nx.DiGraph` subclass), ensures a `global` node, scans `kappasFolder` for kappa objects, and merges each via the `AugmentedFunctionGraph.MergeDisjointSubgraph` method (collision-safe `kappa_<n>_<id>` prefix), wiring global↔kappa edges weighted by the parsed kappa value. Kappa nodes are colored `root_color_kappa`. `filter_active_kappa` prunes inactive kappa subgraphs at load time. Kappa values are generated separately by `add_kappa.py` → `datasets/kappas/kappas.json`.

## UnifiedDataLoader

`UnifiedDataLoader.get_instance(...)` is a **multiton** keyed on `(dataset_name, run_key, mode, add_traces, base_dir, is_synthetic, add_kappa, add_virtual_supernode, csv_path, graphs_path)`. Reuse `get_instance` rather than constructing directly so cache hits work; `clear_instances()` resets the cache (used in tests). It resolves datasets under repo-root `datasets/<run_key>/<name>.{csv,json}`, links CSV rows to GraphML graphs, and auto-enriches missing `x0` start values from graph raw data.

## Naming convention (important, non-obvious)

Public domain-level classes and functions here are **PascalCase** (`ExpressionGraphConverter`, `TopologicalFeatureExtractor`, `AugmentedFunctionGraph`, `LoadAugmentedFunctionGraph`, `GraphDataLoader`). Internal helpers are snake_case (`encode_label`, `to_homogeneous`, `inject_virtual_supernode`, `slice_active_features`). Module constants are UPPER_SNAKE. Match the surrounding style when editing.

## After changing this layer

Run the affected shared tests (see the `gnn-dev-workflow` skill for the exact invocation): `test_graph_utils`, `test_feature_config`, `test_graph_loader`, `test_unified_loader`, `test_augmented_loader`, `test_virtual_nodes`, `test_virtual_supernode`, `test_graphml_import`, `test_dataset_headers`.
