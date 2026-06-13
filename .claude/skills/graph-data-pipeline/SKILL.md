---
name: graph-data-pipeline
description: Work with the shared expression-graph data layer in codebase/src/gnn/shared/ — converting Mathematica/GraphML expression trees into PyG graphs, the node/edge feature catalog, graph modes (graph/tree/tree_derivatives), edge directions, positional encodings, augmented kappa (h-function) graphs, and the UnifiedDataLoader. Use when adding/changing a feature, a converter, a graph mode, the loader, or when a feature-dimension/schema mismatch appears.
---

# Graph data pipeline (shared core)

The `shared/` package is the single source of truth for how raw expression graphs become PyG tensors. Both the supervised and RL workflows depend on it, so changes here ripple everywhere — keep both consumers in mind.

## Where things live

| Concern | File |
| --- | --- |
| GraphML/JSON → PyG conversion, AST edges, virtual nodes, topology features, schemas | `shared/utils/graph_utils.py` |
| Feature catalog + group/PE resolution (shared by supervised & RL) | `shared/utils/feature_config.py` |
| Homogeneous `Data` builder | `shared/utils/homogeneous_converter.py` |
| Heterogeneous `HeteroData` builder | `shared/utils/heterogeneous_converter.py` |
| Augmented graph + kappa (h-function) merging | `shared/utils/graph_utils.py` (`AugmentedFunctionGraph`, `LoadAugmentedFunctionGraph`, `MergeDisjointSubgraph`) |
| Tabular CSV loader | `supervised_learning/dataset.py` (`DatasetLoader`) |
| Graph (PyG) loader | `shared/utils/graph_loader.py` (`GraphDataLoader`) |
| Unified facade over both | `shared/utils/unified_loader.py` (`UnifiedDataLoader`) |

Detailed prose references already exist — read them before deep edits:
`docs/in_depth_reference.md`, `docs/quick_overview.md`, and `changenotes.md` (latest schema decisions, e.g. native edge features, node schema reduced 27→19).

## The three graph modes (`--mode`)

Set once and threaded everywhere; controls which subgraphs load and whether virtual nodes exist.

- **`graph`** — f, f', f'' merged via the `global` node, **plus** virtual nodes (`virtual_current_x`, `virtual_f_x`/`virtual_supernode`, `virtual_y_target`) carrying dynamic solver values. Default.
- **`tree_derivatives`** — f, f', f'' merged via `global`, **no** virtual nodes; dynamic values written directly onto `global`.
- **`tree`** — only the function tree f (derivatives ignored, no virtual nodes); dynamic values as slots on `global`.

## Edge direction (`--edge-direction`)

`top_down` (parent→child, default), `bottom_up` (child→parent), `bidirectional`. Applies only to AST edges. **Virtual/task edges stay bidirectional regardless** — see `is_virtual_task_edge` / `_effective_edge_direction`. Always pass user-facing values through `validate_edge_direction`.

## Feature catalog — change features only via `feature_config.py`

Four feature classes resolved centrally so supervised and RL stay identical: `node`, `topology`, `positional`, `edge`. Positional encoding is **anchor-based** (replaced the old `lpe`/`rwpe`): 5 columns (`anchor_additive`, `anchor_scaling`, `anchor_periodic`, `anchor_exponential`, `anchor_transcendental`), each `1/(1+hops)` to the nearest operator anchor of that semantic group within the node's own function. Defined in `graph_utils.py` (`ANCHOR_GROUP_FEATURES`, `ANCHOR_GROUP_BY_LABEL`, `_compute_anchor_positional_encoding`). **Mutually exclusive with `add_virtual_supernode`** — combining them raises `PositionalSupernodeConflictError` (`validate_positional_supernode_compatibility`), since the supernode collapses the shortest-path distances the encoding relies on.

The `enrich` flag toggles **basic vs enriched** schemas. Authoritative schema constants live in `graph_utils.py`:
`BASIC_NODE_FEATURE_SCHEMA`, `ENRICHED_NODE_FEATURE_SCHEMA`, `BASIC_EDGE_FEATURE_SCHEMA`, `ENRICHED_EDGE_FEATURE_SCHEMA`, plus `CANONICAL_LABEL_VOCAB`.

**To add or remove a node/edge feature, you must keep all of these in sync:**
1. The schema tuple in `graph_utils.py` (defines tensor column order — order is load-bearing).
2. Where the value is written during conversion (`ExpressionGraphConverter.convert`, `TopologicalFeatureExtractor.extract_and_annotate`, `populate_task_virtual_values`).
3. The catalog tuples in `feature_config.py` (`NODE_FEATURES_*`, `TOPOLOGY_FEATURES_*`, etc.) so group toggles resolve correctly.
4. Model input dim — derived from the schema via `GraphConversionPipeline.input_dim` / `get_feature_schema`; `slice_active_features` honors `--active-features` subsets.

A wrong column count surfaces downstream as a shape mismatch in `NodeFeatureEncoder`/`EdgeFeatureEncoder` — trace it back to a schema/catalog desync, not the model.

## Augmented kappa (h-function) graphs

`LoadAugmentedFunctionGraph(graphId, graphsFolder, kappasFolder)` loads a base math graph, ensures a `global` node, scans `kappasFolder` for `id == "kappa"` objects, merges each via `MergeDisjointSubgraph` (collision-safe prefix `kappa_<n>_<id>`), and wires `GlobalToKappa`/`KappaToGlobal` edges weighted by the parsed kappa value. Kappa values are generated separately by `add_kappa.py` → `datasets/kappas/kappas.json`.

## UnifiedDataLoader

`UnifiedDataLoader.get_instance(...)` is a **singleton/multiton** keyed on the full config tuple (`dataset_name`, `run_key`, `mode`, `enrich`, `heterogeneous`, `add_traces`, `base_dir`, `is_synthetic`, `edge_direction`). Reuse `get_instance` rather than constructing directly so cache hits work. It resolves datasets under repo-root `datasets/<run_key>/<name>.{csv,json}` (legacy fallback to `_datasets/`/`graphs/`), links CSV rows to GraphML graphs, and auto-enriches missing `x0` start values from graph raw data.

## Naming convention (important, non-obvious)

Public domain-level classes and functions here are **PascalCase** (`ExpressionGraphConverter`, `TopologicalFeatureExtractor`, `LoadAugmentedFunctionGraph`, `MergeDisjointSubgraph`, `GraphConversionPipeline`). Internal helpers are snake_case (`signed_log_value`, `validate_edge_direction`, `to_homogeneous`, `to_hetero`). Module constants are UPPER_SNAKE. Match the surrounding style when editing.

## After changing this layer

Run the affected shared tests (see `gnn-dev-workflow` skill for the exact invocation): `test_graph_utils`, `test_feature_config`, `test_graph_loader`, `test_unified_loader`, `test_heterogeneous_pipeline`, `test_augmented_loader`, `test_virtual_nodes`, `test_dataset_headers`.
