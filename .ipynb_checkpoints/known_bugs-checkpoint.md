# Known Bugs & Deferred Tasks

Backlog from the pipeline audit (2026-06-13). Items here are **not yet fixed** — they
are recorded as tasks for later. Fixes already applied in the same audit pass are listed
at the bottom for reference.

References are `file:line` at the time of writing; re-grep before acting, line numbers drift.

---

## Reinforcement learning (all RL findings live here)

### RL-1 — Optuna tunes four reward parameters that the reward never uses  [HIGH]
`reinforcement_learning/reward.py:13-32` defines/stores `basis_reward`,
`solver_mismatch_penalty`, `solver_match_bonus`, `solver_wrong_slow_coef`, but
`calculate_episode_rewards` (reward.py:95-144) never references them — reward is purely
`r_time - step_cost_lambda + r_learn`. Yet they are sampled in
`ppo_optuna_search.py:41-49`, carried in `ppo_trial_config.py:18-26`, and read back in
`train_best.py:215-219` / `ppo_optuna_workflow.py:209-211`. **Every trial wastes 4 search
dimensions on no-op knobs**, diluting an expensive (Mathematica-in-the-loop) search.
- **Decision needed:** either (a) implement the solver-match/penalty term in
  `calculate_episode_rewards` (chosen solver in `transition["action"]["solver"]`, benchmark
  in `timeBenchmarkSolver`), or (b) remove the four params from the sampler, trial config,
  reward constructor, and train_best. Bump `OPTUNA_SEARCH_SPACE_SUFFIX` either way.

### RL-2 — `timeout_penalty` hardcoded and inconsistent with error reward  [MED]
`mathematica_env.py:46`, `mathematica_vec_env.py:49`: solver `error`/`non_converged`
yields a fixed `-10.0`, a socket timeout yields `0.0`, while the conceptual sibling
`time_bad_penalty` is Optuna-tuned in `[4,10]`. The `-10` magnitude dominates the
~±2/step shaped reward, so Optuna may end up optimizing error/timeout frequency rather
than solver quality. Expose `timeout_penalty` in config/Optuna and reconcile the
timeout-vs-error treatment (or document why a clean timeout is reward-neutral).

### RL-3 — env-level timeout params are dead; monitor is the real source  [MED]
`MathematicaVecEnv` / `build_mathematica_training_env` (`mathematica_vec_env.py:46-48,
522-538`) accept `timeout_fallback_s/cushion_s/window_size` that are never forwarded; the
actual timeout is `traffic_monitor.roundtrip_timeout`. `train_best.py` uses its own
`train_best_timeout_*` defaults (cushion 2.0) that differ from `main.py`'s. Drop the
unused env-level params and make `GatewayTrafficMonitor` the single source of truth.

### RL-4 — single-env path (`MathematicaGraphEnv`, `_finalize_single_env_episode`) is test-only  [MED]
`_build_training_env` always returns a `MathematicaVecEnv` (even for `n_envs=1`), so
`MathematicaGraphEnv`, `drain_buffered_states`, and `_finalize_single_env_episode`
(`ppo_optuna_workflow.py:264-294`) are exercised only by tests. ~120 lines of parallel
step/pad logic that can silently diverge from the vec path. Consolidate onto
`MathematicaVecEnv(num_envs=1)` or document as intentional test scaffolding.

### RL-5 — ZMQ sockets bind to `tcp://localhost`, ports hardcoded in two files  [MED]
`gateway/network_gateway.py:42-54` binds to `tcp://localhost:<port>`; `bind` should target
an interface (`tcp://127.0.0.1` or `tcp://*`). Ports are duplicated module constants in
`main.py:30-33` and `train_best.py:56-59` with no config override → a second concurrent run
fails to bind ungracefully. Move ports to `config_rl.yaml`, bind to `127.0.0.1`.

### RL-6 — silent NaN/Inf masking can hide solver blow-ups  [LOW]
`observation_sanitize.py:22-29`, `sb3_extractor.py:56-64`, `preprocessor.py:128` map
non-finite obs/GNN outputs to `0.0`. A diverging solver (Inf time/fx) becomes
indistinguishable from a legitimate zero. Add a dedicated non-finite indicator feature, or
at least count/log sanitation hits at the env level.

### RL-7 — `r_learn` tolerance band can out-score a genuine record  [LOW]
`reward.py:109-110,135`: the "within tolerance but slower" branch grants a flat
`0.5*alpha*time_tolerance*record_abs_time` that can exceed the true-record reward
`alpha*(record-final)` for tiny improvements — a mild reward-ordering inversion. Review so a
genuine new record always scores ≥ a near-miss. (Also remove `basis_reward`, subset of RL-1.)

### RL-8 — throughput ceiling: lock-step synchronous `step_wait`  [LOW / known limitation]
`mathematica_vec_env.py:367-413` blocks until all N envs respond before returning a batched
step; per-step wall-clock = slowest Mathematica roundtrip, no overlap of GNN inference with
solver compute. Correct for on-policy PPO; recorded as the known throughput limit. (The now
-deleted `async_collector.py` was an abandoned attempt to fix this.)

### RL-9 — `sb3_extractor.py` uses a bare import  [LOW]
`sb3_extractor.py:7` does `from observation_sanitize import ...` (works only via the
`sys.path` injection in the entry scripts) while siblings use the
`gnn.reinforcement_learning.` package path. Normalize to the package import.

### RL-10 — RL uses homogeneous mode; edge_attr is always None  [HIGH]
`UnifiedDataLoader.get_instance` defaults to `heterogeneous=False`. `to_homogeneous()`
(`shared/utils/homogeneous_converter.py`) calls `from_networkx` without
`group_edge_attrs`, so `data.edge_attr` is always `None` in the RL observation.
`CustomGNNFeaturesExtractor` / `GraphPolicyBackbone` coalesces missing edge_attr to
all-zero tensors of `padded_edge_feature_count` width — the GNN never sees structural
edge features (child_index, direction, relation_type). Severity: all edge information
is silently lost; the backbone trains purely on node topology.
**Fix options**: (a) switch RL to `heterogeneous=True` and wire the hetero edge_attr
into the observation dict and feature extractor, or (b) add `group_edge_attrs` to
`to_homogeneous()` and extend the RL observation space to carry edge_attr.

### RL-11 — solver state no longer injected into GNN  [HIGH]
`populate_task_virtual_values` was removed (it was a no-op after the position-aware
rewrite dropped its target feature columns). The RL Preprocessor still extracts
`currentX / fx / dfx / ddfx` into `global_features` (8 scalars), but these are only
passed through the `GlobalEncoder` linear — the GNN message-passing layers have no
per-node view of the solver's current iterate. The RL agent cannot condition its graph
reasoning on where the solver currently is in the function domain.
**Fix**: redesign solver-state injection — either re-add per-node dynamic columns to
`NODE_FEATURE_SCHEMA` (requiring a converter change) or concatenate the global state
to every node's embedding before the first conv layer.

---

## Shared data layer (deferred)

### SH-1 — `filter_active_kappa` is a no-op for HeteroData  [MED]
`shared/utils/graph_utils.py:2022` early-returns unless `isinstance(data, Data)`; the
docstring/signature advertise `Union[Data, HeteroData]`. Heterogeneous graphs keep all
kappa subgraphs active. Implement the hetero branch or narrow the type and assert.

### SH-2 — `_resolve_source` returns a non-existent path on no match  [MED]
`shared/utils/graph_loader.py:140` returns `repo_root/datasets/{name}.json` even when nothing
matched, so a typo'd dataset name yields a silently empty loader instead of an error. Raise
(or make empty-load callers fail loudly).

### SH-3 — per-config subprocess wipes the in-memory graph cache  [PERF]
`supervised_learning/run_all.py:18-26` trains each grid config in its own subprocess, so the
`UnifiedDataLoader` multiton and in-memory `_converted_cache` are process-local — every config
re-converts every graph (only the on-disk `.pt` cache saves a full re-parse). For a real grid
this is the dominant overhead. Use an in-process orchestrator, or warm the disk cache once.

### SH-4 — kappa folder re-globbed/re-parsed per graph  [PERF]
`graph_loader.py:212` globs once for the gate, then `LoadAugmentedFunctionGraph`
(`graph_utils.py:1939`) re-globs `**/*.json` and re-parses every kappa for every graph: O(N·M)
disk+parse. Parse the kappa map once and cache it on the loader.

### SH-6 — structural dead code left in place (deferred from the cleanup pass)  [LOW]
Verified unused but not removed because each touches multiple call sites or is referenced by
a skill doc — remove carefully with a test run:
- `GraphConversionPipeline` (`shared/utils/graph_utils.py`) — superseded by `GraphDataLoader`,
  no callers. (The `graph-data-pipeline` skill still cites it as the input-dim path; update the
  skill when removing.)
- `sage_stack` / `SAGEStackNetwork` / `_sage_layers` (`shared/models/gnn_backbones.py`) —
  buildable but reachable from no supervised mapping or RL choice; only the deleted
  `time_management/grid.yaml` referenced `sageconv`.

### SH-5 — augmented `.pt` cache key doesn't fingerprint the kappa set  [PERF / correctness]
`graph_loader.py:225-229` distinguishes augmented vs plain only by the `_augmented.pt` suffix;
it does not encode which kappas/values were merged. Rerunning `add_kappa.py` leaves stale
augmented graphs silently reused. Add a kappa count+mtime/hash to the cache filename.

---

## Supervised (deferred)

### SU-1 — `cfg.model.thresh` is read but never registered  [LOW]
`loader_graphgym.py:114-119` reads `getattr(cfg.model, "thresh", 0.5)` (no config key
registers it → always 0.5). The `pos_label==0` vs `==1` branches are asymmetric
(`scores <= 1-thresh` vs `scores > thresh`), only equivalent at exactly 0.5. Latent bug if a
threshold is ever set.

### SU-2 — `main.py` is an orphaned, divergent duplicate of the GraphGym path  [LOW]
`supervised_learning/main.py` reimplements the whole train/eval/curated loop but is referenced
by nothing in `run_all.py`/aggregate/post_eval. It is documented as the "custom training" entry
in the skill, so it is kept — but it can rot out of sync (hardcoded `TEST_SIZE`, seed, manual
best tracking). Decide: bless as the sanctioned standalone entry (and keep it in sync) or remove.

### SU-3 — `benchmark_inference.py` benchmarks an untrained net  [LOW]
`supervised_learning/benchmark_inference.py` is orphaned and constructs
`TestGraphNetwork(...)` with random init (no checkpoint load). If kept as a latency tool, load a
checkpoint; otherwise remove.

---

## Fixed in the audit pass (for reference)
- Non-synthetic `val == test` leakage in `loader_graphgym.py` / `main.py` — fixed (disjoint split).
- Left/right operand not distinguished in homogeneous `relation_type` — fixed (route through
  `get_relation_type`; operand edge types added to the canonical vocab).
- Hardcoded fallback dataset names in `run_results/eval.py` — fixed.
- Dead artifacts removed: `reinforcement_learning/async_collector.py`, `tests_and_archive/`,
  `client.py`, `version_check.py`, `time_management/`, and assorted dead symbols.
- Result/output artifacts gitignored and untracked.
- `LABEL_ID_COL` / `BELONGS_TO_F/D1/D2_COL` / `EDGE_RELATION_TYPE_COL` constants removed from
  `gnn_backbones.py` — they referenced schema columns that no longer exist, causing `ValueError`
  on module import.
- `GraphPolicyBackbone._legacy_forward` subtree aggregator (`belongs_mask` + `aggregator_specs`)
  removed — the `belongs_to_f/d1/d2` columns were dropped from `NODE_FEATURE_SCHEMA`, making the
  block dead code that always fell back to default-true/false masks.
- `populate_task_virtual_values` removed from `graph_utils.py` — was a documented no-op; removed
  the function and all import/call sites in preprocessing.py, rl/preprocessor.py, and tests.
- `kappa_weight` dropped from `EDGE_FEATURE_SCHEMA` — kappa graph topology is still loaded and
  the NetworkX attribute remains on kappa edges, but the column is no longer in the feature tensor.
- `supervised/main.py` switched from `create_graphgym_model` (stock PyG GNN) to
  `ExpressionClassifierNetwork`; `cfg.expression_graph.active_feature_names` now set before
  instantiation so the encoder can locate categorical columns by name.
- `evaluate()` hook guarded with `hasattr(model, "mp")` — Dirichlet energy skipped (0.0) when
  the model does not expose a `.mp` message-passing submodule.
- `feature_layout.py` — clarified that `EDGE_INPUT_DIM_CHOICES` are encoder output dims, not
  raw schema widths.
</content>
</invoke>
