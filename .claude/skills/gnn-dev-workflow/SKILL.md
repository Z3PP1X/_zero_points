---
name: gnn-dev-workflow
description: Environment setup, running the pytest suite, code conventions, config patterns, and linting for this GNN-RL bachelor-thesis repo. Use whenever running tests, fixing import/ModuleNotFound errors, setting up the conda env, linting, or matching the codebase's naming/style conventions.
---

# Dev workflow & conventions

## Environment

Code requires the prepared conda env (it carries `torch` and `torch_geometric`, which the base interpreter lacks):

```bash
conda activate pytorch
```

Workflows are normally run from `codebase/src/gnn/` or a sub-package. The runnable scripts (`main.py`, `train_best.py`, etc.) self-insert `codebase/src` and `codebase/src/gnn` onto `sys.path` at startup, so running them directly works. Tests do **not** — see below.

## Running tests

Tests live in `codebase/src/gnn/tests/` (40 files). Path wiring lives in
`pyproject.toml` (`[tool.pytest.ini_options]` → `pythonpath` + `testpaths`), so a
plain invocation from the repo root just works inside the `pytorch` env — **no
hand-rolled `PYTHONPATH` needed**:

```bash
conda activate pytorch
pytest                                            # whole suite
pytest codebase/src/gnn/tests/test_reward.py      # single file
```

(Pure-Python tests like `test_reward` pass even in the base env; anything importing
torch/PyG needs the `pytorch` env.)

**Why the config is needed** — tests use two import styles: package style
(`from gnn.shared.utils.feature_config import ...`, needs `codebase/src`) and
bare-module style (`from reward import ...`, `from network_gateway import ...`,
`from classifiers import ...`, needs the *module's own directory*). The `pythonpath`
list registers all six roots: `codebase/src`, plus — under `codebase/src/gnn/` —
`shared/utils`, `shared/models`, `reinforcement_learning`,
`reinforcement_learning/gateway`, and `supervised_learning`.

If you add a test, follow the import style of the module under test; if that module
lives in a directory not yet listed, add it to `pythonpath` in `pyproject.toml`.

## Code conventions

- **Naming is mixed by design.** Public, domain-level classes/functions use **PascalCase** (`ExpressionGraphConverter`, `TopologicalFeatureExtractor`, `LoadAugmentedFunctionGraph`, `MergeDisjointSubgraph`, `UnifiedDataLoader`, `RewardCalculator`, `NetworkGateway`). Internal helpers and free functions use snake_case (`signed_log_value`, `validate_edge_direction`, `to_homogeneous`, `read_rl_settings`). Module-level constants are UPPER_SNAKE (`ENRICHED_NODE_FEATURE_SCHEMA`, `RL_EXPERIMENT_CHOICES`). Match the file you are editing.
- `from __future__ import annotations` at the top of modules; type hints throughout (`py312` target).
- **Config pattern:** YAML defaults + argparse CLI overrides, resolved in a dedicated `*_config.py` (`supervised_config.py`, `rl_config.py`). Shared graph CLI args come from `feature_config.add_feature_cli_args` / `add_shared_graph_args` so supervised and RL stay identical. Validate enums (e.g. `validate_edge_direction`). Don't add ad-hoc `argparse` for graph/feature options — reuse the shared helpers.
- Comments in this repo are often German; README and `docs/` are German/English mixed. Keep new comments consistent with the surrounding file's language.

## Linting

Both ruff and flake8 are configured (line length **88**, `py312`):
- `pyproject.toml` → `[tool.ruff]`
- `.flake8` → `max-line-length = 88`, `extend-ignore = E203`

```bash
ruff check codebase/src
ruff format codebase/src        # formatter (88 cols)
flake8 codebase/src
```

## Reference docs (read before structural changes)

- `README.md` — full workflow/CLI reference (German).
- `docs/quick_overview.md`, `docs/in_depth_reference.md` — augmented graph loader API.
- `changenotes.md` — latest schema/backbone decisions (native edge features, node schema 27→19, etc.).

## CI / branch workflow

Branch chain is **`dev` → `main`** (no long-lived feature branches):

- `.github/workflows/tests.yml` runs the full `pytest` suite on every push to
  `dev` (and on PRs). On green it **auto-merges the tested commit into `main`**;
  on red, `main` is left untouched and you fix forward on `dev`.
- CI installs CPU torch (from the PyTorch CPU wheel index) + `requirements-ci.txt`
  (pinned to the `pytorch` env; `torch_scatter` is intentionally absent — the suite
  doesn't exercise it).
- `.github/workflows/docker-build.yml` builds/pushes the RunPod solver image to
  GHCR on push to `main` (needs a `Dockerfile` at the repo root).

Do day-to-day work on `dev` and let the gate promote to `main`. Still run `pytest`
locally before pushing so you don't burn a CI cycle on an obvious failure.

## Related skills
`graph-data-pipeline` (shared data/feature core), `supervised-gnn-training`, `rl-ppo-workflow`.
