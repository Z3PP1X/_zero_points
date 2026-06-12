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

## Running tests (non-obvious — read this before debugging import errors)

Tests live in `codebase/src/gnn/tests/` (27 files) but there is **no conftest or pytest config wiring up paths**, and tests use **two different import styles**:

- Package style: `from gnn.shared.utils.feature_config import ...` → needs `codebase/src` on `PYTHONPATH`.
- Bare-module style: `from reward import ...`, `from graph_utils import ...`, `from feature_layout import ...`, `from unified_loader import ...`, `from network_gateway import ...` → needs the *module's own directory* on `PYTHONPATH` (`shared/utils`, `reinforcement_learning`, `supervised_learning`).

So `python -m pytest codebase/src/gnn/tests` from the repo root fails with `ModuleNotFoundError: No module named 'gnn'`. The working invocation, from the repo root inside the `pytorch` env, sets all roots:

```bash
PYTHONPATH=codebase/src:codebase/src/gnn/shared/utils:codebase/src/gnn/reinforcement_learning:codebase/src/gnn/supervised_learning \
  python -m pytest codebase/src/gnn/tests -q
```

Run a single file the same way, e.g. `... python -m pytest codebase/src/gnn/tests/test_reward.py -q`. (Pure-Python tests like `test_reward` pass even in the base env; anything importing torch/PyG needs the `pytorch` env.) Tip: export that `PYTHONPATH` once in the shell for the session.

If you add a test, follow the existing import style for the module under test and make sure its directory is one of the four roots above.

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

## CI

`.github/workflows/` only builds & pushes a Docker image to GHCR on push to `main` (the RunPod solver workspace). **There is no test job in CI** — run the suite locally before pushing.

## Related skills
`graph-data-pipeline` (shared data/feature core), `supervised-gnn-training`, `rl-ppo-workflow`.
