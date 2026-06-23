# CLAUDE.md — Project Instructions

## Before every `git push`

**ALWAYS run the full test suite before pushing.** This is mandatory, not optional.

```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points
pytest
```

If any test fails: fix it first. Never push with failing tests.

## Quick reference

| Command | What it does |
|---|---|
| `pytest` | Run all 150+ tests from repo root |
| `pytest -k smoke` | Run only smoke/pipeline tests |
| `pytest -x` | Stop at first failure |

## Environments

- **Local (WSL2)**: `/home/zapp1x/GitHub/_bachelor/_zero_points/`
- **Docker container**: `/workspace/_zero_points/` (same filesystem, mounted)
- Conda env: `zero_points` — activate before running anything

## Dataset paths (relative to repo root)

- Curated CSV: `datasets/run_20260604_154509/dataset2_joined.csv`
- Synthetic CSV: `datasets/run_20260604_154509/synthetic_dataset2.csv`
- Graphs JSON: `datasets/graphs/graphs.json` + `synthetic_graphs.json`
- Graph cache: `datasets/graphs/.pt_cache/` — auto-generated, safe to delete on schema changes

## Feature schema

Total features: **32 columns** in `NODE_FEATURE_SCHEMA` (graph_vocab.py):
- node_type (3: global, operator, function) + root_color (5) + label (15, incl. label_Sqrt) + topology (6) + positional (3)
- Virtual supernode (`add_virtual_supernode: true`) wird als `node_type_global=1` encodiert — kein eigenes Feature nötig
- Adding a new label: update `CANONICAL_LABELS`, `LABEL_ONEHOT_NAMES`, `NODE_FEATURE_SCHEMA`, `NODE_FEATURES` — then **delete `.pt_cache/`**

## Stage overview

| Stage | Features | active_features |
|---|---|---|
| 1 — pure AST | 18 (node_type + label) | explicit CSV list |
| 2 — AST roots | 23 (+ root_color) | explicit CSV list |
| 3 — full graph | 32 (all) | `""` (all schema columns) |
| 4 — experiment | 32 default | `""` or explicit |
