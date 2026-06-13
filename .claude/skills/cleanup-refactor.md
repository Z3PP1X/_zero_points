---
name: cleanup-refactor
description: Identify and remove dead code left behind by a rewrite — stale constants, no-op functions, orphaned call sites, and outdated comments — then record any non-trivial removals in known_bugs.md.
---

<cleanup-refactor>

## Purpose

After a significant rewrite (schema change, architecture swap, feature removal) there is always
dead code left behind: constants that reference removed fields, functions stubbed to `pass`,
import/call sites for those stubs, and comments that describe what things used to do.

This skill handles that cleanup pass systematically.

## Workflow

### 1. Identify dead symbols

Search for patterns that indicate dead code:

```bash
# No-op functions (pass-only body)
grep -rn "^\s*pass$" codebase/src/gnn/ --include="*.py" -B3

# Schema-index calls on non-existent columns
grep -rn "\.index(" codebase/src/gnn/shared/models/ --include="*.py"

# Imports of removed symbols
grep -rn "from .* import" codebase/src/gnn/ --include="*.py" | grep -E "(label_id|belongs_to|lpe|rwpe)"

# Stale TODO/REMOVED comments referencing old names
grep -rn "# .*removed\|# .*no.op\|# .*TODO.*rewrite" codebase/src/gnn/ --include="*.py" -i
```

### 2. Classify each candidate

For each dead symbol, determine:

| Question | Action |
|----------|--------|
| Is it a module-level expression that crashes on import? | **Fix immediately** — it blocks the whole pipeline |
| Is the function body `pass` with callers still present? | Remove function AND all import/call sites |
| Is a constant's target column/field gone from the schema? | Remove constant; grep for all usages first |
| Is a comment describing removed behaviour? | Remove or rewrite comment |
| Does a test assert that a call is a no-op? | Remove the no-op test (it tests nothing) |

### 3. Execution order

Always fix in this order to avoid broken intermediate states:

1. **Remove the symbol** (function, constant, class) from its definition file.
2. **Remove imports** in every consumer file.
3. **Remove call sites** in every consumer file.
4. **Remove tests** that only verify the no-op / removed behaviour.
5. **Update `known_bugs.md`** — move resolved items to the "Fixed" section.

### 4. Record non-trivial removals

Add a bullet to the `## Fixed in the audit pass` section of `known_bugs.md` for:
- Anything that was crashing on import.
- Any no-op function with multiple call sites.
- Any schema mismatch that silently produced wrong output.

Skip the log for pure comment/whitespace cleanup.

## Common patterns in this repo

### Stale column-index constants (`gnn_backbones.py`, `classifiers.py`)
Constants like `LABEL_ID_COL = NODE_FEATURE_SCHEMA.index("label_id")` crash at import
when the named column is no longer in `NODE_FEATURE_SCHEMA`. Always verify the target
column exists in the schema before keeping such constants. Remove and grep for all usages.

### No-op schema-injection functions (`graph_utils.py`)
Functions like `populate_task_virtual_values` that are `pass` bodies indicate a schema
feature was removed but call sites were not cleaned up. Remove function + all `import`
and call sites in `preprocessing.py`, `rl/preprocessor.py`, and test files.

### Subtree-routing blocks that lost their mask columns
`_legacy_forward` aggregator blocks that read feature columns by name fall back to
all-ones/all-zeros defaults when the column is absent — they become deterministic no-ops.
Remove the inner helper function and the loop, keeping only the surviving real/virtual
split logic that depends on `node_type`.

### Dead feature references in YAML / config comments
Config files often reference removed feature names (`label_id`, `lpe`, `rwpe`) in
comments or example values. Grep the YAML files for removed names and update.

## What NOT to remove

- `NODE_TYPE_COL` — actively used by tests and the virtual/real split in forwards.
- `kappa_weight` NetworkX edge attribute — still set on kappa edges for debugging; only
  the schema tensor entry was dropped.
- GraphConversionPipeline / SAGEStackNetwork — tracked in `known_bugs.md` SH-6; keep
  until a clean test run confirms nothing calls them.

</cleanup-refactor>
