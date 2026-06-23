# Git branch manifest & health check

## Branch regime

```
dev  ──►  main
```

All day-to-day work lands on **`dev`**.  
`.github/workflows/tests.yml` is the gate:

| Event | What happens |
|---|---|
| Push to `dev` | Full `pytest` suite runs in CI |
| Suite **green** | `promote` job fast-forwards (or --no-ff merges) `dev` commit into `main` automatically |
| Suite **red** | `main` is untouched; fix forward on `dev` |
| Push directly to `main` | **Avoid** — causes branch divergence that blocks the auto-promote |

**Never push directly to `main`.** The docker-build workflow (`docker-build.yml`) fires on `main` pushes; only the `promote` job should write there.

## Diagnosing a blocked promote

Run this to see how many dev commits are stranded:

```bash
git log --oneline origin/main..HEAD
```

Then check what CI actually said:

```bash
gh run list --branch dev --limit 10
gh run view <RUN_ID> --log-failed
```

Common causes and fixes:

| Symptom in log | Root cause | Fix |
|---|---|---|
| `fatal: Not possible to fast-forward` + `CONFLICT` | Someone pushed to `main` directly; branches diverged | Merge `origin/main` into `dev`, resolve conflicts, push |
| `fatal: Not possible to fast-forward` (no conflicts) | `main` has a commit not in `dev` (clean divergence) | `git merge origin/main --no-edit` |
| `pytest` failures | Tests broken by a dev commit | Fix the failing test or source on `dev`, push |
| `pip install` error in CI | Package version in `requirements-ci.txt` doesn't exist | Update the pinned version; check `torch` must use `--index-url https://download.pytorch.org/whl/cpu` |

## Resolving a diverged-branch conflict

```bash
# 1. Pull in the missing main commit
git merge origin/main --no-edit        # may conflict

# 2. If conflicts:  resolve, then
git add <conflicted-files>
git commit                             # completes the merge

# 3. Verify locally BEFORE pushing
conda activate pytorch
pytest                                 # must be 190 passed (or more)

# 4. Push — CI reruns and auto-promotes on green
git push origin dev
```

After step 4 the `promote` job can fast-forward `main` to the new merge commit (because the missing `main` commit is now an ancestor of `dev` HEAD).

## Pre-push checklist (run locally, save a CI cycle)

```bash
conda activate pytorch
pytest                    # whole suite must be green
ruff check codebase/src   # no lint errors
```

If `pytest` is slow, run only the file you changed:

```bash
pytest codebase/src/gnn/tests/test_<relevant>.py -v
```

## Key files

| File | Purpose |
|---|---|
| `.github/workflows/tests.yml` | CI gate + auto-promote logic |
| `.github/workflows/docker-build.yml` | Builds RunPod solver image on `main` push |
| `requirements-ci.txt` | Pinned deps for CI (torch installed separately via CPU wheel index) |
| `pyproject.toml` | `[tool.pytest.ini_options]` — `pythonpath` roots + `testpaths` |

## Related skills

`gnn-dev-workflow` (full env + test invocation), `supervised-gnn-training`, `rl-ppo-workflow`.
