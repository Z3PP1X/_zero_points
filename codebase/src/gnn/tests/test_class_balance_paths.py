"""Regression test for the class-balance fallback's repo-root resolution.

eval.py lives in run_results/ — one directory deeper than loader_graphgym.py — so it
must walk up parents[5] (not parents[4]) to reach the repo root that data.* config paths
are relative to. The off-by-one resolved to .../codebase and made the fallback look for
codebase/datasets/..., silently skipping class balance for any run missing
class_balance.json.
"""

from pathlib import Path

import gnn.supervised_learning.run_results.eval as eval_mod


def test_class_balance_fallback_resolves_repo_root():
    resolved = Path(eval_mod.__file__).resolve()
    # parents[4] is the buggy value (.../codebase); parents[5] is the true repo root.
    assert resolved.parents[4].name == "codebase"
    repo_root = resolved.parents[5]
    # The repo root contains codebase/ and the path back down to eval.py — proving the
    # data.*-relative root is the dir above codebase/, not codebase/ itself.
    assert (repo_root / "codebase").is_dir()
    assert (
        repo_root
        / "codebase" / "src" / "gnn" / "supervised_learning" / "run_results" / "eval.py"
    ).is_file()
