"""YAML-backed configuration helpers for the GNN reinforcement-learning pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gnn.shared.utils.feature_config import (
    FeatureSelection,
    add_feature_cli_args,
    merge_feature_selection,
    parse_feature_selection_from_mapping,
    resolve_active_node_features,
)

RL_MODE_CHOICES = ("tree", "tree_derivatives")

# Default supervised-learning stage whose dataset/graph structure the RL pipeline
# consumes when no --stage is given. Accepts a number or folder name (see
# supervised_learning/config_settings/ and run_all.STAGE_REGISTRY).
# stage4_experiment = all 32 features, kappa off, supernode off — the kappa-free
# "full graph" that matches the RL pipeline's prior behaviour. (stage3_full_graph
# is identical but enables add_kappa, so it requires kappa data; opt in explicitly.)
RL_STAGE_DEFAULT = "stage4_experiment"

# Repo root: codebase/src/gnn/reinforcement_learning/rl_config.py -> parents[4].
# Matches supervised_learning/loader_graphgym.py so the `data:` paths in a stage
# config (which are relative to the repo root) resolve identically for both pipelines.
REPO_ROOT = Path(__file__).resolve().parents[4]


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class StageDataset:
    """Dataset + graph structure for an RL run, sourced from a supervised-learning
    stage config (``config_settings/<stage>/base_config.yaml``).

    Only the ``data`` + ``expression_graph`` blocks are consumed; the SL
    ``train``/``model``/``gnn``/``optim`` keys are irrelevant to RL, whose
    architecture is searched by Optuna. Keeping a single stage registry across
    both pipelines means the dataset structure is defined in exactly one place.
    """

    label: str  # stage folder name, e.g. "stage3_full_graph" (used for run/DB naming)
    mode: str
    add_kappa: bool
    add_virtual_supernode: bool
    expression_graph: dict[str, Any]  # raw block, fed to resolve_rl_features
    dataset_name: str
    graphs_path: Path  # directory holding graphs.json (GraphDataLoader base_dir)
    curated_csv: Path | None
    synthetic_csv: Path | None


def resolve_stage_dataset(stage: str) -> StageDataset:
    """Resolve an SL stage id/folder name to its dataset + graph structure.

    Reuses ``supervised_learning.run_all._resolve_stage`` so the stage registry
    is shared, then reads only the ``data`` + ``expression_graph`` blocks of the
    stage's ``base_config.yaml``. Raises SystemExit (via the SL resolver) on an
    unknown stage.
    """
    import gnn.supervised_learning.run_all as run_all

    sl_dir = Path(run_all.__file__).resolve().parent
    base_config_path, _grid_path = run_all._resolve_stage(str(stage), sl_dir)

    cfg = load_yaml_config(base_config_path)
    expression_graph = cfg.get("expression_graph") or {}
    data = cfg.get("data") or {}
    dataset = cfg.get("dataset") or {}

    graphs_dir = data.get("graphs_dir")
    if not graphs_dir:
        raise ValueError(
            f"Stage config {base_config_path} is missing data.graphs_dir; "
            f"cannot locate graphs.json for the RL pipeline."
        )

    def _abs(rel: Any) -> Path | None:
        return (REPO_ROOT / rel) if rel else None

    return StageDataset(
        label=base_config_path.parent.name,
        mode=str(expression_graph.get("mode", "tree_derivatives")),
        add_kappa=bool(expression_graph.get("add_kappa", False)),
        add_virtual_supernode=bool(expression_graph.get("add_virtual_supernode", False)),
        expression_graph=dict(expression_graph),
        dataset_name=str(dataset.get("name", base_config_path.parent.name)),
        graphs_path=_abs(graphs_dir),
        curated_csv=_abs(data.get("curated_csv")),
        synthetic_csv=_abs(data.get("synthetic_csv")),
    )


def read_rl_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Extract RL runtime settings from a config_rl.yaml-style dict.

    The dataset/graph structure (mode, features, kappa, supernode, data paths) is
    no longer carried here — it comes from the selected supervised-learning stage
    via ``resolve_stage_dataset``. This block only holds the stage selector plus the
    Optuna/gateway/train_best runtime knobs.
    """
    optuna = config.get("optuna") or {}
    gateway = config.get("gateway") or {}
    train_best = config.get("train_best") or {}

    return {
        "stage": config.get("stage", RL_STAGE_DEFAULT),
        "timesteps": int(optuna.get("timesteps", 10000)),
        "n_trials": int(optuna.get("n_trials", 50)),
        "n_envs": int(optuna.get("n_envs", 1)),
        "continue_study": bool(optuna.get("continue_study", False)),
        "timeout_fallback": float(gateway.get("timeout_fallback", 5.0)),
        "timeout_cushion": float(gateway.get("timeout_cushion", 1.0)),
        "timeout_window": int(gateway.get("timeout_window", 100)),
        "train_best_timesteps": int(train_best.get("timesteps", 250000)),
        "train_best_n_envs": int(train_best.get("n_envs", 1)),
        "save_dir": str(train_best.get("save_dir", "models")),
        "model_name": str(train_best.get("model_name", "gnn_ppo_best")),
        "no_torch_compile": bool(train_best.get("no_torch_compile", False)),
        "train_best_timeout_fallback": float(train_best.get("timeout_fallback", 5.0)),
        "train_best_timeout_cushion": float(train_best.get("timeout_cushion", 2.0)),
        "train_best_timeout_window": int(train_best.get("timeout_window", 100)),
    }


def resolve_rl_features(
    experiment: dict[str, Any],
    *,
    feature_groups: list[str] | None = None,
    node_features: list[str] | None = None,
    topology_features: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    active_features: list[str] | None = None,
) -> tuple[FeatureSelection, list[str] | None]:
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping(experiment),
        feature_groups=feature_groups,
        node_features=node_features,
        topology_features=topology_features,
        positional_encoding=positional_encoding,
        active_features=active_features,
    )
    return selection, resolve_active_node_features(selection)


def resolve_rl_setting(
    cli_value: Any,
    config_value: Any,
    *,
    is_flag: bool = False,
    flag_set: bool = False,
) -> Any:
    """Prefer an explicit CLI value; for store_true flags use flag_set."""
    if is_flag:
        return flag_set if flag_set else config_value
    if cli_value is not None:
        return cli_value
    return config_value


def add_shared_graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default="config_rl.yaml",
        help="YAML config with all RL CLI defaults (overridden by explicit flags).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help=(
            "Supervised-learning stage whose dataset/graph structure to use "
            "(number 1-4 or folder name, e.g. stage3_full_graph). Reads the "
            "data + expression_graph blocks of config_settings/<stage>/base_config.yaml."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=list(RL_MODE_CHOICES),
        help="GNN graph mode (tree, tree_derivatives). Overrides the stage's mode.",
    )
    parser.add_argument(
        "--add-kappa",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Merge kappa (h-function) subgraphs into each graph. Overrides the "
            "stage's add_kappa either way (--add-kappa / --no-add-kappa); "
            "unset = inherit the stage value."
        ),
    )
    parser.add_argument(
        "--add-virtual-supernode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Add a fully-connected virtual supernode (bidirectional edges to every node) "
            "to shorten message-passing paths. Overrides the stage's value either way "
            "(--add-virtual-supernode / --no-add-virtual-supernode); unset = inherit the stage."
        ),
    )
    add_feature_cli_args(parser)
