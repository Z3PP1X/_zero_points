"""YAML-backed configuration helpers for the GNN reinforcement-learning pipeline."""

from __future__ import annotations

import argparse
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
from gnn.shared.utils.graph_utils import validate_edge_direction

RL_EXPERIMENT_CHOICES = ("nur_f", "f_fp_roh", "kein_inv")
RL_MODE_CHOICES = ("graph", "tree", "tree_derivatives")
RL_EDGE_DIRECTION_CHOICES = ("top_down", "bottom_up", "bidirectional")


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_rl_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Extract RL runtime settings from a config_rl.yaml-style dict."""
    experiment = config.get("experiment") or {}
    optuna = config.get("optuna") or {}
    gateway = config.get("gateway") or {}
    train_best = config.get("train_best") or {}

    feature_selection, active_features = resolve_rl_features(experiment)

    return {
        "experiment": experiment.get("name", "nur_f"),
        "mode": experiment.get("mode", "graph"),
        "edge_direction": validate_edge_direction(
            experiment.get("edge_direction", "top_down")
        ),
        "add_kappa": bool(experiment.get("add_kappa", False)),
        "active_features": active_features,
        "feature_selection": feature_selection,
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
    edge_features: list[str] | None = None,
    active_features: list[str] | None = None,
) -> tuple[FeatureSelection, list[str] | None]:
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping(experiment),
        feature_groups=feature_groups,
        node_features=node_features,
        topology_features=topology_features,
        positional_encoding=positional_encoding,
        edge_features=edge_features,
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
        "--mode",
        type=str,
        default=None,
        choices=list(RL_MODE_CHOICES),
        help="GNN graph mode (graph, tree, tree_derivatives).",
    )
    parser.add_argument(
        "--edge-direction",
        type=str,
        default=None,
        choices=list(RL_EDGE_DIRECTION_CHOICES),
        help="AST message-passing direction (virtual-node edges stay bidirectional).",
    )
    parser.add_argument(
        "--add-kappa",
        action="store_true",
        help="Merge kappa (h-function) subgraphs from datasets/kappas/ into each graph.",
    )
    add_feature_cli_args(parser)
