"""Shared supervised-learning config helpers for GraphGym YAML and main.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from gnn.shared.utils.feature_config import (
    FeatureSelection,
    active_features_to_csv,
    merge_feature_selection,
    parse_feature_selection_from_mapping,
    resolve_active_node_features,
)
from gnn.shared.utils.graph_utils import (
    EDGE_FEATURE_SCHEMA,
)

SUPERVISED_LAYER_TYPES: tuple[str, ...] = ("ginconv",)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_layer_type(layer_type: str) -> str:
    if layer_type not in SUPERVISED_LAYER_TYPES:
        raise ValueError(
            f"Unsupported gnn.layer_type {layer_type!r}; "
            f"expected one of {list(SUPERVISED_LAYER_TYPES)}"
        )
    return layer_type


def resolve_edge_dim() -> int:
    return len(EDGE_FEATURE_SCHEMA)


def bootstrap_graphgym_cfg(config_path: Path | str, seed: int | None = None):
    """Load GraphGym cfg from YAML and register custom loaders/layers/loss."""
    from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

    import gnn.supervised_learning.loader_graphgym  # noqa: F401

    set_cfg(cfg)
    args = argparse.Namespace(cfg_file=str(config_path), opts=[])
    load_cfg(cfg, args)

    if seed is not None:
        cfg.seed = seed
    cfg.optim.max_epoch = cfg.train.epochs
    if getattr(cfg, "accelerator", "auto") == "auto":
        cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dataset.edge_dim = resolve_edge_dim()
    validate_layer_type(cfg.gnn.layer_type)
    return cfg


def resolve_expression_graph_features(
    expression_graph: dict[str, Any] | None,
    *,
    feature_groups: list[str] | None = None,
    node_features: list[str] | None = None,
    topology_features: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    active_features: list[str] | None = None,
) -> tuple[FeatureSelection, list[str] | None]:
    """Resolve grouped feature toggles into an active node-feature list."""
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping(expression_graph),
        feature_groups=feature_groups,
        node_features=node_features,
        topology_features=topology_features,
        positional_encoding=positional_encoding,
        active_features=active_features,
    )
    return selection, resolve_active_node_features(selection)


def apply_expression_graph_overrides(
    cfg,
    *,
    mode: str | None = None,
    active_features: list[str] | None = None,
    feature_groups: list[str] | None = None,
    node_features: list[str] | None = None,
    topology_features: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    synthetic: bool | None = None,
    synthetic_dataset: str | None = None,
    add_kappa: bool | None = None,
    add_virtual_supernode: bool | None = None,
) -> FeatureSelection:
    """Apply CLI overrides onto a loaded GraphGym cfg."""
    if mode is not None:
        cfg.expression_graph.mode = mode
    from gnn.shared.utils.feature_config import plain_dict

    selection, resolved_features = resolve_expression_graph_features(
        {
            "features": plain_dict(getattr(cfg.expression_graph, "features", {})),
            "active_features": getattr(cfg.expression_graph, "active_features", ""),
        },
        feature_groups=feature_groups,
        node_features=node_features,
        topology_features=topology_features,
        positional_encoding=positional_encoding,
        active_features=active_features,
    )
    cfg.expression_graph.active_features = active_features_to_csv(resolved_features)
    if synthetic is not None:
        cfg.expression_graph.synthetic = synthetic
    if synthetic_dataset is not None:
        cfg.expression_graph.synthetic_dataset = synthetic_dataset
    if add_kappa is not None:
        cfg.expression_graph.add_kappa = add_kappa
    if add_virtual_supernode is not None:
        cfg.expression_graph.add_virtual_supernode = add_virtual_supernode
    return selection


def create_graphgym_model(cfg, dim_in: int, device: str | torch.device):
    """Instantiate the same GraphGym GNN stack used by main_graphgym.py."""
    from torch_geometric.graphgym.model_builder import create_model

    cfg.share.dim_in = dim_in
    cfg.share.dim_out = 2
    model = create_model(to_device=False, dim_in=dim_in)
    return model.to(device)


def read_supervised_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Extract supervised training settings from a GraphGym-style YAML dict."""
    expression_graph = config.get("expression_graph") or {}
    gnn_cfg = config.get("gnn") or {}
    dataset_cfg = config.get("dataset") or {}

    layer_type = validate_layer_type(gnn_cfg.get("layer_type", "ginconv"))
    feature_selection, active_features = resolve_expression_graph_features(
        expression_graph,
    )

    return {
        "dataset_name": dataset_cfg.get("name"),
        "mode": expression_graph.get("mode", "tree_derivatives"),
        "add_kappa": bool(expression_graph.get("add_kappa", False)),
        "add_virtual_supernode": bool(
            expression_graph.get("add_virtual_supernode", False)
        ),
        "layer_type": layer_type,
        "edge_dim": resolve_edge_dim(),
        "synthetic": bool(expression_graph.get("synthetic", False)),
        "synthetic_dataset": expression_graph.get("synthetic_dataset") or None,
        "active_features": active_features,
        "feature_selection": feature_selection,
    }
