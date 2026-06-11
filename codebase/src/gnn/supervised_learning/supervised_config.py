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
    BASIC_EDGE_FEATURE_SCHEMA,
    ENRICHED_EDGE_FEATURE_SCHEMA,
    validate_edge_direction,
)

SUPERVISED_LAYER_TYPES: tuple[str, ...] = (
    "gatv2conv",
    "gineconv",
    "gcnconv",
    "ginconv",
)

LAYER_TYPE_TO_ARCHITECTURE: dict[str, str] = {
    "gatv2conv": "gatv2_stack",
    "gineconv": "gine_stack",
    "gcnconv": "gcn_stack",
    "ginconv": "gin_stack",
}

LAYERS_WITHOUT_EDGE_FEATURES: frozenset[str] = frozenset({"gcnconv", "ginconv"})


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _warn_no_edge_features(layer_type: str) -> None:
    if layer_type in LAYERS_WITHOUT_EDGE_FEATURES:
        architecture = LAYER_TYPE_TO_ARCHITECTURE[layer_type]
        print(
            f"Warning: selected architecture {architecture} does not support edge_features"
        )


def validate_layer_type(layer_type: str) -> str:
    if layer_type not in SUPERVISED_LAYER_TYPES:
        raise ValueError(
            f"Unsupported gnn.layer_type {layer_type!r}; "
            f"expected one of {list(SUPERVISED_LAYER_TYPES)}"
        )
    _warn_no_edge_features(layer_type)
    return layer_type


def architecture_from_layer_type(layer_type: str) -> str:
    return LAYER_TYPE_TO_ARCHITECTURE[validate_layer_type(layer_type)]


def edge_dim_for_enrich(enrich: bool) -> int:
    schema = ENRICHED_EDGE_FEATURE_SCHEMA if enrich else BASIC_EDGE_FEATURE_SCHEMA
    return len(schema)


def get_batch_edge_attr(batch, enrich: bool):
    """Return edge attributes for a PyG batch; require them when enrich=True."""
    edge_attr = getattr(batch, "edge_attr", None)
    if enrich:
        if edge_attr is None:
            raise ValueError(
                "enrich=True requires edge_attr on every graph batch, but edge_attr is missing"
            )
        return edge_attr
    return edge_attr


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

    enrich = bool(getattr(cfg.expression_graph, "enrich", False))
    cfg.dataset.edge_dim = edge_dim_for_enrich(enrich)
    validate_layer_type(cfg.gnn.layer_type)
    return cfg


def resolve_expression_graph_features(
    expression_graph: dict[str, Any] | None,
    *,
    enrich: bool,
    feature_groups: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    active_features: list[str] | None = None,
) -> tuple[FeatureSelection, list[str] | None]:
    """Resolve grouped feature toggles into an active node-feature list."""
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping(expression_graph),
        feature_groups=feature_groups,
        positional_encoding=positional_encoding,
        active_features=active_features,
    )
    return selection, resolve_active_node_features(selection, enrich=enrich)


def apply_expression_graph_overrides(
    cfg,
    *,
    mode: str | None = None,
    enrich: bool | None = None,
    active_features: list[str] | None = None,
    feature_groups: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    synthetic: bool | None = None,
    synthetic_dataset: str | None = None,
    edge_direction: str | None = None,
) -> FeatureSelection:
    """Apply CLI overrides onto a loaded GraphGym cfg."""
    if mode is not None:
        cfg.expression_graph.mode = mode
    current_enrich = enrich if enrich is not None else bool(cfg.expression_graph.enrich)
    if enrich is not None:
        cfg.expression_graph.enrich = enrich
        cfg.dataset.edge_dim = edge_dim_for_enrich(enrich)
    from gnn.shared.utils.feature_config import plain_dict

    selection, resolved_features = resolve_expression_graph_features(
        {
            "features": plain_dict(getattr(cfg.expression_graph, "features", {})),
            "active_features": getattr(cfg.expression_graph, "active_features", ""),
        },
        enrich=current_enrich,
        feature_groups=feature_groups,
        positional_encoding=positional_encoding,
        active_features=active_features,
    )
    cfg.expression_graph.active_features = active_features_to_csv(resolved_features)
    if synthetic is not None:
        cfg.expression_graph.synthetic = synthetic
    if synthetic_dataset is not None:
        cfg.expression_graph.synthetic_dataset = synthetic_dataset
    if edge_direction is not None:
        cfg.expression_graph.edge_direction = validate_edge_direction(edge_direction)
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

    enrich = bool(expression_graph.get("enrich", False))
    layer_type = validate_layer_type(gnn_cfg.get("layer_type", "gatv2conv"))
    feature_selection, active_features = resolve_expression_graph_features(
        expression_graph,
        enrich=enrich,
    )

    return {
        "dataset_name": dataset_cfg.get("name"),
        "mode": expression_graph.get("mode", "graph"),
        "enrich": enrich,
        "edge_direction": validate_edge_direction(
            expression_graph.get("edge_direction", "top_down")
        ),
        "layer_type": layer_type,
        "architecture": architecture_from_layer_type(layer_type),
        "edge_dim": edge_dim_for_enrich(enrich),
        "synthetic": bool(expression_graph.get("synthetic", False)),
        "synthetic_dataset": expression_graph.get("synthetic_dataset") or None,
        "active_features": active_features,
        "feature_selection": feature_selection,
    }
