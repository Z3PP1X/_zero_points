"""Shared supervised-learning config helpers for GraphGym YAML and main.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gnn.shared.utils.graph_utils import (
    BASIC_EDGE_FEATURE_SCHEMA,
    ENRICHED_EDGE_FEATURE_SCHEMA,
)

SUPERVISED_LAYER_TYPES: tuple[str, ...] = ("gatv2conv", "gineconv")

LAYER_TYPE_TO_ARCHITECTURE: dict[str, str] = {
    "gatv2conv": "gatv2_stack",
    "gineconv": "gine_stack",
}


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


def read_supervised_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Extract supervised training settings from a GraphGym-style YAML dict."""
    expression_graph = config.get("expression_graph") or {}
    gnn_cfg = config.get("gnn") or {}
    dataset_cfg = config.get("dataset") or {}

    enrich = bool(expression_graph.get("enrich", False))
    layer_type = validate_layer_type(gnn_cfg.get("layer_type", "gatv2conv"))
    active_features_str = expression_graph.get("active_features") or ""
    active_features = None
    if active_features_str:
        active_features = [
            feature.strip()
            for feature in str(active_features_str).split(",")
            if feature.strip()
        ]

    return {
        "dataset_name": dataset_cfg.get("name"),
        "mode": expression_graph.get("mode", "graph"),
        "enrich": enrich,
        "layer_type": layer_type,
        "architecture": architecture_from_layer_type(layer_type),
        "edge_dim": edge_dim_for_enrich(enrich),
        "synthetic": bool(expression_graph.get("synthetic", False)),
        "synthetic_dataset": expression_graph.get("synthetic_dataset") or None,
        "active_features": active_features,
    }
