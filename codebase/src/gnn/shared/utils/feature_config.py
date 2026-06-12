"""Feature catalog, grouping, and resolution for expression-graph GNN pipelines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable

from gnn.shared.utils.graph_utils import (
    EDGE_FEATURE_SCHEMA,
    NODE_FEATURE_SCHEMA,
)

FEATURE_CLASSES: tuple[str, ...] = ("node", "topology", "positional", "edge")
POSITIONAL_ENCODING_CHOICES: tuple[str, ...] = ("lpe", "rwpe")

NODE_FEATURES: tuple[str, ...] = (
    "node_type",
    "label_id",
    "value",
    "has_value",
    "virtual_current_x_val",
    "virtual_delta_target_val",
    "virtual_d1_x_val",
    "virtual_d2_x_val",
    "belongs_to_f",
    "belongs_to_d1",
    "belongs_to_d2",
)

TOPOLOGY_FEATURES: tuple[str, ...] = (
    "depth",
    "height",
    "subtree_size",
    "out_degree",
    "betweenness_centrality",
)

POSITIONAL_ENCODING_FEATURES: dict[str, tuple[str, ...]] = {
    "lpe": ("lpe_1", "lpe_2", "lpe_3", "lpe_4"),
    "rwpe": ("rwpe_1", "rwpe_2", "rwpe_3", "rwpe_4"),
}

EDGE_FEATURES: tuple[str, ...] = tuple(EDGE_FEATURE_SCHEMA)


def full_node_schema() -> list[str]:
    return list(NODE_FEATURE_SCHEMA)


def plain_dict(value: Any) -> dict[str, Any]:
    """Recursively convert YACS CfgNode-like mappings to plain dicts."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): plain_value(item) for key, item in value.items()}
    return {}


def plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return plain_dict(value)
    if hasattr(value, "items"):
        return plain_dict(value)
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value


def parse_csv_list(value: str | Iterable[str] | None) -> list[str] | None:
    """Parse comma- or space-separated CLI/YAML list values."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
        return parts or None
    return [str(part).strip() for part in value if str(part).strip()] or None


def validate_feature_groups(groups: Iterable[str]) -> list[str]:
    groups_list = list(groups)
    unknown = sorted({group for group in groups_list if group not in FEATURE_CLASSES})
    if unknown:
        raise ValueError(
            f"Unknown feature group(s) {unknown}; expected subset of {list(FEATURE_CLASSES)}"
        )
    return groups_list


def validate_positional_encodings(encodings: Iterable[str]) -> list[str]:
    encodings_list = [encoding.lower() for encoding in encodings]
    unknown = sorted(
        {encoding for encoding in encodings_list if encoding not in POSITIONAL_ENCODING_CHOICES}
    )
    if unknown:
        raise ValueError(
            f"Unknown positional encoding(s) {unknown}; "
            f"expected subset of {list(POSITIONAL_ENCODING_CHOICES)}"
        )
    return encodings_list


# Selectable members per feature category. Each category accepts the same uniform
# flat form in config/CLI: ``true`` (all members), ``false`` (none), or a list subset.
CATEGORY_MEMBERS: dict[str, tuple[str, ...]] = {
    "node": NODE_FEATURES,
    "topology": TOPOLOGY_FEATURES,
    "positional": POSITIONAL_ENCODING_CHOICES,
    "edge": EDGE_FEATURES,
}


def _validate_members(
    values: Iterable[str], members: tuple[str, ...], *, label: str
) -> tuple[str, ...]:
    """Validate a member subset and return it in canonical (catalog) order."""
    lowered = [str(value).lower() for value in values]
    unknown = sorted({value for value in lowered if value not in members})
    if unknown:
        raise ValueError(
            f"Unknown {label} feature(s) {unknown}; expected subset of {list(members)}"
        )
    return tuple(member for member in members if member in lowered)


def _resolve_category_value(
    value: Any, members: tuple[str, ...], *, label: str
) -> bool | tuple[str, ...]:
    """Normalize a flat category config value.

    ``True``/``None`` -> all members, ``False`` or empty list -> none,
    a list/csv-string -> validated ordered subset.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        raise ValueError(
            f"Nested feature config for '{label}' is no longer supported; "
            f"use the flat form '{label}: true | false | [members]'."
        )
    parsed = parse_csv_list(value)
    if parsed is None:
        return False
    return _validate_members(parsed, members, label=label)


def _category_member_list(
    value: bool | tuple[str, ...] | None, members: tuple[str, ...]
) -> list[str]:
    """Expand a normalized category value into its active member names."""
    if value is True:
        return list(members)
    if not value:  # False, None, or empty tuple
        return []
    return list(value)


def _split_positional(value: bool | tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
    """Map a normalized positional value onto (enabled, encodings)."""
    if value is True:
        return True, POSITIONAL_ENCODING_CHOICES
    if not value:
        return False, ()
    return True, tuple(value)


def _coerce_edge(value: bool | tuple[str, ...]) -> bool:
    """Edge slicing is deferred; a member list means 'enabled' (all edge columns)."""
    return bool(value)


def _apply_list_override(
    values: list[str], members: tuple[str, ...], *, label: str
) -> bool | tuple[str, ...]:
    """Interpret a per-category CLI list override (supports ``none``/``all`` sentinels)."""
    if values == ["none"]:
        return False
    if values == ["all"]:
        return True
    return _validate_members(values, members, label=label)


@dataclass
class FeatureSelection:
    """Resolved feature toggles for one experiment run.

    ``node`` and ``topology`` are either a bool (all/none) or a tuple subset of
    member names. ``positional`` is represented by the derived
    ``positional_enabled`` + ``positional_encodings`` pair. ``edge`` stays a bool
    (per-edge-feature slicing is deferred).
    """

    node: bool | tuple[str, ...] = True
    topology: bool | tuple[str, ...] = True
    positional_enabled: bool = True
    positional_encodings: tuple[str, ...] = ("lpe", "rwpe")
    edge: bool = True
    explicit_features: list[str] | None = None

    def enabled_groups(self) -> list[str]:
        groups: list[str] = []
        if self.node:
            groups.append("node")
        if self.topology:
            groups.append("topology")
        if self.positional_enabled and self.positional_encodings:
            groups.append("positional")
        if self.edge:
            groups.append("edge")
        return groups

    def summary(self) -> str:
        if self.explicit_features is not None:
            return f"explicit ({len(self.explicit_features)}): {self.explicit_features}"
        active = resolve_active_node_features(self)
        if active is None:
            return "all node features"
        return f"{len(active)} node features: {active}"


def default_feature_selection() -> FeatureSelection:
    return FeatureSelection()


def parse_feature_selection_from_mapping(
    expression_graph: dict[str, Any] | None,
) -> FeatureSelection:
    """Read grouped feature toggles from an expression_graph YAML/CFG mapping.

    Every category uses the uniform flat form: ``true``/omitted -> all members,
    ``false`` -> none, a list -> only the listed members.
    """
    expression_graph = plain_dict(expression_graph)
    features = plain_dict(expression_graph.get("features"))

    node = _resolve_category_value(
        features.get("node", True), NODE_FEATURES, label="node"
    )
    topology = _resolve_category_value(
        features.get("topology", True), TOPOLOGY_FEATURES, label="topology"
    )
    positional_value = _resolve_category_value(
        features.get("positional", True), POSITIONAL_ENCODING_CHOICES, label="positional"
    )
    edge_value = _resolve_category_value(
        features.get("edge", True), EDGE_FEATURES, label="edge"
    )

    positional_enabled, positional_encodings = _split_positional(positional_value)
    explicit_features = parse_csv_list(expression_graph.get("active_features"))

    return FeatureSelection(
        node=node,
        topology=topology,
        positional_enabled=positional_enabled,
        positional_encodings=positional_encodings,
        edge=_coerce_edge(edge_value),
        explicit_features=explicit_features,
    )


def merge_feature_selection(
    base: FeatureSelection,
    *,
    feature_groups: list[str] | None = None,
    node_features: list[str] | None = None,
    topology_features: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    edge_features: list[str] | None = None,
    active_features: list[str] | None = None,
) -> FeatureSelection:
    """Apply CLI overrides onto a YAML-backed FeatureSelection.

    ``--feature-groups`` sets coarse category on/off first; the per-category list
    overrides (``node_features``/``topology_features``/``positional_encoding``/
    ``edge_features``) then refine individual categories.
    """
    selection = FeatureSelection(
        node=base.node,
        topology=base.topology,
        positional_enabled=base.positional_enabled,
        positional_encodings=base.positional_encodings,
        edge=base.edge,
        explicit_features=base.explicit_features,
    )

    if feature_groups is not None:
        enabled = set(validate_feature_groups(feature_groups))
        selection.node = "node" in enabled
        selection.topology = "topology" in enabled
        selection.edge = "edge" in enabled
        selection.positional_enabled = "positional" in enabled

    if node_features is not None:
        selection.node = _apply_list_override(node_features, NODE_FEATURES, label="node")
    if topology_features is not None:
        selection.topology = _apply_list_override(
            topology_features, TOPOLOGY_FEATURES, label="topology"
        )
    if edge_features is not None:
        selection.edge = _coerce_edge(
            _apply_list_override(edge_features, EDGE_FEATURES, label="edge")
        )

    if positional_encoding is not None:
        if positional_encoding == ["none"]:
            selection.positional_enabled = False
            selection.positional_encodings = ()
        else:
            selection.positional_enabled = True
            selection.positional_encodings = tuple(
                validate_positional_encodings(positional_encoding)
            )

    if active_features is not None:
        selection.explicit_features = active_features

    return selection


def resolve_active_node_features(
    selection: FeatureSelection,
) -> list[str] | None:
    """
    Return active node-feature names in schema order.

    ``None`` means all native node features (no slicing).
    """
    schema = full_node_schema()

    if selection.explicit_features is not None:
        if not selection.explicit_features:
            raise ValueError("active_features cannot be empty; disable groups instead.")
        missing = [
            feature
            for feature in selection.explicit_features
            if feature not in schema
        ]
        if missing:
            raise ValueError(
                f"Unknown active feature(s) {missing}; available: {schema}"
            )
        if selection.explicit_features == schema:
            return None
        return list(selection.explicit_features)

    enabled: list[str] = []
    enabled.extend(_category_member_list(selection.node, NODE_FEATURES))
    enabled.extend(_category_member_list(selection.topology, TOPOLOGY_FEATURES))
    if selection.positional_enabled:
        for encoding in selection.positional_encodings:
            enabled.extend(POSITIONAL_ENCODING_FEATURES[encoding])

    # Preserve schema order and drop duplicates.
    ordered = [feature for feature in schema if feature in enabled]
    if ordered == schema:
        return None
    if not ordered:
        raise ValueError(
            "No node features selected. Enable at least one of "
            f"{list(FEATURE_CLASSES[:-1])} or provide --active-features."
        )
    return ordered


def active_features_to_csv(active_features: list[str] | None) -> str:
    if not active_features:
        return ""
    return ",".join(active_features)


def feature_catalog_markdown() -> str:
    """Human-readable overview of feature classes and members."""
    lines = [
        "Feature classes:",
        f"  node:       {list(NODE_FEATURES)}",
        f"  topology:   {list(TOPOLOGY_FEATURES)}",
        "  positional:",
    ]
    for name, members in POSITIONAL_ENCODING_FEATURES.items():
        lines.append(f"    {name}: {list(members)}")
    lines.append(f"  edge:       {list(EDGE_FEATURES)}")
    return "\n".join(lines)


def add_feature_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=None,
        metavar="GROUP",
        choices=list(FEATURE_CLASSES),
        help=(
            "Enable only these feature classes "
            f"({', '.join(FEATURE_CLASSES)}). Default: all groups from config."
        ),
    )
    parser.add_argument(
        "--node-features",
        nargs="+",
        default=None,
        metavar="FEATURE",
        choices=[*NODE_FEATURES, "none", "all"],
        help=(
            "Node features to load. Subset of "
            f"{', '.join(NODE_FEATURES)}, or 'all'/'none'. "
            "Overrides the config's node category."
        ),
    )
    parser.add_argument(
        "--topology-features",
        nargs="+",
        default=None,
        metavar="FEATURE",
        choices=[*TOPOLOGY_FEATURES, "none", "all"],
        help=(
            "Topology features to load. Subset of "
            f"{', '.join(TOPOLOGY_FEATURES)}, or 'all'/'none'. "
            "Overrides the config's topology category."
        ),
    )
    parser.add_argument(
        "--positional-encoding",
        nargs="+",
        default=None,
        metavar="ENCODING",
        choices=[*POSITIONAL_ENCODING_CHOICES, "none"],
        help=(
            "Positional encodings to use when the positional group is enabled: "
            "lpe (Laplacian), rwpe (random walk), or none."
        ),
    )
    parser.add_argument(
        "--edge-features",
        nargs="+",
        default=None,
        metavar="FEATURE",
        choices=[*EDGE_FEATURES, "none", "all"],
        help=(
            "Edge features to load. 'none' disables edge features; any subset "
            "currently enables all edge columns (per-edge slicing is deferred)."
        ),
    )
    parser.add_argument(
        "--active-features",
        type=lambda value: parse_csv_list(value) or [],
        default=None,
        help=(
            "Explicit comma-separated node feature names. "
            "Overrides --feature-groups and --positional-encoding."
        ),
    )
