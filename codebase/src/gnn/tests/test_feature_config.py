import pytest

from gnn.shared.utils.feature_config import (
    FEATURE_CLASSES,
    PositionalSupernodeConflictError,
    merge_feature_selection,
    parse_feature_selection_from_mapping,
    resolve_active_node_features,
    validate_positional_supernode_compatibility,
)


def test_default_selection_uses_all_enriched_features():
    selection = parse_feature_selection_from_mapping({})
    active = resolve_active_node_features(selection)
    assert active is None


def test_positional_encoding_subset():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping(
            {
                "features": {
                    "node": True,
                    "topology": False,
                    "positional": ["anchor_periodic"],
                    "edge": True,
                }
            }
        ),
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert "anchor_periodic" in active
    assert "anchor_additive" not in active
    assert "depth" not in active


def test_node_category_subset_list():
    selection = parse_feature_selection_from_mapping(
        {
            "features": {
                "node": ["node_type", "value"],
                "topology": False,
                "positional": False,
                "edge": True,
            }
        }
    )
    active = resolve_active_node_features(selection)
    # Only the two listed node features survive; topology/positional dropped.
    assert active == ["node_type", "value"]


def test_topology_subset_and_positional_true():
    selection = parse_feature_selection_from_mapping(
        {
            "features": {
                "node": False,
                "topology": ["depth", "height"],
                "positional": True,
                "edge": True,
            }
        }
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert "depth" in active and "height" in active
    assert "subtree_size" not in active  # not selected
    assert "node_type" not in active  # node disabled
    # positional: true -> all anchor groups
    assert "anchor_additive" in active and "anchor_transcendental" in active


def test_positional_false_disables_pe():
    selection = parse_feature_selection_from_mapping(
        {"features": {"positional": False}}
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert not any(feature.startswith("anchor_") for feature in active)


def test_edge_list_coerces_to_enabled():
    # Edge slicing is deferred: a subset list still enables the edge category.
    selection = parse_feature_selection_from_mapping(
        {"features": {"edge": ["relation_type"]}}
    )
    assert selection.edge is True
    assert "edge" in selection.enabled_groups()


def test_nested_positional_form_rejected():
    with pytest.raises(ValueError, match="Nested feature config"):
        parse_feature_selection_from_mapping(
            {"features": {"positional": {"enabled": True, "encodings": ["anchor_periodic"]}}}
        )


def test_per_category_cli_overrides():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping({}),
        node_features=["node_type"],
        topology_features=["none"],
        edge_features=["none"],
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert active.count("node_type") == 1
    assert "node_type" in active
    assert not any(
        feature in active
        for feature in ("depth", "height", "subtree_size", "out_degree")
    )
    assert selection.edge is False


def test_unknown_node_feature_raises():
    with pytest.raises(ValueError, match="Unknown node feature"):
        merge_feature_selection(
            parse_feature_selection_from_mapping({}),
            node_features=["not_a_feature"],
        )


def test_positional_encoding_none_via_cli():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping({}),
        positional_encoding=["none"],
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert not any(feature.startswith("anchor_") for feature in active)


def test_feature_groups_limit_enabled_classes():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping({}),
        feature_groups=["node", "positional"],
        positional_encoding=["anchor_periodic"],
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert "node_type" in active
    assert "anchor_periodic" in active
    assert "anchor_additive" not in active
    assert "depth" not in active


def test_explicit_active_features_override_groups():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping({}),
        feature_groups=["node"],
        active_features=["node_type", "label_id"],
    )
    active = resolve_active_node_features(selection)
    assert active == ["node_type", "label_id"]


def test_unknown_feature_group_raises():
    with pytest.raises(ValueError, match="Unknown feature group"):
        merge_feature_selection(
            parse_feature_selection_from_mapping({}),
            feature_groups=["invalid"],
        )


def test_feature_classes_cover_catalog():
    assert "node" in FEATURE_CLASSES
    assert "topology" in FEATURE_CLASSES
    assert "positional" in FEATURE_CLASSES
    assert "edge" in FEATURE_CLASSES


def test_positional_with_supernode_raises():
    selection = parse_feature_selection_from_mapping({"features": {"positional": True}})
    with pytest.raises(PositionalSupernodeConflictError):
        validate_positional_supernode_compatibility(selection, add_virtual_supernode=True)


def test_positional_disabled_allows_supernode():
    selection = parse_feature_selection_from_mapping({"features": {"positional": False}})
    # Must not raise: no positional encoding means the supernode is fine.
    validate_positional_supernode_compatibility(selection, add_virtual_supernode=True)


def test_positional_without_supernode_ok():
    selection = parse_feature_selection_from_mapping({"features": {"positional": True}})
    validate_positional_supernode_compatibility(selection, add_virtual_supernode=False)
