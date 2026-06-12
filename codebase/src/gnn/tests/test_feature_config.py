import pytest

from gnn.shared.utils.feature_config import (
    FEATURE_CLASSES,
    merge_feature_selection,
    parse_feature_selection_from_mapping,
    resolve_active_node_features,
)


def test_default_selection_uses_all_enriched_features():
    selection = parse_feature_selection_from_mapping({})
    active = resolve_active_node_features(selection)
    assert active is None


def test_positional_encoding_lpe_only():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping(
            {
                "features": {
                    "node": True,
                    "topology": False,
                    "positional": {"enabled": True, "encodings": ["lpe"]},
                    "edge": True,
                }
            }
        ),
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert "lpe_1" in active
    assert "rwpe_1" not in active
    assert "depth" not in active


def test_positional_encoding_none_via_cli():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping({}),
        positional_encoding=["none"],
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert not any(feature.startswith(("lpe_", "rwpe_")) for feature in active)


def test_feature_groups_limit_enabled_classes():
    selection = merge_feature_selection(
        parse_feature_selection_from_mapping({}),
        feature_groups=["node", "positional"],
        positional_encoding=["rwpe"],
    )
    active = resolve_active_node_features(selection)
    assert active is not None
    assert "node_type" in active
    assert "rwpe_1" in active
    assert "lpe_1" not in active
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
