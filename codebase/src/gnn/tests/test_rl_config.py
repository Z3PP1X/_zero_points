from pathlib import Path

from gnn.reinforcement_learning.rl_config import load_yaml_config, read_rl_settings
from gnn.shared.utils.graph_utils import ANCHOR_GROUP_FEATURES


def test_read_rl_settings_includes_edge_direction():
    config_path = Path(__file__).resolve().parents[1] / "reinforcement_learning" / "config_rl.yaml"
    settings = read_rl_settings(load_yaml_config(config_path))
    assert settings["edge_direction"] == "top_down"
    assert settings["mode"] == "graph"
    assert settings["experiment"] == "nur_f"
    assert settings["active_features"] is None
    assert settings["feature_selection"].positional_encodings == ANCHOR_GROUP_FEATURES
