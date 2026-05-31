import math
import logging
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gnn.reinforcement_learning.feature_layout import (
    NATIVE_GLOBAL_FEATURE_COUNT,
    NATIVE_NODE_FEATURE_COUNT,
    PADDED_GLOBAL_FEATURE_COUNT,
    PADDED_NODE_FEATURE_COUNT,
)
from gnn.reinforcement_learning.gateway.mathematica_state_ingress import MathematicaStateIngress
from gnn.reinforcement_learning.observation_sanitize import sanitize_numpy_features
from gnn.reinforcement_learning.replay_buffer import EpisodeReplayBuffer
from gnn.reinforcement_learning.gateway.state_wait_timeout import StateRoundtripTimeout

logger = logging.getLogger(__name__)


def decode_action_to_solver_tol(action, state_dict: dict) -> tuple:
    chosen_solver = 1 if action[0] > 0 else 0
    base_tol = state_dict.get("tolerance", 1e-15)
    log_tol_min = math.log10(base_tol)
    log_tol_max = log_tol_min + 14.0
    scale_factor = (action[1] + 1.0) / 2.0
    log10_tol = log_tol_min + scale_factor * (log_tol_max - log_tol_min)
    chosen_tol = 10.0 ** log10_tol
    action_dict = {"solver": chosen_solver, "localMaxTolerance": chosen_tol}
    return chosen_solver, chosen_tol, action_dict


class MathematicaGraphEnv(gym.Env):
    def __init__(
        self,
        gateway,
        preprocessor,
        reward_calculator,
        max_nodes=50,
        max_edges=100,
        timeout_fallback_s: float = 5.0,
        timeout_cushion_s: float = 2.0,
        timeout_window_size: int = 100,
        timeout_penalty: float = -10.0,
        state_ingress=None,
    ):
        super().__init__()
        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.timeout_penalty = timeout_penalty
        traffic_monitor = getattr(gateway, "traffic_monitor", None)
        if traffic_monitor is not None and hasattr(traffic_monitor, "roundtrip_timeout"):
            self._roundtrip_timeout = traffic_monitor.roundtrip_timeout
        else:
            self._roundtrip_timeout = StateRoundtripTimeout(
                window_size=timeout_window_size,
                cushion_s=timeout_cushion_s,
                fallback_s=timeout_fallback_s,
            )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_nodes, PADDED_NODE_FEATURE_COUNT),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=self.max_nodes - 1,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(PADDED_GLOBAL_FEATURE_COUNT,),
                    dtype=np.float32,
                ),
                "num_nodes": spaces.Box(
                    low=0, high=self.max_nodes, shape=(1,), dtype=np.int64
                ),
                "num_edges": spaces.Box(
                    low=0, high=self.max_edges, shape=(1,), dtype=np.int64
                ),
            }
        )
        self.replay_buffer = EpisodeReplayBuffer(keep_completed=False)
        self.state_ingress = (
            state_ingress if state_ingress is not None else MathematicaStateIngress(gateway)
        )
        self.current_state_dict = None
        self.current_obs = None
        self.current_uuid = None
        self.total_timeouts = 0

    def _pad_graph(self, pyg_data):
        x = sanitize_numpy_features(pyg_data.x.numpy())
        edge_index = pyg_data.edge_index.numpy()
        global_features = sanitize_numpy_features(
            pyg_data.global_features.numpy().flatten()
        )

        num_nodes = min(x.shape[0], self.max_nodes)
        num_edges = min(edge_index.shape[1], self.max_edges)
        if num_nodes == 0:
            logger.warning(
                "Graph observation has 0 nodes; using one zero node to keep GNN pooling finite."
            )
            num_nodes = 1

        padded_x = np.zeros((self.max_nodes, PADDED_NODE_FEATURE_COUNT), dtype=np.float32)
        node_width = min(x.shape[1], NATIVE_NODE_FEATURE_COUNT, PADDED_NODE_FEATURE_COUNT)
        padded_x[:num_nodes, :node_width] = x[:num_nodes, :node_width]

        padded_edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        padded_edge_index[:, :num_edges] = edge_index[:, :num_edges]

        padded_global = np.zeros((PADDED_GLOBAL_FEATURE_COUNT,), dtype=np.float32)
        global_width = min(
            global_features.shape[0],
            NATIVE_GLOBAL_FEATURE_COUNT,
            PADDED_GLOBAL_FEATURE_COUNT,
        )
        padded_global[:global_width] = global_features[:global_width]

        return {
            "x": padded_x,
            "edge_index": padded_edge_index,
            "global_features": padded_global,
            "num_nodes": np.array([num_nodes], dtype=np.int64),
            "num_edges": np.array([num_edges], dtype=np.int64),
        }

    def _wait_for_next_state(self, timeout_s: float = None):
        if self.current_uuid is None:
            return self.state_ingress.take_next_training_start()
        return self.state_ingress.take_next_for_episode(self.current_uuid, timeout_s)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.current_uuid and self.replay_buffer.has_episode(self.current_uuid):
            self.replay_buffer.clear_episode(self.current_uuid)

        state_dict = self.state_ingress.take_next_training_start()
        self.current_uuid = state_dict.get("uuid")
        self.replay_buffer.start_episode(self.current_uuid)
        self.current_state_dict = state_dict
        pyg_data, _ = self.preprocessor.process(state_dict, dataloader=None)
        self.current_obs = self._pad_graph(pyg_data)
        return self.current_obs, {}

    def _attach_observed_state(self, next_state_dict: dict) -> None:
        if (
            self.current_uuid
            and self.replay_buffer.has_episode(self.current_uuid)
            and self.replay_buffer.get_episode_length(self.current_uuid) > 0
        ):
            transitions = self.replay_buffer.get_transitions(self.current_uuid)
            if transitions[-1].get("next_state") is None:
                self.replay_buffer.set_next_state(self.current_uuid, next_state_dict)

    def _ingest_observed_state(self, next_state_dict: dict):
        self._attach_observed_state(next_state_dict)
        status = next_state_dict.get("status")
        if status in ("error", "non_converged"):
            logger.warning(
                "Solver-Error (UUID: %s, status=%s): %s",
                self.current_uuid,
                status,
                next_state_dict.get("errorMessage"),
            )
            if self.replay_buffer.has_episode(self.current_uuid):
                self.replay_buffer.clear_episode(self.current_uuid)
            return self.current_obs, float(self.timeout_penalty), True, False, {
                "episode_steps": 0,
                "total_reward": float(self.timeout_penalty),
                "error": status,
            }

        if status in ("reward_calc", "finished"):
            transitions = self.replay_buffer.get_transitions(self.current_uuid)
            self.reward_calculator.calculate_episode_rewards(transitions, next_state_dict)
            total_reward = sum((t.get("reward") if t.get("reward") is not None else 0.0) for t in transitions)
            n_steps = len(transitions)
            logger.info(
                "Episode done (UUID: %s) | Steps: %d | Total Reward: %.4f",
                self.current_uuid,
                n_steps,
                total_reward,
            )
            self.replay_buffer.clear_episode(self.current_uuid)
            return self.current_obs, total_reward, True, False, {
                "episode_steps": n_steps,
                "total_reward": total_reward,
            }

        self.current_state_dict = next_state_dict
        pyg_data, _ = self.preprocessor.process(next_state_dict, dataloader=None)
        self.current_obs = self._pad_graph(pyg_data)
        return self.current_obs, 0.0, False, False, {}

    def drain_buffered_states(self) -> bool:
        episode_completed = False
        while True:
            next_state_dict = self.state_ingress.poll_next_for_episode(self.current_uuid)
            if next_state_dict is None:
                return episode_completed
            _, _, terminated, _, _ = self._ingest_observed_state(next_state_dict)
            if terminated:
                episode_completed = True
                return episode_completed

    def step(self, action):
        chosen_solver, chosen_tol, action_dict = decode_action_to_solver_tol(
            action, self.current_state_dict
        )
        self.replay_buffer.add_transition(
            uuid=self.current_uuid,
            current_state=self.current_state_dict,
            action=action_dict,
        )
        self.gateway.send_decision(self.current_state_dict, chosen_solver, chosen_tol)
        wait_started = time.monotonic()
        next_state_dict = self._wait_for_next_state(
            timeout_s=self._roundtrip_timeout.timeout_s()
        )
        if next_state_dict is not None:
            roundtrip_s = time.monotonic() - wait_started
            traffic_monitor = getattr(self.gateway, "traffic_monitor", None)
            if traffic_monitor is not None:
                traffic_monitor.record_roundtrip(roundtrip_s)
            else:
                self._roundtrip_timeout.record(roundtrip_s)
        if next_state_dict is None:
            self.total_timeouts += 1
            traffic_monitor = getattr(self.gateway, "traffic_monitor", None)
            if traffic_monitor is not None:
                traffic_monitor.record_timeout()
            if self.replay_buffer.has_episode(self.current_uuid):
                self.replay_buffer.clear_episode(self.current_uuid)
            return self.current_obs, 0.0, True, False, {
                "episode_steps": 0,
                "total_reward": 0.0,
                "timeout": True,
                "total_timeouts": self.total_timeouts,
            }

        return self._ingest_observed_state(next_state_dict)
