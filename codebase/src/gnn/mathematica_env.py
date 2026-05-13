import math
import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from feature_layout import (
    NATIVE_GLOBAL_FEATURE_COUNT,
    NATIVE_NODE_FEATURE_COUNT,
    PADDED_GLOBAL_FEATURE_COUNT,
    PADDED_NODE_FEATURE_COUNT,
)
from mathematica_state_ingress import MathematicaStateIngress
from replay_buffer import EpisodeReplayBuffer

logger = logging.getLogger(__name__)


def decode_action_to_solver_tol(action, state_dict: dict) -> tuple:
    chosen_solver = 1 if action[0] > 0 else 0
    base_tol = state_dict.get("tolerance", 1e-15)
    log_tol_base = math.log10(base_tol)
    log_tol_min = log_tol_base - 4.0
    log_tol_max = log_tol_base + 4.0
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
        step_timeout_s: float = 30.0,
        timeout_penalty: float = -10.0,
    ):
        super().__init__()
        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.step_timeout_s = step_timeout_s
        self.timeout_penalty = timeout_penalty
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
        self.state_ingress = MathematicaStateIngress(gateway)
        self.current_state_dict = None
        self.current_obs = None
        self.current_uuid = None

    def _pad_graph(self, pyg_data):
        x = pyg_data.x.numpy()
        edge_index = pyg_data.edge_index.numpy()
        global_features = pyg_data.global_features.numpy().flatten()

        num_nodes = min(x.shape[0], self.max_nodes)
        num_edges = min(edge_index.shape[1], self.max_edges)

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
        next_state_dict = self._wait_for_next_state(timeout_s=self.step_timeout_s)
        if next_state_dict is None:
            logger.error(
                "Step Timeout nach %.1fs (UUID: %s). Episode wird abgebrochen.",
                self.step_timeout_s,
                self.current_uuid,
            )
            if self.replay_buffer.has_episode(self.current_uuid):
                self.replay_buffer.clear_episode(self.current_uuid)
            return self.current_obs, float(self.timeout_penalty), True, False, {
                "episode_steps": 0,
                "total_reward": float(self.timeout_penalty),
                "timeout": True,
            }

        self.replay_buffer.set_next_state(self.current_uuid, next_state_dict)
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
            total_reward = sum(t.get("reward", 0.0) for t in transitions)
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
