import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import queue
import time
import logging

from replay_buffer import EpisodeReplayBuffer

logger = logging.getLogger(__name__)


def decode_action_to_solver_tol(action, state_dict: dict) -> tuple:
    """
    Mappt die kontinuierliche Policy-Aktion auf Solver und Toleranz.

    Returns:
        (chosen_solver: int, chosen_tol: float, action_dict: dict)
    """
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
    """
    Gymnasium environment for interacting with Mathematica via ZeroMQ.

    Manages the communication cycle:
        1. Receive state from Mathematica
        2. Process graph via Preprocessor
        3. Record transition in EpisodeReplayBuffer
        4. Send action decision back to Mathematica
        5. At episode end: compute rewards retroactively via RewardCalculator

    Args:
        gateway: NetworkGateway instance for ZeroMQ communication.
        preprocessor: Preprocessor instance for graph loading and feature injection.
        reward_calculator: RewardCalculator instance for episode reward computation.
        max_nodes: Maximum number of nodes for padded observation space.
        max_edges: Maximum number of edges for padded observation space.
    """

    def __init__(
        self,
        gateway,
        preprocessor,
        reward_calculator,
        max_nodes=50,
        max_edges=100,
        step_timeout_s: float = 2.0,
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

        # Action Space: [Solver (Continuous >0.0 is 1), Tolerance (Continuous -1 to 1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space: Padded Graph Dictionary
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, 5), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.max_nodes-1, shape=(2, self.max_edges), dtype=np.int64),
            "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            # Also keep track of valid nodes/edges to unpad in the extractor
            "num_nodes": spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=np.int64),
            "num_edges": spaces.Box(low=0, high=self.max_edges, shape=(1,), dtype=np.int64)
        })

        self.replay_buffer = EpisodeReplayBuffer(keep_completed=False)
        self.current_state_dict = None
        self.current_obs = None
        self.current_uuid = None

    def _pad_graph(self, pyg_data):
        """
        Pads a PyG Data object to fixed dimensions for SB3 compatibility.

        Args:
            pyg_data: PyTorch Geometric Data object from the Preprocessor.

        Returns:
            Dict with padded numpy arrays matching the observation_space.
        """
        x = pyg_data.x.numpy()
        edge_index = pyg_data.edge_index.numpy()
        global_features = pyg_data.global_features.numpy().flatten()

        num_nodes = min(x.shape[0], self.max_nodes)
        num_edges = min(edge_index.shape[1], self.max_edges)

        padded_x = np.zeros((self.max_nodes, 5), dtype=np.float32)
        padded_x[:num_nodes, :] = x[:num_nodes, :]

        padded_edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        padded_edge_index[:, :num_edges] = edge_index[:, :num_edges]

        return {
            "x": padded_x,
            "edge_index": padded_edge_index,
            "global_features": global_features.astype(np.float32),
            "num_nodes": np.array([num_nodes], dtype=np.int64),
            "num_edges": np.array([num_edges], dtype=np.int64)
        }

    def _wait_for_next_state(self, timeout_s: float = None):
        """
        Blocks until the next state arrives from the NetworkGateway queue.

        Args:
            timeout_s: Optional inactivity timeout in seconds. ``None`` blocks
                forever. With a finite value, returns ``None`` if no message
                arrives within that window.

        Returns:
            State dict from Mathematica, or ``None`` if the idle timeout elapses.

        Raises:
            InterruptedError: If the gateway stops running.
        """
        idle_deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            remaining = None if idle_deadline is None else idle_deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            try:
                wait = 0.1 if remaining is None else min(0.1, remaining)
                return self.gateway.network_queue.get(timeout=wait)
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.

        Waits for an initial (non-terminal) state from Mathematica,
        extracts the UUID, and starts a new episode in the replay buffer.

        Args:
            seed: Optional random seed.
            options: Optional reset options.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Clean up any leftover episode from a previous interrupted run
        if self.current_uuid and self.replay_buffer.has_episode(self.current_uuid):
            self.replay_buffer.clear_episode(self.current_uuid)

        # Block until we receive a new non-terminal, uuid-bearing state from Mathematica.
        while True:
            state_dict = self._wait_for_next_state()
            status = state_dict.get("status")
            if status in ("reward_calc", "finished", "error", "non_converged"):
                logger.info(
                    "Skipping legacy/terminal state (ID: %s, status=%s) during reset.",
                    state_dict.get("id", "UNKNOWN"), status,
                )
                continue
            if state_dict.get("uuid") is None:
                logger.warning(
                    "Reset: State ohne UUID verworfen (id=%s, status=%s).",
                    state_dict.get("id"), status,
                )
                continue
            break

        self.current_uuid = state_dict.get("uuid")
        self.replay_buffer.start_episode(self.current_uuid)

        self.current_state_dict = state_dict
        pyg_data, _ = self.preprocessor.process(state_dict, dataloader=None)
        self.current_obs = self._pad_graph(pyg_data)

        return self.current_obs, {}

    def step(self, action):
        """
        Executes one step in the environment.

        1. Decodes the continuous action into solver choice + tolerance.
        2. Records the transition in the replay buffer.
        3. Sends the decision to Mathematica.
        4. Waits for the next state.
        5. If terminal: computes rewards retroactively via RewardCalculator
           and returns the total episode reward.
        6. If intermediate: returns reward=0.0 (sparse reward strategy).

        Args:
            action: numpy array of shape (2,) from SB3's policy.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        chosen_solver, chosen_tol, action_dict = decode_action_to_solver_tol(
            action, self.current_state_dict
        )

        # 2. Record transition in replay buffer (next_state still unknown)
        self.replay_buffer.add_transition(
            uuid=self.current_uuid,
            current_state=self.current_state_dict,
            action=action_dict
        )

        # 3. Send decision to Mathematica
        self.gateway.send_decision(self.current_state_dict, chosen_solver, chosen_tol)

        # 4. Wait for next state (idle timeout avoids deadlock if Mathematica drops a response).
        next_state_dict = self._wait_for_next_state(timeout_s=self.step_timeout_s)
        if next_state_dict is None:
            logger.error(
                "Step Idle-Timeout nach %.1fs ohne Antwort (UUID: %s). Episode wird abgebrochen.",
                self.step_timeout_s, self.current_uuid,
            )
            if self.replay_buffer.has_episode(self.current_uuid):
                self.replay_buffer.clear_episode(self.current_uuid)
            return self.current_obs, float(self.timeout_penalty), True, False, {
                "episode_steps": 0,
                "total_reward": float(self.timeout_penalty),
                "timeout": True,
            }

        # 5. Set next_state on the last transition
        self.replay_buffer.set_next_state(self.current_uuid, next_state_dict)

        status = next_state_dict.get("status")
        if status in ("error", "non_converged"):
            logger.warning(
                "Solver-Error (UUID: %s, status=%s): %s",
                self.current_uuid, status, next_state_dict.get("errorMessage"),
            )
            if self.replay_buffer.has_episode(self.current_uuid):
                self.replay_buffer.clear_episode(self.current_uuid)
            return self.current_obs, float(self.timeout_penalty), True, False, {
                "episode_steps": 0,
                "total_reward": float(self.timeout_penalty),
                "error": status,
            }

        if status in ("reward_calc", "finished"):
            # ── TERMINAL ──
            # Compute ALL per-step rewards retroactively
            transitions = self.replay_buffer.get_transitions(self.current_uuid)
            self.reward_calculator.calculate_episode_rewards(transitions, next_state_dict)

            # Total reward = sum of all per-step rewards
            total_reward = sum(t.get("reward", 0.0) for t in transitions)
            n_steps = len(transitions)

            logger.info(
                "Episode done (UUID: %s) | Steps: %d | Total Reward: %.4f",
                self.current_uuid, n_steps, total_reward
            )

            # Clean up episode from buffer
            self.replay_buffer.clear_episode(self.current_uuid)

            return self.current_obs, total_reward, True, False, {
                "episode_steps": n_steps,
                "total_reward": total_reward
            }

        else:
            # ── INTERMEDIATE ──
            # Sparse reward: 0.0 until episode end
            self.current_state_dict = next_state_dict
            pyg_data, _ = self.preprocessor.process(next_state_dict, dataloader=None)
            self.current_obs = self._pad_graph(pyg_data)

            return self.current_obs, 0.0, False, False, {}
