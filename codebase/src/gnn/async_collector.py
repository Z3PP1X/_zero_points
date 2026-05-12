"""
Asynchronous Gymnasium environment for the SAC-based Mathematica RL pipeline.

Unlike ``MathematicaVecEnv`` (which blocks until **all** N slots respond),
this single-env wrapper processes one UUID at a time while buffering
incoming states from Mathematica for future episodes.  Because SAC is
off-policy, it only needs ``(s, a, r, s', done)`` tuples in the replay
buffer — no lock-step batch is required.

Flow per ``step()`` call:
    1. Decode action → send decision to Mathematica for the current UUID.
    2. Block until the response for **this** UUID arrives.
       While waiting, any events for *other* UUIDs are cached in a fresh-
       state pool so they are not lost.
    3. If terminal → compute retroactive rewards, return ``done=True``.
    4. If intermediate → return the new observation with ``reward=0``.

Args:
    gateway: ``NetworkGateway`` instance (must already be ``init()``'d).
    preprocessor: ``Preprocessor`` for graph loading and feature injection.
    reward_calculator: ``RewardCalculator`` for retroactive episode rewards.
    max_nodes: Padding size for the node-feature tensor.
    max_edges: Padding size for the edge-index tensor.
"""
from __future__ import annotations

import logging
import math
import queue
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network_gateway import CHANNEL_STATE, CHANNEL_TERMINAL
from replay_buffer import EpisodeReplayBuffer

logger = logging.getLogger(__name__)


class AsyncMathematicaEnv(gym.Env):
    """
    Single-env Gymnasium wrapper with an internal fresh-state pool.

    Mathematica can push new problems at any rate; they are buffered in
    ``_fresh_states`` and consumed one-by-one as episodes start/reset.

    Args:
        gateway: NetworkGateway instance for ZeroMQ communication.
        preprocessor: Preprocessor instance for graph loading and
            feature injection.
        reward_calculator: RewardCalculator instance for episode reward
            computation.
        max_nodes: Maximum number of nodes for padded observation space.
        max_edges: Maximum number of edges for padded observation space.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gateway,
        preprocessor,
        reward_calculator,
        max_nodes: int = 200,
        max_edges: int = 1000,
    ):
        super().__init__()
        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.max_nodes, 5), dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0, high=self.max_nodes - 1,
                    shape=(2, self.max_edges), dtype=np.int64,
                ),
                "global_features": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(9,), dtype=np.float32,
                ),
                "num_nodes": spaces.Box(
                    low=0, high=self.max_nodes, shape=(1,), dtype=np.int64,
                ),
                "num_edges": spaces.Box(
                    low=0, high=self.max_edges, shape=(1,), dtype=np.int64,
                ),
            }
        )

        # Episode tracking (retroactive reward computation).
        self.episode_buffer = EpisodeReplayBuffer(keep_completed=False)
        self.current_uuid: Optional[str] = None
        self.current_state_dict: Optional[Dict[str, Any]] = None
        self.current_obs: Optional[Dict[str, np.ndarray]] = None

        # Pool of initial states that arrived while we were busy with
        # another episode.  Consumed FIFO during ``reset()``.
        self._fresh_states: deque = deque()
        self._fresh_uuids: set = set()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _pad_graph(self, pyg_data) -> Dict[str, np.ndarray]:
        """
        Pad a PyG Data object to fixed dimensions for SB3 compatibility.

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
            "num_edges": np.array([num_edges], dtype=np.int64),
        }

    @staticmethod
    def _payload_uuid(payload: Dict[str, Any]) -> str:
        """Extract the UUID from a Mathematica payload dict."""
        uuid = payload.get("uuid")
        if uuid is None:
            uuid = str(payload.get("id", "unknown"))
        return uuid

    def _next_event(self) -> Tuple[str, Dict[str, Any]]:
        """
        Block until the next ``(channel, payload)`` arrives from the gateway.

        Legacy terminal states on the state port are silently skipped.

        Raises:
            InterruptedError: If the gateway stops running.
        """
        while True:
            try:
                channel, payload = self.gateway.event_queue.get(timeout=0.1)
                if (
                    channel == CHANNEL_STATE
                    and payload.get("status") in ("reward_calc", "finished")
                ):
                    continue
                return channel, payload
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def _wait_for_uuid(
        self, target_uuid: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Block until an event for ``target_uuid`` arrives.

        Any events for **other** UUIDs that arrive in the meantime are
        routed into the fresh-state pool so they are not lost.

        Args:
            target_uuid: The UUID to wait for.

        Returns:
            ``(channel, payload)`` for the target UUID.
        """
        while True:
            channel, payload = self._next_event()
            uuid = self._payload_uuid(payload)

            if uuid == target_uuid:
                return channel, payload

            # Not our UUID → buffer as fresh state (if it is a new
            # initial observation) or drop terminals for unknown UUIDs.
            if channel == CHANNEL_TERMINAL:
                logger.debug(
                    "Dropping terminal for non-active UUID %s while "
                    "waiting for %s.",
                    uuid, target_uuid,
                )
                continue

            if uuid not in self._fresh_uuids:
                self._fresh_states.append(payload)
                self._fresh_uuids.add(uuid)

    def _get_fresh_state(self) -> Dict[str, Any]:
        """
        Return the next buffered initial state, blocking if the pool
        is empty until Mathematica sends one.

        Returns:
            A state dict suitable for starting a new episode.
        """
        while not self._fresh_states:
            channel, payload = self._next_event()
            uuid = self._payload_uuid(payload)

            if channel == CHANNEL_TERMINAL:
                logger.debug(
                    "Dropping terminal for UUID %s while waiting for "
                    "a fresh initial state.",
                    uuid,
                )
                continue

            if uuid not in self._fresh_uuids:
                self._fresh_states.append(payload)
                self._fresh_uuids.add(uuid)

        state_dict = self._fresh_states.popleft()
        self._fresh_uuids.discard(self._payload_uuid(state_dict))
        return state_dict

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        """
        Start a new episode by consuming the next available initial state.

        Cleans up the previous episode's replay-buffer entry if present.

        Args:
            seed: Optional random seed (forwarded to ``gym.Env``).
            options: Optional reset options.

        Returns:
            Tuple of ``(observation, info_dict)``.
        """
        super().reset(seed=seed)

        # Clean up leftover episode data.
        if (
            self.current_uuid
            and self.episode_buffer.has_episode(self.current_uuid)
        ):
            self.episode_buffer.clear_episode(self.current_uuid)

        state_dict = self._get_fresh_state()
        self.current_uuid = self._payload_uuid(state_dict)
        self.episode_buffer.start_episode(self.current_uuid)

        self.current_state_dict = state_dict
        pyg_data, _ = self.preprocessor.process(state_dict, dataloader=None)
        self.current_obs = self._pad_graph(pyg_data)

        return self.current_obs, {}

    def step(self, action):
        """
        Execute one environment step.

        1. Decode action → solver choice + tolerance.
        2. Record transition in the episode replay buffer.
        3. Send decision to Mathematica.
        4. Wait for the response for **this** UUID (caching others).
        5. Terminal → retroactive reward computation.
        6. Intermediate → return ``reward=0.0``.

        Args:
            action: numpy array of shape ``(2,)`` from the SAC policy.

        Returns:
            Tuple ``(observation, reward, terminated, truncated, info)``.
        """
        # 1. Decode action
        chosen_solver = 1 if action[0] > 0 else 0

        base_tol = self.current_state_dict.get("tolerance", 1e-15)
        log_tol_base = math.log10(base_tol)
        log_tol_min = log_tol_base - 4.0
        log_tol_max = log_tol_base + 4.0
        scale_factor = (float(action[1]) + 1.0) / 2.0
        log10_tol = log_tol_min + scale_factor * (log_tol_max - log_tol_min)
        chosen_tol = 10.0 ** log10_tol

        action_dict = {
            "solver": chosen_solver,
            "localMaxTolerance": chosen_tol,
        }

        # 2. Record transition (next_state filled later).
        self.episode_buffer.add_transition(
            uuid=self.current_uuid,
            current_state=self.current_state_dict,
            action=action_dict,
        )

        # 3. Send decision.
        self.gateway.send_decision(
            self.current_state_dict, chosen_solver, chosen_tol,
        )

        # 4. Wait for this UUID's response (buffer others).
        next_channel, next_payload = self._wait_for_uuid(self.current_uuid)
        is_terminal = next_channel == CHANNEL_TERMINAL

        # 5. Fill next_state on the last transition.
        self.episode_buffer.set_next_state(self.current_uuid, next_payload)

        if is_terminal:
            transitions = self.episode_buffer.get_transitions(
                self.current_uuid,
            )
            self.reward_calculator.calculate_episode_rewards(
                transitions, next_payload,
            )
            total_reward = sum(
                t.get("reward", 0.0) for t in transitions
            )
            n_steps = len(transitions)

            logger.info(
                "Episode done (UUID: %s) | Steps: %d | Total Reward: %.4f",
                self.current_uuid, n_steps, total_reward,
            )
            self.episode_buffer.clear_episode(self.current_uuid)

            return self.current_obs, total_reward, True, False, {
                "episode_steps": n_steps,
                "total_reward": total_reward,
            }

        # 6. Intermediate step.
        self.current_state_dict = next_payload
        pyg_data, _ = self.preprocessor.process(
            next_payload, dataloader=None,
        )
        self.current_obs = self._pad_graph(pyg_data)

        return self.current_obs, 0.0, False, False, {}
