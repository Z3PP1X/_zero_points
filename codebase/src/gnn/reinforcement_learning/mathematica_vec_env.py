from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from gnn.reinforcement_learning.feature_layout import (
    PADDED_NODE_FEATURE_COUNT,
)
from gnn.reinforcement_learning.mathematica_env import decode_action_to_solver_tol
from gnn.reinforcement_learning.gateway.mathematica_state_ingress import MathematicaStateIngress
from gnn.reinforcement_learning.observation_sanitize import sanitize_numpy_features
from gnn.reinforcement_learning.replay_buffer import EpisodeReplayBuffer
from gnn.reinforcement_learning.reward import RewardCalculator
from gnn.reinforcement_learning.gateway.state_wait_timeout import StateRoundtripTimeout

logger = logging.getLogger(__name__)


def _gateway_channel(payload: Dict[str, Any]) -> str:
    return payload.get("_gateway_channel", "training")


def _is_terminal_response(payload: Dict[str, Any]) -> bool:
    if _gateway_channel(payload) == "reward":
        return True
    return payload.get("status") in ("reward_calc", "finished")


class MathematicaVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        gateway,
        preprocessor,
        reward_calculator: RewardCalculator,
        max_nodes: int = 200,
        max_edges: int = 1000,
        timeout_fallback_s: float = 5.0,
        timeout_cushion_s: float = 2.0,
        timeout_window_size: int = 100,
        timeout_penalty: float = -10.0,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.timeout_penalty = timeout_penalty
        self.state_ingress = MathematicaStateIngress(gateway)
        traffic_monitor = getattr(gateway, "traffic_monitor", None)
        if traffic_monitor is not None and hasattr(traffic_monitor, "roundtrip_timeout"):
            self._roundtrip_timeout = traffic_monitor.roundtrip_timeout
        else:
            self._roundtrip_timeout = StateRoundtripTimeout(
                window_size=timeout_window_size,
                cushion_s=timeout_cushion_s,
                fallback_s=timeout_fallback_s,
            )

        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_space = spaces.Dict(
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
                "num_nodes": spaces.Box(
                    low=0, high=self.max_nodes, shape=(1,), dtype=np.int64
                ),
                "num_edges": spaces.Box(
                    low=0, high=self.max_edges, shape=(1,), dtype=np.int64
                ),
            }
        )

        super().__init__(num_envs, observation_space, action_space)

        self._slot_uuid: List[Optional[str]] = [None] * num_envs
        self._slot_state: List[Optional[Dict[str, Any]]] = [None] * num_envs
        self._slot_obs: List[Optional[Dict[str, np.ndarray]]] = [None] * num_envs
        self._slot_episode_steps: List[int] = [0] * num_envs
        self._uuid_to_slot: Dict[str, int] = {}
        self._slots_pending_refill: set[int] = set()
        self.replay_buffer = EpisodeReplayBuffer(keep_completed=False)
        self._actions: Optional[np.ndarray] = None
        self.total_timeouts = 0
        self._is_finalizing = False

    def _pad_graph(self, pyg_data) -> Dict[str, np.ndarray]:
        x = sanitize_numpy_features(pyg_data.x.numpy())
        edge_index = pyg_data.edge_index.numpy()

        num_nodes = min(x.shape[0], self.max_nodes)
        num_edges = min(edge_index.shape[1], self.max_edges)
        if num_nodes == 0:
            logger.warning(
                "Graph observation has 0 nodes; using one zero node to keep GNN pooling finite."
            )
            num_nodes = 1

        padded_x = np.zeros((self.max_nodes, PADDED_NODE_FEATURE_COUNT), dtype=np.float32)
        node_width = min(x.shape[1], PADDED_NODE_FEATURE_COUNT)
        padded_x[:num_nodes, :node_width] = x[:num_nodes, :node_width]

        padded_edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        padded_edge_index[:, :num_edges] = edge_index[:, :num_edges]

        return {
            "x": padded_x,
            "edge_index": padded_edge_index,
            "num_nodes": np.array([num_nodes], dtype=np.int64),
            "num_edges": np.array([num_edges], dtype=np.int64),
        }

    def _record_roundtrip(self, wait_started: float) -> None:
        roundtrip_s = time.monotonic() - wait_started
        traffic_monitor = getattr(self.gateway, "traffic_monitor", None)
        if traffic_monitor is not None:
            traffic_monitor.record_roundtrip(roundtrip_s)
        else:
            self._roundtrip_timeout.record(roundtrip_s)

    def _record_timeout(self) -> None:
        self.total_timeouts += 1
        traffic_monitor = getattr(self.gateway, "traffic_monitor", None)
        if traffic_monitor is not None:
            traffic_monitor.record_timeout()

    def _fill_slot(self, slot: int) -> None:
        state_dict = self.state_ingress.take_next_training_start()
        uuid = state_dict.get("uuid")
        if uuid is None:
            raise ValueError("Fresh Mathematica state is missing a UUID.")
        uuid = str(uuid)

        self._slot_uuid[slot] = uuid
        self._slot_state[slot] = state_dict
        self._uuid_to_slot[uuid] = slot
        self._slot_episode_steps[slot] = 0

        self.replay_buffer.start_episode(uuid)
        pyg_data, _ = self.preprocessor.process(state_dict)
        self._slot_obs[slot] = self._pad_graph(pyg_data)

    def _stack_obs(self) -> Dict[str, np.ndarray]:
        obs_keys = ("x", "edge_index", "num_nodes", "num_edges")
        stacked: Dict[str, np.ndarray] = {}
        for key in obs_keys:
            arrays = []
            for slot in range(self.num_envs):
                slot_obs = self._slot_obs[slot]
                if slot_obs is not None:
                    arrays.append(slot_obs[key])
                else:
                    arrays.append(
                        np.zeros_like(
                            self.observation_space[key].sample()
                        )
                    )
            stacked[key] = np.stack(arrays, axis=0)
        return stacked

    def _clear_all_slots(self) -> None:
        for slot in range(self.num_envs):
            uuid = self._slot_uuid[slot]
            if uuid is not None and self.replay_buffer.has_episode(uuid):
                self.replay_buffer.clear_episode(uuid)
            self._slot_uuid[slot] = None
            self._slot_state[slot] = None
            self._slot_obs[slot] = None
            self._slot_episode_steps[slot] = 0
        self._uuid_to_slot.clear()
        self._slots_pending_refill.clear()

    def _schedule_slot_refill(self, slot: int) -> None:
        self._slots_pending_refill.add(slot)

    def _refill_pending_slots(self) -> None:
        if self._is_finalizing:
            self._slots_pending_refill.clear()
            return
        for slot in sorted(self._slots_pending_refill):
            self._fill_slot(slot)
        self._slots_pending_refill.clear()

    def _has_open_episodes(self) -> bool:
        return any(
            uuid is not None and self.replay_buffer.has_episode(uuid)
            for uuid in self._slot_uuid
        )

    def _attach_observed_state(self, slot: int, next_state_dict: dict) -> None:
        uuid = self._slot_uuid[slot]
        if (
            uuid
            and self.replay_buffer.has_episode(uuid)
            and self.replay_buffer.get_episode_length(uuid) > 0
        ):
            transitions = self.replay_buffer.get_transitions(uuid)
            if transitions[-1].get("next_state") is None:
                self.replay_buffer.set_next_state(uuid, next_state_dict)

    def _clear_slot(self, slot: int) -> None:
        uuid = self._slot_uuid[slot]
        if uuid is not None:
            self._uuid_to_slot.pop(uuid, None)
        self._slot_uuid[slot] = None
        self._slot_state[slot] = None
        self._slot_obs[slot] = None
        self._slot_episode_steps[slot] = 0

    def _timeout_slot(
        self,
        slot: int,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        uuid = self._slot_uuid[slot]
        self._record_timeout()
        if uuid is not None and self.replay_buffer.has_episode(uuid):
            self.replay_buffer.clear_episode(uuid)
        rewards[slot] = 0.0
        dones[slot] = True
        infos[slot] = {
            "episode_steps": 0,
            "total_reward": 0.0,
            "timeout": True,
            "total_timeouts": self.total_timeouts,
        }
        self._clear_slot(slot)
        self._schedule_slot_refill(slot)

    def drain_pending_responses(self) -> bool:
        progressed = False
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for slot in range(self.num_envs):
            uuid = self._slot_uuid[slot]
            if uuid is None:
                continue
            next_state_dict = self.state_ingress.poll_next_for_episode(uuid)
            if next_state_dict is None:
                continue
            self._handle_response(slot, next_state_dict, rewards, dones, infos)
            progressed = True
        self._refill_pending_slots()
        return progressed

    def finalize_open_episodes(self, max_flush_steps: int = 128) -> None:
        """Drain all in-flight episodes without consuming fresh states.

        Sets ``_is_finalizing`` so that ``_refill_pending_slots`` becomes a
        no-op.  This prevents the flush loop from pulling fresh training
        states out of the shared queue — states that the *next* Optuna trial
        needs. ``max_flush_steps`` caps the loop to prevent infinite flushes.
        """
        self._is_finalizing = True
        try:
            self.drain_pending_responses()
            if not self._has_open_episodes():
                return

            flush_steps = 0
            while self._has_open_episodes() and flush_steps < max_flush_steps:
                self.drain_pending_responses()
                if not self._has_open_episodes():
                    return

                actions = np.stack(
                    [self.action_space.sample() for _ in range(self.num_envs)],
                    axis=0,
                )
                self.step_async(actions)
                self.step_wait()
                flush_steps += 1

            for slot in range(self.num_envs):
                uuid = self._slot_uuid[slot]
                if uuid is not None and self.replay_buffer.has_episode(uuid):
                    self.replay_buffer.clear_episode(uuid)
        finally:
            self._is_finalizing = False

    def reset(self):
        self._clear_all_slots()
        for slot in range(self.num_envs):
            self._fill_slot(slot)
        return self._stack_obs()

    def step_async(self, actions: np.ndarray) -> None:
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected actions of shape ({self.num_envs}, ...), got {actions.shape}"
            )
        self._actions = actions

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async.")
        actions = self._actions
        self._actions = None

        for slot in range(self.num_envs):
            state_dict = self._slot_state[slot]
            uuid = self._slot_uuid[slot]
            if uuid is None or state_dict is None:
                continue
            chosen_solver, chosen_tol, action_dict = decode_action_to_solver_tol(
                actions[slot], state_dict
            )
            self.replay_buffer.add_transition(
                uuid=uuid,
                current_state=state_dict,
                action=action_dict,
            )
            self.gateway.send_decision(state_dict, chosen_solver, chosen_tol)
            self._slot_episode_steps[slot] += 1

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        slots_done = [False] * self.num_envs
        slot_wait_started = [time.monotonic() for _ in range(self.num_envs)]

        while not all(slots_done):
            for slot in range(self.num_envs):
                if slots_done[slot]:
                    continue
                uuid = self._slot_uuid[slot]
                if uuid is None:
                    slots_done[slot] = True
                    continue
                next_state_dict = self.state_ingress.poll_next_for_episode(uuid)
                if next_state_dict is None:
                    continue
                self._record_roundtrip(slot_wait_started[slot])
                self._handle_response(slot, next_state_dict, rewards, dones, infos)
                slots_done[slot] = True

            if all(slots_done):
                break

            pending_slots = [slot for slot in range(self.num_envs) if not slots_done[slot]]
            blocking_slot = min(pending_slots, key=lambda slot: slot_wait_started[slot])
            uuid = self._slot_uuid[blocking_slot]
            remaining = self._roundtrip_timeout.timeout_s() - (
                time.monotonic() - slot_wait_started[blocking_slot]
            )
            if remaining <= 0:
                self._timeout_slot(blocking_slot, rewards, dones, infos)
                slots_done[blocking_slot] = True
                continue

            next_state_dict = self.state_ingress.take_next_for_episode(
                uuid,
                timeout_s=remaining,
            )
            if next_state_dict is None:
                self._timeout_slot(blocking_slot, rewards, dones, infos)
                slots_done[blocking_slot] = True
                continue

            self._record_roundtrip(slot_wait_started[blocking_slot])
            self._handle_response(
                blocking_slot,
                next_state_dict,
                rewards,
                dones,
                infos,
            )
            slots_done[blocking_slot] = True

        self._refill_pending_slots()
        return self._stack_obs(), rewards, dones, infos

    def _handle_response(
        self,
        slot: int,
        payload: Dict[str, Any],
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        uuid = self._slot_uuid[slot]
        status = payload.get("status")

        if status in ("error", "non_converged"):
            logger.warning(
                "Solver-Error (UUID: %s, status=%s): %s",
                uuid,
                status,
                payload.get("errorMessage"),
            )
            if self.replay_buffer.has_episode(uuid):
                self.replay_buffer.clear_episode(uuid)
            rewards[slot] = float(self.timeout_penalty)
            dones[slot] = True
            infos[slot] = {
                "episode_steps": 0,
                "total_reward": float(self.timeout_penalty),
                "episode": {
                    "r": float(self.timeout_penalty),
                    "l": 0,
                    "t": 0.0,
                },
                "error": status,
            }
            self._clear_slot(slot)
            self._schedule_slot_refill(slot)
            return

        self._attach_observed_state(slot, payload)

        if _is_terminal_response(payload):
            transitions = self.replay_buffer.get_transitions(uuid)
            self.reward_calculator.calculate_episode_rewards(transitions, payload)
            total_reward = float(
                sum(float(t.get("reward") or 0.0) for t in transitions)
            )
            n_steps = len(transitions)
            terminal_obs = self._slot_obs[slot]
            infos[slot] = {
                "episode_steps": n_steps,
                "total_reward": total_reward,
                "episode": {"r": total_reward, "l": n_steps, "t": 0.0},
                "terminal_observation": terminal_obs,
            }
            rewards[slot] = total_reward
            dones[slot] = True

            self.replay_buffer.clear_episode(uuid)
            self._clear_slot(slot)
            self._schedule_slot_refill(slot)
            return

        self._slot_state[slot] = payload
        pyg_data, _ = self.preprocessor.process(payload)
        self._slot_obs[slot] = self._pad_graph(pyg_data)
        rewards[slot] = 0.0
        dones[slot] = False
        infos[slot] = {}

    def close(self) -> None:
        """Release all slots and return deferred messages to the shared queue.

        This **must** be called between Optuna trials so that training states
        consumed by this env's ``MathematicaStateIngress`` are returned to
        ``gateway.network_queue`` and available for the next trial's env.
        """
        self._clear_all_slots()
        returned = self.state_ingress.drain_to_queue()
        if returned > 0:
            logger.info(
                "VecEnv close: returned %d deferred messages to the queue.",
                returned,
            )

    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        return [getattr(self, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = self._get_indices(indices)
        for _ in indices:
            setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = self._get_indices(indices)
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = self._get_indices(indices)
        return [False for _ in indices]

    def seed(self, seed=None):
        return [seed for _ in range(self.num_envs)]


def build_mathematica_training_env(
    *,
    gateway,
    preprocessor,
    reward_calculator: RewardCalculator,
    n_envs: int,
    max_nodes: int,
    max_edges: int,
):
    return MathematicaVecEnv(
        num_envs=n_envs,
        gateway=gateway,
        preprocessor=preprocessor,
        reward_calculator=reward_calculator,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
