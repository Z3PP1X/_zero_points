from __future__ import annotations

import logging
import queue
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from feature_layout import (
    NATIVE_GLOBAL_FEATURE_COUNT,
    NATIVE_NODE_FEATURE_COUNT,
    PADDED_GLOBAL_FEATURE_COUNT,
    PADDED_NODE_FEATURE_COUNT,
)
from mathematica_env import decode_action_to_solver_tol
from replay_buffer import EpisodeReplayBuffer
from reward import RewardCalculator

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset(
    {"reward_calc", "finished", "error", "non_converged"}
)


def _payload_uuid(payload: Dict[str, Any]) -> Optional[str]:
    uuid = payload.get("uuid")
    if uuid is None:
        return None
    return str(uuid)


def _gateway_channel(payload: Dict[str, Any]) -> str:
    return payload.get("_gateway_channel", "training")


class MathematicaVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        gateway,
        preprocessor,
        reward_calculator: RewardCalculator,
        max_nodes: int = 200,
        max_edges: int = 1000,
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

        super().__init__(num_envs, observation_space, action_space)

        self._slot_uuid: List[Optional[str]] = [None] * num_envs
        self._slot_state: List[Optional[Dict[str, Any]]] = [None] * num_envs
        self._slot_obs: List[Optional[Dict[str, np.ndarray]]] = [None] * num_envs
        self._slot_episode_steps: List[int] = [0] * num_envs
        self._uuid_to_slot: Dict[str, int] = {}
        self._fresh_states: deque[Dict[str, Any]] = deque()
        self._fresh_uuids: set[str] = set()
        self._pending_step_responses: Dict[str, Dict[str, Any]] = {}
        self.replay_buffer = EpisodeReplayBuffer(keep_completed=False)
        self._actions: Optional[np.ndarray] = None

    def _pad_graph(self, pyg_data) -> Dict[str, np.ndarray]:
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

    def _is_terminal(self, payload: Dict[str, Any]) -> bool:
        if _gateway_channel(payload) == "reward":
            return True
        return payload.get("status") in _TERMINAL_STATUSES

    def _next_payload(self) -> Dict[str, Any]:
        while True:
            try:
                return self.gateway.network_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def _route_payload(self, payload: Dict[str, Any]) -> None:
        uuid = _payload_uuid(payload)
        if uuid is None:
            return

        if uuid in self._uuid_to_slot:
            self._pending_step_responses[uuid] = payload
            return

        if self._is_terminal(payload):
            logger.warning("Dropping terminal event for unknown UUID %s", uuid)
            return

        if uuid in self._fresh_uuids:
            return

        self._fresh_states.append(payload)
        self._fresh_uuids.add(uuid)

    def _get_fresh_state(self) -> Dict[str, Any]:
        while not self._fresh_states:
            payload = self._next_payload()
            self._route_payload(payload)
        state_dict = self._fresh_states.popleft()
        uuid = _payload_uuid(state_dict)
        if uuid is not None:
            self._fresh_uuids.discard(uuid)
        return state_dict

    def _fill_slot(self, slot: int) -> None:
        state_dict = self._get_fresh_state()
        uuid = _payload_uuid(state_dict)
        if uuid is None:
            raise ValueError("Fresh Mathematica state is missing a UUID.")

        self._slot_uuid[slot] = uuid
        self._slot_state[slot] = state_dict
        self._uuid_to_slot[uuid] = slot
        self._slot_episode_steps[slot] = 0

        self.replay_buffer.start_episode(uuid)
        pyg_data, _ = self.preprocessor.process(state_dict, dataloader=None)
        self._slot_obs[slot] = self._pad_graph(pyg_data)

    def _stack_obs(self) -> Dict[str, np.ndarray]:
        return {
            key: np.stack(
                [self._slot_obs[slot][key] for slot in range(self.num_envs)],
                axis=0,
            )
            for key in ("x", "edge_index", "global_features", "num_nodes", "num_edges")
        }

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
        self._fresh_states.clear()
        self._fresh_uuids.clear()
        self._pending_step_responses.clear()

    def _has_open_episodes(self) -> bool:
        return any(
            uuid is not None and self.replay_buffer.has_episode(uuid)
            for uuid in self._slot_uuid
        )

    def drain_pending_responses(self) -> bool:
        progressed = False
        while True:
            try:
                payload = self.gateway.network_queue.get_nowait()
            except queue.Empty:
                break

            uuid = _payload_uuid(payload)
            if uuid is None:
                continue

            slot = self._uuid_to_slot.get(uuid)
            if slot is None:
                self._route_payload(payload)
                progressed = True
                continue

            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
            self._handle_response(slot, payload, rewards, dones, infos)
            progressed = True

        for uuid in list(self._pending_step_responses.keys()):
            slot = self._uuid_to_slot.get(uuid)
            if slot is None:
                continue
            payload = self._pending_step_responses.pop(uuid)
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            infos = [{} for _ in range(self.num_envs)]
            self._handle_response(slot, payload, rewards, dones, infos)
            progressed = True

        return progressed

    def finalize_open_episodes(self, max_flush_steps: int = 128) -> None:
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
        slots_done_this_step = [False] * self.num_envs

        for uuid in list(self._pending_step_responses.keys()):
            slot = self._uuid_to_slot.get(uuid)
            if slot is None or slots_done_this_step[slot]:
                continue
            payload = self._pending_step_responses.pop(uuid)
            self._handle_response(slot, payload, rewards, dones, infos)
            slots_done_this_step[slot] = True

        remaining = sum(1 for done in slots_done_this_step if not done)

        while remaining > 0:
            payload = self._next_payload()
            uuid = _payload_uuid(payload)
            slot = self._uuid_to_slot.get(uuid) if uuid is not None else None

            if slot is None:
                self._route_payload(payload)
                continue

            if slots_done_this_step[slot]:
                self._pending_step_responses[uuid] = payload
                continue

            self._handle_response(slot, payload, rewards, dones, infos)
            slots_done_this_step[slot] = True
            remaining -= 1

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
            del self._uuid_to_slot[uuid]
            self._slot_uuid[slot] = None
            self._slot_state[slot] = None
            self._slot_obs[slot] = None
            self._slot_episode_steps[slot] = 0
            self._fill_slot(slot)
            return

        self.replay_buffer.set_next_state(uuid, payload)

        if self._is_terminal(payload):
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
            del self._uuid_to_slot[uuid]
            self._slot_uuid[slot] = None
            self._slot_state[slot] = None
            self._slot_obs[slot] = None
            self._slot_episode_steps[slot] = 0
            self._fill_slot(slot)
            return

        self._slot_state[slot] = payload
        pyg_data, _ = self.preprocessor.process(payload, dataloader=None)
        self._slot_obs[slot] = self._pad_graph(pyg_data)
        rewards[slot] = 0.0
        dones[slot] = False
        infos[slot] = {}

    def close(self) -> None:
        self._clear_all_slots()

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
