"""
Vectorized environment for the Mathematica RL pipeline.

Maintains ``num_envs`` slots, each holding one in-flight UUID/episode.

Step semantics (per ``step``):
    1. Send ``num_envs`` actions back to Mathematica in one batch
       (one per slot, tagged with the slot's current state dict and UUID).
    2. Receive exactly ``num_envs`` responses (state or terminal). Responses
       may arrive in arbitrary order and are routed to slots via the UUID
       carried in the payload.
    3. Terminal slots have their episode reward computed retroactively and
       are immediately refilled from the pool of fresh initial states that
       Mathematica is streaming.

This lets Mathematica process up to ``num_envs`` problems concurrently
while the Python policy runs a single batched forward pass per step,
hiding most of the Mathematica round-trip latency.

The class deliberately keeps the synchronous SB3 ``VecEnv`` contract: a
call to ``step_wait`` blocks until all slots have advanced once. A truly
asynchronous "dump factory" runtime is a larger redesign and would only
fit naturally with off-policy algorithms.
"""
from __future__ import annotations

import logging
import math
import queue
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from network_gateway import CHANNEL_STATE, CHANNEL_TERMINAL
from replay_buffer import EpisodeReplayBuffer

logger = logging.getLogger(__name__)

# Channel/payload tuple as published by NetworkGateway.event_queue.
Event = Tuple[str, Dict[str, Any]]


class MathematicaVecEnv(VecEnv):
    """
    Slot-based vectorized environment over the existing ``NetworkGateway``.

    Args:
        num_envs: Number of concurrent slots = active in-flight UUIDs.
            Mathematica must be able to keep at least ``num_envs`` problems
            in its outgoing pool, otherwise ``reset`` / refills will block.
        gateway: ``NetworkGateway`` instance (must already be ``init()``'d).
        preprocessor: ``Preprocessor`` instance (graph cache + features).
        reward_calculator: ``RewardCalculator`` for retroactive episode rewards.
        max_nodes: Padding size for the node feature tensor.
        max_edges: Padding size for the edge index tensor.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        gateway,
        preprocessor,
        reward_calculator,
        max_nodes: int = 200,
        max_edges: int = 1000,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_nodes, 5),
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
                    shape=(9,),
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

        # Per-slot mutable state. Index = slot id.
        self._slot_uuid: List[Optional[str]] = [None] * num_envs
        self._slot_state: List[Optional[Dict[str, Any]]] = [None] * num_envs
        self._slot_obs: List[Optional[Dict[str, np.ndarray]]] = [None] * num_envs
        self._slot_episode_steps: List[int] = [0] * num_envs

        # Reverse lookup: UUID currently in flight -> slot id.
        self._uuid_to_slot: Dict[str, int] = {}

        # Pool of received but not yet assigned initial states. Mathematica
        # may push more new problems than we currently have free slots; we
        # buffer them here so refills can proceed without an extra round-trip.
        self._fresh_states: deque = deque()
        self._fresh_uuids: set = set()

        # Step responses that arrived for a slot UUID before we were ready
        # to consume them (e.g. multiple events for the same slot in a row,
        # or events received during reset). Kept here until step_wait drains.
        self._pending_step_responses: Dict[str, Event] = {}

        # Terminal events that arrived for a UUID which is in _fresh_states
        # (received but not yet assigned to a slot). Consumed in _fill_slot.
        self._pending_terminals: Dict[str, Event] = {}

        # Shared episode replay buffer (UUID-keyed); reused across slots.
        self.replay_buffer = EpisodeReplayBuffer(keep_completed=False)

        # Filled by step_async, consumed by step_wait.
        self._actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _decode_action(
        self, action: np.ndarray, state_dict: Dict[str, Any]
    ) -> Tuple[int, float]:
        chosen_solver = 1 if float(action[0]) > 0 else 0
        base_tol = state_dict.get("tolerance", 1e-15)
        log_tol_base = math.log10(base_tol)
        log_tol_min = log_tol_base - 4.0
        log_tol_max = log_tol_base + 4.0
        scale = (float(action[1]) + 1.0) / 2.0
        log10_tol = log_tol_min + scale * (log_tol_max - log_tol_min)
        chosen_tol = 10.0 ** log10_tol
        return chosen_solver, chosen_tol

    def _pad_graph(self, pyg_data) -> Dict[str, np.ndarray]:
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
    def _is_terminal(channel: str, payload: Dict[str, Any]) -> bool:
        return channel == CHANNEL_TERMINAL

    @staticmethod
    def _payload_uuid(payload: Dict[str, Any]) -> str:
        uuid = payload.get("uuid")
        if uuid is None:
            uuid = str(payload.get("id", "unknown"))
        return uuid

    def _next_event(self) -> Event:
        """Block until the next ``(channel, payload)`` arrives from the gateway."""
        while True:
            try:
                channel, payload = self.gateway.event_queue.get(timeout=0.1)
                # Skip legacy terminal states sent on the state port
                if channel == CHANNEL_STATE and payload.get("status") in ("reward_calc", "finished"):
                    continue
                return channel, payload
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def _route_event(self, channel: str, payload: Dict[str, Any]) -> None:
        """
        Classify an event that does not directly answer a step we sent:

        - response for a UUID currently in a slot -> buffer for ``step_wait``
        - new initial state for an unknown UUID   -> push into the fresh pool
        - terminal/reward for an unknown UUID     -> drop with warning
        """
        uuid = self._payload_uuid(payload)

        if uuid in self._uuid_to_slot:
            self._pending_step_responses[uuid] = (channel, payload)
            return

        if self._is_terminal(channel, payload):
            if uuid in self._fresh_uuids:
                # Problem completed before Python assigned it to a slot.
                # Buffer the terminal so _fill_slot can handle it immediately.
                logger.debug(
                    "Buffering early terminal for fresh UUID %s", uuid
                )
                self._pending_terminals[uuid] = (channel, payload)
            else:
                logger.debug(
                    "Dropping terminal/reward event for untracked UUID %s "
                    "(Mathematica processed more problems than num_envs slots).",
                    uuid,
                )
            return

        if uuid in self._fresh_uuids:
            return  # duplicate fresh state for this UUID; ignore.

        self._fresh_states.append(payload)
        self._fresh_uuids.add(uuid)

    def _get_fresh_state(self) -> Dict[str, Any]:
        """Return the next non-terminal initial state, blocking if needed."""
        while not self._fresh_states:
            channel, payload = self._next_event()
            self._route_event(channel, payload)
        state_dict = self._fresh_states.popleft()
        uuid = self._payload_uuid(state_dict)
        self._fresh_uuids.discard(uuid)
        return state_dict

    def _fill_slot(self, slot: int) -> None:
        state_dict = self._get_fresh_state()
        uuid = self._payload_uuid(state_dict)

        self._slot_uuid[slot] = uuid
        self._slot_state[slot] = state_dict
        self._uuid_to_slot[uuid] = slot
        self._slot_episode_steps[slot] = 0

        self.replay_buffer.start_episode(uuid)
        pyg_data, _ = self.preprocessor.process(state_dict, dataloader=None)
        self._slot_obs[slot] = self._pad_graph(pyg_data)

        # If Mathematica already sent a terminal for this UUID before we slotted
        # it, move it to pending_step_responses so step_wait handles it on the
        # very first response cycle for this slot.
        if uuid in self._pending_terminals:
            logger.debug("Recovering buffered terminal for newly-slotted UUID %s", uuid)
            self._pending_step_responses[uuid] = self._pending_terminals.pop(uuid)

    def _stack_obs(self) -> Dict[str, np.ndarray]:
        return {
            key: np.stack(
                [self._slot_obs[s][key] for s in range(self.num_envs)], axis=0
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
        self._pending_terminals.clear()

    # ------------------------------------------------------------------ #
    # VecEnv API
    # ------------------------------------------------------------------ #
    def reset(self):
        self._clear_all_slots()
        for slot in range(self.num_envs):
            self._fill_slot(slot)
        return self._stack_obs()

    def step_async(self, actions: np.ndarray) -> None:
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected actions of shape ({self.num_envs}, ...), "
                f"got {actions.shape}"
            )
        self._actions = actions

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async.")
        actions = self._actions
        self._actions = None

        # 1) Push all N decisions into the network in one go. Mathematica
        #    can then progress the slots concurrently; we overlap solver
        #    work across UUIDs for the duration of one step.
        for slot in range(self.num_envs):
            state_dict = self._slot_state[slot]
            uuid = self._slot_uuid[slot]
            chosen_solver, chosen_tol = self._decode_action(actions[slot], state_dict)
            action_dict = {
                "solver": chosen_solver,
                "localMaxTolerance": chosen_tol,
            }
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

        # 2a) Drain anything that already arrived for active slots while we
        #     were busy elsewhere (e.g. responses that landed during reset).
        for uuid in list(self._pending_step_responses.keys()):
            slot = self._uuid_to_slot.get(uuid)
            if slot is None or slots_done_this_step[slot]:
                continue
            channel, payload = self._pending_step_responses.pop(uuid)
            self._handle_response(slot, channel, payload, rewards, dones, infos)
            slots_done_this_step[slot] = True

        remaining = sum(1 for done in slots_done_this_step if not done)

        # 2b) Block on the gateway until every slot has advanced exactly once.
        while remaining > 0:
            channel, payload = self._next_event()
            uuid = self._payload_uuid(payload)
            slot = self._uuid_to_slot.get(uuid)

            if slot is None:
                self._route_event(channel, payload)
                continue

            if slots_done_this_step[slot]:
                # Two events for the same slot in one step: the slot was
                # already advanced; keep the extra event for the next step.
                self._pending_step_responses[uuid] = (channel, payload)
                continue

            self._handle_response(slot, channel, payload, rewards, dones, infos)
            slots_done_this_step[slot] = True
            remaining -= 1

        return self._stack_obs(), rewards, dones, infos

    def _handle_response(
        self,
        slot: int,
        channel: str,
        payload: Dict[str, Any],
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        uuid = self._slot_uuid[slot]
        self.replay_buffer.set_next_state(uuid, payload)

        if self._is_terminal(channel, payload):
            transitions = self.replay_buffer.get_transitions(uuid)
            self.reward_calculator.calculate_episode_rewards(transitions, payload)
            total_reward = float(sum(t.get("reward", 0.0) for t in transitions))
            n_steps = len(transitions)

            # SB3 reads info["episode"] to fill ep_info_buffer (normally
            # provided by the Monitor wrapper which we do not use here).
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

            # SB3 vec env contract: obs returned for a done env is the
            # initial observation of the next episode. Refill immediately.
            self._fill_slot(slot)
        else:
            self._slot_state[slot] = payload
            pyg_data, _ = self.preprocessor.process(payload, dataloader=None)
            self._slot_obs[slot] = self._pad_graph(pyg_data)
            rewards[slot] = 0.0
            dones[slot] = False
            infos[slot] = {}

    # ------------------------------------------------------------------ #
    # Other VecEnv contract methods
    # ------------------------------------------------------------------ #
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
