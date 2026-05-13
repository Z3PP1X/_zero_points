"""
Vectorisiertes Mathematica-Env mit asynchron eintreffenden States (ein ZMQ-Port).

Alle Nachrichten laufen über dieselbe ``network_queue``; Zuordnung pro PPO-Slot
über die Episode-``uuid`` im State-JSON. ``step_async`` sendet alle Actions;
``step_wait`` sammelt per Collector, bis jeder Slot eine passende Antwort hat
(Reihenfolge beliebig).

Voraussetzungen:
    - Jeder State enthält ``uuid`` (Episode), stabil bis Terminal.
    - Mathematica sendet Antworten mit derselben ``uuid`` wie die zugehörige Episode.
    - Nach ``reset`` kommen ``num_envs`` initiale (nicht-terminale) States; Slots werden
      der Reihe nach 0..n-1 mit eintreffenden Starts belegt (Reihenfolge der Nachrichten
      = Slot-Index). Für ``step_wait`` gilt: Antworten dürfen asynchron kommen, müssen
      aber die erwartete Episode-``uuid`` tragen.

    Wenn in einem Vec-Schritt **mehrere** Slots terminal werden, müssen die folgenden
    **Neustart-States** von Mathematica in **aufsteigender Slot-Reihenfolge** gesendet
    werden (zuerst niedrigster Slot-Index), da sonst die Zuordnung ohne zusätzliches
    Routing-Feld nicht eindeutig wäre.

Robustheit:
    - ``step_wait`` nutzt einen Inaktivitäts-Timeout (Standard 2 s): fehlende Slots
      werden nur beendet, wenn für diese Dauer keine weitere passende Antwort
      eintrifft. Jede erwartete Antwort setzt den Timer zurück.
    - Falls Mathematica eine erwartete Antwort verliert, wird der betroffene Slot
      als Episode-Fehler beendet (done=True, negativer Reward), damit die
      SB3-Trainings-Loop nicht deadlockt.
"""
from __future__ import annotations

import logging
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from gymnasium import spaces

from mathematica_env import decode_action_to_solver_tol
from replay_buffer import EpisodeReplayBuffer

logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.vec_env.base_vec_env import (
        VecEnv,
        VecEnvObs,
        VecEnvStepReturn,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stable_baselines3 wird für AsyncMathematicaVecEnv benötigt."
    ) from e


def _norm_uuid(msg: dict) -> Optional[str]:
    """
    Returns the episode UUID for an incoming message, or ``None`` if absent.

    Important: We DO NOT fall back to ``msg["id"]`` because problem ``id`` (e.g.
    "P1") is shared across thousands of episodes — that fallback would route
    arbitrary problem messages to the first slot expecting that problem,
    silently corrupting episode boundaries.
    """
    u = msg.get("uuid")
    if u is None:
        return None
    return str(u)


_TERMINAL_STATUSES = frozenset({"reward_calc", "finished", "error", "non_converged"})


@dataclass
class _Slot:
    current_state_dict: Optional[dict] = None
    current_obs: Optional[dict] = None
    current_uuid: Optional[str] = None


class AsyncMathematicaVecEnv(VecEnv):
    """
    n parallele logische Umgebungen, ein Gateway, UUID-gestütztes Routing.

    Ein Port reicht: eingehende JSON-Nachrichten werden anhand ``uuid`` dem Slot
    zugeordnet, der genau diese Episode erwartet.
    """

    def __init__(
        self,
        gateway,
        preprocessor,
        reward_calculator,
        num_envs: int,
        max_nodes: int = 200,
        max_edges: int = 1000,
        step_timeout_s: float = 2.0,
        timeout_penalty: float = -10.0,
    ):
        if num_envs < 1:
            raise ValueError("num_envs muss >= 1 sein.")

        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.step_timeout_s = step_timeout_s
        self.timeout_penalty = timeout_penalty

        self._slots: List[_Slot] = [_Slot() for _ in range(num_envs)]
        self.replay_buffer = EpisodeReplayBuffer(keep_completed=False)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_nodes, 5),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max_nodes - 1,
                    shape=(2, max_edges),
                    dtype=np.int64,
                ),
                "global_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
                ),
                "num_nodes": spaces.Box(low=0, high=max_nodes, shape=(1,), dtype=np.int64),
                "num_edges": spaces.Box(low=0, high=max_edges, shape=(1,), dtype=np.int64),
            }
        )

        super().__init__(num_envs, self.observation_space, self.action_space)

        self._saved_actions: Optional[np.ndarray] = None
        self._step_async_pending = False
        self._deferred_by_uuid: Dict[str, dict] = {}
        self._deferred_order: Deque[str] = deque()

    # --- VecEnv: render_mode für super().__init__ ---
    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        indices_list = self._indices_to_list(indices)
        if attr_name == "render_mode":
            return [None for _ in indices_list]
        if attr_name == "current_state_dict":
            return [self._slots[i].current_state_dict for i in indices_list]
        if attr_name == "current_uuid":
            return [self._slots[i].current_uuid for i in indices_list]
        raise AttributeError(f"{attr_name} ist in AsyncMathematicaVecEnv nicht verfügbar.")

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError("AsyncMathematicaVecEnv unterstützt set_attr nicht.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        raise NotImplementedError("AsyncMathematicaVecEnv unterstützt env_method nicht.")

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        """
        SB3 VecEnv-API: Keine eingebetteten gym.Wrapper pro Slot — Wrapping passiert
        außerhalb (z. B. VecMonitor).
        """
        return [False for _ in self._indices_to_list(indices)]

    def close(self) -> None:
        self._step_async_pending = False
        self._saved_actions = None
        self._deferred_by_uuid.clear()
        self._deferred_order.clear()

    # --- Hilfen ---
    def _indices_to_list(self, indices) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def _pad_graph(self, pyg_data) -> dict:
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

    def _defer_incoming(self, msg: dict) -> None:
        uid = _norm_uuid(msg)
        if uid is None:
            logger.warning(
                "Nachricht ohne uuid (status=%s, channel=%s) — verworfen.",
                msg.get("status"),
                msg.get("_gateway_channel"),
            )
            return
        if uid not in self._deferred_by_uuid:
            self._deferred_order.append(uid)
        self._deferred_by_uuid[uid] = msg

    def _pop_deferred(self, uid: str) -> Optional[dict]:
        msg = self._deferred_by_uuid.pop(uid, None)
        if msg is None:
            return None
        self._deferred_order = deque(u for u in self._deferred_order if u != uid)
        return msg

    def _pull_deferred_start_state(self) -> Optional[dict]:
        for uid in list(self._deferred_order):
            msg = self._deferred_by_uuid.get(uid)
            if msg is None:
                continue
            if msg.get("status") in _TERMINAL_STATUSES:
                continue
            if _norm_uuid(msg) is None:
                continue
            return self._pop_deferred(uid)
        return None

    def _fill_received_from_deferred(
        self,
        expected: Dict[str, int],
        received: List[Optional[dict]],
    ) -> None:
        for uid, slot_idx in expected.items():
            if received[slot_idx] is not None:
                continue
            deferred = self._pop_deferred(uid)
            if deferred is not None:
                received[slot_idx] = deferred

    def _recv_state_blocking(self) -> dict:
        deferred = self._pull_deferred_start_state()
        if deferred is not None:
            return deferred
        while True:
            try:
                return self.gateway.network_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def _recv_state_with_deadline(self, deadline: float) -> Optional[dict]:
        """Returns a state dict if one arrives before `deadline`, else ``None``."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return self.gateway.network_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def _stack_obs(self, obs_list: List[dict]) -> dict:
        keys = obs_list[0].keys()
        return {k: np.stack([np.asarray(o[k]) for o in obs_list]) for k in keys}

    def _assign_initial_state(self, slot_idx: int, state_dict: dict) -> None:
        uid = _norm_uuid(state_dict)
        if uid is None:
            raise RuntimeError(
                f"Initial state für Slot {slot_idx} hat keine uuid: {state_dict}"
            )
        slot = self._slots[slot_idx]
        slot.current_uuid = uid
        self.replay_buffer.start_episode(uid)
        slot.current_state_dict = state_dict
        pyg_data, _ = self.preprocessor.process(state_dict, dataloader=None)
        slot.current_obs = self._pad_graph(pyg_data)

    def reset(self) -> VecEnvObs:
        self._step_async_pending = False
        self._saved_actions = None

        seen_uuids: set[str] = set()
        slot_idx = 0
        while slot_idx < self.num_envs:
            msg = self._pull_deferred_start_state()
            if msg is None:
                msg = self._recv_state_blocking()
            if msg.get("status") in ["reward_calc", "finished", "error", "non_converged"]:
                uid = _norm_uuid(msg)
                if uid is not None:
                    self._defer_incoming(msg)
                logger.info(
                    "Überspringe Terminal/Legacy/Error-State uuid=%s beim Vec-Reset.",
                    msg.get("uuid"),
                )
                continue
            uid = _norm_uuid(msg)
            if uid is None:
                logger.warning(
                    "Reset: Nachricht ohne uuid verworfen (id=%s, status=%s).",
                    msg.get("id"), msg.get("status"),
                )
                continue
            if uid in seen_uuids:
                logger.warning("Doppelte uuid %s beim Reset — überspringe.", uid)
                continue
            seen_uuids.add(uid)
            self._assign_initial_state(slot_idx, msg)
            slot_idx += 1

        self.reset_infos = [{} for _ in range(self.num_envs)]
        obs_list = [self._slots[i].current_obs for i in range(self.num_envs)]
        return self._stack_obs(obs_list)

    def step_async(self, actions: np.ndarray) -> None:
        if self._step_async_pending:
            raise RuntimeError("step_async doppelt aufgerufen — step_wait fehlt.")
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions.shape[0]={actions.shape[0]} != num_envs={self.num_envs}")

        self._saved_actions = actions
        for k in range(self.num_envs):
            slot = self._slots[k]
            if slot.current_state_dict is None:
                raise RuntimeError(f"Slot {k} hat keinen State vor step_async.")

            action = actions[k]
            _, _, action_dict = decode_action_to_solver_tol(action, slot.current_state_dict)

            self.replay_buffer.add_transition(
                uuid=slot.current_uuid,
                current_state=slot.current_state_dict,
                action=action_dict,
            )
            self.gateway.send_decision(
                slot.current_state_dict,
                action_dict["solver"],
                action_dict["localMaxTolerance"],
            )

        self._step_async_pending = True

    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for one response per slot, identified by the slot's expected uuid.

        Uses an inactivity timeout (``step_timeout_s``): the wait ends only after
        that many seconds without a newly matched response for this step. Each
        matched reply resets the timer.

        If slots are still missing after the idle period, they are terminated
        with ``timeout_penalty`` and replaced with the next start state.
        """
        if not self._step_async_pending or self._saved_actions is None:
            raise RuntimeError("step_wait ohne step_async.")

        expected: Dict[str, int] = {}
        for k in range(self.num_envs):
            uid = self._slots[k].current_uuid
            if uid is None:
                raise RuntimeError(f"Slot {k} ohne uuid vor step_wait.")
            if uid in expected:
                raise RuntimeError(f"uuid-Kollision zwischen Slots: {uid}")
            expected[uid] = k

        received: List[Optional[dict]] = [None] * self.num_envs
        last_progress = time.monotonic()

        while any(r is None for r in received):
            pending_before = sum(1 for r in received if r is None)
            self._fill_received_from_deferred(expected, received)
            if sum(1 for r in received if r is None) < pending_before:
                last_progress = time.monotonic()
                if not any(r is None for r in received):
                    break

            msg = self._recv_state_with_deadline(last_progress + self.step_timeout_s)
            if msg is None:
                missing = [k for k in range(self.num_envs) if received[k] is None]
                missing_uuids = [self._slots[k].current_uuid for k in missing]
                logger.error(
                    "step_wait Idle-Timeout nach %.1fs ohne neue passende Antwort — "
                    "Slots ohne Antwort: %s (uuids=%s).",
                    self.step_timeout_s, missing, missing_uuids,
                )
                break
            uid = _norm_uuid(msg)
            if uid is None:
                logger.warning(
                    "Nachricht ohne uuid (status=%s) — verworfen.", msg.get("status")
                )
                continue
            if uid not in expected:
                self._defer_incoming(msg)
                logger.debug(
                    "Nachricht uuid=%s (status=%s, channel=%s) zwischengespeichert — "
                    "passt nicht zu den erwarteten Slots dieses step_wait.",
                    uid,
                    msg.get("status"),
                    msg.get("_gateway_channel"),
                )
                continue
            slot_idx = expected[uid]
            if received[slot_idx] is not None:
                logger.warning("Doppelte Antwort für Slot %s — überspringe.", slot_idx)
                continue
            received[slot_idx] = msg
            last_progress = time.monotonic()

        rews = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        terminal_slots: List[int] = []

        for k in range(self.num_envs):
            slot = self._slots[k]
            next_state = received[k]

            if next_state is None:
                # Slot timed out: synthesize a terminal failure transition.
                if self.replay_buffer.has_episode(slot.current_uuid):
                    self.replay_buffer.clear_episode(slot.current_uuid)
                rews[k] = self.timeout_penalty
                dones[k] = True
                infos[k] = {
                    "episode_steps": 0,
                    "total_reward": float(self.timeout_penalty),
                    "timeout": True,
                }
                terminal_slots.append(k)
                continue

            self.replay_buffer.set_next_state(slot.current_uuid, next_state)
            status = next_state.get("status")

            if status in ("error", "non_converged"):
                n_steps_err = self.replay_buffer.get_episode_length(slot.current_uuid)
                if self.replay_buffer.has_episode(slot.current_uuid):
                    self.replay_buffer.clear_episode(slot.current_uuid)
                rews[k] = self.timeout_penalty
                dones[k] = True
                infos[k] = {
                    "episode_steps": n_steps_err,
                    "total_reward": float(self.timeout_penalty),
                    "error": status,
                }
                terminal_slots.append(k)
                continue

            if status in ("reward_calc", "finished"):
                transitions = self.replay_buffer.get_transitions(slot.current_uuid)
                self.reward_calculator.calculate_episode_rewards(transitions, next_state)
                total_reward = float(sum(t.get("reward", 0.0) for t in transitions))
                n_steps = len(transitions)
                self.replay_buffer.clear_episode(slot.current_uuid)

                rews[k] = total_reward
                dones[k] = True
                infos[k] = {"episode_steps": n_steps, "total_reward": total_reward}
                terminal_slots.append(k)
            else:
                slot.current_state_dict = next_state
                pyg_data, _ = self.preprocessor.process(next_state, dataloader=None)
                slot.current_obs = self._pad_graph(pyg_data)
                rews[k] = 0.0
                dones[k] = False

        # Mehrere Neustarts: Mathematica liefert Start-States in aufsteigender Slot-Reihenfolge.
        for slot in sorted(terminal_slots):
            init_msg = self._wait_non_terminal_state_after_done()
            self._assign_initial_state(slot, init_msg)

        obs_list = [self._slots[k].current_obs for k in range(self.num_envs)]
        self._step_async_pending = False
        self._saved_actions = None

        return self._stack_obs(obs_list), rews, dones, infos

    def _wait_non_terminal_state_after_done(self) -> dict:
        while True:
            msg = self._pull_deferred_start_state()
            if msg is None:
                msg = self._recv_state_blocking()
            if msg.get("status") in ("reward_calc", "finished", "error", "non_converged"):
                uid = _norm_uuid(msg)
                if uid is not None:
                    self._defer_incoming(msg)
                logger.info(
                    "Terminal/Error nach Episode — warte auf neuen Start-State (status=%s).",
                    msg.get("status"),
                )
                continue
            uid = _norm_uuid(msg)
            if uid is None:
                logger.warning(
                    "Neustart-Nachricht ohne uuid — verworfen (id=%s).", msg.get("id")
                )
                continue
            return msg

    def flush_unfinished_slots(self, max_steps: int = 10_000) -> None:
        """Zufällige Actions bis alle Slots terminal oder max_steps (für Trainingsabbruch)."""
        steps = 0
        while steps < max_steps:
            busy = [
                k
                for k in range(self.num_envs)
                if self._slots[k].current_state_dict
                and self._slots[k].current_state_dict.get("status")
                not in ["reward_calc", "finished"]
            ]
            if not busy:
                break
            actions = np.array(
                [
                    self.action_space.sample()
                    for _ in range(self.num_envs)
                ],
                dtype=np.float32,
            )
            self.step_async(actions)
            self.step_wait()
            steps += 1

        for k in range(self.num_envs):
            uid = self._slots[k].current_uuid
            if uid and self.replay_buffer.has_episode(uid):
                self.replay_buffer.clear_episode(uid)
