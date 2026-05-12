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
"""
from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces

from mathematica_env import decode_action_to_solver_tol
from replay_buffer import EpisodeReplayBuffer

logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stable_baselines3 wird für AsyncMathematicaVecEnv benötigt."
    ) from e


def _norm_uuid(msg: dict) -> str:
    u = msg.get("uuid")
    if u is None:
        u = msg.get("id")
    return str(u) if u is not None else "unknown"


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
    ):
        if num_envs < 1:
            raise ValueError("num_envs muss >= 1 sein.")

        self.gateway = gateway
        self.preprocessor = preprocessor
        self.reward_calculator = reward_calculator
        self.max_nodes = max_nodes
        self.max_edges = max_edges

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

    def close(self) -> None:
        self._step_async_pending = False
        self._saved_actions = None

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

    def _recv_state_blocking(self) -> dict:
        while True:
            try:
                return self.gateway.network_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")

    def _stack_obs(self, obs_list: List[dict]) -> dict:
        keys = obs_list[0].keys()
        return {k: np.stack([np.asarray(o[k]) for o in obs_list]) for k in keys}

    def _assign_initial_state(self, slot_idx: int, state_dict: dict) -> None:
        uid = _norm_uuid(state_dict)
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
            msg = self._recv_state_blocking()
            if msg.get("status") in ["reward_calc", "finished"]:
                logger.info(
                    "Überspringe Terminal/Legacy-State uuid=%s beim Vec-Reset.",
                    msg.get("uuid"),
                )
                continue
            uid = _norm_uuid(msg)
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

        while any(r is None for r in received):
            msg = self._recv_state_blocking()
            uid = _norm_uuid(msg)
            if uid not in expected:
                logger.warning(
                    "Nachricht uuid=%s passt zu keinem erwarteten Slot — verworfen.",
                    uid,
                )
                continue
            slot_idx = expected[uid]
            if received[slot_idx] is not None:
                logger.warning("Doppelte Antwort für Slot %s — überspringe.", slot_idx)
                continue
            received[slot_idx] = msg

        rews = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        terminal_slots: List[int] = []

        for k in range(self.num_envs):
            next_state = received[k]
            assert next_state is not None
            slot = self._slots[k]
            self.replay_buffer.set_next_state(slot.current_uuid, next_state)

            if next_state.get("status") in ["reward_calc", "finished"]:
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
            msg = self._recv_state_blocking()
            if msg.get("status") in ["reward_calc", "finished"]:
                logger.info("Terminal/Legacy nach Episode — warte auf neuen Start-State.")
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
