from __future__ import annotations

import queue
import time
from collections import deque
from typing import Deque, Dict, Optional

_TERMINAL_STATUSES = frozenset({"reward_calc", "finished", "error", "non_converged"})


def episode_uuid(message: dict) -> Optional[str]:
    uuid = message.get("uuid")
    if uuid is None:
        return None
    return str(uuid)


def gateway_channel(message: dict) -> str:
    return message.get("_gateway_channel", "training")


def is_reward_port_message(message: dict) -> bool:
    return gateway_channel(message) == "reward"


class MathematicaStateIngress:
    def __init__(self, gateway):
        self.gateway = gateway
        self._deferred_by_uuid: Dict[str, dict] = {}
        self._waiting_init_order: Deque[str] = deque()

    def take_next_training_start(self) -> dict:
        while True:
            message = self._take_waiting_init()
            if message is None:
                message = self._recv_blocking()
            if is_reward_port_message(message):
                continue
            status = message.get("status")
            if status in _TERMINAL_STATUSES:
                continue
            if episode_uuid(message) is None:
                continue
            return message

    def poll_next_for_episode(self, active_uuid: str) -> Optional[dict]:
        deferred = self._deferred_by_uuid.pop(active_uuid, None)
        if deferred is not None:
            self._remove_from_waiting_init(active_uuid)
            return deferred

        while True:
            try:
                message = self.gateway.network_queue.get_nowait()
            except queue.Empty:
                return None
            message_uuid = episode_uuid(message)
            if message_uuid == active_uuid:
                return message
            self._defer(message)

    def take_next_for_episode(self, active_uuid: str, timeout_s: Optional[float] = None) -> Optional[dict]:
        message = self.poll_next_for_episode(active_uuid)
        if message is not None:
            return message

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            try:
                wait = 0.1 if remaining is None else min(0.1, remaining)
                message = self.gateway.network_queue.get(timeout=wait)
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")
                continue
            message_uuid = episode_uuid(message)
            if message_uuid == active_uuid:
                return message
            self._defer(message)

    def _take_waiting_init(self) -> Optional[dict]:
        while self._waiting_init_order:
            uuid = self._waiting_init_order.popleft()
            message = self._deferred_by_uuid.pop(uuid, None)
            if message is None:
                continue
            if is_reward_port_message(message):
                continue
            if message.get("status") in _TERMINAL_STATUSES:
                continue
            if episode_uuid(message) is None:
                continue
            return message
        return None

    def _defer(self, message: dict) -> None:
        uuid = episode_uuid(message)
        if uuid is None:
            return
        if (
            uuid not in self._deferred_by_uuid
            and message.get("status") not in _TERMINAL_STATUSES
            and not is_reward_port_message(message)
        ):
            self._waiting_init_order.append(uuid)
        self._deferred_by_uuid[uuid] = message

    def drain_to_queue(self) -> int:
        """Return all deferred messages back into the shared gateway queue.

        This must be called before the ingress is destroyed (e.g. between
        Optuna trials) to prevent message loss.  Messages that were consumed
        from ``network_queue`` during ``poll_next_for_episode`` or
        ``take_next_for_episode`` but belonged to a different UUID are stored
        locally in ``_deferred_by_uuid``.  Without draining, they would be
        silently lost when the ingress goes out of scope.

        Returns:
            The number of messages returned to the queue.
        """
        returned = 0
        for uuid in list(self._waiting_init_order):
            message = self._deferred_by_uuid.pop(uuid, None)
            if message is not None:
                self.gateway.network_queue.put(message)
                returned += 1
        for uuid, message in self._deferred_by_uuid.items():
            self.gateway.network_queue.put(message)
            returned += 1
        self._deferred_by_uuid.clear()
        self._waiting_init_order.clear()
        return returned

    def _remove_from_waiting_init(self, uuid: str) -> None:
        self._waiting_init_order = deque(
            waiting_uuid
            for waiting_uuid in self._waiting_init_order
            if waiting_uuid != uuid
        )

    def _recv_blocking(self) -> dict:
        while True:
            try:
                return self.gateway.network_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.gateway.running:
                    raise InterruptedError("Gateway stopped running.")
