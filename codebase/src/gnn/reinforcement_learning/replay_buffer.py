import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EpisodeReplayBuffer:
    """Episode replay buffer keyed by problem UUID."""

    def __init__(self, keep_completed: bool = False):
        self._buffer: Dict[str, List[Dict]] = {}
        self._completed: Dict[str, List[Dict]] = {}
        self._keep_completed = keep_completed

    def start_episode(self, uuid: str) -> None:
        if uuid in self._buffer:
            logger.warning(
                "Episode for UUID %s already exists (%d transitions). Overwriting.",
                uuid, len(self._buffer[uuid])
            )
        self._buffer[uuid] = []

    def add_transition(
        self,
        uuid: str,
        current_state: dict,
        action: dict
    ) -> None:
        if uuid not in self._buffer:
            raise KeyError(f"No episode started for UUID {uuid}. Call start_episode() first.")
        self._buffer[uuid].append({
            "current_state": current_state,
            "action": action,
            "next_state": None,
            "reward": None
        })

    def set_next_state(self, uuid: str, next_state: dict) -> None:
        if uuid not in self._buffer:
            raise KeyError(f"No episode for UUID {uuid}.")
        if not self._buffer[uuid]:
            raise IndexError(f"Episode for UUID {uuid} is empty.")
        self._buffer[uuid][-1]["next_state"] = next_state

    def get_transitions(self, uuid: str) -> List[Dict]:
        if uuid not in self._buffer:
            raise KeyError(f"No episode for UUID {uuid}.")
        return self._buffer[uuid]

    def clear_episode(self, uuid: str) -> None:
        if uuid in self._buffer:
            if self._keep_completed:
                self._completed[uuid] = self._buffer[uuid]
            del self._buffer[uuid]

    def get_episode_length(self, uuid: str) -> int:
        return len(self._buffer.get(uuid, []))

    def has_episode(self, uuid: str) -> bool:
        return uuid in self._buffer

    @property
    def completed_episodes(self) -> Optional[Dict[str, List[Dict]]]:
        if self._keep_completed:
            return self._completed
        return None
