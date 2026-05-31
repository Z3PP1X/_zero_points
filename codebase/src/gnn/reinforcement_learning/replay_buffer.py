"""
Replay Buffer für die GNN-RL Pipeline.

Speichert Transitions pro Probleminstanz (UUID) als geordnete Liste.
Index 0 = erste Action, Index 1 = zweite Action, etc.

Jede Transition enthält:
    - current_state: Der State VOR der Action (raw dict von Mathematica)
    - action: Die gewählte Action {solver, localMaxTolerance}
    - next_state: Der State NACH der Action (None bis zum nächsten Step)
    - reward: Wird nachträglich durch RewardCalculator gesetzt
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EpisodeReplayBuffer:
    """
    HashMap-basierter Replay Buffer.

    Key: Problem-UUID (str)
    Value: Geordnete Liste von Transition-Dicts

    Args:
        keep_completed: Falls True, werden abgeschlossene Episoden
                        in einem separaten Archiv gespeichert (für Analyse).
    """

    def __init__(self, keep_completed: bool = False):
        self._buffer: Dict[str, List[Dict]] = {}
        self._completed: Dict[str, List[Dict]] = {}
        self._keep_completed = keep_completed

    def start_episode(self, uuid: str) -> None:
        """
        Erzeugt eine neue leere Transition-Liste für die UUID.

        Args:
            uuid: Eindeutige Identifikation der Probleminstanz.

        Raises:
            ValueError: Falls eine Episode mit dieser UUID bereits offen ist.
        """
        if uuid in self._buffer:
            logger.warning(
                "Episode für UUID %s existiert bereits (%d Transitions). "
                "Wird überschrieben.",
                uuid, len(self._buffer[uuid])
            )
        self._buffer[uuid] = []

    def add_transition(
        self,
        uuid: str,
        current_state: dict,
        action: dict
    ) -> None:
        """
        Hängt eine neue Transition an die Liste der UUID an.
        next_state ist initial None und wird über set_next_state gesetzt.

        Args:
            uuid: Problem-UUID.
            current_state: Der State VOR der Action (raw dict von Mathematica).
            action: Die gewählte Action {solver, localMaxTolerance}.

        Raises:
            KeyError: Falls keine Episode für diese UUID gestartet wurde.
        """
        if uuid not in self._buffer:
            raise KeyError(
                f"Keine Episode für UUID {uuid} gestartet. "
                f"Rufe start_episode() zuerst auf."
            )
        self._buffer[uuid].append({
            "current_state": current_state,
            "action": action,
            "next_state": None,
            "reward": None
        })

    def set_next_state(self, uuid: str, next_state: dict) -> None:
        """
        Setzt next_state auf der LETZTEN Transition (die noch None hat).

        Args:
            uuid: Problem-UUID.
            next_state: Der State NACH der Action.

        Raises:
            KeyError: Falls keine Episode für diese UUID existiert.
            IndexError: Falls die Episode leer ist.
        """
        if uuid not in self._buffer:
            raise KeyError(f"Keine Episode für UUID {uuid} vorhanden.")
        if not self._buffer[uuid]:
            raise IndexError(f"Episode für UUID {uuid} ist leer.")
        self._buffer[uuid][-1]["next_state"] = next_state

    def get_transitions(self, uuid: str) -> List[Dict]:
        """
        Gibt die geordnete Liste aller Transitions für die UUID zurück.

        Args:
            uuid: Problem-UUID.

        Returns:
            Liste von Transition-Dicts in chronologischer Reihenfolge.

        Raises:
            KeyError: Falls keine Episode für diese UUID existiert.
        """
        if uuid not in self._buffer:
            raise KeyError(f"Keine Episode für UUID {uuid} vorhanden.")
        return self._buffer[uuid]

    def clear_episode(self, uuid: str) -> None:
        """
        Entfernt die Transitions einer UUID nach Reward-Berechnung.
        Falls keep_completed=True, wird die Episode vorher archiviert.

        Args:
            uuid: Problem-UUID.
        """
        if uuid in self._buffer:
            if self._keep_completed:
                self._completed[uuid] = self._buffer[uuid]
            del self._buffer[uuid]

    def get_episode_length(self, uuid: str) -> int:
        """
        Gibt die Anzahl der Transitions für eine UUID zurück.

        Args:
            uuid: Problem-UUID.

        Returns:
            Anzahl der gespeicherten Transitions.
        """
        return len(self._buffer.get(uuid, []))

    def has_episode(self, uuid: str) -> bool:
        """
        Prüft ob eine Episode für die UUID existiert.

        Args:
            uuid: Problem-UUID.

        Returns:
            True falls Episode existiert.
        """
        return uuid in self._buffer

    @property
    def active_episodes(self) -> int:
        """Anzahl der aktuell offenen Episoden."""
        return len(self._buffer)

    @property
    def completed_episodes(self) -> Optional[Dict[str, List[Dict]]]:
        """Archivierte Episoden (nur falls keep_completed=True)."""
        if self._keep_completed:
            return self._completed
        return None
