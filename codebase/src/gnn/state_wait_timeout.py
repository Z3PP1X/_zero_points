from __future__ import annotations

from collections import deque


class StateRoundtripTimeout:
    def __init__(
        self,
        *,
        window_size: int = 100,
        cushion_s: float = 2.0,
        fallback_s: float = 5.0,
    ):
        self._window = deque(maxlen=window_size)
        self._cushion_s = cushion_s
        self._fallback_s = fallback_s

    def record(self, duration_s: float) -> None:
        if duration_s < 0:
            return
        self._window.append(duration_s)

    @property
    def sample_count(self) -> int:
        return len(self._window)

    def mean_roundtrip_s(self) -> float | None:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)

    def timeout_s(self) -> float:
        if not self._window:
            return self._fallback_s
        return (sum(self._window) / len(self._window)) + self._cushion_s
