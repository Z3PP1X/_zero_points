from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_GATEWAY_INTERNAL_KEYS = frozenset({"_gateway_channel"})


class GatewayStateLogger:
    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            log_path = Path(__file__).resolve().parent / "gateway_states.jsonl"
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._file = self.log_path.open("a", encoding="utf-8")

    def log_incoming(self, state: Any, channel: str) -> None:
        self._write("in", channel, state)

    def log_outgoing(self, state: Any) -> None:
        self._write("out", "training", state)

    def close(self) -> None:
        with self._lock:
            if self._file.closed:
                return
            self._file.close()

    def _write(self, direction: str, channel: str, state: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "channel": channel,
            "state": self._serialize_state(state),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    @staticmethod
    def _serialize_state(state: Any) -> Any:
        if not isinstance(state, dict):
            return state
        return {
            key: value
            for key, value in state.items()
            if key not in _GATEWAY_INTERNAL_KEYS
        }
