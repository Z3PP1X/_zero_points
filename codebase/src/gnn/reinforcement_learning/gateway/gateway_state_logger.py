from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_GATEWAY_INTERNAL_KEYS = frozenset({"_gateway_channel"})


class GatewayStateLogger:
    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            log_path = Path(__file__).resolve().parent / "gateway_states.db"
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self.is_sqlite = self.log_path.suffix == ".db"

        if self.is_sqlite:
            self._conn = sqlite3.connect(str(self.log_path), check_same_thread=False)
            self._create_table()
        else:
            self._file = self.log_path.open("a", encoding="utf-8")

    def _create_table(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS gateway_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    state TEXT NOT NULL
                )
            """)
            self._conn.commit()

    def log_incoming(self, state: Any, channel: str) -> None:
        self._write("in", channel, state)

    def log_outgoing(self, state: Any) -> None:
        self._write("out", "training", state)

    def close(self) -> None:
        with self._lock:
            if self.is_sqlite:
                if hasattr(self, "_conn") and self._conn is not None:
                    self._conn.close()
                    self._conn = None
            else:
                if hasattr(self, "_file") and not self._file.closed:
                    self._file.close()

        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.log_artifact(str(self.log_path), artifact_path="states")
                print(f"[GatewayStateLogger] Successfully logged {self.log_path.name} to MLflow run.")
        except Exception as e:
            print(f"[GatewayStateLogger] Warning: Failed to log state database/file to MLflow: {e}")

    def _write(self, direction: str, channel: str, state: Any) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        serialized_state = self._serialize_state(state)

        if self.is_sqlite:
            state_str = json.dumps(serialized_state, ensure_ascii=False)
            with self._lock:
                if self._conn is not None:
                    self._conn.execute(
                        "INSERT INTO gateway_states (timestamp, direction, channel, state) VALUES (?, ?, ?, ?)",
                        (timestamp, direction, channel, state_str),
                    )
                    self._conn.commit()
        else:
            record = {
                "timestamp": timestamp,
                "direction": direction,
                "channel": channel,
                "state": serialized_state,
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

