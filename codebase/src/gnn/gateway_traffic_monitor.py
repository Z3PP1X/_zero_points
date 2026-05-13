from __future__ import annotations

import sys
import threading
import time
from typing import Optional


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GatewayTrafficMonitor:
    def __init__(self, *, refresh_s: float = 0.5):
        self.refresh_s = refresh_s
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_episode: Optional[str] = None
        self._episode_reward_total = 0.0
        self._reward_state_count = 0
        self._total_timeouts = 0

    def observe(self, message: dict, channel: str) -> None:
        with self._lock:
            if channel == "reward":
                self._reward_state_count += 1
                return

            if channel != "training":
                return

            episode = message.get("episode")
            if episode is None:
                return

            episode_key = str(episode)
            if episode_key != self._current_episode:
                self._current_episode = episode_key
                self._episode_reward_total = 0.0

            reward = _to_float(message.get("reward"))
            if reward is not None:
                self._episode_reward_total += reward

    def record_timeout(self) -> None:
        with self._lock:
            self._total_timeouts += 1

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop,
            name="gateway-traffic-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render_loop(self) -> None:
        while self._running:
            with self._lock:
                episode = self._current_episode or "—"
                episode_reward = self._episode_reward_total
                reward_states = self._reward_state_count
                total_timeouts = self._total_timeouts
            line = (
                f"\r[Gateway] Episode: {episode} | "
                f"Episode-Reward: {episode_reward:>10.4f} | "
                f"Reward-States: {reward_states:>6} | "
                f"Total Timeouts: {total_timeouts:>6}"
            )
            sys.stderr.write(line)
            sys.stderr.flush()
            time.sleep(self.refresh_s)
