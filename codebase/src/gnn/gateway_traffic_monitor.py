from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from state_wait_timeout import StateRoundtripTimeout


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GatewayTrafficMonitor:
    def __init__(
        self,
        *,
        refresh_s: float = 0.5,
        timeout_fallback_s: float = 5.0,
        timeout_cushion_s: float = 2.0,
        timeout_window_size: int = 100,
    ):
        self.refresh_s = refresh_s
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_episode: Optional[str] = None
        self._episode_reward_total = 0.0
        self._reward_state_count = 0
        self._total_timeouts = 0
        self.roundtrip_timeout = StateRoundtripTimeout(
            window_size=timeout_window_size,
            cushion_s=timeout_cushion_s,
            fallback_s=timeout_fallback_s,
        )

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

    def record_roundtrip(self, duration_s: float) -> None:
        with self._lock:
            self.roundtrip_timeout.record(duration_s)

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
                roundtrip_avg = self.roundtrip_timeout.mean_roundtrip_s()
                roundtrip_samples = self.roundtrip_timeout.sample_count
                roundtrip_timeout = self.roundtrip_timeout.timeout_s()
            if roundtrip_avg is None:
                roundtrip_text = f"fallback {roundtrip_timeout:.2f}s"
            else:
                roundtrip_text = (
                    f"{roundtrip_avg:.3f}s (n={roundtrip_samples}) "
                    f"-> timeout {roundtrip_timeout:.2f}s"
                )
            line = (
                f"\r[Gateway] Episode: {episode} | "
                f"Episode-Reward: {episode_reward:>10.4f} | "
                f"Reward-States: {reward_states:>6} | "
                f"Roundtrip avg: {roundtrip_text} | "
                f"Total Timeouts: {total_timeouts:>6}"
            )
            sys.stderr.write(line)
            sys.stderr.flush()
            time.sleep(self.refresh_s)
