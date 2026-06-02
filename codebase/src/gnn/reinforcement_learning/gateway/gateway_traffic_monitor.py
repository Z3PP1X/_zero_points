from __future__ import annotations

import sys
import threading
import time
from typing import Optional
from collections import deque
import numpy as np

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
        timeout_cushion_s: float = 0.5,
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
        self._faster_than_benchmark_history = deque(maxlen=1000)
        self._overshoot_history = deque(maxlen=1000)
        self._convergence_history = deque(maxlen=1000)
        self._steps_history = deque(maxlen=1000)

    def reset_reward_state_count(self) -> None:
        with self._lock:
            self._reward_state_count = 0
            self._faster_than_benchmark_history.clear()
            self._overshoot_history.clear()
            self._convergence_history.clear()
            self._steps_history.clear()

    @staticmethod
    def _benchmark_abs_time(reward_state: dict) -> float:
        for key in (
            "BenchmarkabsTime",
            "BenchmarkAbsTime",
            "benchmarkAbsTime",
        ):
            if key in reward_state and reward_state[key] is not None:
                try:
                    return float(reward_state[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    def get_rolling_metrics(self) -> tuple[Optional[float], Optional[float]]:
        with self._lock:
            if not self._faster_than_benchmark_history:
                return None, None
            faster_avg = float(np.mean(self._faster_than_benchmark_history))
            overshoot_var = float(np.var(self._overshoot_history)) if len(self._overshoot_history) >= 1 else 0.0
            return faster_avg, overshoot_var

    def get_advanced_rolling_metrics(self) -> dict:
        with self._lock:
            conv_rate = float(np.mean(self._convergence_history)) if self._convergence_history else None
            mean_steps = float(np.mean(self._steps_history)) if self._steps_history else None
            mean_roundtrip = self.roundtrip_timeout.mean_roundtrip_s()
            return {
                "convergence_rate": conv_rate,
                "mean_episode_steps": mean_steps,
                "mean_roundtrip_s": mean_roundtrip,
            }

    def observe(self, message: dict, channel: str) -> None:
        with self._lock:
            if channel == "reward":
                self._reward_state_count += 1
                bench = self._benchmark_abs_time(message)
                terminal_abs = _to_float(message.get("absTime"))
                if bench is not None and terminal_abs is not None and bench > 0 and terminal_abs > 0:
                    is_faster = 1.0 if terminal_abs < bench else 0.0
                    overshoot = terminal_abs - bench
                    self._faster_than_benchmark_history.append(is_faster)
                    self._overshoot_history.append(overshoot)
                
                status = message.get("status")
                if status is not None:
                    is_converged = 1.0 if status in ("finished", "reward_calc") else 0.0
                    self._convergence_history.append(is_converged)
                
                steps = message.get("networkStep")
                if steps is not None:
                    steps_val = _to_float(steps)
                    if steps_val is not None:
                        self._steps_history.append(steps_val)
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
                if self._faster_than_benchmark_history:
                    faster_ratio = float(np.mean(self._faster_than_benchmark_history))
                    overshoot_var = float(np.var(self._overshoot_history)) if len(self._overshoot_history) >= 1 else 0.0
                else:
                    faster_ratio = None
                    overshoot_var = None

                conv_rate = float(np.mean(self._convergence_history)) if self._convergence_history else None
                mean_steps = float(np.mean(self._steps_history)) if self._steps_history else None

            if roundtrip_avg is None:
                roundtrip_text = f"fallback {roundtrip_timeout:.2f}s"
            else:
                roundtrip_text = (
                    f"{roundtrip_avg:.3f}s (n={roundtrip_samples}) "
                    f"-> timeout {roundtrip_timeout:.2f}s"
                )
            faster_text = f"{faster_ratio:.3f}" if faster_ratio is not None else "—"
            overshoot_text = f"{overshoot_var:.3f}" if overshoot_var is not None else "—"
            conv_text = f"{conv_rate:.3f}" if conv_rate is not None else "—"
            steps_text = f"{mean_steps:.1f}" if mean_steps is not None else "—"
            line = (
                f"\r[Gateway] Episode: {episode} | "
                f"Episode-Reward: {episode_reward:>10.4f} | "
                f"Reward-States: {reward_states:>6} | "
                f"Faster Ratio: {faster_text} | "
                f"Overshoot Var: {overshoot_text} | "
                f"Conv Rate: {conv_text} | "
                f"Mean Steps: {steps_text} | "
                f"Roundtrip avg: {roundtrip_text} | "
                f"Total Timeouts: {total_timeouts:>6}"
            )
            sys.stderr.write(line)
            sys.stderr.flush()
            time.sleep(self.refresh_s)
