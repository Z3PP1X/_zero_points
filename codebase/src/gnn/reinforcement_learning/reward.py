from __future__ import annotations

import math
from typing import Any, Dict, Optional


class RewardCalculator:

    RECORD_SENTINEL_THRESHOLD = 1.0e8

    def __init__(
        self,
        basis_reward: float = 1.0,
        gamma: float = 0.99,
        alpha: float = 1.0,
        time_tolerance: float = 0.03,
        step_cost_lambda: float = 0.01,
        time_bad_penalty: float = 1.0,
        solver_mismatch_penalty: float = 0.5,
        solver_match_bonus: float = 0.05,
        solver_wrong_slow_coef: float = 0.5,
        abs_time_eps: float = 1e-9,
    ):
        self.basis_reward = basis_reward
        self.gamma = gamma
        self.alpha = alpha
        self.time_tolerance = time_tolerance
        self.step_cost_lambda = step_cost_lambda
        self.time_bad_penalty = time_bad_penalty
        self.solver_mismatch_penalty = solver_mismatch_penalty
        self.solver_match_bonus = solver_match_bonus
        self.solver_wrong_slow_coef = solver_wrong_slow_coef
        self.abs_time_eps = abs_time_eps

    @staticmethod
    def _to_float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _raw_time_score(
        self, delta_time: float, time_benchmark: float
    ) -> float:
        if delta_time > 0 and time_benchmark > 0:
            raw = math.log(time_benchmark / delta_time)
            return max(min(raw, 2.0), -2.0)
        return -self.time_bad_penalty

    def _time_benchmark_for_step(
        self,
        i: int,
        current_state: dict,
        next_state: Optional[Dict[str, Any]],
        reward_state: Dict[str, Any],
    ) -> float:
        if i == 0 and next_state is not None:
            raw = next_state.get("timeBenchmarkSolver")
        else:
            raw = current_state.get("timeBenchmarkSolver")
        if raw is None:
            raw = reward_state.get("timeBenchmarkSolver")
        return self._to_float(raw, 0.0)

    def calculate_episode_rewards(self, episode_transitions: list, reward_state: dict):
        if not episode_transitions:
            return

        T = reward_state.get("networkStep", len(episode_transitions))

        record_abs_time = self._to_float(reward_state.get("recordAbsTime"), 0.0)
        final_abs_time = self._to_float(reward_state.get("absTime"), 0.0)
        if record_abs_time >= self.RECORD_SENTINEL_THRESHOLD or record_abs_time <= 0:
            r_learn = 0.0
        else:
            relative_diff = (final_abs_time - record_abs_time) / record_abs_time
            if relative_diff <= 0:
                r_learn = self.alpha * (record_abs_time - final_abs_time)
            elif relative_diff <= self.time_tolerance:
                r_learn = 0.5 * self.alpha * self.time_tolerance * record_abs_time
            else:
                r_learn = 0.0

        for i, transition in enumerate(episode_transitions):
            current_state = transition["current_state"]
            next_state = transition.get("next_state")

            t = current_state.get("networkStep", i + 1)

            time_benchmark = self._time_benchmark_for_step(
                i, current_state, next_state, reward_state
            )

            if next_state:
                delta_time = (self._to_float(next_state.get("absTime"), 0.0) -
                              self._to_float(
                    current_state.get("absTime"), 0.0
                ))
            else:
                delta_time = final_abs_time - self._to_float(
                    current_state.get("absTime"), 0.0
                )

            s_raw = self._raw_time_score(delta_time, time_benchmark)
            time_weight = self.gamma ** (T - t)
            r_time = s_raw * time_weight

            r_step = r_time - self.step_cost_lambda

            total_reward = r_step
            if i == len(episode_transitions) - 1:
                total_reward += r_learn

            transition["reward"] = total_reward
