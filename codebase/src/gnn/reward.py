from __future__ import annotations

import math
from typing import Any, Dict, Optional

"""
Reward-Berechnung für die GNN-RL Pipeline.

Terminal-State (Mathematica), u. a.:
    - BenchmarkabsTime, Benchmarksolver, recordAbsTime, absTime (kumuliert)

Zwischen-States:
    - absTime (kumuliert), ab Schritt 2 i. d. R. timeBenchmarkSolver am current_state

r_base (nur erste Transition):
    Erste Action-Solver vs. Benchmarksolver (nur im Terminal).
    Bei Treffer: basis_reward * (BenchmarkabsTime / max(absTime_terminal, eps))
    (größerer Faktor bei kürzerer Endzeit vs. Benchmark = stärkerer Bonus).
    Bei Fehltreffer: -basis_reward mit demselben Faktor (symmetrisch skaliert).

r_step:
    Log-Ratio aus Delta absTime (next - current) und timeBenchmarkSolver.
    Für die erste Transition: timeBenchmarkSolver vom *next_state* (nach erstem Schritt),
    danach vom current_state (jeweils „nach dem ersten“ Zustand).

r_learn (nur letzte Transition):
    alpha * (recordAbsTime - absTime_terminal) — Rekord vs. finale kumulierte Zeit.
"""


class RewardCalculator:
    """
    Berechnet Episode-Rewards rückwirkend über alle Transitions.

    Args:
        basis_reward: Skalierung für r_base (Solver der ersten Action vs. Benchmarksolver).
        gamma: Zeitgewichtung reward_gamma ** (T - t) für den r_step-Zeitterm.
        alpha: Skalierung r_learn (recordAbsTime vs. finales absTime).
        step_cost_lambda: Wird pro Transition von r_step abgezogen.
        time_bad_penalty: S_raw = -time_bad_penalty bei ungültigen Zeiten.
        solver_mismatch_penalty / solver_match_bonus / solver_wrong_slow_coef:
            Zusatz-Solver-Shaping ab der zweiten Transition (optional).
    """

    # Values >= this threshold are treated as "no record yet" sentinel from
    # Mathematica's setInitStateConfig (1.0e9). Keeps r_learn = 0 on the
    # first epoch instead of paying out a huge artificial bonus.
    RECORD_SENTINEL_THRESHOLD = 1.0e8

    def __init__(
        self,
        basis_reward: float = 1.0,
        gamma: float = 0.99,
        alpha: float = 1.0,
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
        self.step_cost_lambda = step_cost_lambda
        self.time_bad_penalty = time_bad_penalty
        self.solver_mismatch_penalty = solver_mismatch_penalty
        self.solver_match_bonus = solver_match_bonus
        self.solver_wrong_slow_coef = solver_wrong_slow_coef
        self.abs_time_eps = abs_time_eps

    @staticmethod
    def _float_eq(a, b) -> bool:
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _to_float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

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

    def _terminal_time_scale(
        self, reward_state: dict
    ) -> float:
        """
        Faktor für r_base: BenchmarkabsTime / absTime_terminal.
        Kurze Endzeit (gut) -> größerer Faktor -> mehr Basis-Bonus bei richtigem Solver.

        (Entspricht der üblichen Lesart von „größerer Zeitgewinn -> mehr Reward“;
        die reine Ratio absTime/Benchmark wäre bei kürzerer Endzeit kleiner.)
        """
        bench = self._benchmark_abs_time(reward_state)
        terminal_abs = self._to_float(reward_state.get("absTime"), 0.0)
        if bench <= 0 or terminal_abs <= 0:
            return 1.0
        return bench / max(terminal_abs, self.abs_time_eps)

    def _raw_time_score(
        self, delta_time: float, time_benchmark: float
    ) -> float:
        if delta_time > 0 and time_benchmark > 0:
            return math.log(time_benchmark / delta_time)
        return -self.time_bad_penalty

    def _time_benchmark_for_step(
        self,
        i: int,
        current_state: dict,
        next_state: Optional[Dict[str, Any]],
        reward_state: Dict[str, Any],
    ) -> float:
        """
        timeBenchmarkSolver: laut Spezifikation ab Zuständen „nach dem ersten“.
        - Erste Transition (i==0): Wert vom next_state (nach erstem Schritt).
        - Später: Wert vom current_state, Fallback Terminal.
        """
        if i == 0 and next_state is not None:
            raw = next_state.get("timeBenchmarkSolver")
        else:
            raw = current_state.get("timeBenchmarkSolver")
        if raw is None:
            raw = reward_state.get("timeBenchmarkSolver")
        return self._to_float(raw, 0.0)

    def _solver_shaping(
        self, i: int, chosen_solver, benchmark_solver, s_raw: float
    ) -> float:
        if i == 0:
            return 0.0
        if benchmark_solver is None or chosen_solver is None:
            return 0.0
        if self._float_eq(chosen_solver, benchmark_solver):
            return self.solver_match_bonus
        penalty = -self.solver_mismatch_penalty
        if s_raw < 0 and self.solver_wrong_slow_coef > 0:
            penalty -= self.solver_wrong_slow_coef * abs(s_raw)
        return penalty

    def calculate_episode_rewards(self, episode_transitions: list, reward_state: dict):
        if not episode_transitions:
            return

        T = reward_state.get("networkStep", len(episode_transitions))
        benchmark_solver = reward_state.get("Benchmarksolver")

        record_abs_time = self._to_float(reward_state.get("recordAbsTime"), 0.0)
        final_abs_time = self._to_float(reward_state.get("absTime"), 0.0)
        # First epoch: recordAbsTime is the sentinel (>=1e8); don't reward against it.
        if record_abs_time >= self.RECORD_SENTINEL_THRESHOLD:
            r_learn = 0.0
        else:
            r_learn = self.alpha * (record_abs_time - final_abs_time)

        time_scale = self._terminal_time_scale(reward_state)

        for i, transition in enumerate(episode_transitions):
            current_state = transition["current_state"]
            next_state = transition.get("next_state")

            r_base = 0.0
            if i == 0:
                action0 = transition.get("action", {})
                chosen0 = action0.get("solver", current_state.get("solver"))
                if chosen0 is not None and benchmark_solver is not None:
                    if self._float_eq(chosen0, benchmark_solver):
                        r_base = self.basis_reward * time_scale
                    else:
                        r_base = -self.basis_reward * time_scale

            t = current_state.get("networkStep", i + 1)

            time_benchmark = self._time_benchmark_for_step(
                i, current_state, next_state, reward_state
            )

            if next_state:
                delta_time = self._to_float(next_state.get("absTime"), 0.0) - self._to_float(
                    current_state.get("absTime"), 0.0
                )
            else:
                delta_time = final_abs_time - self._to_float(
                    current_state.get("absTime"), 0.0
                )

            s_raw = self._raw_time_score(delta_time, time_benchmark)
            time_weight = self.gamma ** (T - t)
            r_time = s_raw * time_weight

            action = transition.get("action", {})
            chosen_solver = action.get("solver", current_state.get("solver"))
            r_solver = self._solver_shaping(i, chosen_solver, benchmark_solver, s_raw)

            r_step = r_time + r_solver - self.step_cost_lambda

            total_reward = r_base + r_step
            if i == len(episode_transitions) - 1:
                total_reward += r_learn

            transition["reward"] = total_reward
