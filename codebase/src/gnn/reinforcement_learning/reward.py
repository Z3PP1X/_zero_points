from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Potential-based reward shaping (PBRS) for the RL solver controller.

    Per transition ``t`` (current state ``s_t`` -> next state ``s_{t+1}``)::

        r_total(t) = F_shaping(t) + r_time(t) + (r_record + r_over   if last step)

        Phi(s)       = -log10( clip(lastStepError, tolerance, error_ref) / tolerance )
        F_shaping(t) = lambda_s * ( gamma * Phi(s_{t+1}) * (1 - is_last) - Phi(s_t) )
        r_time(t)    = - w_time   * (delta_absTime / timeBudget)
        r_record     = + w_record * (recordAbsTime - absTime) / recordAbsTime  (new record only)
        r_over       = - w_over   * max(0, (absTime - timeBudget) / timeBudget)

    * ``Phi`` is goal-anchored (0 at convergence, ~ -15 far away) and log-scaled, so
      every order-of-magnitude gained toward ``tolerance`` is worth ~1 unit of shaping.
    * ``F_shaping`` is the PBRS term of Ng, Harada & Russell (1999): ``gamma`` multiplies
      ``Phi(s_{t+1})`` and **must equal the PPO discount** for policy invariance. The
      ``(1 - is_last)`` mask forces ``Phi(terminal) = 0`` (Grzes 2017) so the shaping
      cannot change the optimal policy; its telescoping sum is path-independent, so the
      agent cannot farm it by stalling/oscillating.
    * The genuine objective lives in ``r_time`` (per-step wall-clock cost), ``r_record``
      (beat the best-known time -- "push the limits") and ``r_over`` (budget hinge).
      Episodes are assumed to always converge, so there is no terminal success/failure
      term (it would be policy-invariant and contribute nothing).

    Design rationale, decisions and citations:
    ``supervised_learning/handover_20260627_120213/reward-redesign-plan.md``.
    """

    # recordAbsTime sentinel: ">= this" (or <= 0) means "no valid record yet" -> no bonus.
    RECORD_SENTINEL_THRESHOLD = 1.0e8

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_s: float = 0.1,
        w_time: float = 1.0,
        w_record: float = 2.0,
        w_over: float = 2.0,
        error_ref: float = 1.0,
    ):
        # gamma is the PBRS discount; tie it to the PPO discount at the call site.
        self.gamma = gamma
        self.lambda_s = lambda_s      # PBRS convergence-shaping scale
        self.w_time = w_time          # per-step wall-clock cost weight
        self.w_record = w_record      # record-beating bonus weight
        self.w_over = w_over          # budget-overrun penalty weight
        self.error_ref = error_ref    # upper clamp on lastStepError for Phi (structural const)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _to_float(val, default: float = 0.0) -> float:
        """Soft numeric read with a default (used for guarded terminal scalars)."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _require_float(state: Dict[str, Any], key: str) -> float:
        """Read a load-bearing numeric state key or fail loud.

        The Mathematica gateway historically returned missing keys as a silent
        ``0.0`` (``dict.get``), which corrupted the reward without ever raising.
        PBRS depends on ``lastStepError`` / ``tolerance`` / ``absTime`` being real
        numbers, so a missing/non-numeric one is a hard error here -- never a zero.
        """
        if key not in state or state[key] is None:
            raise ValueError(
                f"RewardCalculator: required state key '{key}' is missing/None "
                f"(state keys present: {sorted(state)})"
            )
        try:
            return float(state[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RewardCalculator: state key '{key}'={state[key]!r} is not numeric"
            ) from exc

    def _time_budget(self, reward_state: Dict[str, Any]) -> float:
        """Per-episode total time budget; warn + fall back if absent/non-positive.

        ``timeBudget`` is a per-episode total (the time the solver is allowed to
        spend). If the producer has not populated it yet, fall back to the record
        time (or a neutral 1.0) so time normalization never divides by zero -- but
        warn loudly rather than silently distorting the cost signal.
        """
        raw = reward_state.get("timeBudget")
        budget = self._to_float(raw, 0.0)
        if budget > 0:
            return budget
        record = self._to_float(reward_state.get("recordAbsTime"), 0.0)
        fallback = record if 0 < record < self.RECORD_SENTINEL_THRESHOLD else 1.0
        logger.warning(
            "RewardCalculator: 'timeBudget' missing/non-positive (%r); falling back "
            "to %.6g for time normalization.", raw, fallback,
        )
        return fallback

    def _potential(self, state: Dict[str, Any]) -> float:
        """Phi(s) = -log10( clip(lastStepError, tolerance, error_ref) / tolerance ).

        0 at/under convergence (lastStepError <= tolerance), most negative far from
        the goal. The clamp keeps Phi finite and kills log(0)/overshoot.
        """
        err = abs(self._require_float(state, "lastStepError"))
        tol = self._require_float(state, "tolerance")
        if tol <= 0:
            raise ValueError(f"RewardCalculator: tolerance must be > 0, got {tol!r}")
        upper = max(self.error_ref, tol)          # keep the clamp range non-degenerate
        err_clamped = min(max(err, tol), upper)
        return -math.log10(err_clamped / tol)

    # ----------------------------------------------------------------- main API
    def calculate_episode_rewards(
        self,
        episode_transitions: List[Dict[str, Any]],
        reward_state: Dict[str, Any],
    ) -> None:
        """Assign ``transition['reward']`` in place for one (always-converging) episode."""
        if not episode_transitions:
            return

        n = len(episode_transitions)
        final_abs_time = self._to_float(reward_state.get("absTime"), 0.0)
        record_abs_time = self._to_float(reward_state.get("recordAbsTime"), 0.0)
        time_budget = self._time_budget(reward_state)  # guaranteed > 0

        # --- terminal record bonus: reward beating the best-known time -------------
        if (
            0 < record_abs_time < self.RECORD_SENTINEL_THRESHOLD
            and final_abs_time < record_abs_time
        ):
            r_record = self.w_record * (record_abs_time - final_abs_time) / record_abs_time
        else:
            r_record = 0.0

        # --- terminal budget hinge: punish only when the solve ran over budget -----
        if final_abs_time > time_budget:
            r_over = -self.w_over * (final_abs_time - time_budget) / time_budget
        else:
            r_over = 0.0

        for i, transition in enumerate(episode_transitions):
            current_state = transition["current_state"]
            next_state: Optional[Dict[str, Any]] = transition.get("next_state")
            is_last = i == n - 1

            # PBRS convergence shaping; Phi(terminal) = 0 on the last/truncated step.
            phi_current = self._potential(current_state)
            if is_last or not next_state:
                phi_next = 0.0
            else:
                phi_next = self.gamma * self._potential(next_state)
            r_shaping = self.lambda_s * (phi_next - phi_current)

            # Per-step wall-clock cost = absTime delta of the last decision.
            cur_abs = self._require_float(current_state, "absTime")
            nxt_abs = (
                self._require_float(next_state, "absTime")
                if next_state else final_abs_time
            )
            delta_time = max(0.0, nxt_abs - cur_abs)
            r_time = -self.w_time * (delta_time / time_budget)

            total_reward = r_shaping + r_time
            if is_last:
                total_reward += r_record + r_over

            transition["reward"] = total_reward


# ===========================================================================
# LEGACY (V2 "tolerance" reward) -- obsolete, kept commented for reference only.
# Replaced by the PBRS reward above (see reward-redesign-plan.md). It scored each
# step by a time-benchmark log-ratio and is NOT used in future experiments:
#
#   def _raw_time_score(self, delta_time, time_benchmark):
#       if delta_time > 0 and time_benchmark > 0:
#           raw = math.log(time_benchmark / delta_time)
#           return max(min(raw, 2.0), -2.0)
#       return -self.time_bad_penalty
#
#   # per step: s_raw = _raw_time_score(delta_time, timeBenchmarkSolver)
#   #           r_time = s_raw * gamma ** (T - t)          # manual extra discount
#   #           r_step = r_time - step_cost_lambda
#   # terminal: r_learn from recordAbsTime vs absTime (asymmetric near-miss band)
#
# Why retired:
#   * `timeBenchmarkSolver` was renamed to `timeBudget`, which is a CAP not a
#     measured reference, so log(benchmark / delta_time) was meaningless against it.
#   * the manual gamma**(T - t) double-discounted on top of PPO/GAE.
#   * the solver-shaping knobs (basis_reward, solver_mismatch_penalty, ...) were
#     dead (known_bugs.md RL-1).
# ===========================================================================
