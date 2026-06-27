import math

import pytest
from reward import RewardCalculator

SENTINEL = RewardCalculator.RECORD_SENTINEL_THRESHOLD


def _state(err, tol=1e-15, abs_time=0.0, **extra):
    """Minimal solver state carrying the keys the PBRS reward needs."""
    s = {"lastStepError": err, "tolerance": tol, "absTime": abs_time}
    s.update(extra)
    return s


def _reward_state(abs_time, time_budget=100.0, record=SENTINEL):
    return {"absTime": abs_time, "timeBudget": time_budget, "recordAbsTime": record}


# --------------------------------------------------------------------- potential
def test_potential_goal_anchored_and_clamped():
    calc = RewardCalculator(error_ref=1.0)
    # At/under tolerance -> Phi = 0 (clamped); never positive.
    assert calc._potential(_state(1e-15)) == 0.0
    assert calc._potential(_state(1e-20)) == 0.0
    # error == error_ref (1.0), tol = 1e-15 -> Phi = -log10(1e15) = -15.
    assert math.isclose(calc._potential(_state(1.0)), -15.0)
    # Above error_ref clamps to the same floor.
    assert math.isclose(calc._potential(_state(1e3)), -15.0)
    # One decade from goal: error 1e-14 -> Phi = -log10(1e-14/1e-15) = -1.
    assert math.isclose(calc._potential(_state(1e-14)), -1.0)


# --------------------------------------------------- shaping sign + terminal mask
def test_shaping_progress_positive_and_terminal_masked():
    # Isolate the shaping term: no time cost, no terminal bonuses, gamma = 1.
    calc = RewardCalculator(gamma=1.0, lambda_s=1.0, w_time=0.0, w_record=0.0, w_over=0.0)
    s0 = _state(1e-5)   # Phi = -10
    s1 = _state(1e-7)   # Phi = -8
    s2 = _state(1e-15)  # Phi = 0 (converged)
    transitions = [
        {"current_state": s0, "next_state": s1},
        {"current_state": s1, "next_state": s2},
    ]
    calc.calculate_episode_rewards(transitions, _reward_state(abs_time=0.0))
    # Non-last step: F = gamma*Phi(s1) - Phi(s0) = -8 - (-10) = +2 (progress rewarded).
    assert math.isclose(transitions[0]["reward"], 2.0)
    # Last step: Phi(terminal) masked to 0 -> F = 0 - Phi(s1) = +8.
    assert math.isclose(transitions[1]["reward"], 8.0)


def test_shaping_telescopes_to_constant_path_independent():
    # At gamma = 1 the per-step shaping sums to -lambda_s*Phi(s0) regardless of the
    # intermediate path -> stalling/oscillating cannot farm reward.
    calc = RewardCalculator(gamma=1.0, lambda_s=0.3, w_time=0.0, w_record=0.0, w_over=0.0)

    def episode_total(midpoints):
        errors = [1e-3] + midpoints + [1e-15]
        transitions = [
            {"current_state": _state(a), "next_state": _state(b)}
            for a, b in zip(errors[:-1], errors[1:])
        ]
        calc.calculate_episode_rewards(transitions, _reward_state(abs_time=0.0))
        return sum(t["reward"] for t in transitions)

    expected = -0.3 * calc._potential(_state(1e-3))  # = -0.3 * (-12) = 3.6
    assert math.isclose(episode_total([1e-6]), expected)                 # monotone
    assert math.isclose(episode_total([1e-1, 1e-9, 1e-6]), expected)     # non-monotone


# ------------------------------------------------------------- per-step time cost
def test_per_step_time_cost_uses_abs_time_delta_over_budget():
    calc = RewardCalculator(lambda_s=0.0, w_time=2.0, w_record=0.0, w_over=0.0)
    transitions = [
        {"current_state": _state(1e-3, abs_time=0.0), "next_state": _state(1e-6, abs_time=1.0)},
        {"current_state": _state(1e-6, abs_time=1.0), "next_state": _state(1e-9, abs_time=4.0)},
    ]
    calc.calculate_episode_rewards(transitions, _reward_state(abs_time=4.0, time_budget=8.0))
    assert math.isclose(transitions[0]["reward"], -2.0 * (1.0 / 8.0))  # delta 1.0
    assert math.isclose(transitions[1]["reward"], -2.0 * (3.0 / 8.0))  # delta 3.0


def test_time_budget_fallback_to_record_when_missing():
    calc = RewardCalculator(lambda_s=0.0, w_time=1.0, w_record=0.0, w_over=0.0)
    transitions = [
        {"current_state": _state(1e-3, abs_time=0.0), "next_state": _state(1e-15, abs_time=2.0)}
    ]
    # No timeBudget -> falls back to recordAbsTime (5.0) as the denominator.
    reward_state = {"absTime": 2.0, "recordAbsTime": 5.0}
    calc.calculate_episode_rewards(transitions, reward_state)
    assert math.isclose(transitions[0]["reward"], -1.0 * (2.0 / 5.0))


# ----------------------------------------------------------- terminal bonus/hinge
def test_record_bonus_only_when_faster_than_a_valid_record():
    calc = RewardCalculator(lambda_s=0.0, w_time=0.0, w_record=4.0, w_over=0.0)

    def final_reward(final_abs, record):
        tr = [{"current_state": _state(1e-3, abs_time=0.0),
               "next_state": _state(1e-15, abs_time=final_abs)}]
        calc.calculate_episode_rewards(tr, _reward_state(abs_time=final_abs, record=record))
        return tr[0]["reward"]

    # New record (9 < 10): bonus = 4 * (10 - 9) / 10 = 0.4.
    assert math.isclose(final_reward(9.0, 10.0), 0.4)
    # Slower than record: no bonus.
    assert final_reward(11.0, 10.0) == 0.0
    # No valid record yet (sentinel): no bonus.
    assert final_reward(1.0, SENTINEL) == 0.0


def test_budget_overrun_hinge():
    calc = RewardCalculator(lambda_s=0.0, w_time=0.0, w_record=0.0, w_over=3.0)

    def final_reward(final_abs, budget):
        tr = [{"current_state": _state(1e-3, abs_time=0.0),
               "next_state": _state(1e-15, abs_time=final_abs)}]
        calc.calculate_episode_rewards(tr, _reward_state(abs_time=final_abs, time_budget=budget))
        return tr[0]["reward"]

    # Over budget (12 > 10): penalty = -3 * (12 - 10) / 10 = -0.6.
    assert math.isclose(final_reward(12.0, 10.0), -0.6)
    # Within budget: no penalty.
    assert final_reward(8.0, 10.0) == 0.0


# --------------------------------------------------------------------- fail loud
def test_missing_state_key_raises_instead_of_silently_zeroing():
    calc = RewardCalculator()
    transitions = [
        {"current_state": {"tolerance": 1e-15, "absTime": 0.0}, "next_state": None}
    ]  # lastStepError absent
    with pytest.raises(ValueError, match="lastStepError"):
        calc.calculate_episode_rewards(transitions, _reward_state(abs_time=1.0))


def test_empty_episode_is_a_noop():
    RewardCalculator().calculate_episode_rewards([], _reward_state(abs_time=0.0))
