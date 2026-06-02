import math
from reward import RewardCalculator


def test_raw_time_score_clipping():
    calc = RewardCalculator(time_bad_penalty=1.5)

    # Normal log ratio
    # log(10 / 2) = log(5) = 1.6094... (within [-2, 2])
    score = calc._raw_time_score(delta_time=2.0, time_benchmark=10.0)
    assert math.isclose(score, math.log(5.0))

    # Upper clipping: log(10 / 0.5) = log(20) = 2.9957... -> clips to 2.0
    score_upper = calc._raw_time_score(delta_time=0.5, time_benchmark=10.0)
    assert score_upper == 2.0

    # Lower clipping: log(10 / 100) = log(0.1) = -2.3025... -> clips to -2.0
    score_lower = calc._raw_time_score(delta_time=100.0, time_benchmark=10.0)
    assert score_lower == -2.0

    # Bad penalty cases
    assert calc._raw_time_score(delta_time=-1.0, time_benchmark=10.0) == -1.5
    assert calc._raw_time_score(delta_time=2.0, time_benchmark=-5.0) == -1.5


def test_r_learn_asymmetry_and_tolerance():
    # Setup with alpha = 2.0, time_tolerance = 0.05 (5%)
    calc = RewardCalculator(alpha=2.0, time_tolerance=0.05)

    # 1. Sentinel check
    transitions = [{"current_state": {"absTime": 0.0, "networkStep": 1}}]
    reward_state = {
        "recordAbsTime": calc.RECORD_SENTINEL_THRESHOLD + 100,
        "absTime": 8.0,
        "networkStep": 1,
    }
    calc.calculate_episode_rewards(transitions, reward_state)
    # Since record is sentinel, r_learn is 0.0. Total reward = r_step + r_learn = r_step
    # With delta_time = 8.0, time_benchmark = 0.0 (returns -time_bad_penalty = -1.0)
    # r_time = -1.0, r_step = -1.0 - 0.01 = -1.01
    assert math.isclose(transitions[-1]["reward"], -1.01)

    # 2. True new record: final_abs_time <= record_abs_time
    # record = 10.0, final = 9.0
    # expected r_learn = alpha * (record - final) = 2.0 * (10.0 - 9.0) = 2.0
    reward_state = {
        "recordAbsTime": 10.0,
        "absTime": 9.0,
        "networkStep": 1,
    }
    transitions = [{"current_state": {"absTime": 0.0, "networkStep": 1}}]
    calc.calculate_episode_rewards(transitions, reward_state)
    # s_raw = math.log(0.0/0.0) -> -1.0, r_time = -1.0, r_step = -1.01
    # total reward = -1.01 + 2.0 = 0.99
    assert math.isclose(transitions[-1]["reward"], 0.99)

    # 3. Within tolerance: 0 < relative_diff <= time_tolerance
    # record = 10.0, final = 10.3 (relative diff = 0.03 <= 0.05)
    # expected constant preservation bonus: 0.5 * alpha * tolerance * record = 0.5 * 2.0 * 0.05 * 10.0 = 0.5
    reward_state = {
        "recordAbsTime": 10.0,
        "absTime": 10.3,
        "networkStep": 1,
    }
    transitions = [{"current_state": {"absTime": 0.0, "networkStep": 1}}]
    calc.calculate_episode_rewards(transitions, reward_state)
    # total reward = -1.01 + 0.5 = -0.51
    assert math.isclose(transitions[-1]["reward"], -0.51)

    # 4. Outside tolerance: relative_diff > time_tolerance
    # record = 10.0, final = 11.0 (relative diff = 0.10 > 0.05)
    # expected r_learn = 0.0, no penalty
    reward_state = {
        "recordAbsTime": 10.0,
        "absTime": 11.0,
        "networkStep": 1,
    }
    transitions = [{"current_state": {"absTime": 0.0, "networkStep": 1}}]
    calc.calculate_episode_rewards(transitions, reward_state)
    # total reward = -1.01 + 0.0 = -1.01
    assert math.isclose(transitions[-1]["reward"], -1.01)


def test_no_r_base_and_no_solver_shaping():
    # Setup RewardCalculator, where basis_reward and solver_mismatch_penalty are configured
    # but should be completely ignored in V2
    calc = RewardCalculator(
        basis_reward=5.0,
        solver_mismatch_penalty=1.0,
        solver_match_bonus=2.0,
        step_cost_lambda=0.05,
    )

    transitions = [
        # First step (i = 0): previously rewarded basis_reward if matching solver
        {
            "current_state": {
                "absTime": 0.0,
                "networkStep": 1,
                "solver": 1.0,
                "timeBenchmarkSolver": 10.0,
            },
            "next_state": {
                "absTime": 2.0,
                "solver": 1.0,
                "timeBenchmarkSolver": 10.0,
            },
            "action": {"solver": 1.0},
        },
        # Second step (i = 1)
        {
            "current_state": {
                "absTime": 2.0,
                "networkStep": 2,
                "solver": 2.0,
                "timeBenchmarkSolver": 10.0,
            },
            "next_state": {
                "absTime": 5.0,
                "solver": 2.0,
                "timeBenchmarkSolver": 10.0,
            },
            "action": {"solver": 2.0},
        },
    ]

    reward_state = {
        "Benchmarksolver": 1.0,
        "recordAbsTime": calc.RECORD_SENTINEL_THRESHOLD,  # r_learn is 0.0
        "absTime": 5.0,
        "networkStep": 2,
    }

    calc.calculate_episode_rewards(transitions, reward_state)

    # 1. Check transition 0:
    # delta_time = 2.0 - 0.0 = 2.0
    # time_benchmark = timeBenchmarkSolver from next_state for i == 0 -> 10.0
    # s_raw = math.log(10.0 / 2.0) = math.log(5.0) = 1.6094379...
    # gamma = 0.99, T = 2, t = 1 -> time_weight = 0.99 ** (2 - 1) = 0.99
    # r_time = s_raw * 0.99 = 1.5933435...
    # r_step = r_time - 0.05 = 1.5433435...
    # In V1, we would have added r_base = 5.0 * time_scale. In V2, r_base is 0.0.
    # So reward should be exactly r_step
    expected_r0 = math.log(5.0) * 0.99 - 0.05
    assert math.isclose(transitions[0]["reward"], expected_r0)

    # 2. Check transition 1:
    # delta_time = 5.0 - 2.0 = 3.0
    # time_benchmark = timeBenchmarkSolver from current_state for i > 0 -> 10.0
    # s_raw = math.log(10.0 / 3.0) = 1.2039728...
    # gamma = 0.99, T = 2, t = 2 -> time_weight = 0.99 ** 0 = 1.0
    # r_time = s_raw * 1.0 = 1.2039728...
    # r_step = r_time - 0.05 = 1.1539728...
    # In V1, chosen_solver (2.0) != benchmark_solver (1.0) would trigger mismatch penalty.
    # In V2, solver shaping is deleted.
    # So reward should be exactly r_step (plus r_learn = 0.0)
    expected_r1 = math.log(10.0 / 3.0) - 0.05
    assert math.isclose(transitions[1]["reward"], expected_r1)
