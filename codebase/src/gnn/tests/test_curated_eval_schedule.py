import pytest

from gnn.supervised_learning.curated_eval_schedule import (
    CuratedEvalSchedule,
    parse_curated_eval_schedule,
    should_evaluate_curated,
)


def test_parse_curated_eval_schedule_defaults():
    schedule = parse_curated_eval_schedule({})
    assert schedule.period == 5
    assert schedule.on_test_highscore is True


def test_parse_curated_eval_schedule_from_yaml_mapping():
    schedule = parse_curated_eval_schedule(
        {"curated_eval_period": 10, "curated_eval_on_test_highscore": False}
    )
    assert schedule.period == 10
    assert schedule.on_test_highscore is False


def test_periodic_eval_every_fifth_epoch():
    schedule = CuratedEvalSchedule(period=5, on_test_highscore=False)
    assert should_evaluate_curated(4, schedule) == (True, "period=5")
    assert should_evaluate_curated(3, schedule) == (False, None)


def test_eval_on_test_highscore_only():
    schedule = CuratedEvalSchedule(period=0, on_test_highscore=True)
    assert should_evaluate_curated(2, schedule, is_new_test_highscore=True) == (
        True,
        "test_highscore",
    )
    assert should_evaluate_curated(2, schedule, is_new_test_highscore=False) == (
        False,
        None,
    )


def test_combined_period_and_highscore_reasons():
    schedule = CuratedEvalSchedule(period=5, on_test_highscore=True)
    should_run, reason = should_evaluate_curated(
        4,
        schedule,
        is_new_test_highscore=True,
    )
    assert should_run is True
    assert reason == "period=5+test_highscore"


def test_invalid_period_raises():
    with pytest.raises(ValueError, match="curated_eval_period"):
        CuratedEvalSchedule(period=-1)
