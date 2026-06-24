"""Tests for curated holdout eval scheduling, incl. the warmup gate.

The warmup gate suppresses curated holdout evaluation for the first N epochs,
where an underfit model on the out-of-distribution curated set can produce
anomalous — even inverted — metrics that would mislead diagnostics.
"""

import pytest

from gnn.supervised_learning.curated_eval_schedule import (
    CuratedEvalSchedule,
    parse_curated_eval_schedule,
    should_evaluate_curated,
)


def test_default_schedule_has_no_warmup():
    """Backwards-compatible default: warmup disabled (eval allowed from epoch 0)."""
    schedule = CuratedEvalSchedule(period=1)
    assert schedule.warmup == 0
    should_run, reason = should_evaluate_curated(0, schedule)
    assert should_run is True
    assert reason == "period=1"


def test_warmup_suppresses_periodic_eval():
    """During warmup, periodic triggers are gated even when the period matches."""
    schedule = CuratedEvalSchedule(period=1, warmup=5)
    # epochs 0..4 (1-based 1..5) are inside warmup -> never run
    for epoch in range(5):
        assert should_evaluate_curated(epoch, schedule) == (False, None)
    # first epoch >= warmup runs
    assert should_evaluate_curated(5, schedule) == (True, "period=1")


def test_warmup_overrides_test_highscore():
    """A new highscore inside the warmup window must NOT trigger curated eval."""
    schedule = CuratedEvalSchedule(period=0, on_test_highscore=True, warmup=3)
    assert should_evaluate_curated(1, schedule, is_new_test_highscore=True) == (
        False,
        None,
    )
    # after warmup, the highscore trigger fires again
    assert should_evaluate_curated(3, schedule, is_new_test_highscore=True) == (
        True,
        "test_highscore",
    )


def test_parse_reads_curated_eval_warmup():
    schedule = parse_curated_eval_schedule(
        {
            "curated_eval_period": 1,
            "curated_eval_on_test_highscore": True,
            "curated_eval_warmup": 5,
        }
    )
    assert schedule.warmup == 5


def test_parse_defaults_warmup_to_zero_when_absent():
    schedule = parse_curated_eval_schedule({"curated_eval_period": 1})
    assert schedule.warmup == 0


def test_negative_warmup_rejected():
    with pytest.raises(ValueError, match="curated_eval_warmup"):
        CuratedEvalSchedule(period=1, warmup=-1)
