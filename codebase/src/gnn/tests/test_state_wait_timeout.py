from state_wait_timeout import StateRoundtripTimeout


def test_timeout_uses_fallback_without_samples():
    estimator = StateRoundtripTimeout(fallback_s=5.0, cushion_s=2.0)

    assert estimator.timeout_s() == 5.0


def test_timeout_uses_rolling_average_plus_cushion():
    estimator = StateRoundtripTimeout(fallback_s=5.0, cushion_s=2.0, window_size=100)
    estimator.record(0.5)
    estimator.record(1.5)

    assert estimator.timeout_s() == 3.0


def test_mean_roundtrip_s_is_none_without_samples():
    estimator = StateRoundtripTimeout()

    assert estimator.mean_roundtrip_s() is None
    assert estimator.sample_count == 0


def test_mean_roundtrip_s_tracks_recorded_samples():
    estimator = StateRoundtripTimeout()
    estimator.record(0.5)
    estimator.record(1.5)

    assert estimator.mean_roundtrip_s() == 1.0
    assert estimator.sample_count == 2


def test_timeout_window_limits_history():
    estimator = StateRoundtripTimeout(fallback_s=5.0, cushion_s=0.0, window_size=2)
    estimator.record(1.0)
    estimator.record(1.0)
    estimator.record(3.0)

    assert estimator.timeout_s() == 2.0
