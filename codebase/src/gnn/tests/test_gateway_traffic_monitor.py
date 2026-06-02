import pytest
from gateway_traffic_monitor import GatewayTrafficMonitor


def test_gateway_traffic_monitor_advanced_metrics():
    # Initialize monitor
    monitor = GatewayTrafficMonitor()
    
    # 1. Initially they should be None
    metrics = monitor.get_advanced_rolling_metrics()
    assert metrics["convergence_rate"] is None
    assert metrics["mean_episode_steps"] is None
    assert metrics["mean_roundtrip_s"] is None

    # 2. Observe converged states
    monitor.observe({"status": "finished", "networkStep": 10}, "reward")
    metrics = monitor.get_advanced_rolling_metrics()
    assert metrics["convergence_rate"] == 1.0
    assert metrics["mean_episode_steps"] == 10.0

    # 3. Observe non-converged state
    monitor.observe({"status": "non_converged", "networkStep": 20}, "reward")
    metrics = monitor.get_advanced_rolling_metrics()
    assert metrics["convergence_rate"] == 0.5
    assert metrics["mean_episode_steps"] == 15.0

    # 4. Observe reward_calc state (should count as converged) and ignore invalid step
    monitor.observe({"status": "reward_calc", "networkStep": "invalid"}, "reward")
    metrics = monitor.get_advanced_rolling_metrics()
    assert metrics["convergence_rate"] == 2.0 / 3.0
    # Mean steps should still be 15.0 since "invalid" is ignored
    assert metrics["mean_episode_steps"] == 15.0

    # 5. Observe error state (should count as non-converged)
    monitor.observe({"status": "error", "networkStep": 30}, "reward")
    metrics = monitor.get_advanced_rolling_metrics()
    assert metrics["convergence_rate"] == 0.5
    assert metrics["mean_episode_steps"] == 20.0

    # 6. Test reset_reward_state_count
    monitor.reset_reward_state_count()
    metrics = monitor.get_advanced_rolling_metrics()
    assert metrics["convergence_rate"] is None
    assert metrics["mean_episode_steps"] is None
