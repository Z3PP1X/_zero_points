from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from stable_baselines3.common.monitor import Monitor

from mathematica_env import MathematicaGraphEnv
from ppo_optuna_workflow import PpoOptunaWorkflow


class _GatewayStub:
    def __init__(self):
        self.network_queue = Queue()
        self.running = True
        self.sent_decisions = []

    def send_decision(self, original_state, solver, local_max_tolerance):
        self.sent_decisions.append(
            {
                "state": original_state,
                "solver": solver,
                "local_max_tolerance": local_max_tolerance,
            }
        )


def _build_env(gateway):
    pyg_data = SimpleNamespace(
        x=MagicMock(),
        edge_index=MagicMock(),
    )
    pyg_data.x.numpy.return_value = np.zeros((1, 1), dtype=np.float32)
    pyg_data.edge_index.numpy.return_value = np.zeros((2, 0), dtype=np.int64)
    preprocessor = MagicMock()
    preprocessor.process.return_value = (pyg_data, None)
    reward_calculator = MagicMock()
    return MathematicaGraphEnv(
        gateway=gateway,
        preprocessor=preprocessor,
        reward_calculator=reward_calculator,
    )


def test_drain_buffered_states_completes_episode_without_sending_decision():
    gateway = _GatewayStub()
    env = _build_env(gateway)
    env.current_uuid = "episode-a"
    env.current_state_dict = {"uuid": "episode-a", "status": "running", "id": "P1"}
    env.current_obs = {
        "x": np.zeros((env.max_nodes, 1), dtype=np.float32),
        "edge_index": np.zeros((2, env.max_edges), dtype=np.int64),
        "num_nodes": np.array([1], dtype=np.int64),
        "num_edges": np.array([0], dtype=np.int64),
    }
    env.replay_buffer.start_episode("episode-a")
    env.replay_buffer.add_transition(
        uuid="episode-a",
        current_state=env.current_state_dict,
        action={"solver": 0, "localMaxTolerance": 1e-12},
    )
    gateway.network_queue.put(
        {
            "uuid": "episode-a",
            "status": "reward_calc",
            "id": "P1",
            "networkStep": 1,
        }
    )

    completed = env.drain_buffered_states()

    assert completed is True
    assert gateway.sent_decisions == []
    assert not env.replay_buffer.has_episode("episode-a")
    env.reward_calculator.calculate_episode_rewards.assert_called_once()


def test_finalize_drains_buffered_terminal_state_before_sampling_actions():
    gateway = _GatewayStub()
    env = Monitor(_build_env(gateway))
    unwrapped = env.unwrapped
    unwrapped.current_uuid = "episode-a"
    unwrapped.current_state_dict = {
        "uuid": "episode-a",
        "status": "running",
        "id": "P1",
    }
    unwrapped.current_obs = {
        "x": np.zeros((unwrapped.max_nodes, 1), dtype=np.float32),
        "edge_index": np.zeros((2, unwrapped.max_edges), dtype=np.int64),
        "num_nodes": np.array([1], dtype=np.int64),
        "num_edges": np.array([0], dtype=np.int64),
    }
    unwrapped.replay_buffer.start_episode("episode-a")
    unwrapped.replay_buffer.add_transition(
        uuid="episode-a",
        current_state=unwrapped.current_state_dict,
        action={"solver": 0, "localMaxTolerance": 1e-12},
    )
    gateway.network_queue.put(
        {
            "uuid": "episode-a",
            "status": "finished",
            "id": "P1",
            "networkStep": 1,
        }
    )
    workflow = PpoOptunaWorkflow(
        gateway=gateway,
        preprocessor=MagicMock(),
        experiment_name="stage3_full_graph",
        timesteps_per_trial=100,
    )

    workflow._finalize_episode_state(env)

    assert gateway.sent_decisions == []
    assert not unwrapped.replay_buffer.has_episode("episode-a")
