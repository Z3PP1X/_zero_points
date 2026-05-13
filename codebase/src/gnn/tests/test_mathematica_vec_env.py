from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mathematica_vec_env import build_mathematica_training_env, iter_monitored_envs


class _GatewayStub:
    def __init__(self, messages=None):
        self.network_queue = Queue()
        for message in messages or []:
            self.network_queue.put(message)
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


def _build_preprocessor():
    pyg_data = SimpleNamespace(
        x=MagicMock(),
        edge_index=MagicMock(),
        global_features=MagicMock(),
    )
    pyg_data.x.numpy.return_value = np.zeros((1, 1), dtype=np.float32)
    pyg_data.edge_index.numpy.return_value = np.zeros((2, 0), dtype=np.int64)
    pyg_data.global_features.numpy.return_value = np.zeros((1,), dtype=np.float32)
    preprocessor = MagicMock()
    preprocessor.process.return_value = (pyg_data, None)
    return preprocessor


def test_build_mathematica_training_env_shares_state_ingress_across_workers():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running", "id": "P1"},
            {"uuid": "episode-b", "status": "running", "id": "P2"},
        ]
    )
    reward_calculator = MagicMock()
    vec_env = build_mathematica_training_env(
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=reward_calculator,
        n_envs=2,
        max_nodes=50,
        max_edges=100,
    )

    assert hasattr(vec_env, "envs")
    assert len(iter_monitored_envs(vec_env)) == 2
    ingress = vec_env.envs[0].unwrapped.state_ingress
    assert ingress is vec_env.envs[1].unwrapped.state_ingress

    vec_env.reset()
    active_uuids = {env.unwrapped.current_uuid for env in vec_env.envs}

    assert active_uuids == {"episode-a", "episode-b"}


def test_build_mathematica_training_env_returns_single_monitor_for_one_worker():
    gateway = _GatewayStub(
        [{"uuid": "episode-a", "status": "running", "id": "P1"}]
    )
    reward_calculator = MagicMock()
    env = build_mathematica_training_env(
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=reward_calculator,
        n_envs=1,
        max_nodes=50,
        max_edges=100,
    )

    assert not hasattr(env, "envs")
    assert len(iter_monitored_envs(env)) == 1
    env.reset()
    assert env.unwrapped.current_uuid == "episode-a"
