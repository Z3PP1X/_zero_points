from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mathematica_vec_env import MathematicaVecEnv, build_mathematica_training_env


class _GatewayStub:
    def __init__(self, messages=None):
        self.network_queue = Queue()
        for message in messages or []:
            self._enqueue(message)

        self.running = True
        self.sent_decisions = []

    def _enqueue(self, message, channel="training"):
        payload = dict(message)
        payload["_gateway_channel"] = channel
        self.network_queue.put(payload)

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
    pyg_data.x.numpy.return_value = np.zeros((1, 8), dtype=np.float32)
    pyg_data.edge_index.numpy.return_value = np.zeros((2, 0), dtype=np.int64)
    pyg_data.global_features.numpy.return_value = np.zeros((12,), dtype=np.float32)
    preprocessor = MagicMock()
    preprocessor.process.return_value = (pyg_data, None)
    return preprocessor


def _build_vec_env(gateway, *, n_envs=2):
    return MathematicaVecEnv(
        num_envs=n_envs,
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=MagicMock(),
        max_nodes=50,
        max_edges=100,
    )


def test_reset_fills_slots_from_fresh_state_pool():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running", "id": "P1"},
            {"uuid": "episode-b", "status": "running", "id": "P2"},
        ]
    )
    env = _build_vec_env(gateway)

    env.reset()

    assert set(env._slot_uuid) == {"episode-a", "episode-b"}


def test_build_mathematica_training_env_returns_slot_vec_env():
    gateway = _GatewayStub(
        [{"uuid": "episode-a", "status": "running", "id": "P1"}]
    )
    env = build_mathematica_training_env(
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=MagicMock(),
        n_envs=1,
        max_nodes=50,
        max_edges=100,
    )

    assert isinstance(env, MathematicaVecEnv)
    env.reset()
    assert env._slot_uuid == ["episode-a"]


def test_step_sends_all_decisions_before_waiting_for_responses():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running", "id": "P1"},
            {"uuid": "episode-b", "status": "running", "id": "P2"},
        ]
    )
    env = _build_vec_env(gateway)
    env.reset()
    gateway.sent_decisions.clear()

    actions = np.zeros((2, 2), dtype=np.float32)
    env.step_async(actions)
    gateway._enqueue(
        {"uuid": "episode-b", "status": "running", "id": "P1", "networkStep": 1}
    )
    gateway._enqueue(
        {"uuid": "episode-a", "status": "running", "id": "P1", "networkStep": 1}
    )
    env.step_wait()

    assert len(gateway.sent_decisions) == 2
    assert {decision["state"]["uuid"] for decision in gateway.sent_decisions} == {
        "episode-a",
        "episode-b",
    }


def test_step_wait_finishes_other_slots_before_refill():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running", "id": "P1"},
            {"uuid": "episode-b", "status": "running", "id": "P2"},
            {"uuid": "episode-c", "status": "running", "id": "P3"},
        ]
    )
    env = _build_vec_env(gateway)
    env.reset()
    env.step_async(np.zeros((2, 2), dtype=np.float32))
    gateway._enqueue(
        {"uuid": "episode-a", "status": "finished", "id": "P1", "networkStep": 1}
    )
    gateway._enqueue(
        {"uuid": "episode-b", "status": "running", "id": "P2", "networkStep": 1}
    )
    _, _, dones, _ = env.step_wait()

    assert bool(dones[0]) is True
    assert bool(dones[1]) is False
    assert env._slot_uuid[0] == "episode-c"
    assert env._slot_uuid[1] == "episode-b"


def test_reward_port_state_completes_episode():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running", "id": "P1"},
            {"uuid": "episode-b", "status": "running", "id": "P2"},
            {"uuid": "episode-c", "status": "running", "id": "P3"},
        ]
    )
    reward_calculator = MagicMock()
    env = MathematicaVecEnv(
        num_envs=1,
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=reward_calculator,
        max_nodes=50,
        max_edges=100,
    )
    env.reset()
    env.step_async(np.zeros((1, 2), dtype=np.float32))
    gateway._enqueue(
        {"uuid": "episode-a", "status": "running", "id": "P1", "networkStep": 1},
        channel="reward",
    )
    _, _, dones, infos = env.step_wait()

    assert bool(dones[0]) is True
    assert infos[0]["episode"]["l"] == 1
    assert env._slot_uuid == ["episode-b"]
    reward_calculator.calculate_episode_rewards.assert_called_once()


def test_terminal_response_refills_slot_from_fresh_pool():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running", "id": "P1"},
            {"uuid": "episode-b", "status": "running", "id": "P2"},
            {"uuid": "episode-c", "status": "running", "id": "P3"},
        ]
    )
    reward_calculator = MagicMock()
    env = MathematicaVecEnv(
        num_envs=1,
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=reward_calculator,
        max_nodes=50,
        max_edges=100,
    )
    env.reset()
    assert env._slot_uuid == ["episode-a"]

    env.step_async(np.zeros((1, 2), dtype=np.float32))
    gateway._enqueue(
        {"uuid": "episode-a", "status": "finished", "id": "P1", "networkStep": 1}
    )
    _, rewards, dones, infos = env.step_wait()

    assert rewards[0] == 0.0
    assert bool(dones[0]) is True
    assert infos[0]["episode"]["l"] == 1
    assert env._slot_uuid == ["episode-b"]
    reward_calculator.calculate_episode_rewards.assert_called_once()
