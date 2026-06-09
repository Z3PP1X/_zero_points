"""Tests for the trial-switch pipeline stall fix.

Validates that:
1. ``drain_to_queue`` returns deferred messages to the shared queue.
2. ``finalize_open_episodes`` does NOT consume fresh states from the queue.
3. ``close()`` returns deferred messages so the next trial can use them.
4. A simulated two-trial scenario does not deadlock.
"""
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mathematica_state_ingress import MathematicaStateIngress
from mathematica_vec_env import MathematicaVecEnv


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
    pyg_data.x.numpy.return_value = np.zeros((1, 24), dtype=np.float32)
    pyg_data.edge_index.numpy.return_value = np.zeros((2, 0), dtype=np.int64)
    pyg_data.global_features.numpy.return_value = np.zeros((9,), dtype=np.float32)
    preprocessor = MagicMock()
    preprocessor.process.return_value = (pyg_data, None)
    return preprocessor


def _build_vec_env(gateway, *, n_envs=1):
    return MathematicaVecEnv(
        num_envs=n_envs,
        gateway=gateway,
        preprocessor=_build_preprocessor(),
        reward_calculator=MagicMock(),
        max_nodes=50,
        max_edges=100,
    )


# ---------- drain_to_queue tests ----------


def test_drain_to_queue_returns_deferred_messages():
    """Deferred messages must be returned to the shared queue."""
    gateway = _GatewayStub()
    ingress = MathematicaStateIngress(gateway)
    ingress._deferred_by_uuid["ep-a"] = {"uuid": "ep-a", "status": "running"}
    ingress._deferred_by_uuid["ep-b"] = {"uuid": "ep-b", "status": "running"}
    ingress._waiting_init_order.append("ep-a")
    ingress._waiting_init_order.append("ep-b")

    returned = ingress.drain_to_queue()

    assert returned == 2
    assert gateway.network_queue.qsize() == 2
    assert len(ingress._deferred_by_uuid) == 0
    assert len(ingress._waiting_init_order) == 0


def test_drain_to_queue_handles_empty_ingress():
    """Draining an empty ingress should be a no-op."""
    gateway = _GatewayStub()
    ingress = MathematicaStateIngress(gateway)

    returned = ingress.drain_to_queue()

    assert returned == 0
    assert gateway.network_queue.qsize() == 0


# ---------- finalize without refill tests ----------


def test_finalize_and_close_preserve_fresh_states():
    """Finalize + close must NOT lose fresh states from the queue.

    The fresh state may get deferred by the ingress during finalize's
    ``step_wait``.  ``close()`` must return it to the shared queue.
    """
    gateway = _GatewayStub(
        [
            {"uuid": "ep-a", "status": "running", "id": "P1"},
            {"uuid": "ep-b", "status": "running", "id": "P2"},
        ]
    )
    env = _build_vec_env(gateway, n_envs=1)
    env.reset()
    assert env._slot_uuid == ["ep-a"]
    assert gateway.network_queue.qsize() == 1

    env.step_async(np.zeros((1, 2), dtype=np.float32))
    gateway._enqueue(
        {"uuid": "ep-a", "status": "finished", "id": "P1", "networkStep": 1}
    )
    env.step_wait()
    assert env._slot_uuid == ["ep-b"]
    assert gateway.network_queue.qsize() == 0

    gateway._enqueue(
        {"uuid": "ep-fresh", "status": "running", "id": "P3"},
    )
    env.step_async(np.zeros((1, 2), dtype=np.float32))
    gateway._enqueue(
        {"uuid": "ep-b", "status": "finished", "id": "P2", "networkStep": 1}
    )
    env.finalize_open_episodes()
    env.close()

    assert gateway.network_queue.qsize() == 1
    msg = gateway.network_queue.get_nowait()
    assert msg["uuid"] == "ep-fresh"


# ---------- close() drains deferred ----------


def test_close_drains_deferred_messages_to_queue():
    """close() must return deferred messages so the next env can use them."""
    gateway = _GatewayStub(
        [
            {"uuid": "ep-a", "status": "running", "id": "P1"},
        ]
    )
    env = _build_vec_env(gateway, n_envs=1)
    env.reset()
    env.state_ingress._deferred_by_uuid["ep-leftover"] = {
        "uuid": "ep-leftover",
        "status": "running",
    }
    env.state_ingress._waiting_init_order.append("ep-leftover")

    env.close()

    assert gateway.network_queue.qsize() == 1
    msg = gateway.network_queue.get_nowait()
    assert msg["uuid"] == "ep-leftover"


# ---------- Two-trial simulation ----------


def test_two_trial_simulation_no_deadlock():
    """Simulate two consecutive Optuna trials; the second must not starve."""
    gateway = _GatewayStub(
        [
            {"uuid": "t1-ep-a", "status": "running", "id": "P1"},
            {"uuid": "t2-ep-a", "status": "running", "id": "P2"},
        ]
    )

    # --- Trial 1 ---
    env1 = _build_vec_env(gateway, n_envs=1)
    env1.reset()
    assert env1._slot_uuid == ["t1-ep-a"]
    env1.step_async(np.zeros((1, 2), dtype=np.float32))
    gateway._enqueue(
        {"uuid": "t1-ep-a", "status": "finished", "id": "P1", "networkStep": 1}
    )
    env1.finalize_open_episodes()
    env1.close()

    # --- Trial 2 ---
    env2 = _build_vec_env(gateway, n_envs=1)
    env2.reset()
    assert env2._slot_uuid == ["t2-ep-a"]


def test_stack_obs_handles_none_slots():
    """_stack_obs must produce valid arrays even when some slots are None."""
    gateway = _GatewayStub(
        [
            {"uuid": "ep-a", "status": "running", "id": "P1"},
        ]
    )
    env = _build_vec_env(gateway, n_envs=2)
    env._slot_obs[0] = None
    env._slot_obs[1] = None

    obs = env._stack_obs()

    assert obs["x"].shape == (2, 50, 24)
    assert obs["edge_index"].shape == (2, 2, 100)
    assert obs["num_nodes"].shape == (2, 1)
