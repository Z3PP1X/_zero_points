from queue import Queue

from mathematica_state_ingress import MathematicaStateIngress


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


def test_take_next_training_start_skips_reward_port_message():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running"},
        ]
    )
    gateway._enqueue(
        {"uuid": "episode-b", "status": "finished"},
        channel="reward",
    )
    ingress = MathematicaStateIngress(gateway)

    state = ingress.take_next_training_start()

    assert state == {"uuid": "episode-a", "status": "running", "_gateway_channel": "training"}


def test_poll_next_for_episode_returns_buffered_state_without_blocking():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running"},
            {"uuid": "episode-b", "status": "running"},
        ]
    )
    ingress = MathematicaStateIngress(gateway)

    state = ingress.poll_next_for_episode("episode-a")

    assert state == {
        "uuid": "episode-a",
        "status": "running",
        "_gateway_channel": "training",
    }
    assert gateway.network_queue.qsize() == 1


def test_poll_next_for_episode_prefers_deferred_state():
    gateway = _GatewayStub([])
    ingress = MathematicaStateIngress(gateway)
    ingress._deferred_by_uuid["episode-a"] = {
        "uuid": "episode-a",
        "status": "reward_calc",
    }

    state = ingress.poll_next_for_episode("episode-a")

    assert state["status"] == "reward_calc"
