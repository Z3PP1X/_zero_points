from queue import Queue

from mathematica_state_ingress import MathematicaStateIngress


class _GatewayStub:
    def __init__(self, messages):
        self.network_queue = Queue()
        for message in messages:
            self.network_queue.put(message)
        self.running = True


def test_poll_next_for_episode_returns_buffered_state_without_blocking():
    gateway = _GatewayStub(
        [
            {"uuid": "episode-a", "status": "running"},
            {"uuid": "episode-b", "status": "running"},
        ]
    )
    ingress = MathematicaStateIngress(gateway)

    state = ingress.poll_next_for_episode("episode-a")

    assert state == {"uuid": "episode-a", "status": "running"}
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
