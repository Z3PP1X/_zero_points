import json
from pathlib import Path
from unittest.mock import MagicMock

from gateway_state_logger import GatewayStateLogger
from network_gateway import NetworkGateway


def test_gateway_state_logger_writes_incoming_and_outgoing_states(tmp_path):
    log_path = tmp_path / "gateway_states.jsonl"
    logger = GatewayStateLogger(log_path=log_path)

    logger.log_incoming({"uuid": "episode-a", "status": "running"}, "training")
    logger.log_incoming({"uuid": "episode-a", "status": "finished"}, "reward")
    logger.log_outgoing({"uuid": "episode-a", "solver": 2})
    logger.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert [record["direction"] for record in records] == ["in", "in", "out"]
    assert [record["channel"] for record in records] == ["training", "reward", "training"]
    assert records[0]["state"] == {"uuid": "episode-a", "status": "running"}
    assert records[1]["state"] == {"uuid": "episode-a", "status": "finished"}
    assert records[2]["state"] == {"uuid": "episode-a", "solver": 2}


def test_network_gateway_logs_states_on_enqueue_and_send_decision(tmp_path):
    log_path = tmp_path / "gateway_states.jsonl"
    logger = GatewayStateLogger(log_path=log_path)
    gateway = NetworkGateway(
        receiver_port=0,
        sender_port=0,
        control_port=0,
        reward_port=0,
        state_logger=logger,
    )
    gateway.sender = MagicMock()

    gateway._enqueue_message({"uuid": "episode-a", "status": "running"}, "training")
    gateway._enqueue_message({"uuid": "episode-a", "status": "finished"}, "reward")
    gateway.send_decision({"uuid": "episode-a", "status": "running"}, solver=1, local_max_tolerance=0.01)
    logger.close()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert [record["direction"] for record in records] == ["in", "in", "out"]
    assert [record["channel"] for record in records] == ["training", "reward", "training"]
    assert records[2]["state"]["solver"] == 1
    assert records[2]["state"]["localMaxTolerance"] == 0.01
    assert "_gateway_channel" not in records[0]["state"]
