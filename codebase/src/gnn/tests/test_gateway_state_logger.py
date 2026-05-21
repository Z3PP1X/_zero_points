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


def test_gateway_state_logger_sqlite(tmp_path):
    import sqlite3
    db_path = tmp_path / "gateway_states.db"
    logger = GatewayStateLogger(log_path=db_path)

    logger.log_incoming({"uuid": "episode-a", "status": "running"}, "training")
    logger.log_incoming({"uuid": "episode-a", "status": "finished"}, "reward")
    logger.log_outgoing({"uuid": "episode-a", "solver": 2})
    logger.close()

    # Query the SQLite DB
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, direction, channel, state FROM gateway_states ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 3
    
    # row 0
    assert rows[0][1] == "in"
    assert rows[0][2] == "training"
    assert json.loads(rows[0][3]) == {"uuid": "episode-a", "status": "running"}

    # row 1
    assert rows[1][1] == "in"
    assert rows[1][2] == "reward"
    assert json.loads(rows[1][3]) == {"uuid": "episode-a", "status": "finished"}

    # row 2
    assert rows[2][1] == "out"
    assert rows[2][2] == "training"
    assert json.loads(rows[2][3]) == {"uuid": "episode-a", "solver": 2}


def test_gateway_state_logger_mlflow_upload(tmp_path, monkeypatch):
    import sys
    # Mock mlflow
    mock_mlflow = MagicMock()
    mock_mlflow.active_run.return_value = MagicMock()
    
    # We patch sys.modules so that 'import mlflow' inside close() returns our mock
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    db_path = tmp_path / "gateway_states.db"
    logger = GatewayStateLogger(log_path=db_path)
    logger.log_incoming({"uuid": "episode-a"}, "training")
    logger.close()

    # Verify that log_artifact was called with the path and states folder
    mock_mlflow.log_artifact.assert_called_once_with(str(db_path), artifact_path="states")


