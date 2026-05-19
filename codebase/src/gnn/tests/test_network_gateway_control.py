from unittest.mock import MagicMock

from network_gateway import CONTROL_FRESH_TRIAL_ENV, NetworkGateway


def test_send_control_publishes_json_payload():
    gateway = NetworkGateway(
        receiver_port=0,
        sender_port=0,
        control_port=0,
        reward_port=0,
    )
    gateway.controller = MagicMock()

    gateway.send_control(CONTROL_FRESH_TRIAL_ENV)

    gateway.controller.send_json.assert_called_once_with(
        {"control": CONTROL_FRESH_TRIAL_ENV}
    )
