import time
import zmq
import threading
from queue import Queue
from typing import Optional

from gateway_state_logger import GatewayStateLogger
from gateway_traffic_monitor import GatewayTrafficMonitor

# Instructs the Mathematica pipeline to stream fresh initial states for a new trial.
CONTROL_FRESH_TRIAL_ENV = 3


class NetworkGateway():

    def __init__(
        self,
        receiver_port,
        sender_port,
        control_port,
        reward_port,
        *,
        traffic_monitor: Optional[GatewayTrafficMonitor] = None,
        state_logger: Optional[GatewayStateLogger] = None,
    ):
        self.context = zmq.Context()
        self.receiver_port = receiver_port
        self.sender_port = sender_port
        self.control_port = control_port
        self.reward_port = reward_port
        self.traffic_monitor = traffic_monitor
        self.state_logger = state_logger
        self.receiver = None
        self.sender = None
        self.controller = None
        self.poller = None
        self.running = False
        self._thread = None
        self._ready = threading.Event()
        self.network_queue = Queue()

    def _set_up_sender(self):
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.bind(f"tcp://localhost:{self.sender_port}")


    def _set_up_reward_receiver(self):
        self.reward_receiver = self.context.socket(zmq.PULL)
        self.reward_receiver.bind(f"tcp://localhost:{self.reward_port}")

    def _set_up_receiver(self):
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://localhost:{self.receiver_port}")

    def _set_up_controller(self):
        self.controller = self.context.socket(zmq.PUB)
        self.controller.bind(f"tcp://localhost:{self.control_port}")

    def _set_up_poller(self):
        self.poller = zmq.Poller()
        self.poller.register(self.receiver, zmq.POLLIN)
        self.poller.register(self.reward_receiver, zmq.POLLIN)

    def init(self):
        self._set_up_sender()
        self._set_up_controller()
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        try:
            self._set_up_receiver()
            self._set_up_reward_receiver()
            self._set_up_poller()
        except Exception as e:
            print(f"[Gateway Debug] FATAL EXCEPTION DURING SOCKET SETUP: {e}")
            return

        self._ready.set()
        print(f"[Gateway Debug] Loop is live! Polling on ports {self.receiver_port} and {self.reward_port}...")
        
        try:
            while self.running:                   
                socks = dict(self.poller.poll(timeout=100))
                if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                    message = self.receiver.recv_json()
                    self._enqueue_message(message, "training")
                if (
                    self.reward_receiver in socks
                    and socks[self.reward_receiver] == zmq.POLLIN
                ):
                    reward_state = self.reward_receiver.recv_json()
                    self._enqueue_message(reward_state, "reward")

        except Exception as e:
            print(f"[Gateway] CRITICAL ERROR IN POLL LOOP: {e}")
        finally:
            self._cleanup_receivers()

    def _enqueue_message(self, message, channel: str) -> None:
        if isinstance(message, dict):
            if self.state_logger is not None:
                self.state_logger.log_incoming(message, channel)
            message["_gateway_channel"] = channel
            if self.traffic_monitor is not None:
                self.traffic_monitor.observe(message, channel)
        self.network_queue.put(message)

    def send_control(self, control: int) -> None:
        if not self.controller:
            raise RuntimeError("Control socket not initialized; call init() first.")
        self.controller.send_json({"control": int(control)})

    def stop(self):
        control_parameter = {"pipeline_status": 0}
        self.controller.send_json(control_parameter)
        self.running = False
        if self._thread:
            self._thread.join()

    def send(self, message):
        if self.sender:
            if self.state_logger is not None:
                self.state_logger.log_outgoing(message)
            self.sender.send_json(message)

    def send_decision(self, original_state: dict, solver: int, local_max_tolerance: float):
        """
        Fügt die vom GNN getroffenen Entscheidungen in den originalen State ein 
        und sendet das aktualisierte Dictionary zurück an Mathematica.
        """
        response_state = original_state.copy()  # Kopie zur Vermeidung von Seiteneffekten
        response_state["solver"] = int(solver)
        response_state["localMaxTolerance"] = float(local_max_tolerance)
        
        if self.sender:
            if self.state_logger is not None:
                self.state_logger.log_outgoing(response_state)
            self.sender.send_json(response_state)

    def _cleanup_receivers(self):
        if self.receiver:
            self.receiver.close()
        if self.reward_receiver:
            self.reward_receiver.close()

    def cleanup(self):
        if self.sender:
            self.sender.close()
        if self.controller:
            self.controller.close()
        self.context.term()
    