import time
import zmq
import threading
from queue import Queue

class NetworkGateway():

    def __init__(self, receiver_port, sender_port, control_port, reward_port):
        self.context = zmq.Context()
        self.receiver_port = receiver_port
        self.sender_port = sender_port
        self.control_port = control_port
        self.reward_port = reward_port
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
                    self.network_queue.put(message)
                if (
                    self.reward_receiver in socks
                    and socks[self.reward_receiver] == zmq.POLLIN
                ):
                    reward_state = self.reward_receiver.recv_json()
                    self.network_queue.put(reward_state)

        except Exception as e:
            print(f"[Gateway] CRITICAL ERROR IN POLL LOOP: {e}")
        finally:
            self._cleanup_receivers()

    def stop(self):
        control_parameter = {"pipeline_status": 0}
        self.controller.send_json(control_parameter)
        self.running = False
        if self._thread:
            self._thread.join()

    def send(self, message):
        if self.sender:
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
    