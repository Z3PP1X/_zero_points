import socket
import json
import time
import queue
import threading


class SocketStream:
    def __init__(self, port, timeout_sec=10.0):
        self.port = port
        self.buffer = ""
        self.server = None
        self.connection = None
        self.client_address = None
        self.received_queue = queue.PriorityQueue()
        self.send_queue = queue.PriorityQueue()

        self.running = False
        self.timeout_sec = timeout_sec
        self.last_contact = time.time()

    def _wrap_envelope(self, msg_type, payload=None):
        return (
            json.dumps({"type": msg_type, "payload": payload, "timestamp": time.time()})
            + "\n"
        )

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("localhost", self.port))
        self.server.listen(1)
        print(f"Waiting for Mathematica on Port: {self.port}...")

        self.connection, self.client_address = self.server.accept()
        print(f"Connected with: {self.client_address}")

        self.running = True
        self.last_contact = time.time()

        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._health_loop, daemon=True).start()

    def send_data(self, payload):
        envelope = self._wrap_envelope("data", payload)
        self.send_queue.put((1, envelope))

    def _health_loop(self):
        while self.running:
            time.sleep(5.0)
            if time.time() - self.last_contact > self.timeout_sec:
                print("TIMEOUT!")
                self.stop()
                break
            self.send_queue.put((0, self._wrap_envelope("ping")))

    def _listen_loop(self):
        while self.running:
            try:
                data = self.connection.recv(8192).decode("utf-8")
                if not data:
                    print("No data, closing connection to client...")
                    self.stop()
                    break

                self.buffer += data

                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    if line.strip():
                        self.last_contact = time.time()
                        envelope = json.loads(line)

                        msg_type = envelope.get("type")
                        if msg_type == "ping":
                            self.send_queue.put((0, self._wrap_envelope("pong")))
                        elif msg_type == "pong":
                            pass
                        elif msg_type == "data":
                            self.received_queue.put((1, envelope.get("payload")))

            except Exception as e:
                print(f"Network Error: {e}")
                self.stop()

    def _send_loop(self):
        while self.running:
            try:
                priority, msg = self.send_queue.get(timeout=1.0)
                if self.connection:
                    self.connection.sendall(msg.encode("utf-8"))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Send Error: {e}")
                self.stop()

    def get_message(self, block=True, timeout=None):
        try:
            priority, payload = self.received_queue.get(block=block, timeout=timeout)
            return payload
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.connection:
            print("Closing connection...")
            try:
                self.connection.close()
            except Exception as e:
                print(f"Error: {e}")
        if self.server:
            print("Shutting down server...")
            try:
                self.server.close()
            except Exception as e:
                print(f"Error: {e}")
