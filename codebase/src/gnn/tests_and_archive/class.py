import zmq
import time
import sys


context = zmq.Context()

RECEIVER_PORT = 5650
receiver = context.socket(zmq.PULL)
receiver.bind(f"tcp://localhost:{RECEIVER_PORT}")

RESULTS_PORT = 5693
results = context.socket(zmq.PULL)
results.bind(f"tcp://localhost:{RESULTS_PORT}")

SENDER_PORT = 5651
sender = context.socket(zmq.PUSH)
sender.bind(f"tcp://localhost:{SENDER_PORT}")

test = {"moin" : "Hello World!"}

poller = zmq.Poller()
poller.register(receiver, zmq.POLLIN)
poller.register(results, zmq.POLLIN)

try: 
    while True:
        socks = dict(poller.poll())

        if socks.get(receiver) == zmq.POLLIN:
            message = receiver.recv_json()
            print(message)
            sender.send_json(message)
            sys.stdout.write(".")
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\nBeende Worker...")

finally:
    receiver.close()
    sender.close()
    context.term()