import zmq

test = {
    "id": "P4",
    "x0": 2.5,
    "yRange": [0.1, 1.0e4],
    "function": "P4",
    "yTarget": 8.9889990990991e3,
    "status": "init",
    "initialx0": 2.5,
    "uuid": "_id_P4_yTarget_8989.0_x0_",
    "epoch": 0,
    "networkStep": 0,
    "parentStateId": "initialState",
    "tolerance": 1.0e-15,
    "maxIter": 1500,
    "yMin": 0.1,
    "yMax": 1.0e4,
    "timeBenchmarkSolver": 0,
    "stateId": "P4_yTarget_8989.0_epoch_0_networkstep_0",
    "mask": 1,
    "solver": 0,
    "localMaxTolerance": 0.0,
    "lastStepError": 0.0,
    "fx": 2.5,
    "dfx": 0.0,
    "ddfx": 0.0,
    "kappa": 0,
    "lastKappa": 0,
}

inPORT = 6666
outPORT = 6667

context = zmq.Context()

hello = "Hello World!"

sink = context.socket(zmq.PUSH)
sink.bind(f"tcp://127.0.0.1:{outPORT}")

try:

    while True:

        sink.send_string(hello)

except KeyboardInterrupt:
    print("\nSkript durch Benutzer abgebrochen.")
finally:
    sink.close()
    sink.close()
    context.term()
    print("Alle ZMQ-Verbindungen sicher geschlossen.")