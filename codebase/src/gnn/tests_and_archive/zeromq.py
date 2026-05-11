import zmq
import time
import json

# Konfiguration
# Tipp: Überprüfe, wer 'bind' und wer 'connect' macht. 
# Üblicherweise bindet der Service (Python) die Ports.
PROVIDER_PORT = 5650  # Eingang von Mathematica
SINK_PORT = 5651      # Ausgang zu Mathematica

def run_nn_node():
    context = zmq.Context()

    # --- 1. Provider-Socket (PULL) ---
    # Wir empfangen Daten. 
    provider = context.socket(zmq.PULL)
    try:
        # Falls Mathematica 'SocketOpen' (Listen) macht, ist connect richtig.
        # Falls Mathematica sich verbindet, sollte Python hier .bind nutzen.
        provider.connect(f"tcp://127.0.0.1:{PROVIDER_PORT}")
        print(f"[*] Verbunden mit Provider (PULL) auf Port {PROVIDER_PORT}")
    except zmq.ZMQError as e:
        print(f"Fehler beim Provider-Setup: {e}")
        return

    # --- 2. Sink-Socket (PUSH) ---
    # Wir senden Daten zurück.
    sink = context.socket(zmq.PUSH)
    try:
        sink.bind(f"tcp://127.0.0.1:{SINK_PORT}")
        print(f"[*] Sink (PUSH) gebunden auf Port {SINK_PORT}")
    except zmq.ZMQError as e:
        print(f"Fehler beim Sink-Setup: {e}")
        return

    # Poller initialisieren (erlaubt nicht-blockierendes Warten)
    poller = zmq.Poller()
    poller.register(provider, zmq.POLLIN)

    print("\n=== NN-Node bereit! Warte auf Daten von Mathematica... ===\n")

    try:
        while True:
            # Warte bis zu 1000ms auf neue Nachrichten
            socks = dict(poller.poll(1000))

            if provider in socks and socks[provider] == zmq.POLLIN:
                # Nachricht empfangen
                message = provider.recv_json() # Automatische Deserialisierung falls JSON
                print(f"Empfangen: {message}")

                # --- Platzhalter für Neuronales Netz ---
                # Beispiel: Wir hängen nur einen Zeitstempel an
                result = {
                    "original_data": message,
                    "status": "processed",
                    "timestamp": time.time()
                }
                # ---------------------------------------

                # Ergebnis zurücksenden
                sink.send_json(result)
                print(f" -> Ergebnis gesendet an Port {SINK_PORT}\n")
            else:
                # Hier könnte man andere Hintergrundaufgaben erledigen, 
                # während man auf Daten wartet.
                pass

    except KeyboardInterrupt:
        print("\n[*] Node wird heruntergefahren...")
    finally:
        # Ressourcen sauber freigeben
        provider.close()
        sink.close()
        context.term()
        print("[+] ZMQ-Ressourcen bereinigt.")

if __name__ == "__main__":
    run_nn_node()