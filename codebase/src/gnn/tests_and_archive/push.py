import zmq
import time

PORT = 6550

def start_sender():
    context = zmq.Context()
    
    # PUSH Socket erstellen
    # In diesem Test-Szenario 'bindet' der Sender den Port
    socket = context.socket(zmq.PUSH)
    socket.bind(f"tcp://127.0.0.1:{PORT}")
    
    print(f"[SENDER] PUSH-Socket bereit auf Port {PORT}. Sende Zeitstempel...")
    
    try:
        while True:
            # Aktuellen Zeitstempel erfassen
            current_time = time.time()
            
            # Zeitstempel als einfaches JSON-Objekt senden
            socket.send_json({"timestamp": current_time})
            print(f"[SENDER] Gesendet: {current_time}")
            
            # 1 Sekunde warten
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[SENDER] Beendet.")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    start_sender()