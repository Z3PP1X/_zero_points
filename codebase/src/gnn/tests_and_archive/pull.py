import zmq

PORT = 6551

def start_receiver():
    context = zmq.Context()
    
    # PULL Socket erstellen
    # Der Empfänger 'verbindet' sich mit dem gebundenen Port des Senders
    socket = context.socket(zmq.PULL)
    socket.connect(f"tcp://127.0.0.1:{PORT}")
    
    print(f"[RECEIVER] PULL-Socket verbunden mit Port {PORT}. Warte auf Daten...")
    
    try:
        while True:
            # Blockiert, bis eine Nachricht eintrifft
            data = socket.recv_string()
            print(f"[RECEIVER] Empfangen: {data}")
            
    except KeyboardInterrupt:
        print("\n[RECEIVER] Beendet.")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    start_receiver()