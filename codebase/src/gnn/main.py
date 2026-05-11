import torch
from network_gateway import NetworkGateway
from preprocessor import Preprocessor
from dataloader import GraphDataLoader
from replay_buffer import ReplayBuffer
from model import TestGraphNetwork
from reward import RewardCalculator
import time
import queue
import mlflow
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Start GNN RL Pipeline")
parser.add_argument(
    "--experiment", 
    type=str, 
    default="nur_f", 
    choices=["nur_f", "f_fp_roh", "kein_inv"], 
    help="Welcher Graph-Struktur-Ordner verwendet werden soll."
)
args = parser.parse_args()

RECEIVER_PORT = 5650
RESULTS_PORT = 5693
SENDER_PORT = 5651
CONTROL_PORT = 6000

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"GNN_RL_Pipeline_{args.experiment}")

# Initialisierung aller Pipeline-Komponenten
pipeline = NetworkGateway(receiver_port=RECEIVER_PORT, sender_port=SENDER_PORT, reward_port=RESULTS_PORT, control_port=CONTROL_PORT)

graphs_path = os.path.join("graphs", args.experiment)
print(f"Starte Pipeline mit Graphen aus: {graphs_path}")

preprocessor = Preprocessor(graphs_dir=graphs_path) 
# batch_size=1: Online-RL erfordert sofortige Verarbeitung pro State.
# Jede Entscheidung muss zurück nach Mathematica, bevor der nächste State eintrifft.
# Mit batch_size>1 entsteht ein Race Condition: Der Reward-State kann eintreffen,
# bevor der Batch voll ist — dann ist der ReplayBuffer leer und das Training wird übersprungen.
dataloader = GraphDataLoader(batch_size=1)
replay_buffer = ReplayBuffer()
reward_calculator = RewardCalculator(basis_reward=1.0, gamma=0.99, alpha=1.0)

# Initialisierung des Netzwerks & Optimizers
gnn_model = TestGraphNetwork(input_dim=5, hidden_dim=128, global_dim=9, heads=4)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)


# RAM-Zwischenspeicher für States, während sie im Dataloader auf ihr Batch warten
original_states = {}

pipeline.init()

try:
    print("Gateway läuft. Warte auf Nachrichten von Mathematica...")
    with mlflow.start_run():
        mlflow.set_tag("graph_topology", args.experiment)
        while pipeline.running:
            try:
                # 1. State vom Gateway empfangen (nicht-blockierend mit Timeout)
                state_dict = pipeline.network_queue.get(timeout=0.1)
                
                # --- Terminal States (Training Loop) ---
                if state_dict.get("status") in ["reward_calc", "finished"]:
                    uuid = state_dict.get("uuid")
                    
                    print(f"\n[Main] Terminal State empfangen für UUID {uuid}")
                    print(f"Status: {state_dict.get('status')}")
                    print(f"Terminal State Content: networkStep={state_dict.get('networkStep')}, absTime={state_dict.get('absTime')}, recordAbsTime={state_dict.get('recordAbsTime')}, Benchmarksolver={state_dict.get('Benchmarksolver')}")
                    
                    # Episode aus dem Buffer holen
                    episode = replay_buffer.get_episode(uuid)
                    
                    if len(episode) > 0:
                        # Rewards für die Episode berechnen
                        reward_calculator.calculate_episode_rewards(episode, state_dict)
                        
                        # Re-run Forward Pass for the whole episode to get fresh log_probs
                        # This avoids the "inplace operation" error caused by shared batch graphs
                        optimizer.zero_grad()
                        episode_loss = 0.0
                        total_reward = sum(t.get("reward", 0.0) for t in episode)
                        
                        from torch_geometric.data import Batch
                        from torch.distributions import Bernoulli
                        
                        pyg_graphs = []
                        valid_transitions = []
                        for transition in episode:
                            try:
                                G, _ = preprocessor.process(transition["current_state"], None)
                                pyg_graphs.append(G)
                                valid_transitions.append(transition)
                            except Exception as e:
                                print(f"[Main] Error preprocessing state for backward pass: {e}")
                                
                        if len(pyg_graphs) > 0:
                            batch = Batch.from_data_list(pyg_graphs)
                            solver_probs, tol_scales = gnn_model(batch.x, batch.edge_index, batch.batch, batch.global_features)
                            
                            for i, transition in enumerate(valid_transitions):
                                action = transition.get("action", {})
                                reward = transition.get("reward", 0.0)
                                chosen_solver = action.get("solver")
                                
                                if chosen_solver is not None:
                                    solver_prob = solver_probs[i]
                                    dist_solver = Bernoulli(solver_prob)
                                    log_prob_solver = dist_solver.log_prob(torch.tensor(float(chosen_solver)))
                                    
                                    # REINFORCE: loss = -log_prob * reward
                                    episode_loss = episode_loss - (log_prob_solver * reward)
                                    
                            if isinstance(episode_loss, torch.Tensor):
                                episode_loss.backward()
                                optimizer.step()
                            
                            # MLflow Logging
                            mlflow.log_metric("episode_reward", total_reward)
                            mlflow.log_metric("episode_loss", episode_loss.item())
                            mlflow.log_metric("episode_length", len(episode))
                            print(f"[Main] Training Step! Steps: {len(episode)}, Total Reward: {total_reward:.2f}, Loss: {episode_loss.item():.4f}")
                        
                        # Episode aus dem Buffer entfernen, um mehrfaches Training (und Crashes) zu vermeiden
                        replay_buffer.remove_episode(uuid)
                    
                    continue # Diesen State nicht weiter durchs Netz jagen!
                
                # Überprüfen, ob es ein gültiger State ist
                if "stateId" in state_dict:
                    state_id = state_dict["stateId"]
                    # Original im RAM parken
                    original_states[state_id] = state_dict
                    
                    # 2. Preprocessing & Ab in den Dataloader
                    G, features = preprocessor.process(state_dict, dataloader)
                    dataloader.add_graph(G)
                    
            except queue.Empty:
                # Wenn die Queue für 0.1s leer war, einfach weitermachen (verhindert Deadlocks)
                pass
            
            # 3. Wenn der Batch voll ist -> Forward Pass
            if dataloader.has_batch():
                batch = dataloader.get_batch()
                
                # Forward Pass durch das Modell
                # batch.x, batch.edge_index und batch.batch werden vom Dataloader bereitgestellt
                solver_probs, tol_scales = gnn_model(batch.x, batch.edge_index, batch.batch, batch.global_features)
                
                num_graphs = len(batch.state_id)
                
                # 4. Loop über die Ergebnisse im Batch
                for i in range(num_graphs):
                    sid = batch.state_id[i]
                    
                    # Originalen State wieder hervorholen und aus dem RAM löschen
                    if sid in original_states:
                        orig_state = original_states.pop(sid)
                        
                        # --- Entscheidung 1: Solver (Binär) ---
                        from torch.distributions import Bernoulli, Normal
                        
                        solver_prob = solver_probs[i]
                        # Deterministische Wahl via Schwellenwert
                        chosen_solver = int(solver_prob.item() > 0.5)
                        
                        # Berechnung der Log-Wahrscheinlichkeit für Backpropagation
                        dist_solver = Bernoulli(solver_prob)
                        log_prob_solver = dist_solver.log_prob(torch.tensor(float(chosen_solver)))
                        
                        # --- Entscheidung 2: Tolerance (Log-Space Mapping) ---
                        # Wir zentrieren den Log-Raum um die vom Problem gegebene globale Toleranz.
                        import math
                        base_tol = orig_state.get("tolerance", 1e-15)
                        LOG_TOL_BASE = math.log10(base_tol) # z.B. -15.0
                        
                        # Das Netz darf die Toleranz um z.B. +/- 4 Größenordnungen variieren
                        LOG_TOL_MIN = LOG_TOL_BASE - 4.0
                        LOG_TOL_MAX = LOG_TOL_BASE + 4.0
                        
                        scale_factor = tol_scales[i]  # Sigmoid-Output in (0, 1)
                        # Untrainiert (Sigmoid = 0.5) => log10_tol = LOG_TOL_BASE
                        log10_tol = LOG_TOL_MIN + scale_factor.item() * (LOG_TOL_MAX - LOG_TOL_MIN)
                        chosen_tol = 10.0 ** log10_tol
                        
                        # REINFORCE log_prob operiert auf dem Sigmoid-Output (dem echten Policy-Parameter).
                        # Normal-Verteilung mit kleiner Varianz um den aktuellen Wert herum.
                        dist_tol = Normal(scale_factor, 0.1)
                        log_prob_tol = dist_tol.log_prob(scale_factor.detach())
                        
                        # Action als Dictionary im Replay Buffer speichern inkl. der log_probs
                        action_dict = {
                            "solver": chosen_solver, 
                            "localMaxTolerance": chosen_tol,
                            "log_prob_solver": log_prob_solver,
                            "log_prob_tol": log_prob_tol
                        }
                        replay_buffer.store(state=orig_state, action=action_dict)
                        
                        response_state = orig_state.copy()
                        response_state["solver"] = int(chosen_solver)
                        response_state["localMaxTolerance"] = float(chosen_tol)
                        
                        import json
                        print(f"\n--- STATE VOR SENDEN ---")
                        # Nur die wichtigsten Metadaten und die Entscheidungen ausgeben, 
                        # anstatt die riesigen Arrays (graphs) im Log zu spiegeln:
                        print_state = {k: v for k, v in response_state.items() if k not in ["nodes", "edges"]}
                        print(json.dumps(print_state, indent=2))
                        print(f"------------------------\n")
                        
                        # Zurück nach Mathematica funken!
                        pipeline.send_decision(orig_state, chosen_solver, chosen_tol)
                        # print(f"[Main] Entscheidung für State {sid} gesendet: Solver={chosen_solver}, Tol={chosen_tol}")
                        
except KeyboardInterrupt:
    print("\nBeende Gateway und räume auf...")
finally:
    pipeline.stop()
    pipeline.cleanup()