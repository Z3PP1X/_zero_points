import sys
import argparse
import os
from pathlib import Path

import mlflow

# Dynamic sys.path resolution to support package imports when run as scripts
gnn_root = Path(__file__).resolve().parents[1]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.reinforcement_learning.gateway.gateway_state_logger import GatewayStateLogger
from gnn.reinforcement_learning.gateway.network_gateway import NetworkGateway
from gnn.reinforcement_learning.gateway.gateway_traffic_monitor import GatewayTrafficMonitor
from gnn.reinforcement_learning.ppo_optuna_workflow import PpoOptunaWorkflow
from gnn.reinforcement_learning.preprocessor import Preprocessor

RECEIVER_PORT = 5650
RESULTS_PORT = 5693
SENDER_PORT = 5651
CONTROL_PORT = 6000


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start GNN RL Pipeline with SB3 & Optuna")
    parser.add_argument(
        "--experiment",
        type=str,
        default="nur_f",
        choices=["nur_f", "f_fp_roh", "kein_inv"],
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument(
        "--timeout-fallback",
        type=float,
        default=5.0,
        help="Initiale Timeout-Wartezeit in Sekunden ohne Roundtrip-Historie.",
    )
    parser.add_argument(
        "--timeout-cushion",
        type=float,
        default=1.0,
        help="Puffer in Sekunden auf den gleitenden Roundtrip-Durchschnitt.",
    )
    parser.add_argument(
        "--timeout-window",
        type=int,
        default=100,
        help="Anzahl erfolgreicher Roundtrips für den gleitenden Durchschnitt.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Anzahl paralleler Mathematica-Slots für SB3-Training "
            "(gemeinsames Senden/Sammeln pro VecEnv-Schritt)."
        ),
    )
    parser.add_argument(
        "--continue-study",
        action="store_true",
        help="Set this flag to continue the last not finished study, otherwise start a new one.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="graph",
        choices=["graph", "tree", "tree_derivatives"],
        help="Select GNN experiment mode: graph (with virtual nodes), tree (features on global node, f only) or tree_derivatives (f, f', f'' connected via global node)"
    )
    parser.add_argument(
        "--active-features",
        type=str,
        default=None,
        help="Comma-separated list of active GNN node features to use (dynamically adapts dimensions)."
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    mlflow.set_experiment(f"GNN_RL_Optuna_{args.experiment}")

    traffic_monitor = GatewayTrafficMonitor(
        timeout_fallback_s=args.timeout_fallback,
        timeout_cushion_s=args.timeout_cushion,
        timeout_window_size=args.timeout_window,
    )
    state_logger = GatewayStateLogger()
    gateway = NetworkGateway(
        receiver_port=RECEIVER_PORT,
        sender_port=SENDER_PORT,
        reward_port=RESULTS_PORT,
        control_port=CONTROL_PORT,
        traffic_monitor=traffic_monitor,
        state_logger=state_logger,
    )
    print(
        f"Optuna: {args.n_trials} Trials × {args.timesteps} Schritte | "
        f"Experiment: {args.experiment} | Parallel-Envs: {args.n_envs} | "
        f"Continue Study: {args.continue_study}"
    )
    active_features = None
    if args.active_features is not None:
        active_features = [f.strip() for f in args.active_features.split(",") if f.strip()]
        print(f"Aktivierte Features ({len(active_features)}): {active_features}")
    
    from gnn.shared.utils.unified_loader import UnifiedDataLoader
    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=args.experiment,
        mode=args.mode,
        enrich=True,
    )
    loader = unified_loader.graph_loader
    
    preprocessor = Preprocessor(loader=loader, mode=args.mode, active_features=active_features)
    print(
        f"Graph-Templates: {len(preprocessor.known_problem_ids)} Problem-IDs indexiert, "
        f"lazy LRU-Cache aktiv (mode: {args.mode})"
    )
    gateway.init()
    traffic_monitor.start()

    workflow = PpoOptunaWorkflow(
        gateway=gateway,
        preprocessor=preprocessor,
        experiment_name=args.experiment,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
    )

    try:
        study = workflow.optimize(n_trials=args.n_trials, continue_study=args.continue_study)
        print("\n--- OPTUNA STUDY COMPLETED ---")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
        try:
            study = workflow.study
            if study is not None:
                completed = sum(
                    1
                    for trial in study.trials
                    if trial.state.name == "COMPLETE"
                )
                print(
                    f"Letzter Stand: {completed} abgeschlossene Trials | "
                    f"Study Best: {study.best_value:.3f} (Trial {study.best_trial.number})"
                )
        except (AttributeError, ValueError):
            pass
        print("Gateway wird beendet...")
    finally:
        traffic_monitor.stop()
        gateway.stop()
        gateway.cleanup()
        state_logger.close()


if __name__ == "__main__":
    main()
