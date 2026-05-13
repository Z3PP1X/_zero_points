import argparse
import os

import mlflow

from network_gateway import NetworkGateway
from gateway_traffic_monitor import GatewayTrafficMonitor
from ppo_optuna_workflow import PpoOptunaWorkflow
from preprocessor import Preprocessor

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
        default=2.0,
        help="Puffer in Sekunden auf den gleitenden Roundtrip-Durchschnitt.",
    )
    parser.add_argument(
        "--timeout-window",
        type=int,
        default=100,
        help="Anzahl erfolgreicher Roundtrips für den gleitenden Durchschnitt.",
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
    gateway = NetworkGateway(
        receiver_port=RECEIVER_PORT,
        sender_port=SENDER_PORT,
        reward_port=RESULTS_PORT,
        control_port=CONTROL_PORT,
        traffic_monitor=traffic_monitor,
    )
    graphs_path = os.path.join("graphs", args.experiment)
    print(f"Starte Pipeline mit Graphen aus: {graphs_path}")
    print(
        f"Optuna: {args.n_trials} Trials × {args.timesteps} Schritte | "
        f"Experiment: {args.experiment}"
    )
    preprocessor = Preprocessor(graphs_dir=graphs_path)
    print(
        f"Graph-Templates: {len(preprocessor.known_problem_ids)} Problem-IDs indexiert, "
        f"lazy LRU-Cache aktiv"
    )
    gateway.init()
    traffic_monitor.start()

    workflow = PpoOptunaWorkflow(
        gateway=gateway,
        preprocessor=preprocessor,
        experiment_name=args.experiment,
        timesteps_per_trial=args.timesteps,
    )

    try:
        study = workflow.optimize(n_trials=args.n_trials)
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


if __name__ == "__main__":
    main()
