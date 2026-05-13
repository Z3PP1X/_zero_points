import argparse
import os

import mlflow

from network_gateway import NetworkGateway
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
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    mlflow.set_experiment(f"GNN_RL_Optuna_{args.experiment}")

    gateway = NetworkGateway(
        receiver_port=RECEIVER_PORT,
        sender_port=SENDER_PORT,
        reward_port=RESULTS_PORT,
        control_port=CONTROL_PORT,
    )
    graphs_path = os.path.join("graphs", args.experiment)
    print(f"Starte Pipeline mit Graphen aus: {graphs_path}")
    print(
        f"Optuna: {args.n_trials} Trials × {args.timesteps} Schritte | "
        f"Experiment: {args.experiment}"
    )
    preprocessor = Preprocessor(graphs_dir=graphs_path)
    gateway.init()

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
        gateway.stop()
        gateway.cleanup()


if __name__ == "__main__":
    main()
