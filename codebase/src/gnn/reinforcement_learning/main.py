import argparse
from pathlib import Path
import torch

import mlflow

from gnn.reinforcement_learning.gateway.gateway_state_logger import GatewayStateLogger
from gnn.reinforcement_learning.gateway.network_gateway import NetworkGateway
from gnn.reinforcement_learning.gateway.gateway_traffic_monitor import GatewayTrafficMonitor
from gnn.reinforcement_learning.ppo_optuna_workflow import PpoOptunaWorkflow
from gnn.reinforcement_learning.preprocessor import Preprocessor
from gnn.shared.utils.feature_config import validate_positional_supernode_compatibility
from gnn.reinforcement_learning.rl_config import (
    RL_EXPERIMENT_CHOICES,
    add_shared_graph_args,
    load_yaml_config,
    read_rl_settings,
    resolve_rl_features,
    resolve_rl_setting,
)

RECEIVER_PORT = 5650
RESULTS_PORT = 5693
SENDER_PORT = 5651
CONTROL_PORT = 6000


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start GNN RL Pipeline with SB3 & Optuna")
    add_shared_graph_args(parser)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=list(RL_EXPERIMENT_CHOICES),
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument(
        "--timeout-fallback",
        type=float,
        default=None,
        help="Initial timeout in seconds when no roundtrip history is available.",
    )
    parser.add_argument(
        "--timeout-cushion",
        type=float,
        default=None,
        help="Buffer in seconds added to the rolling roundtrip average.",
    )
    parser.add_argument(
        "--timeout-window",
        type=int,
        default=None,
        help="Window size for the rolling roundtrip average.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel Mathematica slots for SB3 training (shared send/collect per VecEnv step).",
    )
    parser.add_argument(
        "--continue-study",
        action="store_true",
        help="Set this flag to continue the last not finished study, otherwise start a new one.",
    )
    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    parser = build_argument_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args.config
    settings = read_rl_settings(load_yaml_config(config_path))

    experiment = resolve_rl_setting(args.experiment, settings["experiment"])
    mode = resolve_rl_setting(args.mode, settings["mode"])
    edge_direction = resolve_rl_setting(args.edge_direction, settings["edge_direction"])
    add_kappa = resolve_rl_setting(
        None, settings["add_kappa"], is_flag=True, flag_set=args.add_kappa
    )
    add_virtual_supernode = resolve_rl_setting(
        None,
        settings["add_virtual_supernode"],
        is_flag=True,
        flag_set=args.add_virtual_supernode,
    )
    timesteps = int(resolve_rl_setting(args.timesteps, settings["timesteps"]))
    n_trials = int(resolve_rl_setting(args.n_trials, settings["n_trials"]))
    n_envs = int(resolve_rl_setting(args.n_envs, settings["n_envs"]))
    continue_study = resolve_rl_setting(
        None,
        settings["continue_study"],
        is_flag=True,
        flag_set=args.continue_study,
    )
    timeout_fallback = float(
        resolve_rl_setting(args.timeout_fallback, settings["timeout_fallback"])
    )
    timeout_cushion = float(
        resolve_rl_setting(args.timeout_cushion, settings["timeout_cushion"])
    )
    timeout_window = int(
        resolve_rl_setting(args.timeout_window, settings["timeout_window"])
    )

    feature_selection, active_features = resolve_rl_features(
        load_yaml_config(config_path).get("experiment") or {},
        feature_groups=args.feature_groups,
        node_features=args.node_features,
        topology_features=args.topology_features,
        positional_encoding=args.positional_encoding,
        edge_features=args.edge_features,
        active_features=args.active_features,
    )

    # Anchor positional encoding and the fully-connected supernode are mutually exclusive.
    validate_positional_supernode_compatibility(feature_selection, add_virtual_supernode)

    mlflow.set_experiment(f"GNN_RL_Optuna_{experiment}")

    traffic_monitor = GatewayTrafficMonitor(
        timeout_fallback_s=timeout_fallback,
        timeout_cushion_s=timeout_cushion,
        timeout_window_size=timeout_window,
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
        f"Optuna: {n_trials} trials × {timesteps} steps | "
        f"Experiment: {experiment} | Mode: {mode} | Edge direction: {edge_direction} | "
        f"Add kappa: {add_kappa} | Add supernode: {add_virtual_supernode} | "
        f"Parallel envs: {n_envs} | Continue study: {continue_study} | Config: {config_path.name}"
    )
    print(f"Feature groups: {feature_selection.enabled_groups()}")
    print(f"Positional encodings: {list(feature_selection.positional_encodings)}")
    print(f"Active node features: {feature_selection.summary()}")

    from gnn.shared.utils.unified_loader import UnifiedDataLoader

    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=experiment,
        mode=mode,
        edge_direction=edge_direction,
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
    )
    loader = unified_loader.graph_loader

    preprocessor = Preprocessor(loader=loader, mode=mode, active_features=active_features)
    print(
        f"Graph templates: {len(preprocessor.known_problem_ids)} problem IDs indexed, "
        f"lazy LRU-cache active (mode: {mode})"
    )
    gateway.init()
    traffic_monitor.start()

    workflow = PpoOptunaWorkflow(
        gateway=gateway,
        preprocessor=preprocessor,
        experiment_name=experiment,
        timesteps_per_trial=timesteps,
        n_envs=n_envs,
    )

    try:
        study = workflow.optimize(n_trials=n_trials, continue_study=continue_study)
        print("\n--- OPTUNA STUDY COMPLETED ---")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        try:
            study = workflow.study
            if study is not None:
                completed = sum(
                    1
                    for trial in study.trials
                    if trial.state.name == "COMPLETE"
                )
                print(
                    f"Last state: {completed} completed trials | "
                    f"Study best: {study.best_value:.3f} (trial {study.best_trial.number})"
                )
        except (AttributeError, ValueError):
            pass
        print("Shutting down gateway...")
    finally:
        traffic_monitor.stop()
        gateway.stop()
        gateway.cleanup()
        state_logger.close()


if __name__ == "__main__":
    main()
