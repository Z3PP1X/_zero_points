#!/usr/bin/env python3
"""
train_best.py

Starts a full training run of the GNN-RL pipeline using the best hyperparameters 
discovered during an Optuna study. Loads the parameters from a selected SQLite 
database, sets up the environments, initializes the GNN-PPO model, and starts 
a long training run with MLflow tracking and automated checkpoints.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

import numpy as np
import optuna
import torch
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Dynamic sys.path resolution to support package imports when run as scripts
from pathlib import Path
sys_path_root = Path(__file__).resolve().parents[1]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))
sys_path_src = Path(__file__).resolve().parents[2]
if str(sys_path_src) not in sys.path:
    sys.path.insert(0, str(sys_path_src))

# Pipeline imports
from gnn.reinforcement_learning.gateway.gateway_state_logger import GatewayStateLogger
from gnn.reinforcement_learning.gateway.network_gateway import NetworkGateway, CONTROL_FRESH_TRIAL_ENV
from gnn.reinforcement_learning.gateway.gateway_traffic_monitor import GatewayTrafficMonitor
from gnn.reinforcement_learning.preprocessor import Preprocessor
from gnn.reinforcement_learning.reward import RewardCalculator
from gnn.reinforcement_learning.mathematica_vec_env import build_mathematica_training_env, MathematicaVecEnv
from gnn.shared.models.gnn_backbones import build_graph_policy_backbone
from gnn.reinforcement_learning.sb3_extractor import CustomGNNFeaturesExtractor
from gnn.reinforcement_learning.feature_layout import FeatureLayout, EDGE_INPUT_DIM_CHOICES
from gnn.reinforcement_learning.ppo_trial_config import PpoHyperparameters, RewardShapingParameters, GnnPolicySpec, TrialConfiguration
from gnn.reinforcement_learning.rl_config import (
    RL_EXPERIMENT_CHOICES,
    add_shared_graph_args,
    load_yaml_config,
    read_rl_settings,
    resolve_rl_features,
    resolve_rl_setting,
)

# ZMQ Port configuration
RECEIVER_PORT = 5650
RESULTS_PORT = 5693
SENDER_PORT = 5651
CONTROL_PORT = 6000


class TrainingCallback(BaseCallback):
    """
    Callback for logging detailed metrics to MLflow, printing live updates to the console,
    and saving the best model and regular checkpoints.
    """
    def __init__(
        self,
        check_freq: int = 1000,
        save_path: str = "models",
        model_name: str = "gnn_ppo_model",
        traffic_monitor: Optional[GatewayTrafficMonitor] = None,
    ):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_name = model_name
        self.best_mean_reward = -float("inf")
        self.traffic_monitor = traffic_monitor
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        ep_buf = self.model.ep_info_buffer
        n_episodes = len(ep_buf)

        faster_ratio = None
        overshoot_var = None
        convergence_rate = None
        mean_episode_steps = None
        mean_roundtrip_s = None
        if self.traffic_monitor is not None:
            faster_ratio, overshoot_var = self.traffic_monitor.get_rolling_metrics()
            adv_metrics = self.traffic_monitor.get_advanced_rolling_metrics()
            convergence_rate = adv_metrics.get("convergence_rate")
            mean_episode_steps = adv_metrics.get("mean_episode_steps")
            mean_roundtrip_s = adv_metrics.get("mean_roundtrip_s")

        if n_episodes > 0:
            rewards = [ep["r"] for ep in ep_buf]
            mean_reward = float(np.mean(rewards))
            best_ep_reward = float(np.max(rewards))
            worst_ep_reward = float(np.min(rewards))

            # Log metrics to MLflow
            mlflow.log_metric("mean_episode_reward", mean_reward, step=self.num_timesteps)
            mlflow.log_metric("best_episode_reward", best_ep_reward, step=self.num_timesteps)
            mlflow.log_metric("worst_episode_reward", worst_ep_reward, step=self.num_timesteps)
            if faster_ratio is not None:
                mlflow.log_metric("faster_than_benchmark_ratio", faster_ratio, step=self.num_timesteps)
            if overshoot_var is not None:
                mlflow.log_metric("overshoot_variance", overshoot_var, step=self.num_timesteps)
            if convergence_rate is not None:
                mlflow.log_metric("convergence_rate", convergence_rate, step=self.num_timesteps)
            if mean_episode_steps is not None:
                mlflow.log_metric("mean_episode_steps", mean_episode_steps, step=self.num_timesteps)
            if mean_roundtrip_s is not None:
                mlflow.log_metric("mean_roundtrip_s", mean_roundtrip_s, step=self.num_timesteps)

            faster_text = f"{faster_ratio:.3f}" if faster_ratio is not None else "—"
            overshoot_text = f"{overshoot_var:.3f}" if overshoot_var is not None else "—"
            conv_text = f"{convergence_rate:.3f}" if convergence_rate is not None else "—"
            steps_text = f"{mean_episode_steps:.1f}" if mean_episode_steps is not None else "—"
            latency_text = f"{mean_roundtrip_s:.3f}s" if mean_roundtrip_s is not None else "—"

            print(
                f"[Step {self.num_timesteps:>6}] "
                f"Completed Episodes: {n_episodes:>3} | "
                f"Mean Reward: {mean_reward:>8.3f} | "
                f"Best Ep: {best_ep_reward:>8.3f} | "
                f"Worst Ep: {worst_ep_reward:>8.3f} | "
                f"Faster Ratio: {faster_text} | "
                f"Overshoot Var: {overshoot_text} | "
                f"Conv Rate: {conv_text} | "
                f"Mean Steps: {steps_text} | "
                f"Latency: {latency_text}"
            )

            # Check and save the best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.save_path, f"{self.model_name}_best.zip")
                self.model.save(best_model_path)
                print(f"  ↳ Saved new best model to {best_model_path} (Mean Reward: {mean_reward:.3f})")
                try:
                    mlflow.log_artifact(best_model_path, artifact_path="models")
                except Exception as e:
                    print(f"  ↳ [Callback] Warning: Failed to log best model to MLflow: {e}")
        else:
            print(f"[Step {self.num_timesteps:>6}] No completed episodes in buffer yet...")

        # Save regular checkpoints (e.g. every 10k steps)
        if self.num_timesteps % (self.check_freq * 10) == 0:
            checkpoint_path = os.path.join(self.save_path, f"{self.model_name}_step_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            print(f"  ↳ Saved checkpoint to {checkpoint_path}")
            try:
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            except Exception as e:
                print(f"  ↳ [Callback] Warning: Failed to log checkpoint to MLflow: {e}")

        return True


def load_best_trial_params(db_path: str, study_name: str | None = None) -> dict:
    """Loads the best trial parameters from a specified SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Optuna database file not found: {db_path}")

    storage = f"sqlite:///{db_path}"
    
    if study_name is None:
        # Auto-discover study name
        summaries = optuna.get_all_study_summaries(storage=storage)
        if not summaries:
            raise ValueError(f"No studies found in database {db_path}")
        study_name = summaries[0].study_name
        print(f"[Loader] Auto-selected study: {study_name}")

    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    
    if best_trial is None or best_trial.value is None:
        raise ValueError(f"No completed trials found in study {study_name}")

    print(f"\n[Loader] Successfully loaded study: '{study_name}'")
    print(f"  Best Trial Number: {best_trial.number}")
    print(f"  Best Mean Reward: {best_trial.value:.4f}")
    print("  Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    print("-" * 50)
    
    return best_trial.params


def build_trial_configuration(params: dict, override_seed: int | None = None, padded_node_feature_count: int = 25) -> TrialConfiguration:
    """Constructs a structured TrialConfiguration from flat Optuna params."""
    random_seed = override_seed if override_seed is not None else int(params.get("random_seed", 42))
    
    ppo = PpoHyperparameters(
        learning_rate=float(params["learning_rate"]),
        gamma=float(params.get("gamma", 0.9841845574038558)),
        ent_coef=float(params.get("ent_coef", 5.267281408304776e-06)),
        n_steps=2048,  # Standard rollout length
        random_seed=random_seed,
    )
    
    reward = RewardShapingParameters(
        alpha=float(params["alpha"]),
        basis_reward=float(params["basis_reward"]),
        reward_gamma=float(params["reward_gamma"]),
        step_cost_lambda=float(params["step_cost_lambda"]),
        time_bad_penalty=float(params["time_bad_penalty"]),
        solver_mismatch_penalty=float(params["solver_mismatch_penalty"]),
        solver_match_bonus=float(params["solver_match_bonus"]),
        solver_wrong_slow_coef=float(params["solver_wrong_slow_coef"]),
        time_tolerance=float(params.get("time_tolerance", 0.03)),
    )
    
    layout = FeatureLayout(
        node_input_dim=int(params["node_input_dim"]),
        global_input_dim=int(params["global_input_dim"]),
        edge_input_dim=int(params.get("edge_input_dim", EDGE_INPUT_DIM_CHOICES[0])),
        padded_node_feature_count=padded_node_feature_count,
    )
    
    policy = GnnPolicySpec(
        architecture=params["gnn_architecture"],
        activation=params.get("gnn_activation", "leaky_relu"),
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_gnn_layers"]),
        heads=int(params["heads"]),
        layout=layout,
    )
    
    return TrialConfiguration(ppo=ppo, reward=reward, policy=policy)


def set_random_seeds(random_seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start Standalone GNN RL PPO Training with Best Hyperparameters")
    add_shared_graph_args(parser)
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the SQLite database containing Optuna studies (e.g. optuna_kein_inv_n4g6_20260520_011237.db)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=list(RL_EXPERIMENT_CHOICES),
        help="The experiment name / graph directory to use.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional study name within the DB. If omitted, first study will be loaded.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total environment timesteps for the full training run.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel Mathematica environments.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the checkpoints and the final model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base name for the saved model files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed used during training (otherwise loads the best trial seed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: loads and parses the study/hyperparameters and outputs configuration without starting ZMQ/training.",
    )
    parser.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="Disable torch.compile on the GNN extractor backbone.",
    )
    parser.add_argument(
        "--timeout-fallback",
        type=float,
        default=None,
        help="Initial fallback timeout in seconds.",
    )
    parser.add_argument(
        "--timeout-cushion",
        type=float,
        default=None,
        help="Cushion added to rolling roundtrip average.",
    )
    parser.add_argument(
        "--timeout-window",
        type=int,
        default=None,
        help="Rolling window size for timeout average.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args.config
    settings = read_rl_settings(load_yaml_config(config_path))

    experiment = resolve_rl_setting(args.experiment, settings["experiment"])
    mode = resolve_rl_setting(args.mode, settings["mode"])
    edge_direction = resolve_rl_setting(args.edge_direction, settings["edge_direction"])
    timesteps = int(
        resolve_rl_setting(args.timesteps, settings["train_best_timesteps"])
    )
    n_envs = int(resolve_rl_setting(args.n_envs, settings["train_best_n_envs"]))
    save_dir = str(resolve_rl_setting(args.save_dir, settings["save_dir"]))
    model_name = str(resolve_rl_setting(args.model_name, settings["model_name"]))
    timeout_fallback = float(
        resolve_rl_setting(args.timeout_fallback, settings["train_best_timeout_fallback"])
    )
    timeout_cushion = float(
        resolve_rl_setting(args.timeout_cushion, settings["train_best_timeout_cushion"])
    )
    timeout_window = int(
        resolve_rl_setting(args.timeout_window, settings["train_best_timeout_window"])
    )
    no_torch_compile = resolve_rl_setting(
        None,
        settings["no_torch_compile"],
        is_flag=True,
        flag_set=args.no_torch_compile,
    )

    feature_selection, active_features = resolve_rl_features(
        load_yaml_config(config_path).get("experiment") or {},
        enrich=True,
        feature_groups=args.feature_groups,
        positional_encoding=args.positional_encoding,
        active_features=args.active_features,
    )
    print(f"[Pipeline] Feature groups: {feature_selection.enabled_groups()}")
    print(f"[Pipeline] Positional encodings: {list(feature_selection.positional_encodings)}")
    print(f"[Pipeline] Active node features: {feature_selection.summary(enrich=True)}")

    padded_node_feature_count = len(active_features) if active_features is not None else 25

    # 1. Load and parse hyperparameter configuration
    try:
        best_params = load_best_trial_params(args.db, args.study_name)
        trial_config = build_trial_configuration(best_params, args.seed, padded_node_feature_count)
    except Exception as e:
        print(f"[Error] Failed to load/parse study parameters: {e}")
        sys.exit(1)

    print("\n--- GNN RL RUN CONFIGURATION ---")
    print(f"  Config:           {config_path.name}")
    print(f"  Experiment:       {experiment}")
    print(f"  Mode:             {mode}")
    print(f"  Edge direction:   {edge_direction}")
    print(f"  GNN Architecture: {trial_config.policy.architecture}")
    print(f"  GNN Activation:   {trial_config.policy.activation}")
    print(f"  Hidden Dim:       {trial_config.policy.hidden_dim}")
    print(f"  Num Layers:       {trial_config.policy.num_layers}")
    print(f"  Heads:            {trial_config.policy.heads}")
    print(f"  Learning Rate:    {trial_config.ppo.learning_rate:.2e}")
    print(f"  Gamma (PPO):      {trial_config.ppo.gamma:.4f}")
    print(f"  Entropy Coef:     {trial_config.ppo.ent_coef:.2e}")
    print(f"  Random Seed:      {trial_config.ppo.random_seed}")
    print("-" * 50)

    if args.dry_run:
        print("[Dry Run] Hyperparameters parsed successfully. Exiting without training.")
        sys.exit(0)

    # Set up random seeds
    set_random_seeds(trial_config.ppo.random_seed)

    # 2. Initialize Gateway, Monitor, and Preprocessor
    mlflow.set_experiment(f"GNN_RL_Full_Training_{experiment}")

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
    from gnn.shared.utils.unified_loader import UnifiedDataLoader
    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=experiment,
        mode=mode,
        enrich=True,
        edge_direction=edge_direction,
    )
    loader = unified_loader.graph_loader
    preprocessor = Preprocessor(loader=loader, mode=mode, active_features=active_features)

    print(f"[Pipeline] Initializing ZMQ NetworkGateway on receiver={RECEIVER_PORT}, sender={SENDER_PORT}...")
    gateway.init()
    traffic_monitor.start()

    # Signal Mathematica to prepare a fresh environment stream
    print("[Pipeline] Sending fresh environment control signal...")
    gateway.send_control(CONTROL_FRESH_TRIAL_ENV)

    # Define build components
    reward_calculator = RewardCalculator(
        basis_reward=trial_config.reward.basis_reward,
        gamma=trial_config.reward.reward_gamma,
        alpha=trial_config.reward.alpha,
        time_tolerance=trial_config.reward.time_tolerance,
        step_cost_lambda=trial_config.reward.step_cost_lambda,
        time_bad_penalty=trial_config.reward.time_bad_penalty,
        solver_mismatch_penalty=trial_config.reward.solver_mismatch_penalty,
        solver_match_bonus=trial_config.reward.solver_match_bonus,
        solver_wrong_slow_coef=trial_config.reward.solver_wrong_slow_coef,
    )

    # 3. Create VecEnv
    env = build_mathematica_training_env(
        gateway=gateway,
        preprocessor=preprocessor,
        reward_calculator=reward_calculator,
        n_envs=n_envs,
        max_nodes=200,
        max_edges=1000,
    )

    # 4. Create Custom GNN Policy Backbone
    gnn_model = build_graph_policy_backbone(
        layout=trial_config.policy.layout,
        architecture=trial_config.policy.architecture,
        activation=trial_config.policy.activation,
        hidden_dim=trial_config.policy.hidden_dim,
        num_layers=trial_config.policy.num_layers,
        heads=trial_config.policy.heads,
    )

    # Attempt torch compile if requested
    if not no_torch_compile:
        from gnn.shared.models.gnn_backbones import maybe_torch_compile
        gnn_model = maybe_torch_compile(gnn_model, enabled=True)

    policy_kwargs = {
        "features_extractor_class": CustomGNNFeaturesExtractor,
        "features_extractor_kwargs": {
            "gnn_model": gnn_model,
            "features_dim": trial_config.policy.hidden_dim,
        },
    }

    # 5. Build PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=trial_config.ppo.learning_rate,
        gamma=trial_config.ppo.gamma,
        ent_coef=trial_config.ppo.ent_coef,
        n_steps=trial_config.ppo.n_steps,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=trial_config.ppo.random_seed,
    )

    # Callback and training start
    training_callback = TrainingCallback(
        check_freq=1000,
        save_path=save_dir,
        model_name=model_name,
        traffic_monitor=traffic_monitor,
    )

    print(f"\n[Training] Starting full training of {timesteps} steps...")
    print(f"[Training] Checkpoints and best model will be saved to '{save_dir}/'")

    # Start MLflow run
    with mlflow.start_run(run_name=f"Full_Training_Run_{model_name}"):
        try:
            # Log all parameters to MLflow
            mlflow.log_param("experiment", experiment)
            mlflow.log_param("gnn_architecture", trial_config.policy.architecture)
            mlflow.log_param("gnn_activation", trial_config.policy.activation)
            mlflow.log_param("hidden_dim", trial_config.policy.hidden_dim)
            mlflow.log_param("num_layers", trial_config.policy.num_layers)
            mlflow.log_param("heads", trial_config.policy.heads)
            mlflow.log_param("learning_rate", trial_config.ppo.learning_rate)
            mlflow.log_param("gamma", trial_config.ppo.gamma)
            mlflow.log_param("ent_coef", trial_config.ppo.ent_coef)
            mlflow.log_param("seed", trial_config.ppo.random_seed)
            mlflow.log_param("n_envs", n_envs)
            mlflow.log_param("timesteps", timesteps)
            mlflow.log_param("mode", mode)
            mlflow.log_param("edge_direction", edge_direction)

            # Log reward parameters
            mlflow.log_param("reward_version", "v2_tolerance")
            mlflow.log_param("reward_alpha", trial_config.reward.alpha)
            mlflow.log_param("reward_gamma", trial_config.reward.reward_gamma)
            mlflow.log_param("time_tolerance", trial_config.reward.time_tolerance)
            mlflow.log_param("step_cost_lambda", trial_config.reward.step_cost_lambda)
            mlflow.log_param("time_bad_penalty", trial_config.reward.time_bad_penalty)

            # Start learning
            model.learn(total_timesteps=timesteps, callback=training_callback)

            # Save the final model
            final_path = os.path.join(save_dir, f"{model_name}_final.zip")
            model.save(final_path)
            print(f"\n[Training] Training completed successfully!")
            print(f"[Training] Saved final model to: {final_path}")
            try:
                mlflow.log_artifact(final_path, artifact_path="models")
                print(f"[Training] Logged final model artifact to MLflow.")
            except Exception as e:
                print(f"[Training] Warning: Failed to log final model to MLflow: {e}")

        except KeyboardInterrupt:
            print("\n[Training] Training interrupted by user.")
        finally:
            print("[Pipeline] Flushing open episodes and closing environments...")
            if isinstance(env, MathematicaVecEnv):
                try:
                    env.finalize_open_episodes()
                except Exception as e:
                    print(f"[Pipeline] Error during vec env episode flushing: {e}")
                env.close()

            print("[Pipeline] Stopping traffic monitor and closing ZMQ network gateway...")
            traffic_monitor.stop()
            gateway.stop()
            gateway.cleanup()
            state_logger.close()
            print("[Pipeline] Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
