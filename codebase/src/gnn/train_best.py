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
import numpy as np
import optuna
import torch
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Pipeline imports
from gateway_state_logger import GatewayStateLogger
from network_gateway import NetworkGateway, CONTROL_FRESH_TRIAL_ENV
from gateway_traffic_monitor import GatewayTrafficMonitor
from preprocessor import Preprocessor
from reward import RewardCalculator
from mathematica_vec_env import build_mathematica_training_env, MathematicaVecEnv
from gnn_policy_backbone import build_graph_policy_backbone
from sb3_extractor import CustomGNNFeaturesExtractor
from feature_layout import FeatureLayout
from ppo_trial_config import PpoHyperparameters, RewardShapingParameters, GnnPolicySpec, TrialConfiguration

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
        model_name: str = "gnn_ppo_model"
    ):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_name = model_name
        self.best_mean_reward = -float("inf")
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        ep_buf = self.model.ep_info_buffer
        n_episodes = len(ep_buf)

        if n_episodes > 0:
            rewards = [ep["r"] for ep in ep_buf]
            mean_reward = float(np.mean(rewards))
            best_ep_reward = float(np.max(rewards))
            worst_ep_reward = float(np.min(rewards))

            # Log metrics to MLflow
            mlflow.log_metric("mean_episode_reward", mean_reward, step=self.num_timesteps)
            mlflow.log_metric("best_episode_reward", best_ep_reward, step=self.num_timesteps)
            mlflow.log_metric("worst_episode_reward", worst_ep_reward, step=self.num_timesteps)

            print(
                f"[Step {self.num_timesteps:>6}] "
                f"Completed Episodes: {n_episodes:>3} | "
                f"Mean Reward: {mean_reward:>8.3f} | "
                f"Best Ep: {best_ep_reward:>8.3f} | "
                f"Worst Ep: {worst_ep_reward:>8.3f}"
            )

            # Check and save the best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.save_path, f"{self.model_name}_best.zip")
                self.model.save(best_model_path)
                print(f"  ↳ Saved new best model to {best_model_path} (Mean Reward: {mean_reward:.3f})")
        else:
            print(f"[Step {self.num_timesteps:>6}] No completed episodes in buffer yet...")

        # Save regular checkpoints (e.g. every 10k steps)
        if self.num_timesteps % (self.check_freq * 10) == 0:
            checkpoint_path = os.path.join(self.save_path, f"{self.model_name}_step_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            print(f"  ↳ Saved checkpoint to {checkpoint_path}")

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


def build_trial_configuration(params: dict, override_seed: int | None = None) -> TrialConfiguration:
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
    )
    
    layout = FeatureLayout(
        node_input_dim=int(params["node_input_dim"]),
        global_input_dim=int(params["global_input_dim"]),
    )
    
    policy = GnnPolicySpec(
        architecture=params["gnn_architecture"],
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
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the SQLite database containing Optuna studies (e.g. optuna_kein_inv_n4g6_20260520_011237.db)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="kein_inv",
        choices=["nur_f", "f_fp_roh", "kein_inv"],
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
        default=250000,
        help="Total environment timesteps for the full training run.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel Mathematica environments.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save the checkpoints and the final model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gnn_ppo_best",
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
        default=5.0,
        help="Initial fallback timeout in seconds.",
    )
    parser.add_argument(
        "--timeout-cushion",
        type=float,
        default=2.0,
        help="Cushion added to rolling roundtrip average.",
    )
    parser.add_argument(
        "--timeout-window",
        type=int,
        default=100,
        help="Rolling window size for timeout average.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    # 1. Load and parse hyperparameter configuration
    try:
        best_params = load_best_trial_params(args.db, args.study_name)
        trial_config = build_trial_configuration(best_params, args.seed)
    except Exception as e:
        print(f"[Error] Failed to load/parse study parameters: {e}")
        sys.exit(1)

    print("\n--- GNN RL RUN CONFIGURATION ---")
    print(f"  Experiment:       {args.experiment}")
    print(f"  GNN Architecture: {trial_config.policy.architecture}")
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
    mlflow.set_experiment(f"GNN_RL_Full_Training_{args.experiment}")

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

    graphs_path = os.path.join("graphs", args.experiment)
    preprocessor = Preprocessor(graphs_dir=graphs_path)

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
        n_envs=args.n_envs,
        max_nodes=200,
        max_edges=1000,
    )

    # 4. Create Custom GNN Policy Backbone
    gnn_model = build_graph_policy_backbone(
        layout=trial_config.policy.layout,
        architecture=trial_config.policy.architecture,
        hidden_dim=trial_config.policy.hidden_dim,
        num_layers=trial_config.policy.num_layers,
        heads=trial_config.policy.heads,
    )

    # Attempt torch compile if requested
    if not args.no_torch_compile:
        from gnn_architectures import maybe_torch_compile
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
        save_path=args.save_dir,
        model_name=args.model_name
    )

    print(f"\n[Training] Starting full training of {args.timesteps} steps...")
    print(f"[Training] Checkpoints and best model will be saved to '{args.save_dir}/'")

    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"Full_Training_Run_{args.model_name}"):
            # Log all parameters to MLflow
            mlflow.log_param("experiment", args.experiment)
            mlflow.log_param("gnn_architecture", trial_config.policy.architecture)
            mlflow.log_param("hidden_dim", trial_config.policy.hidden_dim)
            mlflow.log_param("num_layers", trial_config.policy.num_layers)
            mlflow.log_param("heads", trial_config.policy.heads)
            mlflow.log_param("learning_rate", trial_config.ppo.learning_rate)
            mlflow.log_param("gamma", trial_config.ppo.gamma)
            mlflow.log_param("ent_coef", trial_config.ppo.ent_coef)
            mlflow.log_param("seed", trial_config.ppo.random_seed)
            mlflow.log_param("n_envs", args.n_envs)
            mlflow.log_param("timesteps", args.timesteps)

            # Start learning
            model.learn(total_timesteps=args.timesteps, callback=training_callback)

            # Save the final model
            final_path = os.path.join(args.save_dir, f"{args.model_name}_final.zip")
            model.save(final_path)
            print(f"\n[Training] Training completed successfully!")
            print(f"[Training] Saved final model to: {final_path}")

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
