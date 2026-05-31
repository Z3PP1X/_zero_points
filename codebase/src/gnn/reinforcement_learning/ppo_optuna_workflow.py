from __future__ import annotations

import random
from typing import Optional

import mlflow
import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO

from gnn.reinforcement_learning.feature_layout import OPTUNA_SEARCH_SPACE_SUFFIX
from gnn.shared.models.gnn_backbones import build_graph_policy_backbone
from gnn.reinforcement_learning.mathematica_vec_env import MathematicaVecEnv, build_mathematica_training_env
from gnn.reinforcement_learning.gateway.network_gateway import CONTROL_FRESH_TRIAL_ENV, NetworkGateway
from gnn.reinforcement_learning.ppo_optuna_callback import (
    OptunaEpisodeRewardCallback,
    OptunaStudyProgressCallback,
    _format_study_best,
)
from gnn.reinforcement_learning.ppo_optuna_search import sample_trial_configuration
from gnn.reinforcement_learning.ppo_trial_config import TrialConfiguration
from gnn.reinforcement_learning.preprocessor import Preprocessor
from gnn.reinforcement_learning.reward import RewardCalculator
from gnn.reinforcement_learning.sb3_extractor import CustomGNNFeaturesExtractor


class PpoOptunaWorkflow:
    def __init__(
        self,
        *,
        gateway: NetworkGateway,
        preprocessor: Preprocessor,
        experiment_name: str,
        timesteps_per_trial: int,
        max_nodes: int = 200,
        max_edges: int = 1000,
        optuna_check_freq: int = 500,
        n_envs: int = 1,
    ):
        self.gateway = gateway
        self.preprocessor = preprocessor
        self.experiment_name = experiment_name
        self.timesteps_per_trial = timesteps_per_trial
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.optuna_check_freq = optuna_check_freq
        self.n_envs = n_envs
        self.study: Optional[optuna.Study] = None
        self.total_trials = 0

    def create_study(self, continue_study: bool = False) -> optuna.Study:
        study_name = f"gnn_rl_{self.experiment_name}_{OPTUNA_SEARCH_SPACE_SUFFIX}"
        storage_name = (
            f"sqlite:///optuna_{self.experiment_name}_{OPTUNA_SEARCH_SPACE_SUFFIX}.db"
        )
        print(f"[PpoOptunaWorkflow] Optuna study={study_name!r} storage={storage_name}")
        
        if not continue_study:
            try:
                optuna.delete_study(study_name=study_name, storage=storage_name)
                print(f"[PpoOptunaWorkflow] Deleted existing study {study_name!r} to start fresh.")
            except Exception:
                pass

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True,
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2000,
                interval_steps=1000,
            ),
        )
        return self.study

    def objective(self, trial: optuna.Trial) -> float:
        if self.study is None:
            raise RuntimeError("create_study must be called before optimize.")

        trial_index = trial.number + 1
        print(
            f"\n--- Trial {trial.number} startet "
            f"({trial_index}/{self.total_trials}) | "
            f"Study Best: {_format_study_best(self.study)} ---"
        )
        self.gateway.send_control(CONTROL_FRESH_TRIAL_ENV)
        print(
            f"  Pipeline control signal sent "
            f"(control={CONTROL_FRESH_TRIAL_ENV}, fresh trial environment)"
        )
        if self.gateway.traffic_monitor is not None:
            self.gateway.traffic_monitor.reset_reward_state_count()
            print("  Reward-States counter reset to 0.")
        trial_config = sample_trial_configuration(trial)
        print(
            f"  Hyperparameter: lr={trial_config.ppo.learning_rate:.2e}, "
            f"gamma={trial_config.ppo.gamma:.3f}, "
            f"ent_coef={trial_config.ppo.ent_coef:.2e}, "
            f"arch={trial_config.policy.architecture}, "
            f"hidden={trial_config.policy.hidden_dim}"
        )
        self._set_random_seeds(trial_config.ppo.random_seed)

        reward_calculator = self._build_reward_calculator(trial_config)
        env = self._build_training_env(reward_calculator)
        model = self._build_ppo_model(env, trial_config)
        callback = OptunaEpisodeRewardCallback(
            trial,
            self.study,
            total_timesteps=self.timesteps_per_trial,
            check_freq=self.optuna_check_freq,
            traffic_monitor=self.gateway.traffic_monitor,
        )

        with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
            try:
                mlflow.log_params(trial.params)
                mlflow.log_param("reward_version", "v2_tolerance")
                model.learn(total_timesteps=self.timesteps_per_trial, callback=callback)
                self._finalize_episode_state(env)
                env.close()

                if callback.is_pruned:
                    mlflow.set_tag("status", "pruned")
                    raise optuna.TrialPruned()

                final_mean_reward = self._mean_episode_reward(model)
                mlflow.log_metric("final_mean_reward", final_mean_reward)
                mlflow.set_tag("status", "completed")
                return final_mean_reward
            finally:
                if self.gateway.state_logger is not None:
                    try:
                        mlflow.log_artifact(
                            str(self.gateway.state_logger.log_path),
                            artifact_path="states",
                        )
                        print(
                            "[PpoOptunaWorkflow] Successfully logged "
                            f"{self.gateway.state_logger.log_path.name} "
                            f"to Trial {trial.number} MLflow run."
                        )
                    except Exception as e:
                        print(
                            "[PpoOptunaWorkflow] Warning: Failed to log state database "
                            f"to Trial {trial.number} MLflow: {e}"
                        )

    def optimize(self, n_trials: int, continue_study: bool = False) -> optuna.Study:
        self.total_trials = n_trials
        study = self.create_study(continue_study=continue_study)
        with mlflow.start_run(run_name=f"Optuna_Study_{self.experiment_name}"):
            try:
                study.optimize(
                    self.objective,
                    n_trials=n_trials,
                    callbacks=[OptunaStudyProgressCallback(total_trials=n_trials)],
                )
            finally:
                if self.gateway.state_logger is not None:
                    try:
                        mlflow.log_artifact(
                            str(self.gateway.state_logger.log_path),
                            artifact_path="states",
                        )
                        print(
                            "[PpoOptunaWorkflow] Successfully logged "
                            f"{self.gateway.state_logger.log_path.name} to study MLflow"
                            " run."
                        )
                    except Exception as e:
                        print(
                            "[PpoOptunaWorkflow] Warning: Failed to log state "
                            f"database to study MLflow: {e}"
                        )
        return study

    @staticmethod
    def _set_random_seeds(random_seed: int) -> None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    def _build_reward_calculator(
        self, trial_config: TrialConfiguration
    ) -> RewardCalculator:
        reward = trial_config.reward
        return RewardCalculator(
            basis_reward=reward.basis_reward,
            gamma=reward.reward_gamma,
            alpha=reward.alpha,
            time_tolerance=reward.time_tolerance,
            step_cost_lambda=reward.step_cost_lambda,
            time_bad_penalty=reward.time_bad_penalty,
            solver_mismatch_penalty=reward.solver_mismatch_penalty,
            solver_match_bonus=reward.solver_match_bonus,
            solver_wrong_slow_coef=reward.solver_wrong_slow_coef,
        )

    def _build_training_env(self, reward_calculator: RewardCalculator):
        return build_mathematica_training_env(
            gateway=self.gateway,
            preprocessor=self.preprocessor,
            reward_calculator=reward_calculator,
            n_envs=self.n_envs,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )

    def _build_ppo_model(self, env, trial_config: TrialConfiguration) -> PPO:
        policy = trial_config.policy
        ppo = trial_config.ppo
        gnn_model = build_graph_policy_backbone(
            layout=policy.layout,
            architecture=policy.architecture,
            activation=policy.activation,
            hidden_dim=policy.hidden_dim,
            num_layers=policy.num_layers,
            heads=policy.heads,
        )
        policy_kwargs = {
            "features_extractor_class": CustomGNNFeaturesExtractor,
            "features_extractor_kwargs": {
                "gnn_model": gnn_model,
                "features_dim": policy.hidden_dim,
            },
        }
        return PPO(
            "MultiInputPolicy",
            env,
            learning_rate=ppo.learning_rate,
            gamma=ppo.gamma,
            ent_coef=ppo.ent_coef,
            n_steps=ppo.n_steps,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=ppo.random_seed,
        )

    def _finalize_episode_state(self, env) -> None:
        if isinstance(env, MathematicaVecEnv):
            print("[PpoOptunaWorkflow] Flushing unfinished Mathematica episodes...")
            env.finalize_open_episodes()
            return

        self._finalize_single_env_episode(env)

    def _finalize_single_env_episode(self, env) -> None:
        unwrapped_env = env.unwrapped
        if (
            unwrapped_env.current_uuid is None
            or not unwrapped_env.replay_buffer.has_episode(unwrapped_env.current_uuid)
        ):
            return

        print("[PpoOptunaWorkflow] Flushing unfinished Mathematica episode...")
        unwrapped_env.drain_buffered_states()
        max_flush_steps = 128
        flush_steps = 0
        while (
            unwrapped_env.replay_buffer.has_episode(unwrapped_env.current_uuid)
            and flush_steps < max_flush_steps
        ):
            unwrapped_env.drain_buffered_states()
            if not unwrapped_env.replay_buffer.has_episode(unwrapped_env.current_uuid):
                return

            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            flush_steps += 1
            if terminated or truncated:
                return

        if unwrapped_env.replay_buffer.has_episode(unwrapped_env.current_uuid):
            print(
                "[PpoOptunaWorkflow] Could not finish Mathematica episode after "
                f"{flush_steps} flush steps; clearing replay buffer."
            )
            unwrapped_env.replay_buffer.clear_episode(unwrapped_env.current_uuid)

    @staticmethod
    def _mean_episode_reward(model: PPO) -> float:
        if len(model.ep_info_buffer) == 0:
            return -float("inf")
        return float(np.mean([episode["r"] for episode in model.ep_info_buffer]))
