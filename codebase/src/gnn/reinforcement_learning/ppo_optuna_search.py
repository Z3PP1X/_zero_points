from __future__ import annotations

import optuna

from gnn.reinforcement_learning.feature_layout import (
    EDGE_INPUT_DIM_CHOICES,
    GAT_HEAD_COUNT_CHOICES,
    GLOBAL_INPUT_DIM_CHOICES,
    GNN_ACTIVATION_CHOICES,
    GNN_ARCHITECTURE_CHOICES,
    GNN_LAYER_COUNT_CHOICES,
    HIDDEN_DIM_CHOICES,
    NODE_INPUT_DIM_CHOICES,
    FeatureLayout,
)
from gnn.reinforcement_learning.ppo_trial_config import (
    GnnPolicySpec,
    PpoHyperparameters,
    RewardShapingParameters,
    TrialConfiguration,
)


def sample_trial_configuration(
    trial: optuna.Trial,
    *,
    target_rollout: int = 2048,
    padded_node_feature_count: int = 25,
    active_feature_names: tuple[str, ...] | None = None,
) -> TrialConfiguration:
    random_seed = trial.suggest_int("random_seed", 0, 99_999)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 9e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    n_steps = target_rollout

    reward = RewardShapingParameters(
        alpha=trial.suggest_float("alpha", 0.1, 5.0),
        basis_reward=trial.suggest_float("basis_reward", 0.1, 2.0),
        reward_gamma=trial.suggest_float("reward_gamma", 0.97, 0.999),
        step_cost_lambda=trial.suggest_float("step_cost_lambda", 1e-4, 0.5, log=True),
        time_bad_penalty=trial.suggest_float("time_bad_penalty", 4, 10),
        solver_mismatch_penalty=trial.suggest_float(
            "solver_mismatch_penalty", 0.30, 0.5
        ),
        solver_match_bonus=trial.suggest_float("solver_match_bonus", 0.0, 0.5),
        solver_wrong_slow_coef=trial.suggest_float("solver_wrong_slow_coef", 1, 2),
        time_tolerance=trial.suggest_float("time_tolerance", 0.0, 0.1),
    )

    policy = GnnPolicySpec(
        architecture=trial.suggest_categorical(
            "gnn_architecture", GNN_ARCHITECTURE_CHOICES
        ),
        activation=trial.suggest_categorical("gnn_activation", GNN_ACTIVATION_CHOICES),
        hidden_dim=trial.suggest_categorical("hidden_dim", HIDDEN_DIM_CHOICES),
        num_layers=trial.suggest_categorical("num_gnn_layers", GNN_LAYER_COUNT_CHOICES),
        heads=trial.suggest_categorical("heads", GAT_HEAD_COUNT_CHOICES),
        layout=FeatureLayout(
            node_input_dim=trial.suggest_categorical(
                "node_input_dim", NODE_INPUT_DIM_CHOICES
            ),
            global_input_dim=trial.suggest_categorical(
                "global_input_dim", GLOBAL_INPUT_DIM_CHOICES
            ),
            edge_input_dim=trial.suggest_categorical(
                "edge_input_dim", EDGE_INPUT_DIM_CHOICES
            ),
            padded_node_feature_count=padded_node_feature_count,
            active_feature_names=active_feature_names,
        ),
    )

    return TrialConfiguration(
        ppo=PpoHyperparameters(
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            random_seed=random_seed,
        ),
        reward=reward,
        policy=policy,
    )
