from __future__ import annotations

import optuna

from gnn.reinforcement_learning.feature_layout import (
    FIXED_DIM_INNER,
    FIXED_DROPOUT,
    FIXED_GNN_ACTIVATION,
    FIXED_GNN_LAYER_COUNT,
    GLOBAL_INPUT_DIM_CHOICES,
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

    # PBRS reward weights (reward.py). gamma is NOT searched here -- it is tied to
    # the PPO discount (ppo.gamma) for policy invariance. error_ref stays at its
    # fixed default.
    reward = RewardShapingParameters(
        lambda_s=trial.suggest_float("lambda_s", 0.01, 1.0, log=True),
        w_time=trial.suggest_float("w_time", 0.1, 5.0, log=True),
        w_record=trial.suggest_float("w_record", 0.5, 10.0),
        w_over=trial.suggest_float("w_over", 0.5, 10.0),
    )

    # GNN architecture is fixed (not searched): dim_inner, message-passing depth,
    # dropout and activation are constants. edge_input_dim is dropped entirely
    # (GIN convs ignore edge features), so FeatureLayout falls back to its default.
    policy = GnnPolicySpec(
        activation=FIXED_GNN_ACTIVATION,
        hidden_dim=FIXED_DIM_INNER,
        num_layers=FIXED_GNN_LAYER_COUNT,
        dropout=FIXED_DROPOUT,
        layout=FeatureLayout(
            global_input_dim=trial.suggest_categorical(
                "global_input_dim", GLOBAL_INPUT_DIM_CHOICES
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
