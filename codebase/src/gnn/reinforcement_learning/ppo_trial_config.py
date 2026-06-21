from __future__ import annotations

from dataclasses import dataclass

from gnn.reinforcement_learning.feature_layout import FeatureLayout


@dataclass(frozen=True)
class PpoHyperparameters:
    learning_rate: float
    gamma: float
    ent_coef: float
    n_steps: int
    random_seed: int


@dataclass(frozen=True)
class RewardShapingParameters:
    alpha: float
    basis_reward: float
    reward_gamma: float
    step_cost_lambda: float
    time_bad_penalty: float
    solver_mismatch_penalty: float
    solver_match_bonus: float
    solver_wrong_slow_coef: float
    time_tolerance: float = 0.03


@dataclass(frozen=True)
class GnnPolicySpec:
    activation: str
    hidden_dim: int
    num_layers: int
    layout: FeatureLayout


@dataclass(frozen=True)
class TrialConfiguration:
    ppo: PpoHyperparameters
    reward: RewardShapingParameters
    policy: GnnPolicySpec
