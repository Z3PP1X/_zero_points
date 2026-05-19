from __future__ import annotations

from dataclasses import dataclass

from feature_layout import FeatureLayout


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


@dataclass(frozen=True)
class GnnPolicySpec:
    architecture: str
    hidden_dim: int
    num_layers: int
    heads: int
    layout: FeatureLayout


@dataclass(frozen=True)
class TrialConfiguration:
    ppo: PpoHyperparameters
    reward: RewardShapingParameters
    policy: GnnPolicySpec
