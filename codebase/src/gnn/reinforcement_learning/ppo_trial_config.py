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
    """Weights for the potential-based (PBRS) reward (see reward.py).

    lambda_s/w_time/w_record/w_over are Optuna-searched; error_ref is a fixed
    structural constant. The PBRS gamma is NOT here -- it is tied to ppo.gamma
    at the RewardCalculator call site for policy invariance.
    """

    lambda_s: float
    w_time: float
    w_record: float
    w_over: float
    error_ref: float = 1.0


@dataclass(frozen=True)
class GnnPolicySpec:
    activation: str
    hidden_dim: int
    num_layers: int
    dropout: float
    layout: FeatureLayout


@dataclass(frozen=True)
class TrialConfiguration:
    ppo: PpoHyperparameters
    reward: RewardShapingParameters
    policy: GnnPolicySpec
