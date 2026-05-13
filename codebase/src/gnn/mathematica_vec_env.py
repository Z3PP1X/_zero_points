from __future__ import annotations

from collections.abc import Callable, Sequence

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mathematica_env import MathematicaGraphEnv
from mathematica_state_ingress import MathematicaStateIngress
from reward import RewardCalculator


def build_mathematica_training_env(
    *,
    gateway,
    preprocessor,
    reward_calculator: RewardCalculator,
    n_envs: int,
    max_nodes: int,
    max_edges: int,
):
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1.")

    state_ingress = MathematicaStateIngress(gateway)

    def make_env() -> Callable[[], Monitor]:
        def _init() -> Monitor:
            base_env = MathematicaGraphEnv(
                gateway=gateway,
                preprocessor=preprocessor,
                reward_calculator=reward_calculator,
                max_nodes=max_nodes,
                max_edges=max_edges,
                state_ingress=state_ingress,
            )
            return Monitor(base_env)

        return _init

    if n_envs == 1:
        return make_env()()

    return DummyVecEnv([make_env() for _ in range(n_envs)])


def iter_monitored_envs(env) -> Sequence[Monitor]:
    if hasattr(env, "envs"):
        return env.envs
    return [env]
