"""
SAC-based entry point for the GNN-RL Mathematica pipeline.

Replaces the synchronous PPO + VecEnv approach (``main.py``) with an
off-policy SAC trainer that naturally supports asynchronous data
collection.  The ``AsyncMathematicaEnv`` never blocks on a full batch;
it processes one UUID at a time while caching incoming states from
Mathematica for subsequent episodes.

Usage (drop-in replacement for the PPO command)::

    rm optuna_nur_f.db && python3 main_sac.py \\
        --experiment nur_f \\
        --n_trials 20 \\
        --timesteps 10000
"""
import os
import argparse
import random
import logging

import mlflow
import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner

from network_gateway import NetworkGateway
from preprocessor import Preprocessor
from gnn_architectures import ARCHITECTURE_NAMES, build_gnn, maybe_torch_compile

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from async_collector import AsyncMathematicaEnv
from sb3_extractor import CustomGNNFeaturesExtractor
from reward import RewardCalculator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Optuna Callback (reused pattern from main.py)
# ------------------------------------------------------------------ #
class SACOptunaCallback(BaseCallback):
    """
    SB3 Callback that reports metrics to Optuna and prints live
    training statistics to the console.

    Args:
        trial: The current Optuna trial.
        study: The Optuna study (used to show best-so-far).
        check_freq: How often (in timesteps) to print and report.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        study: optuna.Study,
        check_freq: int = 500,
    ):
        super().__init__()
        self.trial = trial
        self.study = study
        self.check_freq = check_freq
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        ep_buf = self.model.ep_info_buffer
        n_episodes = len(ep_buf)

        if n_episodes > 0:
            rewards = [ep["r"] for ep in ep_buf]
            mean_reward = np.mean(rewards)
            best_ep_reward = np.max(rewards)
            worst_ep_reward = np.min(rewards)

            self.trial.report(mean_reward, self.num_timesteps)

            try:
                study_best = self.study.best_value
            except ValueError:
                study_best = float("nan")

            print(
                f"  [Trial {self.trial.number:>3}] "
                f"Steps: {self.num_timesteps:>6} | "
                f"Episodes: {n_episodes:>3} | "
                f"Mean R: {mean_reward:>8.3f} | "
                f"Best Ep: {best_ep_reward:>8.3f} | "
                f"Worst Ep: {worst_ep_reward:>8.3f} | "
                f"Study Best: {study_best:>8.3f}"
            )

            if self.trial.should_prune():
                print(
                    f"  [Trial {self.trial.number}] "
                    f"PRUNED at step {self.num_timesteps}"
                )
                self.is_pruned = True
                return False
        else:
            print(
                f"  [Trial {self.trial.number:>3}] "
                f"Steps: {self.num_timesteps:>6} | "
                f"No episodes completed yet..."
            )
        return True


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(
    description="Start GNN RL Pipeline with SB3 SAC & Optuna (async)"
)
parser.add_argument(
    "--experiment",
    type=str,
    default="nur_f",
    choices=["nur_f", "f_fp_roh", "kein_inv"],
)
parser.add_argument(
    "--timesteps",
    type=int,
    default=10_000,
    help="Timesteps per trial",
)
parser.add_argument(
    "--n_trials",
    type=int,
    default=50,
    help="Number of Optuna trials",
)
parser.add_argument(
    "--no-torch-compile",
    action="store_true",
    help="Disable torch.compile on the GNN.",
)
parser.add_argument(
    "--learning_starts",
    type=int,
    default=None,
    help=(
        "Override SAC learning_starts (default: SAC built-in 100). "
        "Steps of random exploration before training begins."
    ),
)
args = parser.parse_args()

# ------------------------------------------------------------------ #
# Network & Preprocessor
# ------------------------------------------------------------------ #
RECEIVER_PORT = 5650
RESULTS_PORT = 5693
SENDER_PORT = 5651
CONTROL_PORT = 6000

mlflow.set_experiment(f"GNN_RL_SAC_{args.experiment}")

pipeline = NetworkGateway(
    receiver_port=RECEIVER_PORT,
    sender_port=SENDER_PORT,
    reward_port=RESULTS_PORT,
    control_port=CONTROL_PORT,
)

graphs_path = os.path.join("graphs", args.experiment)
print(f"Starte SAC-Pipeline mit Graphen aus: {graphs_path}")
preprocessor = Preprocessor(graphs_dir=graphs_path)
pipeline.init()

# Will be set in __main__ before study.optimize()
study = None


# ------------------------------------------------------------------ #
# Objective
# ------------------------------------------------------------------ #
def objective(trial):
    """Optuna objective: one SAC training run."""
    print(f"\n--- Starting SAC Trial {trial.number} ---")

    # 1. Seed
    random_seed = trial.suggest_int("random_seed", 0, 99_999)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # 2. SAC Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", "0.1", "0.01"])
    # SB3 SAC accepts "auto" or a float for ent_coef
    ent_coef_value = ent_coef if ent_coef == "auto" else float(ent_coef)

    learning_starts = trial.suggest_int("learning_starts", 50, 500)
    if args.learning_starts is not None:
        learning_starts = args.learning_starts

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # 3. GNN Hyperparameters
    gnn_architecture = trial.suggest_categorical(
        "gnn_architecture", ARCHITECTURE_NAMES,
    )
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    heads = trial.suggest_categorical("heads", [2, 4, 8])

    print(
        f"  Trial {trial.number}: arch={gnn_architecture} "
        f"seed={random_seed} lr={learning_rate:.2e} "
        f"tau={tau:.4f} ent={ent_coef}"
    )

    # 4. Reward Shaping
    alpha = trial.suggest_float("alpha", 0.1, 5.0)
    basis_reward = trial.suggest_float("basis_reward", 0.1, 2.0)
    reward_gamma = trial.suggest_float("reward_gamma", 0.9, 0.999)

    reward_calc = RewardCalculator(
        basis_reward=basis_reward,
        gamma=reward_gamma,
        alpha=alpha,
    )

    # 5. Environment
    base_env = AsyncMathematicaEnv(
        gateway=pipeline,
        preprocessor=preprocessor,
        reward_calculator=reward_calc,
        max_nodes=200,
        max_edges=1000,
    )
    env = Monitor(base_env)

    # 6. GNN Model
    gnn_model = build_gnn(
        gnn_architecture,
        input_dim=5,
        hidden_dim=hidden_dim,
        global_dim=9,
        heads=heads,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomGNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            gnn_model=gnn_model, features_dim=hidden_dim,
        ),
    )

    # 7. SAC Model
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef_value,
        batch_size=batch_size,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=random_seed,
    )

    if not args.no_torch_compile:
        fe = model.policy.features_extractor
        fe.gnn = maybe_torch_compile(fe.gnn, enabled=True)

    # 8. Training
    optuna_callback = SACOptunaCallback(trial, study, check_freq=500)

    with mlflow.start_run(run_name=f"SAC_Trial_{trial.number}"):
        mlflow.log_params(trial.params)

        model.learn(
            total_timesteps=args.timesteps,
            callback=optuna_callback,
        )

        # 9. Cleanup unfinished episode
        unwrapped = env.unwrapped
        if (
            unwrapped.current_uuid
            and unwrapped.episode_buffer.has_episode(unwrapped.current_uuid)
        ):
            unwrapped.episode_buffer.clear_episode(unwrapped.current_uuid)

        # 10. Handle pruning
        if optuna_callback.is_pruned:
            mlflow.set_tag("status", "pruned")
            raise optuna.TrialPruned()

        # 11. Final metric
        if len(model.ep_info_buffer) > 0:
            final_mean_reward = np.mean(
                [ep["r"] for ep in model.ep_info_buffer]
            )
        else:
            final_mean_reward = -float("inf")

        mlflow.log_metric("final_mean_reward", final_mean_reward)
        mlflow.set_tag("status", "completed")

        return final_mean_reward


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    try:
        study_name = f"gnn_rl_sac_{args.experiment}"
        storage_name = f"sqlite:///optuna_sac_{args.experiment}.db"

        study = optuna.create_study(
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

        # Make study accessible in the objective function
        import __main__
        __main__.study = study

        study.optimize(objective, n_trials=args.n_trials)

        print("\n--- OPTUNA SAC STUDY COMPLETED ---")
        print("Best trial:")
        best = study.best_trial
        print(f"  Value: {best.value}")
        print("  Params: ")
        for key, value in best.params.items():
            print(f"    {key}: {value}")

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer. Gateway wird beendet...")
    finally:
        pipeline.stop()
        pipeline.cleanup()
