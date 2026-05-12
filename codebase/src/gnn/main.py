import os
import argparse
import random
import mlflow
import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner

from network_gateway import NetworkGateway
from preprocessor import Preprocessor
from gnn_architectures import ARCHITECTURE_NAMES, build_gnn, maybe_torch_compile

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from mathematica_env import MathematicaGraphEnv
from sb3_extractor import CustomGNNFeaturesExtractor
from reward import RewardCalculator

# --- Custom Optuna Callback ---
class CustomOptunaCallback(BaseCallback):
    """
    SB3 Callback that reports metrics to Optuna and prints live
    training statistics to the console.

    Args:
        trial: The current Optuna trial.
        study: The Optuna study (used to show best-so-far).
        check_freq: How often (in timesteps) to print and report.
    """
    def __init__(self, trial: optuna.Trial, study: optuna.Study, check_freq: int = 500):
        super().__init__()
        self.trial = trial
        self.study = study
        self.check_freq = check_freq
        self.is_pruned = False
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            ep_buf = self.model.ep_info_buffer
            n_episodes = len(ep_buf)

            if n_episodes > 0:
                rewards = [ep["r"] for ep in ep_buf]
                mean_reward = np.mean(rewards)
                best_ep_reward = np.max(rewards)
                worst_ep_reward = np.min(rewards)

                # Report to Optuna
                self.trial.report(mean_reward, self.num_timesteps)

                # Best study value so far
                try:
                    study_best = self.study.best_value
                except ValueError:
                    study_best = float('nan')

                print(
                    f"  [Trial {self.trial.number:>3}] "
                    f"Steps: {self.num_timesteps:>6} | "
                    f"Episodes: {n_episodes:>3} | "
                    f"Mean R: {mean_reward:>8.3f} | "
                    f"Best Ep: {best_ep_reward:>8.3f} | "
                    f"Worst Ep: {worst_ep_reward:>8.3f} | "
                    f"Study Best: {study_best:>8.3f}"
                )

                # Check for pruning
                if self.trial.should_prune():
                    print(f"  [Trial {self.trial.number}] PRUNED at step {self.num_timesteps}")
                    self.is_pruned = True
                    return False  # Stops training gracefully
            else:
                print(
                    f"  [Trial {self.trial.number:>3}] "
                    f"Steps: {self.num_timesteps:>6} | "
                    f"No episodes completed yet..."
                )
        return True

# --- Globals and Argument Parsing ---
parser = argparse.ArgumentParser(description="Start GNN RL Pipeline with SB3 & Optuna")
parser.add_argument("--experiment", type=str, default="nur_f", choices=["nur_f", "f_fp_roh", "kein_inv"])
parser.add_argument("--timesteps", type=int, default=10000, help="Timesteps per trial")
parser.add_argument("--n_trials", type=int, default=50, help="Number of optuna trials")
parser.add_argument(
    "--no-torch-compile",
    action="store_true",
    help="Disable torch.compile on the GNN (default: try compile with dynamic shapes, fall back on error).",
)
args = parser.parse_args()

RECEIVER_PORT = 5650
RESULTS_PORT = 5693
SENDER_PORT = 5651
CONTROL_PORT = 6000

# Setup MLflow
mlflow.set_experiment(f"GNN_RL_Optuna_{args.experiment}")

# Initialisierung aller Pipeline-Komponenten
pipeline = NetworkGateway(
    receiver_port=RECEIVER_PORT, 
    sender_port=SENDER_PORT, 
    reward_port=RESULTS_PORT, 
    control_port=CONTROL_PORT
)

graphs_path = os.path.join("graphs", args.experiment)
print(f"Starte Pipeline mit Graphen aus: {graphs_path}")
preprocessor = Preprocessor(graphs_dir=graphs_path) 
pipeline.init()

# Will be set in __main__ before study.optimize()
study = None

def objective(trial):
    print(f"\n--- Starting Trial {trial.number} ---")

    # 1. Sample Hyperparameters
    random_seed = trial.suggest_int("random_seed", 0, 99_999)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # PPO Parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)

    # GNN Parameters
    gnn_architecture = trial.suggest_categorical("gnn_architecture", ARCHITECTURE_NAMES)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    heads = trial.suggest_categorical("heads", [2, 4, 8])

    print(
        f"  Trial {trial.number}: arch={gnn_architecture} seed={random_seed} "
        f"torch_compile={not args.no_torch_compile}"
    )
    # Reward Shaping Parameters
    alpha = trial.suggest_float("alpha", 0.1, 5.0)
    basis_reward = trial.suggest_float("basis_reward", 0.1, 2.0)
    reward_gamma = trial.suggest_float("reward_gamma", 0.9, 0.999)
    
    # 2. RewardCalculator und Environment initialisieren
    reward_calc = RewardCalculator(
        basis_reward=basis_reward,
        gamma=reward_gamma,
        alpha=alpha
    )
    
    base_env = MathematicaGraphEnv(
        gateway=pipeline, 
        preprocessor=preprocessor, 
        reward_calculator=reward_calc,
        max_nodes=200,   
        max_edges=1000
    )
    # Wrap in Monitor to track episode rewards in model.ep_info_buffer
    env = Monitor(base_env)
    
    # 3. GNN Modell initialisieren
    gnn_model = build_gnn(
        gnn_architecture,
        input_dim=5,
        hidden_dim=hidden_dim,
        global_dim=9,
        heads=heads,
    )
    
    policy_kwargs = dict(
        features_extractor_class=CustomGNNFeaturesExtractor,
        features_extractor_kwargs=dict(gnn_model=gnn_model, features_dim=hidden_dim)
    )
    
    # 4. PPO Modell erstellen
    model = PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs, 
        verbose=0,
        seed=random_seed,
    )

    if not args.no_torch_compile:
        fe = model.policy.features_extractor
        fe.gnn = maybe_torch_compile(fe.gnn, enabled=True)
    
    # 5. Training
    optuna_callback = CustomOptunaCallback(trial, study, check_freq=500)
    
    with mlflow.start_run(run_name=f"Trial_{trial.number}"):
        mlflow.log_params(trial.params)
        
        # Train
        model.learn(total_timesteps=args.timesteps, callback=optuna_callback)
        
        # 6. Handle Episode State after training (or pruning)
        # Check if we need to flush the environment (Mathematica blocked mid-episode)
        unwrapped_env = env.unwrapped
        if unwrapped_env.current_state_dict is not None:
            status = unwrapped_env.current_state_dict.get("status")
            if status not in ["reward_calc", "finished"]:
                print("[Main] Flushing unfinished Mathematica episode...")
                while unwrapped_env.current_state_dict.get("status") not in ["reward_calc", "finished"]:
                    obs, reward, term, trunc, info = env.step(env.action_space.sample())
                    if term or trunc:
                        break
        
        # Clean up any leftover replay buffer entries
        if unwrapped_env.current_uuid and unwrapped_env.replay_buffer.has_episode(unwrapped_env.current_uuid):
            unwrapped_env.replay_buffer.clear_episode(unwrapped_env.current_uuid)
        
        # Handle Pruning
        if optuna_callback.is_pruned:
            mlflow.set_tag("status", "pruned")
            raise optuna.TrialPruned()
        
        # Calculate Final Metric
        if len(model.ep_info_buffer) > 0:
            final_mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])
        else:
            final_mean_reward = -float('inf')
            
        mlflow.log_metric("final_mean_reward", final_mean_reward)
        mlflow.set_tag("status", "completed")
        
        return final_mean_reward

if __name__ == "__main__":
    try:
        # Create SQLite DB for Optuna to resume later
        study_name = f"gnn_rl_{args.experiment}"
        storage_name = f"sqlite:///optuna_{args.experiment}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2000, interval_steps=1000)
        )
        
        # Make study accessible in the objective function
        import __main__
        __main__.study = study
        
        study.optimize(objective, n_trials=args.n_trials)
        
        print("\n--- OPTUNA STUDY COMPLETED ---")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer. Gateway wird beendet...")
    finally:
        pipeline.stop()
        pipeline.cleanup()