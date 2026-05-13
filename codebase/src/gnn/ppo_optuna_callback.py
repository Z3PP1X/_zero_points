from __future__ import annotations

import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback


class OptunaEpisodeRewardCallback(BaseCallback):
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

        episode_buffer = self.model.ep_info_buffer
        episode_count = len(episode_buffer)
        if episode_count == 0:
            print(
                f"  [Trial {self.trial.number:>3}] "
                f"Steps: {self.num_timesteps:>6} | "
                f"No episodes completed yet..."
            )
            return True

        rewards = [episode["r"] for episode in episode_buffer]
        mean_reward = float(np.mean(rewards))
        best_episode_reward = float(np.max(rewards))
        worst_episode_reward = float(np.min(rewards))
        self.trial.report(mean_reward, self.num_timesteps)

        try:
            study_best = self.study.best_value
        except ValueError:
            study_best = float("nan")

        print(
            f"  [Trial {self.trial.number:>3}] "
            f"Steps: {self.num_timesteps:>6} | "
            f"Episodes: {episode_count:>3} | "
            f"Mean R: {mean_reward:>8.3f} | "
            f"Best Ep: {best_episode_reward:>8.3f} | "
            f"Worst Ep: {worst_episode_reward:>8.3f} | "
            f"Study Best: {study_best:>8.3f}"
        )

        if self.trial.should_prune():
            print(f"  [Trial {self.trial.number}] PRUNED at step {self.num_timesteps}")
            self.is_pruned = True
            return False

        return True
