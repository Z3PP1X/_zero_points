from __future__ import annotations

import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback


def _format_study_best(study: optuna.Study) -> str:
    try:
        best_trial = study.best_trial
    except ValueError:
        return "noch keiner"
    return f"{study.best_value:.3f} (Trial {best_trial.number})"


class OptunaEpisodeRewardCallback(BaseCallback):
    def __init__(
        self,
        trial: optuna.Trial,
        study: optuna.Study,
        *,
        total_timesteps: int,
        check_freq: int = 500,
    ):
        super().__init__()
        self.trial = trial
        self.study = study
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        progress_pct = 100.0 * self.num_timesteps / self.total_timesteps
        episode_buffer = self.model.ep_info_buffer
        episode_count = len(episode_buffer)
        if episode_count == 0:
            print(
                f"  [Trial {self.trial.number:>3}] "
                f"{progress_pct:5.1f}% ({self.num_timesteps}/{self.total_timesteps}) | "
                f"noch keine Episoden abgeschlossen"
            )
            return True

        rewards = [episode["r"] for episode in episode_buffer]
        mean_reward = float(np.mean(rewards))
        best_episode_reward = float(np.max(rewards))
        worst_episode_reward = float(np.min(rewards))
        self.trial.report(mean_reward, self.num_timesteps)

        print(
            f"  [Trial {self.trial.number:>3}] "
            f"{progress_pct:5.1f}% ({self.num_timesteps}/{self.total_timesteps}) | "
            f"Episoden: {episode_count:>3} | "
            f"Mean R: {mean_reward:>8.3f} | "
            f"Best Ep: {best_episode_reward:>8.3f} | "
            f"Worst Ep: {worst_episode_reward:>8.3f} | "
            f"Study Best: {_format_study_best(self.study)}"
        )

        if self.trial.should_prune():
            print(
                f"  [Trial {self.trial.number}] PRUNED bei Schritt "
                f"{self.num_timesteps} | Mean R: {mean_reward:.3f}"
            )
            self.is_pruned = True
            return False

        return True


class OptunaStudyProgressCallback:
    def __init__(self, *, total_trials: int):
        self.total_trials = total_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = sum(
            1
            for finished_trial in study.trials
            if finished_trial.state == optuna.trial.TrialState.COMPLETE
        )
        pruned = sum(
            1
            for finished_trial in study.trials
            if finished_trial.state == optuna.trial.TrialState.PRUNED
        )
        value_text = (
            f"{trial.value:.3f}"
            if trial.value is not None
            else "—"
        )
        print(
            f"\n--- Trial {trial.number} abgeschlossen ({trial.state.name}) | "
            f"Wert: {value_text} | "
            f"Fertig: {completed}/{self.total_trials} | "
            f"Pruned: {pruned} | "
            f"Study Best: {_format_study_best(study)} ---"
        )
