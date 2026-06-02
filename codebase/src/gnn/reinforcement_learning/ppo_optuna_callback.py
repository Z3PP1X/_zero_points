from __future__ import annotations

import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Any
from gnn.reinforcement_learning.gateway.gateway_traffic_monitor import GatewayTrafficMonitor


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
        traffic_monitor: Optional[GatewayTrafficMonitor] = None,
    ):
        super().__init__()
        self.trial = trial
        self.study = study
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.is_pruned = False
        self.traffic_monitor = traffic_monitor

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
        faster_ratio = None
        overshoot_var = None
        convergence_rate = None
        mean_episode_steps = None
        mean_roundtrip_s = None
        if self.traffic_monitor is not None:
            faster_ratio, overshoot_var = self.traffic_monitor.get_rolling_metrics()
            adv_metrics = self.traffic_monitor.get_advanced_rolling_metrics()
            convergence_rate = adv_metrics.get("convergence_rate")
            mean_episode_steps = adv_metrics.get("mean_episode_steps")
            mean_roundtrip_s = adv_metrics.get("mean_roundtrip_s")

        self.trial.report(mean_reward, self.num_timesteps)

        try:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.log_metric(
                    "mean_episode_reward", mean_reward, step=self.num_timesteps
                )
                mlflow.log_metric(
                    "best_episode_reward", best_episode_reward, step=self.num_timesteps
                )
                mlflow.log_metric(
                    "worst_episode_reward",
                    worst_episode_reward,
                    step=self.num_timesteps,
                )
                if faster_ratio is not None:
                    mlflow.log_metric(
                        "faster_than_benchmark_ratio",
                        faster_ratio,
                        step=self.num_timesteps,
                    )
                if overshoot_var is not None:
                    mlflow.log_metric(
                        "overshoot_variance",
                        overshoot_var,
                        step=self.num_timesteps,
                    )
                if convergence_rate is not None:
                    mlflow.log_metric(
                        "convergence_rate",
                        convergence_rate,
                        step=self.num_timesteps,
                    )
                if mean_episode_steps is not None:
                    mlflow.log_metric(
                        "mean_episode_steps",
                        mean_episode_steps,
                        step=self.num_timesteps,
                    )
                if mean_roundtrip_s is not None:
                    mlflow.log_metric(
                        "mean_roundtrip_s",
                        mean_roundtrip_s,
                        step=self.num_timesteps,
                    )
        except Exception:
            pass

        faster_text = f"{faster_ratio:.3f}" if faster_ratio is not None else "—"
        overshoot_text = f"{overshoot_var:.3f}" if overshoot_var is not None else "—"
        conv_text = f"{convergence_rate:.3f}" if convergence_rate is not None else "—"
        steps_text = f"{mean_episode_steps:.1f}" if mean_episode_steps is not None else "—"
        latency_text = f"{mean_roundtrip_s:.3f}s" if mean_roundtrip_s is not None else "—"

        print(
            f"  [Trial {self.trial.number:>3}] "
            f"{progress_pct:5.1f}% ({self.num_timesteps}/{self.total_timesteps}) | "
            f"Episoden: {episode_count:>3} | "
            f"Mean R: {mean_reward:>8.3f} | "
            f"Best Ep: {best_episode_reward:>8.3f} | "
            f"Worst Ep: {worst_episode_reward:>8.3f} | "
            f"Faster Ratio: {faster_text} | "
            f"Overshoot Var: {overshoot_text} | "
            f"Conv Rate: {conv_text} | "
            f"Mean Steps: {steps_text} | "
            f"Latency: {latency_text} | "
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
        value_text = f"{trial.value:.3f}" if trial.value is not None else "—"
        print(
            f"\n--- Trial {trial.number} abgeschlossen ({trial.state.name}) | "
            f"Wert: {value_text} | "
            f"Fertig: {completed}/{self.total_trials} | "
            f"Pruned: {pruned} | "
            f"Study Best: {_format_study_best(study)} ---"
        )
