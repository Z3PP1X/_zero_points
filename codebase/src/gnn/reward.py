"""
Reward-Berechnung für die GNN-RL Pipeline.

Implementiert die drei Reward-Komponenten:
    - r_base:  Bonus/Strafe für Solver-Wahl vs. Benchmark (nur erster Schritt)
    - r_step:  Per-Action Benchmark-Vergleich mit zeitlichem Discount
    - r_learn: Gesamtverbesserung gegenüber dem Rekord
"""
import math


class RewardCalculator:
    """
    Berechnet Episode-Rewards rückwirkend über alle Transitions.

    Args:
        basis_reward: Betrag des Bonus/Strafe für korrekte Solver-Wahl.
        gamma: Discount-Faktor für r_step (zeitliche Gewichtung).
        alpha: Skalierungsfaktor für r_learn (Rekord-Verbesserung).
    """

    def __init__(self, basis_reward: float = 1.0, gamma: float = 0.99, alpha: float = 1.0):
        self.basis_reward = basis_reward
        self.gamma = gamma
        self.alpha = alpha

    def calculate_episode_rewards(self, episode_transitions: list, reward_state: dict):
        """
        Berechnet Rewards für alle Transitions einer Episode.

        Setzt transition["reward"] für jede Transition in der Liste.

        Reward-Struktur pro Transition i:
            - i == 0:           r_base + r_step
            - 0 < i < T-1:     r_step
            - i == T-1:         r_step + r_learn

        Args:
            episode_transitions: Geordnete Liste von Transition-Dicts mit
                                 keys: current_state, next_state, action.
            reward_state: Terminal-State von Mathematica mit Benchmark-Daten.
        """
        if not episode_transitions:
            return

        T = reward_state.get("networkStep", len(episode_transitions))

        record_abs_time = reward_state.get("recordAbsTime", 0.0)
        final_abs_time = reward_state.get("absTime", 0.0)
        r_learn = self.alpha * (record_abs_time - final_abs_time)

        for i, transition in enumerate(episode_transitions):
            current_state = transition["current_state"]
            next_state = transition.get("next_state")

            r_base = 0.0
            if i == 0:
                action = transition.get("action", {})
                chosen_solver = action.get("solver", current_state.get("solver"))
                benchmark_solver = reward_state.get("Benchmarksolver")

                if chosen_solver is not None and benchmark_solver is not None:
                    if float(chosen_solver) == float(benchmark_solver):
                        r_base = self.basis_reward
                    else:
                        r_base = -self.basis_reward

            t = current_state.get("networkStep", i + 1)

            time_benchmark = current_state.get(
                "timeBenchmarkSolver",
                reward_state.get("timeBenchmarkSolver", 0.0)
            )

            if next_state:
                delta_time = next_state.get("absTime", 0.0) - current_state.get("absTime", 0.0)
            else:
                delta_time = final_abs_time - current_state.get("absTime", 0.0)

            if delta_time > 0 and time_benchmark > 0:
                S = math.log(time_benchmark / delta_time)
            else:
                S = 0.0

            r_step = S * (self.gamma ** (T - t))

            total_reward = r_base + r_step

            if i == len(episode_transitions) - 1:
                total_reward += r_learn

            transition["reward"] = total_reward
