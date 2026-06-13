---
name: rl-ppo-workflow
description: Run, extend, or debug the reinforcement-learning workflow in codebase/src/gnn/reinforcement_learning/ — a PPO agent (Stable-Baselines3) with a GNN feature extractor that controls a Mathematica solver over a network gateway. Covers the two phases (Optuna tuning via main.py, final training via train_best.py), the Mathematica gateway/env, reward shaping, and config_rl.yaml. Use when launching trials, resuming a study, tuning rewards, or touching the gateway/env code.
---

# Reinforcement learning (PPO) workflow

A PPO policy learns to steer a Mathematica solver. The GNN backbone (shared with supervised) extracts features from the live expression graph; observations arrive from Mathematica over a socket gateway.

**First:** `conda activate pytorch` and `cd codebase/src/gnn/reinforcement_learning`. The shared `--mode / --edge-direction / --feature-groups / --positional-encoding / --active-features` flags behave exactly as in supervised — see `graph-data-pipeline` and `gnn-dev-workflow` skills.

## Two-phase design

### Phase 1 — Optuna hyperparameter tuning (`main.py`)
Searches PPO structure, GNN backbone, and reward coefficients in parallel; results persist to a SQLite Optuna DB (`optuna_<experiment>_<...>.db`).

```bash
python main.py --config config_rl.yaml --experiment kein_inv --n_trials 50 --timesteps 16384 --n-envs 4
python main.py --config config_rl.yaml --edge-direction bidirectional
python main.py --config config_rl.yaml --experiment kein_inv --continue-study --n-envs 4   # resume
```
Experiments (graph folders): `nur_f`, `f_fp_roh`, `kein_inv`. Flags: `--timesteps` (PPO steps/trial), `--n_trials`, `--n-envs` (parallel Mathematica envs), `--continue-study`.

### Phase 2 — final training of the best trial (`train_best.py`)
Reads the best trial from the Optuna DB and trains long.

```bash
python train_best.py --config config_rl.yaml --db optuna_kein_inv_n45g69_20260527_163125.db --dry-run
python train_best.py --config config_rl.yaml --db optuna_kein_inv.db --timesteps 300000 --positional-encoding anchor_periodic --n-envs 4
```
`--db` is **required**. Other flags: `--timesteps` (default 250000 from yaml), `--save-dir`, `--model-name`, `--dry-run` (load hparams + init only), `--no-torch-compile`.

## Config resolution (config_rl.yaml → CLI override)

`config_rl.yaml` is the single source of defaults, organized into `experiment:`, `optuna:`, `gateway:`, `train_best:` blocks. `rl_config.py` resolves it: `read_rl_settings()` flattens the YAML; explicit CLI flags override YAML via `resolve_rl_setting()`; shared graph args are added by `add_shared_graph_args()`. Always route edge-direction through `validate_edge_direction`. Choice tuples (`RL_EXPERIMENT_CHOICES`, `RL_MODE_CHOICES`, `RL_EDGE_DIRECTION_CHOICES`) live in `rl_config.py`.

## The Mathematica gateway & env (the moving parts)

The agent does not call Mathematica directly — it exchanges state/actions over sockets. Default ports in `main.py`: `RECEIVER_PORT=5650`, `SENDER_PORT=5651`, `RESULTS_PORT=5693`, `CONTROL_PORT=6000`.

| Component | File | Role |
| --- | --- | --- |
| `NetworkGateway` | `gateway/network_gateway.py` | socket control plane to Mathematica |
| `GatewayStateLogger` | `gateway/gateway_state_logger.py` | persists state stream (`gateway_states.db`) |
| `GatewayTrafficMonitor` | `gateway/gateway_traffic_monitor.py` | adaptive timeout / traffic stats |
| state ingress | `gateway/mathematica_state_ingress.py` | parses incoming Mathematica states |
| timeout policy | `gateway/state_wait_timeout.py` | bounded wait for next state (`gateway.timeout_*` config) |
| Gym env | `mathematica_env.py`, `mathematica_vec_env.py` | single & vectorized SB3 environments |
| SB3 feature extractor | `sb3_extractor.py` | wraps the GNN backbone for PPO |
| obs sanitation | `observation_sanitize.py`, `preprocessor.py` | clean/normalize observations |
| reward | `reward.py` (`RewardCalculator`) | episode reward shaping |
| Optuna driver | `ppo_optuna_workflow.py`, `ppo_optuna_search.py`, `ppo_optuna_callback.py`, `ppo_trial_config.py` | trial orchestration |

## Reward shaping (`reward.py`)

`RewardCalculator` combines a per-step term and a learning/record term. Key params: `time_bad_penalty`, `alpha`, `time_tolerance`, `RECORD_SENTINEL_THRESHOLD`. Time score is a clipped log-ratio of `delta_time`/`time_benchmark` (clipped to [-2, 2]); negative/invalid times return `-time_bad_penalty`. `r_learn` only fires on a genuine new record (`final_abs_time <= record_abs_time`), scaled by `alpha`. When changing reward logic, update `test_reward.py` — it asserts exact numeric values.

## Logging
MLflow tracks epoch rewards and policy losses: `mlflow ui --host 0.0.0.0 --port 5000`.

## Tests to run after changes
`test_reward`, `test_rl_config`, `test_network_gateway_control`, `test_gateway_state_logger`, `test_gateway_traffic_monitor`, `test_mathematica_env_finalize`, `test_mathematica_vec_env`, `test_mathematica_state_ingress`, `test_state_wait_timeout`, `test_observation_sanitize`, `test_gnn_policy_backbone`, `test_trial_switch` (invocation in `gnn-dev-workflow`). Most are pure-Python and run without Mathematica.
