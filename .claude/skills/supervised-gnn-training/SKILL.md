---
name: supervised-gnn-training
description: Run, extend, or debug the supervised GNN solver-classifier workflow in codebase/src/gnn/supervised_learning/ — predicting which Mathematica solver (Newton=0 vs gMGF=1) converges faster. Covers main.py custom training, the GraphGym grid (main_graphgym.py / run_all.py / grid.yaml), aggregation, and the eval/plot artifacts. Use when training a model, sweeping hyperparameters, adding an architecture, or interpreting run_results output.
---

# Supervised GNN training workflow

Goal: binary classification — given an expression graph + start/target values, predict the faster solver (Newton `0` vs gMGF `1`).

**First:** `conda activate pytorch` and `cd codebase/src/gnn/supervised_learning`. See the `gnn-dev-workflow` skill for environment/PYTHONPATH details. The shared graph/feature layer is documented in the `graph-data-pipeline` skill — `--mode`, `--edge-direction`, `--feature-groups`, `--positional-encoding`, `--active-features`, `--enrich` all behave identically here and in RL.

## Two entry paths

### A. Custom training — `main.py`
Reads graphs from CSV/GraphML, train/val splits, trains a GATv2 (or other) classifier.

```bash
python main.py --dry-run                                  # validate dataset load only
python main.py --dataset run_20260408_160456/dataset_4 --dry-run
python main.py --config config_supervised.yaml --mode graph
python main.py --config config_supervised.yaml --mode tree --enrich
python main.py --config config_supervised.yaml --edge-direction bottom_up
python main.py --config config_supervised.yaml --feature-groups node topology --positional-encoding none
python main.py --config config_supervised.yaml --active-features "node_type,depth,value,virtual_current_x_val"
```

### B. GraphGym grid — recommended end-to-end
`run_all.py` runs: config-gen from `grid.yaml` → train every config via `main_graphgym.py` → aggregate to `run_results/<exp>/agg/` → full evaluation to `run_results/<exp>/eval_plots/`.

```bash
python run_all.py --experiment-name res_with_enrich
python run_all.py --experiment-name res_with_enrich --parallel -n 4   # 4 configs at once
python run_all.py --experiment-name res_with_enrich --skip-training   # re-eval only
```

Manual steps when you need them:
```bash
python main_graphgym.py --cfg config_supervised.yaml      # one model
python configs_gen.py                                     # grid.yaml -> configs/*.yaml
python aggregate_graphgym.py res_with_enrich --eval --top-k 5
python run_results/post_eval.py res_with_enrich
python run_results/eval.py res_with_enrich
```

`run_all.py` flags: `--skip-training`, `--skip-eval`, `--full-eval` (all 9 CSV variants), `--top-k N`, `--parallel`, `-n/--num`.

## Config files

- `config_supervised.yaml` — base GraphGym config. Key blocks: `gnn:` (`layers_mp`, `dim_inner`, `layer_type`, `act`), `dataset.name`, and `expression_graph:` (`mode`, `enrich`, `edge_direction`, `features:`). `loader_graphgym.py` reads `expression_graph` keys via dependency injection and wires them into the shared pipeline.
- `grid.yaml` — hyperparameter sweep, e.g. `gnn.layer_type: [sageconv, gcnconv, ginconv, gatv2conv]`, `gnn.layers_mp: [2,3]`, `model.graph_pooling: [add, mean]`.

## Architectures

Supported layer types and their stacks are declared in `supervised_config.py`:
`gatv2conv→gatv2_stack`, `gineconv→gine_stack`, `gcnconv→gcn_stack`, `ginconv→gin_stack`.
**`gcnconv` and `ginconv` ignore edge features** (`LAYERS_WITHOUT_EDGE_FEATURES`) — don't expect native edge_attr to matter for those. Backbones live in `shared/models/gnn_backbones.py`; classifier heads in `shared/models/classifiers.py` (`SupervisedGraphClassifier`). To add an architecture, register it in `supervised_config.py`, add the stack in `gnn_backbones.py`, and add the option to `grid.yaml`.

## Model selection & split semantics (don't mislabel these)

Best model is chosen by **`val_pr_auc`** on the **unseen synthetic holdout**; the curated real-data set is final generalization test only. In every plot:
- `train_*` → training (synthetic)
- `val_*` → validation **synthetic** (holdout — this drives model selection)
- `test_*` → validation **curated** (real data — generalization only)

## Output artifacts (`run_results/<experiment>/`)

`agg/` holds aggregated CSVs (train/val/test × last/best/bestepoch). `eval_plots/` holds `split_comparison.png`, `generalization_gap.png`, `training_curves_overview.png`, `leaderboard.{csv,png}`, per-run heatmaps (mean & max), `summary_bars.png`, and `top_configs/rank_N_.../` diagnostics (training curves, confusion/ROC/PR for both synthetic and curated splits).

## Logging

MLflow logs everything in real time: `mlflow ui --host 0.0.0.0 --port 5000` → http://localhost:5000.

## Tests to run after changes
`test_supervised_classifier`, `test_graphgym_logger`, `test_curated_eval_schedule`, `test_eval_metrics`, `test_eval_confidence_metrics`, `test_synthetic_mode` (invocation in `gnn-dev-workflow`).
