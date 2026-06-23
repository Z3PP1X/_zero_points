---
name: critical-path
description: Critical path audit for the GNN data-loading → kappa-augmentation → feature-resolution → training → eval pipeline. Use when debugging feature-dimension mismatches, silent kappa-augmentation failures, stale cache issues, broken eval plots, or adding new features/stages.
---

# Critical Path: Data → Kappa → Features → Training → Eval

## Overview

```
CSV + JSON-Graphs
    ↓ UnifiedDataLoader (unified_loader.py)
    ↓ build_kappa_map()           ← kappa Spalte muss in CSV vorhanden sein
    ↓ GraphDataLoader.get_graph() ← Cache-Check + Augmentierungsentscheidung
    ↓ LoadAugmentedFunctionGraph() / ExpressionGraphConverter.convert()
    ↓ to_homogeneous()            → data.x: (N, 32) Tensor
    ↓ slice_active_features()     → data.x: (N, K) wenn active_features gesetzt
    ↓ ExpressionGNN (input_dim=K) → Logits → Loss
    ↓ aggregate_graphgym.py       → agg/{train,val,test}/stats.json + CSVs
    ↓ eval.py / training_curves.py / diagnostics.py / report.py
```

---

## Phase 1 — CSV laden

**Entry:** `UnifiedDataLoader.get_instance()` (unified_loader.py:23)

- CSV-Pfade kommen aus `cfg.data.curated_csv` / `cfg.data.synthetic_csv`
- Aktuelle Pfade (alle Stage-Configs):
  - Curated: `datasets/run_20260604_154509/dataset2_joined.csv`
  - Synthetic: `datasets/run_20260604_154509/synthetic_dataset2.csv`
- Label: `faster_algorithm` = 0 (gMGF) / 1 (Newton) — berechnet via `FeatureEngineering._tag_faster_algorithm()` (preprocessing.py:26)

**Kappa-Map:** `UnifiedDataLoader.build_kappa_map()` (unified_loader.py:190)
- Liest Spalte `kappa` aus CSV → `{graph_id: float}` Dict
- **⚠️ SILENT FAIL SF-1:** Wenn `kappa`-Spalte fehlt → leeres Dict, kein Fehler, kein Warning
- Woran erkennbar: `add_kappa=True` gesetzt, aber `root_color_kappa`-Spalte in `batch.x` ist überall 0.0
- Fix: `python add_kappa.py` im Repo-Root ausführen (einmalig, schreibt `kappa` in CSV zurück)

---

## Phase 2 — Graphs laden & Cache

**Entry:** `GraphDataLoader.__init__()` (graph_loader.py:24)

Graph-JSON-Quellen:
- Curated: `datasets/graphs/graphs.json`
- Synthetic: `datasets/graphs/synthetic_graphs.json`

**Cache-Verzeichnis:** `datasets/graphs/.pt_cache/` (graph_loader.py:65–68)

Cache-Key-Format (graph_loader.py:210–219):
```
{gid}_{mode}_{edge_direction}[_sn][_k{kappa_int}][_augmented].pt
```

**⚠️ SILENT FAIL SF-2: Stale Cache nach Schema-Änderung**
- Cache-Key enthält **keinen Hash** von `NODE_FEATURE_SCHEMA`
- Wenn Feature hinzugefügt/entfernt wird und `.pt_cache/` nicht gelöscht → alter Tensor geladen, Dimension stimmt nicht mit Modell überein
- **Fix nach jeder Schema-Änderung:** `rm -rf datasets/graphs/.pt_cache/`

---

## Phase 3 — Kappa-Augmentierung

**Entscheidungslogik** (graph_loader.py:199–208):
```python
use_augmented = (
    self.add_kappa
    and self.kappas_dir.exists()
    and any(self.kappas_dir.glob("**/*.json"))
    and (kappas_dir_explicit or not is_temp_path)
)
```

- `kappas_dir` = `datasets/kappas/` (50 Einträge, Werte -25..+24, Format: GraphML)
- In pytest-Runs: `is_temp_path=True` → `use_augmented=False` — absichtlich, kein Bug

**Augmentierungspfad** (kappa_loader.py:378):
1. `LoadGraphFromLocalStructure()` → lädt Haupt-Graph aus JSON
2. `_load_normalized_kappas()` (kappa_loader.py:78) → cached im Modul-Level `_kappa_graph_cache`
3. `MergePrenormalizedSubgraph()` (kappa_loader.py:243) → fügt passenden kappa-Subgraph ein
4. `_tag_and_connect_kappa()` (kappa_loader.py:117) → Root-Knoten bekommt `root_color=4` (kappa), Edges: GlobalToKappa/KappaToGlobal

**⚠️ SILENT FAIL SF-3:** `kappa_map.get(gid, 0.0)` (graph_loader.py:275)
- Graph-IDs ohne CSV-kappa-Eintrag bekommen `kappa_value=0.0`
- kappa=0 existiert nicht in kappas.json → Warning "Kappa value 0.0 not found", kein Fehler

---

## Phase 4 — Feature-Extraktion

**Entry:** `ExpressionGraphConverter.convert()` (graph_converter.py:392)

**NODE_FEATURE_SCHEMA** (graph_vocab.py:89) — 32 Spalten in dieser Reihenfolge:

| Gruppe | Spalten (N) | Inhalt |
|---|---|---|
| node_type | 3 | global, operator, function |
| root_color | 5 | none, f, d1, d2, kappa |
| label | 15 | UNK, CONSTANT, x, E, Log, Pi, Sin, Cos, Tan, Plus, Minus, Times, Divide, Power, Sqrt |
| topology | 2 | subtree_size, subtree_depth |
| histogram | 4 | hist_trigonometric, hist_exponential, hist_variables, hist_constants |
| positional | 3 | anchor_trigonometric, anchor_exponential, anchor_variable |

**Aufbau in `_enrich_nodes()`** (graph_converter.py:338):
- One-hot Encoding für node_type, root_color, label
- `TopologicalFeatureExtractor.extract_and_annotate()` (feature_extraction.py:118): subtree_sizes, subtree_depth, Histogramme
- `_compute_anchor_positional_encoding()` (feature_extraction.py:21): BFS-basierte anchor_* Features
- `_normalize_graph_attrs()` (graph_converter.py:548): paddet auf einheitliche Keys
- `to_homogeneous()` (homogeneous_converter.py:9): `from_networkx(G, group_node_attrs=NODE_FEATURE_SCHEMA)` → `data.x: (N, 32)`

**Neues Label hinzufügen:** `CANONICAL_LABELS`, `LABEL_ONEHOT_NAMES`, `NODE_FEATURE_SCHEMA`, `NODE_FEATURES` in graph_vocab.py updaten → **dann `.pt_cache/` löschen**

---

## Phase 5 — Feature-Slicing (active_features)

**Entry:** `ProblemRunDataset.__getitem__()` (preprocessing.py:272)
- Wenn `active_features` gesetzt: `slice_active_features(data.x, active_features)` (feature_extraction.py:252)
- Wirft `ValueError` wenn Feature-Name nicht in `NODE_FEATURE_SCHEMA` — kein Silent Fail ✓

**active_features Resolution** (supervised_config.py:71 → feature_config.py:323):
- YAML: `active_features: "col1,col2,..."` → `list[str]`
- YAML: `active_features: ""` → `None` → alle 32 Spalten

**Stage-Feature-Counts:**
| Stage | active_features | Spalten |
|---|---|---|
| 1 — pure AST | CSV-Liste (node_type + label) | 18 |
| 2 — AST roots | CSV-Liste (+ root_color) | 23 |
| 3 — full graph | `""` (alle) | 32 |
| 4 — experiment | `""` oder explizit | 32 default |

**Active feature names werden gesetzt:** `cfg.expression_graph.active_feature_names` (loader_graphgym.py:454)

---

## Phase 6 — Modell-Input-Dimension

**`ExpressionClassifierNetwork.__init__()`** (loader_graphgym.py:249):
```python
names = list(cfg.expression_graph.active_feature_names or [])
ExpressionGNN(input_dim=(len(names) or dim_in), ...)
```

→ `input_dim` = Anzahl active features (K). `batch.x` hat Shape (N, K). Dimension stimmt.

**`ExpressionGNN.forward()`** (gnn_backbones.py:162):
1. `node_encoder`: LayerNorm(K) → Linear(K, hidden_dim) → activation
2. `num_layers × GINConv` (kein edge_attr)
3. `pool_fn(x, batch)` → (batch_size, hidden_dim)
4. `tail` → `head` → (batch_size, 2)

**⚠️ Latentes Risiko SF-4 (nicht silent):** `ExpressionNodeEncoder` in loader_graphgym.py:219 hardcoded auf 32. Wenn `model.type = "gnn"` (stock GraphGym) statt `"expression_classifier"` → RuntimeError bei K < 32. Aktuell kein Problem, da alle Stage-Configs `model.type: expression_classifier` setzen.

---

## Phase 7 — Eval-Pipeline

### Datenfluss nach Training
```
Training: agg/{train,val,test}/stats.json (pro Run)
           ↓
aggregate_graphgym.py → agg/*.csv (9 CSVs: {train,val,test}×{last,best,bestepoch})
           ↓
eval.py:GNNResultEvaluator → eval_plots/: heatmaps, leaderboard, split_comparison, generalization_gap, pareto
           ↓
training_curves.py:TrainingCurvePlotter → per-Epoch-Kurven
           ↓
diagnostics.py:DiagnosticPlotter → confusion, ROC, PR, reliability (Top-K Runs, braucht Checkpoint)
           ↓
report.py:generate_report() → summary.md + summary.json
           ↓
feature_importance.py → feature_importance Heatmap
```

**Orchestrierung:** `post_eval.py:run_post_evaluation()` ruft alle Steps auf.

### Split-Semantik (nicht verwechseln)
| CSV-Stem | Bedeutung | Plots-Label |
|---|---|---|
| `train_bestepoch` | Training (Synthetic) | Train |
| `val_bestepoch` | Validation Synthetic → **Model Selection** | Validation Synthetic |
| `test_bestepoch` | Validation Curated → **nur Generalisierungstest** | Validation Curated |

Diese Semantik ist in `eval.py:SPLIT_STEMS` und `report.py:SPLIT_STEMS` konsistent implementiert.

### Metriken
Berechnet in `eval_metrics.py`: `pr_auc`, `auc`, `f1`, `precision`, `recall`, `accuracy`, `loss`, `mean_confidence`, `brier_score`, `ece`, `mean_margin`, `mean_entropy`
- **Leaderboard:** alle außer `ece`, `mean_margin`, `mean_entropy` (nur calibrierte Metriken)
- **Model Selection:** nach `val_pr_auc` (höher = besser)
- `positive_label = 1 = Newton` (baselines.py:25, eval.py:353) — konsistent

---

## Bekannte Silent-Fail-Risiken (Zusammenfassung)

| ID | Risiko | Datei:Zeile | Symptom | Fix |
|---|---|---|---|---|
| SF-1 | `add_kappa=True` ohne kappa-Spalte in CSV | unified_loader.py:190 | `root_color_kappa=0` überall | `python add_kappa.py` |
| SF-2 | Stale `.pt_cache/` nach Schema-Änderung | graph_loader.py:210 | Falsche Feature-Dimension, RuntimeError | `rm -rf datasets/graphs/.pt_cache/` |
| SF-3 | Graph-ID ohne kappa-Map-Eintrag → kappa=0.0 | graph_loader.py:275 | Warning + falscher kappa-Wert | CSV-Vollständigkeit sicherstellen |
| SF-4 | `ExpressionNodeEncoder` hardcoded 32 | loader_graphgym.py:219 | RuntimeError bei model.type=gnn+K<32 | Nicht auf stock GNN wechseln |
| SF-5 | `class_balance.json` mehrfach überschrieben | loader_graphgym.py:627 | Letzter Parallel-Run gewinnt | Kein Problem bei sequentiellem Training |

---

## Test-Commands

```bash
# Ganzes Test-Suite (kritischer Pfad ist abgedeckt)
conda activate pytorch && pytest

# Nur kritischer Pfad
pytest -k "smoke or augmented or feature_config or graph_loader or unified_loader"

# Nach Schema-Änderung: erst Cache löschen
rm -rf datasets/graphs/.pt_cache/
pytest
```

### Test-Coverage-Gaps (noch offen)

| Lücke | Was fehlt |
|---|---|
| Kein Smoke-Test mit add_kappa=True | add_kappa=True + echte kappa_map → root_color_kappa≠0 prüfen |
| Keine Dimensions-Konsistenzprüfung | len(active_feature_names) == model.net.node_encoder[1].in_features |
| Kein Warning-Test für leere kappa_map | add_kappa=True + {} → sollte Warning loggen |
| Keine Tests für training_curves.py | _iter_run_dirs, plot_all_configs, plot_top_configs |
| Keine Tests für eval.py | GNNResultEvaluator, generate_leaderboard, generate_heatmaps |
| Keine Tests für aggregate_graphgym.py | custom_agg_batch → CSV-Spaltenschema |

---

## Konfigurationsmuster

```yaml
# Kappa aktivieren (benötigt: kappa-Spalte in CSV + datasets/kappas/ vorhanden)
expression_graph:
  add_kappa: true

# Stage 1: nur AST (18 Features)
active_features: "node_type_global,node_type_operator,node_type_function,label_UNK,label_CONSTANT,label_x,label_E,label_Log,label_Pi,label_Sin,label_Cos,label_Tan,label_Plus,label_Minus,label_Times,label_Divide,label_Power,label_Sqrt"

# Stage 3+: alle 32 Features
active_features: ""

# Supernode (INKOMPATIBEL mit positional encoding!)
add_virtual_supernode: true
features:
  positional: false   # MUSS false sein — sonst PositionalSupernodeConflictError
```

## Preprocessing (einmalig bei neuen Daten)

```bash
# Kappa in CSV berechnen (kappa = clamp(round(f''(x₀) / f'(x₀)²), -25, 25))
python add_kappa.py   # Repo-Root, schreibt kappa-Spalte in CSV zurück

# Cache invalidieren (nach Schema-Änderungen immer!)
rm -rf datasets/graphs/.pt_cache/
```

## Wichtige Datei-Referenzen

| Komponente | Datei:Zeile |
|---|---|
| NODE_FEATURE_SCHEMA (32 Spalten) | graph_vocab.py:89–127 |
| CANONICAL_LABELS / LABEL_ONEHOT_NAMES | graph_vocab.py:5–42 |
| ROOT_COLOR_VOCAB (kappa=4) | graph_vocab.py:61 |
| resolve_active_node_features | feature_config.py:323 |
| UnifiedDataLoader.get_instance | unified_loader.py:23 |
| build_kappa_map (SF-1) | unified_loader.py:190 |
| GraphDataLoader.get_graph + Cache-Key | graph_loader.py:185–265 |
| use_augmented Entscheidungslogik | graph_loader.py:199 |
| LoadAugmentedFunctionGraph | kappa_loader.py:378 |
| _tag_and_connect_kappa (root_color=kappa) | kappa_loader.py:117 |
| _enrich_nodes (Feature-Vektor) | graph_converter.py:338 |
| inject_virtual_supernode | feature_extraction.py:179 |
| slice_active_features | feature_extraction.py:252 |
| to_homogeneous (→ data.x Tensor) | homogeneous_converter.py:9 |
| ExpressionClassifierNetwork (dim_in) | loader_graphgym.py:249 |
| active_feature_names gesetzt | loader_graphgym.py:454 |
| ExpressionNodeEncoder (hardcoded 32) | loader_graphgym.py:219 |
| ExpressionGNN.forward | gnn_backbones.py:162 |
| aggregate_graphgym.py (CSV-Schema) | aggregate_graphgym.py |
| Split-Semantik (SPLIT_STEMS) | eval.py:SPLIT_STEMS, report.py:SPLIT_STEMS |
| Leaderboard-Metriken | eval.py:LEADERBOARD_METRICS |
| _iter_run_dirs (datetime-Namen) | training_curves.py:83 |
| feature_importance dir-Filter | feature_importance.py:230 |
