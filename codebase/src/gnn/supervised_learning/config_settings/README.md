# `config_settings/` — Architektur-Evolutionsstufen

Jede Stufe ist ein eigenständiges Config-Paar (`base_config.yaml` + `grid.yaml`), das
genau die für diese Stufe relevanten Parameter exponiert. Die Stufen 1–3 sind **feste
Experimente** mit wachsendem Feature-Umfang; Stufe 4 ist die **freie Experimentierzone**.

Auswahl über `run_all.py`:

```bash
python run_all.py --list-stages              # Stufen anzeigen
python run_all.py --stage 1 --dry-run        # nur Configs generieren (kein Training)
python run_all.py --stage 1 --experiment-name stufe1_lauf1
python run_all.py --stage stage1_pure_ast    # Ordnername geht auch
```

Alle Stufen nutzen **ExpressionGNN** (`model.type: expression_classifier`, `gnn.layer_type: ginconv`).
Features werden über `active_features` gesteuert — eine kommaseparierte Liste von
Spaltennamen aus `NODE_FEATURE_SCHEMA` (28 Spalten gesamt). Leeres `active_features`
aktiviert alle 28 Features.

## Stufen-Übersicht

| Stufe | Ordner | Features | Anzahl | Grid |
|---|---|---|---|---|
| 1 | `stage1_pure_ast` | node_type + label | 14 | 8 Configs |
| 2 | `stage2_ast_roots` | + root_color | 19 | 8 Configs |
| 3 | `stage3_full_graph` | alle (+ Topologie, Histogramm, Anchor-PE) | 28 | 24 Configs |
| 4 | `stage4_experiment` | frei konfigurierbar | — | frei |

## Feature-Schema (NODE_FEATURE_SCHEMA, 28 Spalten)

```
node_type_global, node_type_operator, node_type_root, node_type_supernode   (4)
root_color_none, root_color_f, root_color_d1, root_color_d2, root_color_kappa  (5)
label_UNK, label_CONSTANT, label_GLOBAL, label_x, label_E,
label_Log, label_Pi, label_Sin, label_Cos, label_Tan                        (10)
subtree_size, subtree_depth                                                  (2)
hist_trigonometric, hist_exponential, hist_variables, hist_constants         (4)
anchor_trigonometric, anchor_exponential, anchor_variable                    (3)
```

## Datensatz

Alle Stufen verwenden `datasets/run_20260604_154509/dataset_joined.csv`
(8 934 Zeilen nach Bereinigung, ~50/50 Klassenverteilung Newton vs. gMGF).
