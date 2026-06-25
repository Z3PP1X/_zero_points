# GNN-RL & Supervised Expression Pipeline

Dieses Repository enthält eine vollständige Pipeline für maschinelles Lernen auf mathematischen Symbolik-Graphen. Das Projekt umfasst:
1. **Supervised Learning Workflow**: Evaluierung und Training von Graph Neural Networks (GNNs) zur Vorhersage des optimalen mathematischen Solvers (Newton vs. gMGF).
2. **Reinforcement Learning Workflow**: Steuerung und Optimierung eines Mathematica-Solvers über einen PPO-Agenten mit GNN-Feature-Extractor (Zwei-Phasen-Workflow: Optuna-Tuning & dediziertes Training).

---

## 📂 Repository-Struktur und wichtige Pfade

* `codebase/src/gnn/supervised_learning/`: Dateien für Supervised Learning.
  * `main.py`: Benutzerdefiniertes GNN-Training und Evaluierung.
  * `main_graphgym.py` & `loader_graphgym.py`: Integration des PyTorch Geometric GraphGym Frameworks.
  * `run_all.py`: End-to-End-Orchestrator (Grid-Training + Aggregation + Evaluation).
  * `aggregate_graphgym.py`: Aggregation der GraphGym-Läufe in CSVs.
  * `run_results/`: Post-Training-Evaluation (`post_eval.py`, `eval.py`, `training_curves.py`, `diagnostics.py`).
  * `config_supervised.yaml` & `grid.yaml`: Basis-Config und Hyperparameter-Grid für GraphGym.
* `codebase/src/gnn/reinforcement_learning/`: Dateien für Reinforcement Learning.
  * `main.py`: Einstiegspunkt für Phase 1 (Optuna Hyperparameter-Tuning).
  * `train_best.py`: Einstiegspunkt für Phase 2 (Bestes PPO-Modell trainieren).
  * `config_rl.yaml` & `rl_config.py`: Zentrale YAML-Defaults und CLI-Override-Auflösung für alle RL-Einstellungen.
* `codebase/src/gnn/shared/utils/`:
  * `graph_utils.py`: Kernmodul zur Graphml-Konvertierung, Feature-Extraktion, AST-Kantenrichtung und Verknüpfung von Ableitungen.
  * `feature_config.py`: Feature-Katalog (node / topology / positional / edge) und Auflösung gruppenbasierter Feature-Toggles.
  * `unified_loader.py`: Vereinheitlichter `UnifiedDataLoader` (Singleton/Multiton-Muster) zur synchronen Verwaltung von Tabellen- und Graphen-Daten.

---

## 🗃️ Datasets & Unified Loader

Um Redundanzen und Pfadprobleme zu vermeiden, wurden alle Datenquellen vereinheitlicht:
* **Pfad**: Beide Loader suchen standardmäßig im Ordner `/datasets/` an der Repository-Wurzel:
  * Tabulardaten (CSV): `/datasets/<run_key>/<dataset_name>.csv`
  * Graphdaten (JSON): `/datasets/<run_key>/<dataset_name>.json`
  * Traces (JSONL): `/datasets/<run_key>/traces/`
  *(Falls `/datasets/` nicht existiert, fallen die Loader abwärtskompatibel auf die alten Verzeichnisse `_datasets/` bzw. `graphs/` zurück).*
* **UnifiedDataLoader**: Verwaltet Tabellen- und Graph-Daten unter einer konsistenten API. Er lädt CSV-Dateien und verknüpft sie mit den entsprechenden GraphML-Graphen. Fehlende Startwerte (`x0`) werden automatisch aus den Graph-Rohdaten angereichert.

---

## 🛠️ Vorbereitung & System-Setup

Aktivieren Sie das vorbereitete Conda-Environment im WSL / Linux-Terminal:
```bash
conda activate pytorch
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn
```

---

## 📊 1. Supervised Learning Workflow

Das Ziel des Supervised Learnings ist die Klassifizierung, welcher Solver (Newton: `0` oder gMGF: `1`) bei einem gegebenen Graphen und Start-/Zielwert schneller konvergiert.

### A. Klassisches GNN-Training (`supervised_learning/main.py`)
Dieses Skript liest Graphen aus CSV/GraphML-Verzeichnissen ein, teilt sie in Train/Val-Splits auf und trainiert ein GNN-Modell (GATv2-Klassifikator).

#### 🏃 Befehlsbeispiele
```bash
# Wechseln ins Verzeichnis:
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn/supervised_learning

# 1. Konfiguration prüfen (Dry-Run):
python main.py --dry-run

# 2. Spezifisches Dataset laden (Dry-Run oder Training):
python main.py --dataset run_20260408_160456/dataset_4 --dry-run

# 3. Standard GNN-Lauf (f, f', f'' über den global-Knoten verknüpft):
python main.py --config config_supervised.yaml --mode tree_derivatives

# 4. GNN-Lauf im Tree-Modus (reiner Funktionstree f, ohne Ableitungen):
python main.py --config config_supervised.yaml --mode tree

# 5. Bottom-up Message Passing auf AST-Kanten:
python main.py --config config_supervised.yaml --edge-direction bottom_up

# 6. Nur eine Anchor-Positional-Encoding-Gruppe (z. B. periodische Operatoren):
python main.py --config config_supervised.yaml --positional-encoding anchor_periodic

# 7. Keine Positional Encodings, nur Node- und Topologie-Features:
python main.py --config config_supervised.yaml --feature-groups node topology --positional-encoding none

# 8. Explizite Feature-Liste (überschreibt Gruppen-Toggles):
python main.py --config config_supervised.yaml --active-features "node_type,depth,value,virtual_current_x_val"

# 9. Kappa- (h-Funktions-) Subgraphen aus datasets/kappas/ einbinden:
python main.py --config config_supervised.yaml --add-kappa
```

#### 📋 CLI-Optionen
| Parameter | Standard | Optionen | Beschreibung |
| :--- | :--- | :--- | :--- |
| `--config` | `config_supervised.yaml` | String | GraphGym-YAML mit Architektur-, Feature- und Graph-Einstellungen. |
| `--dataset` | aus Config | String | Pfad zum Dataset (z. B. `run_key/dataset_name`). |
| `--dry-run` | *aus* | Flag | Lädt den Datensatz kurz, um die Struktur zu validieren (ohne volles Training). |
| `--mode` | `tree_derivatives` | `tree`, `tree_derivatives` | Bestimmt, welche Teilgraphen kompiliert werden:<br>• `tree`: Nur der Funktionstree f (keine Ableitungen).<br>• `tree_derivatives`: f, f' und f'' geladen und über den `global`-Knoten verknüpft (jede Teilbaum-Wurzel wird per `root_color` als f/d1/d2 markiert). |
| `--edge-direction` | `top_down` | `top_down`, `bottom_up`, `bidirectional` | AST-Message-Passing-Richtung (parent→child, child→parent oder beides). |
| `--add-kappa` | *aus* | Flag | Fügt Kappa- (h-Funktions-) Subgraphen aus `datasets/kappas/` über `GlobalToKappa`/`KappaToGlobal`-Kanten in jeden Graphen ein. Standardmäßig deaktiviert (Opt-in). |
| `--feature-groups` | alle (aus Config) | Liste | Aktiviert nur bestimmte Feature-Klassen: `node`, `topology`, `positional`, `edge`. |
| `--positional-encoding` | alle Gruppen | Liste | Anchor-Positional-Encoding-Gruppen: `anchor_additive`, `anchor_scaling`, `anchor_periodic`, `anchor_exponential`, `anchor_transcendental`, oder `none`. Inkompatibel mit `--add-virtual-supernode`. |
| `--active-features` | `None` | String (Komma-separiert) | Explizite Teilmenge an Knoten-Features; überschreibt `--feature-groups` und `--positional-encoding`. |
| `--enrich` | aus Config | Flag | Schaltet zwischen **12 Basis-Knoten-Features** und **21 angereicherten Knoten-Features** (plus native Edge-Features) um. |

---

### B. GraphGym-Workflow (`supervised_learning/main_graphgym.py`)
GraphGym ist ein Framework von PyTorch Geometric zur standardisierten Modell-Evaluierung über YAML-Dateien. Das beste Modell wird per **`val_pr_auc`** auf dem ungesehenen synthetischen Holdout gewählt; der finale Test auf kuratierten Realdaten nutzt diesen Checkpoint.

#### 🏃 Empfohlener End-to-End-Lauf (Grid + automatische Auswertung)

```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn/supervised_learning

# 1) Grid trainieren, aggregieren und alle Plots automatisch erzeugen
python run_all.py --experiment-name res_with_enrich

# Parallel: 4 Grid-Configs gleichzeitig trainieren
python run_all.py --experiment-name res_with_enrich --parallel -n 4

# Optional: nur Auswertung, wenn Training bereits abgeschlossen ist
python run_all.py --experiment-name res_with_enrich --skip-training
```

`run_all.py` führt nacheinander aus:
1. **Config-Generierung** aus `grid.yaml` → `configs/*.yaml`
2. **Training** aller Grid-Konfigurationen via `main_graphgym.py`
3. **Aggregation** der Ergebnisse nach `run_results/<experiment>/agg/`
4. **Vollständige Evaluation** nach `run_results/<experiment>/eval_plots/`

#### 📊 Erzeugte Auswertungs-Artefakte

```
run_results/<experiment>/
├── agg/                              # Aggregierte CSVs (train/val/test × last/best/bestepoch)
└── eval_plots/
    ├── split_comparison.png          # Train vs. Val Synthetic vs. Val Curated
    ├── generalization_gap.png        # PR-AUC Generalisierungslücke pro Architektur
    ├── training_curves_overview.png  # Mittlere Trainingskurven über alle Configs
    ├── leaderboard.csv / .png        # Top-K nach val_bestepoch pr_auc
    ├── <run>/heatmaps_mean.png       # Hyperparameter-Heatmaps (MEAN)
    ├── <run>/heatmaps_max.png        # Hyperparameter-Heatmaps (MAX, best pr_auc pro Zelle)
    ├── <run>/summary_bars.png        # Architektur-/Pooling-Vergleich
    └── top_configs/rank_N_.../       # Top-K Diagnostik (best checkpoint)
        ├── training_curves.png
        ├── confusion_validation_synthetic.png
        ├── confusion_validation_curated.png
        ├── roc_validation_synthetic.png
        ├── roc_validation_curated.png
        ├── pr_validation_synthetic.png
        └── pr_validation_curated.png
```

**Split-Benennung in den Plots:**
- `train_*` → Training (synthetisch)
- `val_*` → Validation Synthetic (ungesehenes synthetisches Holdout, Modellauswahl)
- `test_*` → Validation Curated (kuratierte Realdaten, nur Generalisierung)

#### 🧩 Einzelne Schritte (manuell)

```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn/supervised_learning

# Nur ein einzelnes Modell trainieren
python main_graphgym.py --cfg config_supervised.yaml

# Grid-Configs erzeugen (schreibt out_dir unter run_results/<experiment>/)
python configs_gen.py

# Grid manuell trainieren
for conf in configs/*.yaml; do
  python main_graphgym.py --cfg "$conf"
done

# Aggregation + vollständige Evaluation
python aggregate_graphgym.py res_with_enrich --eval --top-k 5

# Nur Plots (wenn agg/ bereits existiert)
python run_results/post_eval.py res_with_enrich
python run_results/eval.py res_with_enrich
```

#### 📋 Wichtige CLI-Flags

| Skript | Flag | Beschreibung |
| :--- | :--- | :--- |
| `run_all.py` | `--experiment-name` | Ordnername unter `run_results/` |
| `run_all.py` | `--parallel` | Training parallel starten (Standard: sequenziell) |
| `run_all.py` | `-n` / `--num` | Anzahl paralleler Jobs mit `--parallel` (Standard: 2) |
| `run_all.py` | `--skip-training` | Nur Aggregation + Evaluation |
| `run_all.py` | `--skip-eval` | Nur Training, keine Plots |
| `run_all.py` | `--full-eval` | Alle 9 Run-CSV-Varianten plotten |
| `run_all.py` | `--top-k` | Anzahl Top-Configs für CM/ROC/PR (Standard: 5) |
| `aggregate_graphgym.py` | `--eval` | Nach Aggregation direkt alle Plots erzeugen |
| `run_results/post_eval.py` | `--skip-diagnostics` | Ohne Confusion Matrix / ROC / PR |
| `run_results/eval.py` | `--skip-slices` | Nur Top-Level-Plots, keine Architektur-Slices |

#### ⚙️ Grid und Basis-Config anpassen

Hyperparameter-Sweep in `grid.yaml` (Beispiel):

```yaml
gnn.layer_type: [sageconv, gcnconv, ginconv, gatv2conv]
gnn.layers_mp: [2, 3]
gnn.dim_inner: [256]
gnn.dropout: [0.2]
model.graph_pooling: [add, mean]
```

#### ⚙️ Pooling-/Skip-Ablation (`model.type: expression_classifier`)

Neben dem PyG-Standard-`GNN` (`model.type: gnn`) kann der Supervised-Workflow über die
geteilte `TestGraphNetwork`-Backbone laufen. Das macht zwei orthogonale Achsen aus
`gnn_backbones.py` (`UniformPoolMixin`) im Grid sweepbar:

| Achse | Werte | Wirkung |
| :--- | :--- | :--- |
| `gnn.variant` | `legacy` | ursprünglicher Conv-Stack + Real/Virtual-Split-Pooling (Baseline) |
| | `pooling` | hierarchisches Pooling zwischen den Blöcken (Readout aus letztem Block) |
| | `pooling_skip` | hierarchisches Pooling + JK-artige Skip-Aggregation der Block-Readouts |
| `gnn.pool_type` | `topk` | `TopKPooling` pro Block, Readout = mean ‖ max |
| | `diffpool` | `DenseSAGEConv` Soft-Clustering (Cluster `16, 4`) + Link-/Entropie-Aux-Loss |

```yaml
# config_supervised.yaml
model:
  type: expression_classifier   # auf `gnn` zurücksetzen für PyG-Standard-GNN
gnn:
  att_heads: 4                  # über alle Ablations-Arme konstant gehalten
  aux_loss_weight: 1.0          # Gewicht des DiffPool-Aux-Loss

# grid.yaml
gnn.variant:   [legacy, pooling, pooling_skip]
gnn.pool_type: [topk, diffpool]
gnn.layer_type: [gatv2conv]     # nur kantenbewusste Stacks (gatv2conv | gineconv)
```

Hinweise: `legacy` ignoriert `pool_type` (die beiden `legacy`-Kombis sind identisch);
`model.graph_pooling` / `gnn.stage_type` sind für `expression_classifier` inert. DiffPool
verdichtet Batches (`to_dense_batch`) — ggf. `train.batch_size` für DiffPool-Arme senken.

#### ⚙️ Anpassung der Einstellungen in `config_supervised.yaml`
Sie können Parameter direkt in `config_supervised.yaml` modifizieren:
* **GNN-Schichten & Dimensionen**:
  ```yaml
  gnn:
    layers_mp: 3       # Anzahl Graph-Masseging-Passing Schichten
    dim_inner: 128     # Latente Dimension
    layer_type: sageconv  # sageconv, gatv2conv, gcnconv, ginconv
    act: gelu          # relu, leaky_relu, gelu, elu, tanh
  ```
* **Dependency Injection (DI)**: In `loader_graphgym.py` werden folgende Parameter direkt aus der YAML ausgelesen und an die Pipeline injiziert:
  ```yaml
  dataset:
    name: run_20260408_160456/dataset_4  # Pfad zum Datensatz
  expression_graph:
    mode: graph
    enrich: True
    edge_direction: top_down   # top_down | bottom_up | bidirectional
    add_kappa: false           # Kappa-Subgraphen aus datasets/kappas/ einbinden (Opt-in)
    features:
      node: true               # node_type, label_id, value, virtual_*, belongs_to_*
      topology: true           # depth, height, subtree_size, out_degree, betweenness
      positional: true         # Anchor-PE-Gruppen (anchor_additive, …, anchor_transcendental);
                               # true = alle, false = keine, [a, b] = Teilmenge.
                               # Inkompatibel mit add_virtual_supernode.
      edge: true               # native edge_attr (child_index, direction, …)
    active_features: ""        # Optional: explizite Override-Liste
  ```

#### 🧩 Feature-Klassen (Übersicht)

| Klasse | Enriched-Features |
| :--- | :--- |
| **node** | `node_type`, `label_id`, `value`, `has_value`, `virtual_*`, `belongs_to_*` |
| **topology** | `depth`, `height`, `subtree_size`, `out_degree`, `betweenness_centrality` |
| **positional** | Anchor-PE: `anchor_additive`, `anchor_scaling`, `anchor_periodic`, `anchor_exponential`, `anchor_transcendental` (je `1/(1+hops)` zum nächsten Operator-Anker der Gruppe, pro Funktion) |
| **edge** | `child_index`, `direction`, `relation_type`, `edge_betweenness_centrality` |

Die Auflösung erfolgt zentral über `shared/utils/feature_config.py` und ist für Supervised und RL identisch.

---

## 🎮 2. Reinforcement Learning Workflow

Der RL-Workflow lernt eine optimale Policy zur Steuerung des mathematischen Solvers über ein PPO-Modell.

### Phase 1: Hyperparameter-Tuning (`main.py`)
Startet eine Optuna-Studie zur parallelen Suche der besten Hyperparameter (PPO-Struktur, GNN-Backbone und Belohnungskoeffizienten).

#### 🏃 Befehlsbeispiele
```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn/reinforcement_learning

# Tuning-Lauf mit zentraler YAML-Config und 4 parallelen Slots:
python main.py --config config_rl.yaml --experiment kein_inv --n_trials 50 --timesteps 16384 --n-envs 4

# Bidirektionales AST-Message-Passing:
python main.py --config config_rl.yaml --edge-direction bidirectional

# Kappa- (h-Funktions-) Subgraphen einbinden:
python main.py --config config_rl.yaml --add-kappa

# Nur eine Anchor-PE-Gruppe (z. B. exponentielle Operatoren):
python main.py --config config_rl.yaml --positional-encoding anchor_exponential

# Pausierte Studie fortsetzen (Resume):
python main.py --config config_rl.yaml --experiment kein_inv --continue-study --n-envs 4
```

#### 📋 CLI-Optionen
| Parameter | Standard | Beschreibung |
| :--- | :--- | :--- |
| `--config` | `config_rl.yaml` | Zentrale YAML mit allen RL-Defaults (Experiment, Optuna, Gateway, train_best). |
| `--experiment` | aus Config | Graph-Ordner: `nur_f`, `f_fp_roh`, `kein_inv`. |
| `--mode` | `tree_derivatives` | `tree`, `tree_derivatives` (welche Teilgraphen über den `global`-Knoten verknüpft werden). |
| `--edge-direction` | `top_down` | AST-Kantenrichtung (parent→child, child→parent oder beides). |
| `--add-kappa` | *aus* | Flag: Kappa- (h-Funktions-) Subgraphen aus `datasets/kappas/` einbinden (Opt-in). |
| `--feature-groups` | alle | Feature-Klassen: `node`, `topology`, `positional`, `edge`. |
| `--positional-encoding` | alle Gruppen | Anchor-PE-Gruppen (`anchor_additive`, …, `anchor_transcendental`) oder `none`. Inkompatibel mit `--add-virtual-supernode`. |
| `--active-features` | — | Explizite Knoten-Feature-Liste (überschreibt Gruppen). |
| `--timesteps` | `10000` | PPO-Schritte pro Trial. |
| `--n_trials` | `50` | Anzahl Optuna-Trials. |
| `--n-envs` | `1` | Parallele Mathematica-Umgebungen. |
| `--continue-study` | *aus* | Letzte Studie fortsetzen statt neu starten. |

---

### Phase 2: Finales Training (`train_best.py`)
Liest den besten Trial aus der SQLite-Optuna-Datenbank aus und trainiert das optimale GNN-PPO-Modell über einen ausgedehnten Zeitraum.

#### 🏃 Befehlsbeispiele
```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn/reinforcement_learning

# 1. Parameter-Überprüfung des besten Trials (Dry-Run):
python train_best.py --config config_rl.yaml --db optuna_kein_inv_n45g69_20260527_163125.db --dry-run

# 2. Ausgedehntes Training mit Feature-Experiment (nur periodische Anchor-PE):
python train_best.py --config config_rl.yaml --db optuna_kein_inv.db --timesteps 300000 --positional-encoding anchor_periodic --n-envs 4
```

#### 📋 CLI-Optionen
* `--db`: **Erforderlich**. Pfad zur SQLite-Optuna-Datenbankdatei.
* `--config`: YAML-Defaults (`config_rl.yaml`); explizite CLI-Flags überschreiben YAML-Werte.
* `--edge-direction`, `--add-kappa`, `--feature-groups`, `--positional-encoding`, `--active-features`: wie im Supervised-Workflow.
* `--timesteps`: Gesamte Anzahl der Trainingsschritte (Standard aus `config_rl.yaml`: `250000`).
* `--save-dir` / `--model-name`: Ausgabeordner und Modellname.
* `--dry-run`: Lädt nur die Hyperparameter und initialisiert das Modell zu Testzwecken.
* `--no-torch-compile`: Deaktiviert `torch.compile` für die GNN-Module.

---

## 📈 MLflow & Visualisierung

Die Protokollierung aller Trainingsläufe (Supervised & RL) erfolgt in Echtzeit über **MLflow**.

1. Starten Sie den MLflow-Server im WSL-Hintergrund oder in einer separaten Shell:
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
2. Öffnen Sie Ihren Browser auf Windows oder WSL unter: **`http://localhost:5000`**
3. Vergleichen Sie Metriken wie Loss-Kurven, Accuracy, F1-Scores (für Supervised) oder Epochen-Belohnungen und Policy-Verluste (für RL).

---

## 🧮 Ableitungsverknüpfung & XML-Container-Import

Das System unterstützt den Import von kombinierten mathematischen Graphen aus XML-Container-JSON-Dateien. Ein solcher Datensatz enthält XML-GraphML-Strings für die Funktion, die 1. Ableitung und die 2. Ableitung (`graphml_f`, `graphml_derivative1`, `graphml_derivative2`).

### Funktionsweise
Beim Einlesen wird das Python-Modul `create_virtual_global_node` aufgerufen:
1. **Renamimg / Namensräume**: Die Knoten der einzelnen Teilgraphen werden mit Präfixen (`f_`, `d1_`, `d2_`) versehen, um ID-Kollisionen auszuschließen.
2. **Klassifizierung**: Die XML-Knotennamen werden geparst (z.B. `{Plus, {}}` -> `Plus`), in Knotentypen (`operator`, `constant`, etc.) übersetzt und die Werte (z.B. Brüche wie `1/5` -> `0.2`) extrahiert.
3. **Virtueller globaler Knoten**: Ein zentraler virtueller Knoten (`global`) wird erzeugt.
4. **Verknüpfung**: Das Modul sucht die Wurzelknoten (Knoten mit In-Degree = 0) der drei Teilgraphen und verknüpft sie direkt mit dem globalen Knoten über spezifische Kanten:
   - `global -> f_root` (Kanten-Typ: `belongs_to_f`)
   - `global -> d1_root` (Kanten-Typ: `belongs_to_d1`)
   - `global -> d2_root` (Kanten-Typ: `belongs_to_d2`)

### Graphen-Kompilierung basierend auf `--mode`
Welche Teilgraphen geladen und wie sie strukturiert werden, hängt direkt vom `--mode` Argument ab (gilt gleichermaßen für Supervised Learning und Reinforcement Learning):
* **`tree`**: Es wird ausschließlich der Funktionstree $f$ (`graphml_f`) geladen. Ableitungsgraphen werden ignoriert.
* **`tree_derivatives`**: Es werden alle drei mathematischen Trees ($f$, $f'$, $f''$) geladen und über den `global`-Knoten verknüpft; jede Teilbaum-Wurzel wird per `root_color` als f/d1/d2 markiert (keine separaten Aggregator-Knoten).

> **Kappa-Augmentierung (`--add-kappa`)** ist orthogonal zum `--mode`: Ist sie aktiviert, werden zusätzlich Kappa- (h-Funktions-) Subgraphen aus `datasets/kappas/` über `GlobalToKappa`/`KappaToGlobal`-Kanten an den `global`-Knoten angebunden. Standardmäßig deaktiviert.
