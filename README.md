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
* `codebase/src/gnn/`: Dateien für Reinforcement Learning.
  * `main.py`: Einstiegspunkt für Phase 1 (Optuna Hyperparameter-Tuning).
  * `train_best.py`: Einstiegspunkt für Phase 2 (Bestes PPO-Modell trainieren).
  * `shared/utils/graph_utils.py`: Kernmodul zur Graphml-Konvertierung, Feature-Extraktion und Verknüpfung von Ableitungen.
  * `shared/utils/unified_loader.py`: Der neue vereinheitlichte `UnifiedDataLoader` (Singleton/Multiton-Muster) zur synchronen Verwaltung von Tabellen- und Graphen-Daten.

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

# 3. Standard GNN-Lauf im Graph-Modus (mit virtuellen Knoten) und 8 Basis-Features:
python main.py --mode graph

# 4. GNN-Lauf im Tree-Modus (Globale Features) mit 19 angereicherten (Enriched) Features:
python main.py --mode tree --enrich

# 5. GNN-Lauf mit dynamischer Feature-Filterung (Slicing):
python main.py --mode graph --enrich --active-features "node_type,depth,value,virtual_current_x_val"
```

#### 📋 CLI-Optionen
| Parameter | Standard | Optionen | Beschreibung |
| :--- | :--- | :--- | :--- |
| `--dataset` | `run_20260408_160456/dataset_4` | String | Pfad zum Dataset (z. B. `run_key/dataset_name`). |
| `--dry-run` | *aus* | Flag | Lädt den Datensatz kurz, um die Struktur zu validieren (ohne volles Training). |
| `--mode` | `graph` | `graph`, `tree`, `tree_derivatives` | Bestimmt das Graphen- und Features-Layout:<br>• `graph`: f, f' und f'' verbunden über den globalen Knoten + alle 3 virtuellen Knoten für dynamische Solver-Werte.<br>• `tree`: Nur der Funktionstree f (keine Ableitungen, keine Ableitungsknoten geladen) ohne virtuelle Knoten; dynamische Werte werden direkt als Slots auf dem globalen Knoten platziert.<br>• `tree_derivatives`: f, f' und f'' verbunden über den globalen Knoten (ohne virtuelle Knoten); dynamische Werte werden direkt auf dem globalen Knoten platziert. |
| `--enrich` | *aus* | Flag | Schaltet zwischen **8 Basis-Features** (ohne Flag) und **19 angereicherten topologischen/spektralen Features** (mit Flag) um. |
| `--active-features` | `None` | String (Komma-separiert) | Erlaubt die Übergabe einer exakten Teilmenge an Features (z. B. `"node_type,depth,value"`). Passt die Eingangsdimensionen des GNNs automatisch an. |

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
    mode: graph        # graph oder tree
    enrich: True       # Feature-Enrichment aktivieren/deaktivieren
    active_features: "node_type,value"   # Optional: Feature-Selektion
  ```

---

## 🎮 2. Reinforcement Learning Workflow

Der RL-Workflow lernt eine optimale Policy zur Steuerung des mathematischen Solvers über ein PPO-Modell.

### Phase 1: Hyperparameter-Tuning (`main.py`)
Startet eine Optuna-Studie zur parallelen Suche der besten Hyperparameter (PPO-Struktur, GNN-Backbone und Belohnungskoeffizienten).

#### 🏃 Befehlsbeispiele
```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn

# Tuning-Lauf mit 4 parallelen mathematischen Slots starten:
python main.py --experiment kein_inv --n_trials 50 --timesteps 16384 --n-envs 4

# Pausierte Studie fortsetzen (Resume):
python main.py --experiment kein_inv --n_trials 50 --timesteps 16384 --continue-study --n-envs 4
```

#### 📋 CLI-Optionen
* `--experiment`: Datensatz/Graph-Ordner unter `graphs/` (`nur_f`, `f_fp_roh`, `kein_inv`).
* `--timesteps`: PPO-Schritte pro Trial (Standard: `10000`).
* `--n_trials`: Gesamtanzahl der Suchen (Standard: `50`).
* `--n-envs`: Anzahl paralleler mathematischer Umgebungen (Standard: `1`).
* `--continue-study`: Setzen Sie dieses Flag, um die letzte nicht abgeschlossene Studie fortzusetzen, anstatt sie zu überschreiben.

---

### Phase 2: Finales Training (`train_best.py`)
Liest den besten Trial aus der SQLite-Optuna-Datenbank aus und trainiert das optimale GNN-PPO-Modell über einen ausgedehnten Zeitraum.

#### 🏃 Befehlsbeispiele
```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn

# 1. Parameter-Überprüfung des besten Trials (Dry-Run):
python train_best.py --db optuna_kein_inv_n45g69_20260527_163125.db --experiment kein_inv --dry-run

# 2. Ausgedehntes Training (z.B. 300.000 Schritte) starten:
python train_best.py --db optuna_kein_inv_n45g69_20260527_163125.db --experiment kein_inv --timesteps 300000 --n-envs 4
```

#### 📋 CLI-Optionen
* `--db`: **Erforderlich**. Pfad zur SQLite-Optuna-Datenbankdatei.
* `--timesteps`: Gesamte Anzahl der Trainingsschritte für das beste Modell (Standard: `250000`).
* `--save-dir`: Speicherverzeichnis für Kontrollpunkte (Checkpoints) und Finalmodelle.
* `--dry-run`: Lädt nur die Hyperparameter und initialisiert das Modell zu Testzwecken.
* `--no-torch-compile`: Deaktiviert `torch.compile` für die GNN-Module (falls Kompatibilitätsprobleme auftreten).

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
* **`tree`**: Es wird ausschließlich der Funktionstree $f$ (`graphml_f`) geladen. Ableitungsgraphen werden ignoriert, und es werden keine virtuellen Knoten injiziert.
* **`tree_derivatives`**: Es werden alle drei mathematischen Trees ($f$, $f'$, $f''$) geladen und über den `global`-Knoten verknüpft, jedoch ohne virtuelle Knoten.
* **`graph`**: Es werden alle drei mathematischen Trees ($f$, $f'$, $f''$) geladen und über den `global`-Knoten verknüpft, zusätzlich werden die drei virtuellen Knoten (`virtual_current_x`, `virtual_f_x`, `virtual_y_target`) und ihre bidirektionalen Verbindungen erzeugt.
