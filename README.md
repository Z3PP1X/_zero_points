# GNN-RL & Supervised Expression Pipeline

Dieses Repository enthält eine vollständige Pipeline für maschinelles Lernen auf mathematischen Symbolik-Graphen. Das Projekt umfasst:
1. **Supervised Learning Workflow**: Evaluierung und Training von Graph Neural Networks (GNNs) zur Vorhersage des optimalen mathematischen Solvers (Newton vs. gMGF).
2. **Reinforcement Learning Workflow**: Steuerung und Optimierung eines Mathematica-Solvers über einen PPO-Agenten mit GNN-Feature-Extractor (Zwei-Phasen-Workflow: Optuna-Tuning & dediziertes Training).

---

## 📂 Repository-Struktur und wichtige Pfade

* `codebase/src/gnn/supervised_learning/`: Dateien für Supervised Learning.
  * `main.py`: Benutzerdefiniertes GNN-Training und Evaluierung.
  * `main_graphgym.py` & `loader_graphgym.py`: Integration des PyTorch Geometric GraphGym Frameworks.
  * `config_supervised.yaml`: Konfigurationsdatei für GraphGym.
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
GraphGym ist ein Framework von PyTorch Geometric zur standardisierten Modell-Evaluierung über YAML-Dateien.

#### 🏃 Ausführen mit GraphGym
```bash
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn/supervised_learning
python main_graphgym.py --cfg config_supervised.yaml
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
