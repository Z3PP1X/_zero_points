# GNN-RL-Pipeline (Mathematica · ZeroMQ · Stable-Baselines3 · Optuna)

Diese Pipeline trainiert einen **PPO-Agenten** mit einem **graph-neuralen Feature-Extractor (GNN)** auf Symbolik-Graphen. Der Agent steuert einen externen **Mathematica**-Solver über **ZeroMQ**; Hyperparameter werden mit **Optuna** optimiert, Lauf-Metadaten mit **MLflow** protokolliert.

---

## 🎯 Hauptfunktionen

1. **Flexible Feature-Layouts**: Die Pipeline projiziert Rohgraphen mit beliebiger Knotendimension dynamisch in einen optimierbaren latenten Raum (`node_input_dim` und `global_input_dim`).
2. **Dynamische Aktivierungsfunktionen**: Unterstützt die Optimierung der Aktivierungsfunktion im GNN-Backbone und den MLP-Tails (`relu`, `leaky_relu`, `elu`, `tanh`, `gelu`).
3. **Zwei-Phasen-Workflow**:
   - **Phase 1: Tuning** mit Optuna (`main.py`) zur automatischen Suche der optimalen Parameter.
   - **Phase 2: Training** mit den besten Parametern (`train_best.py`) für einen ausgedehnten Trainingslauf mit MLflow-Tracking.
4. **Abwärtskompatibilität**: Ältere Studien-Datenbanken, die den Aktivierungsparameter noch nicht enthielten, werden beim Laden in Phase 2 automatisch auf `"leaky_relu"` zurückgesetzt.
5. **Trial-Kontrolle & Fortsetzbarkeit**:
   - Der `reward-states`-Zähler wird bei jedem Start eines neuen Optuna-Trials automatisch auf `0` zurückgesetzt.
   - Studien können pausiert und mit dem `--continue-study` CLI-Flag nahtlos fortgesetzt werden, anstatt bei jedem Neustart eine neue Studie zu erzwingen.

---

## 🛠️ Phase 1: Optuna Hyperparameter-Tuning (`main.py`)

Das Skript `main.py` startet eine Optuna-Studie, die systematisch den Hyperparameter-Suchraum exploriert.

### 🏃 Starten des Tunings
Führen Sie das Skript aus dem `gnn`-Verzeichnis aus. Aktivieren Sie vorab das entsprechende Conda-Environment (z. B. `pytorch`).

```bash
# In WSL / Linux Terminal wechseln und Environment aktivieren:
conda activate pytorch
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn

# Optuna Run starten:
python main.py --experiment kein_inv --n_trials 50 --timesteps 16384 --n-envs 4
```

### 📋 CLI-Parameter für `main.py`

| Parameter | Standard | Beschreibung |
| :--- | :--- | :--- |
| `--experiment` | `nur_f` | Datensatz/Graph-Ordner: `nur_f`, `f_fp_roh` oder `kein_inv` (Unterordner von `graphs/`). |
| `--timesteps` | `10000` | PPO-Umgebungsschritte pro Optuna-Trial. |
| `--n_trials` | `50` | Gesamtanzahl der zu suchenden Optuna-Trials. |
| `--n-envs` | `1` | Anzahl paralleler Mathematica-Slots in der vektorisierten Umgebung (`VecEnv`). |
| `--continue-study` | *aus (False)* | Flag zum Fortsetzen der letzten nicht beendeten Studie. Wenn nicht gesetzt, wird die existierende Studie gelöscht und neu gestartet. |
| `--timeout-fallback` | `5.0` | Initialer Timeout in Sekunden für Mathematica-Rückmeldungen. |
| `--timeout-cushion` | `2.0` | Pufferzeit in Sekunden, die auf den gleitenden Antwortzeit-Durchschnitt addiert wird. |
| `--timeout-window` | `100` | Fenstergröße für den gleitenden Durchschnitt der Antwortzeiten. |

### 🔍 Der Optuna-Suchraum
Folgende Parameter werden dynamisch gesucht und in der SQLite-Datenbank abgespeichert:

* **PPO Hyperparameter**:
  * `learning_rate` (logarithmisch zwischen `1e-6` und `9e-3`)
  * `gamma` (PPO-Diskontierungsfaktor, kontinuierlich zwischen `0.9` und `0.999`)
  * `ent_coef` (Entropie-Koeffizient, logarithmisch zwischen `1e-8` und `1e-2`)
* **Reward-Shaping**:
  * `alpha` (Skalierung der Rekord-Belohnung, kontinuierlich zwischen `0.1` und `5.0`)
  * `basis_reward` (Basis-Belohnung, kontinuierlich zwischen `0.1` und `2.0`)
  * `reward_gamma` (Diskontierung für gelöste Pfade, `0.97` bis `0.999`)
  * `step_cost_lambda` (Gewichtung der Schrittkosten, `1e-4` bis `0.5` logarithmisch)
  * `time_bad_penalty` (Strafe für schlechte Rechenzeit, `4` bis `10`)
  * `solver_mismatch_penalty` / `solver_match_bonus`
* **GNN & Feature-Layout**:
  * `gnn_architecture` (Auswahl aus: `gatv2_stack`, `gcn_stack`, `sage_stack`, `gin_stack`)
  * `gnn_activation` (Auswahl aus: `relu`, `leaky_relu`, `elu`, `tanh`, `gelu`)
  * `hidden_dim` (Projektionsdimension: `64`, `128` oder `256`)
  * `num_gnn_layers` (Anzahl GNN-Schichten: `2`, `3` oder `4`)
  * `heads` (GATv2 Attention Heads: `2`, `4` oder `8`)
  * `node_input_dim` (Knoten-Einbettung: `4` oder `5`)
  * `global_input_dim` (Globale-Einbettung: `6` oder `9`)

---

## 🚀 Phase 2: Training mit den besten Parametern (`train_best.py`)

Nach Abschluss der Optuna-Studie ermittelt `train_best.py` den besten Trial aus der SQLite-Datenbank, baut das exakte GNN-Modell und trainiert einen dedizierten Agenten über viele Schritte inklusive Checkpointing und MLflow-Visualisierung.

### 🏃 Starten des Trainings
```bash
# Wechseln Sie ins Verzeichnis
cd /home/zapp1x/GitHub/_bachelor/_zero_points/codebase/src/gnn

# Besten Run laden und trainieren (z. B. für 250k Schritte)
python train_best.py --db optuna_kein_inv_n4g6.db --experiment kein_inv --timesteps 250000
```

### 📋 CLI-Parameter für `train_best.py`

| Parameter | Standard | Beschreibung |
| :--- | :--- | :--- |
| `--db` | *Erforderlich* | Pfad zur Optuna SQLite-Datenbankdatei (z. B. `optuna_kein_inv.db`). |
| `--experiment` | `kein_inv` | Experiment-Name / Graph-Pfad: `nur_f`, `f_fp_roh` oder `kein_inv`. |
| `--study-name` | `None` | Name der Studie. Wenn `None`, wird die erste in der DB gefundene Studie geladen. |
| `--timesteps` | `250000` | Gesamte Anzahl an Trainings-Schritten für den optimalen Agenten. |
| `--n-envs` | `1` | Anzahl paralleler Mathematica-Instanzen. |
| `--save-dir` | `models` | Ordner zum Speichern von Zwischen-Checkpoints und des finalen Agenten. |
| `--model-name` | `gnn_ppo_best` | Basisname für gespeicherte ZIP-Modelldateien. |
| `--seed` | `None` | Ermöglicht das Überschreiben des besten Trial-Seeds mit einem benutzerdefinierten Seed. |
| `--dry-run` | aus | Führt nur das Parsen und Laden der Parameter aus (ohne ZeroMQ-Sockets zu binden oder das Training zu starten). Perfekt zum Testen von DBs! |
| `--no-torch-compile` | aus | Deaktiviert `torch.compile` für das GNN-Backbone. |

---

## 💡 Nützliche Terminal-Befehle & Beispiele

### 1. Einen schnellen Test-Tuning-Lauf (Optuna) machen:
```bash
python main.py --experiment kein_inv --n_trials 3 --timesteps 2000
```

### 2. Eine bestehende Optuna-Datenbank überprüfen (Dry-Run):
Dies lädt die beste Trial-Konfiguration und gibt alle Parameter sauber im Terminal aus, ohne ZeroMQ-Verbindungen aufzubauen.
```bash
python train_best.py --db optuna_kein_inv_n4g6.db --experiment kein_inv --dry-run
```

### 3. Vektorisiertes Tuning mit 4 parallelen Mathematica-Instanzen:
```bash
python main.py --experiment kein_inv --n_trials 50 --timesteps 16384 --n-envs 4
```

### 4. MLflow Benutzeroberfläche starten:
Visualisieren Sie Trainingskurven, vergleichen Sie Trials und bewerten Sie Hyperparameter direkt im Browser.
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Öffnen Sie anschließend `http://localhost:5000` in Ihrem Webbrowser.

### 5. Eine bestehende Optuna-Studie fortsetzen (Resume):
Setzen Sie eine unterbrochene oder pausierte Tuning-Studie nahtlos fort:
```bash
python main.py --experiment kein_inv --n_trials 50 --timesteps 16384 --continue-study
```

---

## 📂 Wichtige Modul-Dateien

* `main.py`: Haupteinstiegspunkt für Phase 1 (Optuna-Tuning).
* `train_best.py`: Haupteinstiegspunkt für Phase 2 (Bestes Modell trainieren).
* `shared/utils/unified_loader.py`: Der vereinheitlichte `UnifiedDataLoader` (Singleton/Multiton-Muster) zur synchronen Verwaltung von Tabellen- und Graphen-Daten.
* `feature_layout.py`: Definition des Suchraums, der Dimensionen und Aktivierungsfunktionen.
* `ppo_trial_config.py`: Datenstrukturen (`dataclasses`) für Modell- und Reward-Parameter.
* `ppo_optuna_search.py`: Logik zur Erzeugung von Parametern durch den Optuna-Sampler.
* `gnn_policy_backbone.py`: Die PyTorch-GNN-Netzwerkarchitektur mit dynamischer Aktivierungswahl und robustem Fallback.
* `ppo_optuna_workflow.py`: Verbindungsklasse zur Initialisierung des SB3-PPO-Modells.
* `mathematica_vec_env.py` / `mathematica_env.py`: Gymnasium-Umgebungen zur Kommunikation mit Mathematica.
* `reward.py`: Nachberechnung der asymmetrischen Rewards und Schrittkosten.

