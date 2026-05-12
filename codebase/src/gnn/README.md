# GNN-RL-Pipeline (Mathematica · ZeroMQ · Stable-Baselines3 · Optuna)

Diese Pipeline trainiert einen **PPO-Agenten** mit **graph-neuralem Feature-Extractor** auf Symbolik-Graphen. Der Agent steuert einen externen **Mathematica**-Solver über **ZeroMQ**; Hyperparameter werden mit **Optuna** gesucht, Lauf-Metadaten mit **MLflow** protokolliert.

---

## Was die Pipeline macht (Kurzüberblick)

1. **NetworkGateway** bindet lokale ZeroMQ-Ports und multiplext Zustands- vs. Terminal-/Reward-Nachrichten in eine gemeinsame Warteschlange.
2. **Preprocessor** lädt pro Graph-ID die statische Topologie aus `graphs/<experiment>/` (Cache), ergänzt pro Schritt skalare **global_features** aus der Mathematica-Nachricht und baut PyTorch-Geometric-`Data`.
3. **MathematicaGraphEnv** (Gymnasium) führt die Episode, puffert Transitionen und triggert nach Episode-Ende die rückwirkende Reward-Berechnung.
4. **RewardCalculator** setzt `basis_reward`, zeitlich diskontierte Schritt-Rewards und einen Rekord-Verbesserungsanteil (`alpha`, `reward_gamma`).
5. **CustomGNNFeaturesExtractor** (SB3) führt die gewählte GNN-Architektur aus und liefert feste Feature-Dimensionen an die Policy.
6. **Optuna** variiert Seeds, PPO-, GNN- und Reward-Parameter; **MedianPruner** kann schlechte Trials früh beenden.
7. **MLflow** legt pro Trial einen Run an, loggt alle gesampelten Parameter und die Metrik `final_mean_reward` (sowie Tags `status`: `completed` / `pruned`).

Zwischenstände der mittleren Episoden-Belohnung werden an Optuna gemeldet (`trial.report` über **Timesteps**) — für Kurven und Pruning-Analysen die Optuna-Study-DB verwenden.

---

## Voraussetzungen

- **Python** mit u. a. `torch`, `torch-geometric`, `stable-baselines3`, `gymnasium`, `pyzmq`, `optuna`, `mlflow`, `numpy`.
- **Mathematica-Seite**, die die konfigurierten Ports nutzt und mit dem Nachrichtenformat der Umgebung zusammenpasst.
- Graph-Daten unter `codebase/src/gnn/graphs/<experiment>/` (z. B. `P1_meta.json` / `P1_ast.graphml` — der Preprocessor löst `*_meta.json` bevorzugt).

Standard-**ZeroMQ-Ports** (in `main.py` fest, bei Bedarf im Code anpassen):

| Rolle        | Port |
|-------------|------|
| State (in)  | 5650 |
| Actions out | 5651 |
| Terminal / Reward (in) | 5693 |
| Control (PUB) | 6000 |

---

## Start aus dem GNN-Verzeichnis

Alle Pfade beziehen sich auf das Verzeichnis `codebase/src/gnn/` (damit `graphs/` und relative Pfade stimmen).

```bash
cd codebase/src/gnn
python main.py [OPTIONEN]
```

---

## CLI-Argumente (`python main.py`)

| Option | Standard | Beschreibung |
|--------|----------|--------------|
| `--experiment` | `nur_f` | Datensatz/Graph-Ordner: `nur_f`, `f_fp_roh` oder `kein_inv` (Unterordner von `graphs/`). |
| `--timesteps` | `10000` | PPO-Umgebungsschritte pro Optuna-Trial. |
| `--n_trials` | `50` | Anzahl Optuna-Trials pro Studien-Lauf. |
| `--no-torch-compile` | aus | Wenn gesetzt: kein `torch.compile` auf dem GNN (sonst Versuch mit dynamischen Shapes, bei Fehler Fallback laut `maybe_torch_compile`). |
| `--num_envs` | `1` | Anzahl gleichzeitig aktiver Slots (= in-flight UUIDs) in der **vektorisierten** Umgebung. `1` = bisheriges synchrones Verhalten. Höhere Werte lassen Mathematica mehrere Probleme parallel bearbeiten und bündeln die Policy-Inferenz pro PPO-Step. Sollte ≤ Mathematica-Batchgröße sein. |
| `--n_steps` | PPO-Default (2048) | PPO-Rolloutlänge **pro Slot**. Mit `--num_envs > 1` wächst der Rollout-Buffer auf `n_steps * num_envs`; entsprechend `n_steps` reduzieren, um Speicher und Update-Frequenz vernünftig zu halten. |

Hilfe anzeigen:

```bash
python main.py --help
```

---

## Optuna-Studie und Speicher

- **Storage:** SQLite-Datei pro Experiment im GNN-Ordner, z. B. `sqlite:///optuna_nur_f.db` für `--experiment nur_f`.
- **`load_if_exists=True`:** Studie wird fortgesetzt, wenn die DB schon existiert (gleicher `study_name`).
- **Richtung:** Maximierung des Rückgabewerts der Objective-Funktion (Mittelwert der letzten Episoden-Rewards im `ep_info_buffer` nach Training bzw. vor Pruning-Abbruch).

Nach einem Lauf zeigt die Konsole den besten Trial und dessen Parameter.

---

## MLflow

- **Experimentname:** `GNN_RL_Optuna_<experiment>` (z. B. `GNN_RL_Optuna_nur_f`).
- Es wird kein explizites `set_tracking_uri` im GNN-`main.py` gesetzt — Standard ist das lokale Backend (typisch `./mlruns` im Arbeitsverzeichnis oder konfigurierter URI über Umgebungsvariablen).

UI starten (üblich):

```bash
mlflow ui
```

Dann im Browser das Tracking-UI öffnen und das Experiment auswählen.

---

## GNN-Architekturen (Optuna-kategorial)

`gnn_architectures.py` definiert genau vier Varianten (`ARCHITECTURE_NAMES`):

- `gatv2_stack` — GATv2, nutzt `heads`
- `gcn_stack` — GCN (`heads` wird ignoriert, API bleibt gleich)
- `sage_stack` — GraphSAGE
- `gin_stack` — GIN

Weitere gesampelte Netz-Hyperparameter: `hidden_dim` ∈ {64, 128, 256}, `heads` ∈ {2, 4, 8} (wo relevant).

---

## Wichtige Modul-Dateien

| Datei | Rolle |
|-------|--------|
| `main.py` | Einstieg, Argumente, Gateway/Preprocessor, Optuna-Objective, MLflow-Runs |
| `mathematica_env.py` | Gymnasium-Env, Beobachtungsraum, Episode-Logik (single-stream) |
| `mathematica_vec_env.py` | Slot-basierte SB3-`VecEnv` über denselben Gateway: hält `num_envs` UUIDs gleichzeitig in-flight, sendet pro PPO-Step `num_envs` Actions als Batch, routet Antworten per UUID auf Slots zurück und füllt fertige Slots aus dem Pool frischer Initial-States nach. |
| `network_gateway.py` | ZeroMQ-Bridge, Event-Queue |
| `preprocessor.py` | Graph laden/cachen, Features aus Nachricht |
| `reward.py` | Episode-Reward-Shaping |
| `sb3_extractor.py` | SB3-Feature-Extractor mit GNN |
| `gnn_architectures.py` | Modellfabrik und `torch.compile`-Hilfe |
| `replay_buffer.py` | Episoden-Puffer für Reward-Nachrechnung |
| `graph_utils.py` | Konvertierung Rohgraph → PyG |

Tests und ältere ZeroMQ-Hilfsskripte liegen unter `tests_and_archive/` (kein Pflichtteil der Trainings-Pipeline).

---

## Typische Probleme

- **Ports belegt:** Andere Instanz beenden oder Ports in `main.py` / Mathematica anpassen.
- **Graph nicht gefunden:** `id` in der Nachricht muss zu `graphs/<experiment>/<id>_meta.json` (oder Fallback `.json`) passen.
- **Pruning sehr aggressiv:** `MedianPruner`-Parameter und `--timesteps` in `main.py` prüfen.
- **KeyboardInterrupt:** Gateway wird im `finally`-Block gestoppt und aufgeräumt.

---

## Kurzbeispiele

Nur kleines Experiment, wenige Trials:

```bash
cd codebase/src/gnn
python main.py --experiment nur_f --n_trials 5 --timesteps 2000
```

Anderes Graph-Set, Torch-Compile deaktivieren (Debugging / ältere GPUs):

```bash
python main.py --experiment f_fp_roh --no-torch-compile
```

Vectorized-Run mit 64 parallelen Slots und kürzerem Rollout pro Slot (Mathematica muss mind. 64 Probleme gleichzeitig liefern können):

```bash
python main.py --num_envs 64 --n_steps 64 --timesteps 50000
```
