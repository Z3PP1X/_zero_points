# `config_settings/` — Architektur-Evolutionsstufen

Jede Unterstufe ist ein **eigenständiges, zugeschnittenes** Config-Paar
(`base_config.yaml` + `grid.yaml`), das nur die für diese Stufe relevanten und
**tatsächlich wirksamen** Parameter exponiert. Das macht die Entwicklung der
Netzwerkarchitektur nachvollziehbar und die Experimente reproduzierbar.

Auswahl über `run_all.py`:

```bash
python run_all.py --list-stages              # Stufen anzeigen
python run_all.py --stage 3 --dry-run        # nur Configs generieren (kein Training)
python run_all.py --stage 3 --experiment-name stufe3_lauf1
python run_all.py --stage stage3_graph_features   # Ordnername geht auch
```

Ohne `--stage` nutzt `run_all.py` weiterhin die Default-`config_supervised.yaml` +
`grid.yaml` (rückwärtskompatibel). `--config`/`--grid` überschreiben eine Stufe explizit.

## Überblick

| Stufe | Ordner | Neu in dieser Stufe | Backbone | Sweepbare Parameter (grid.yaml) |
|---|---|---|---|---|
| 1 | `stage1_tree_basic`        | Tree, Basisfeatures | `gnn` (edge-blind) | `layer_type{gcnconv,ginconv}`, `dim_inner`, `dropout`, `layers_mp` |
| 2 | `stage2_tree_derivatives`  | Tree-Derivatives, **edge_direction** | `gnn` (edge-blind) | + `edge_direction{top_down,bottom_up,bidirectional}` |
| 3 | `stage3_graph_features`    | **Graphen**, **Supernode** an/aus, **Kappa** an/aus, **erweiterte/Anchor-Features** | `gnn` (edge-blind) | `dim_inner`, `layers_mp`, `add_virtual_supernode`, `add_kappa`, `features.positional` |
| 4 | `stage4_edge_networks`     | **Edge-Networks** (nur `gatv2conv`/`gineconv`), Edge-Features | `expression_classifier` | `layer_type{gatv2conv,gineconv}`, `dim_inner` |
| 5 | `stage5_heterogeneous`     | **Heterogene Netzwerke** (nur hetero) | `expression_classifier` (hetero) | `dim_inner`, `layers_mp` |
| 6 | `stage6_hetero_diffpool`   | **Diffpool für hetero** (geplant) | `expression_classifier` (hetero) | wie Stufe 5; Pooling-Achse vorbereitet |

## Wichtige Hinweise zu wirksamen Parametern

Pro Stufe werden bewusst nur Parameter gesweept, die im jeweiligen Code-Pfad **etwas
bewirken** — sonst entstünden identische Doppelläufe:

- **Stufen 1–3** nutzen PyG's Stock-GNN (`model.type: gnn`) mit edge-blinden Layern.
  `gnn.dim_inner`, `gnn.dropout` und `gnn.layers_mp` werden hier voll honoriert.
- **Stufe 4** nutzt `expression_classifier` → `TestGraphNetwork`. Dieses hardcodet
  aktuell **3 MP-Layer** und verdrahtet `dropout` nicht; `layers_mp`/`dropout` sind
  daher **inert** und werden nicht gesweept (in der base_config gepinnt, kommentiert).
- **Stufen 5–6**: Der `HeteroExpressionClassifier` nutzt ausschließlich `SAGEConv` und
  macht heute flaches Message-Passing + Readout. Wirksam sind hier nur `gnn.dim_inner` und
  `gnn.layers_mp` (= `num_layers`). **Inert** und daher gepinnt (nicht gesweept):
  `gnn.layer_type`/`att_heads` (SAGEConv hat keine Conv-/Kopf-Wahl) und `gnn.dropout`
  (der Head hardcodet Dropout). Hierarchical Pooling (`topk`/`diffpool`) für heterogene
  Netze ist **noch nicht im Code verdrahtet** (`variant`/`pool_type` werden ignoriert);
  diese Stufen sind daher **Gerüste** mit auskommentierter Pooling-Achse, die aktiv wird,
  sobald der Hetero-Pooling-Code existiert.
- Heterogene Stufen erfordern das **volle Node-Schema** (`features.positional: true`): eine
  Feature-Teilmenge ist mit `heterogeneous: true` aktuell nicht unterstützt (das
  Dataset-seitige Slicing greift auf `data.x` zu, das es auf `HeteroData` nicht gibt).

## Constraint: Supernode ↔ Anchor-Features

`add_virtual_supernode: true` ist **inkompatibel** mit aktivem `features.positional`
(Anchor-PE) — ein voll vernetzter Supernode zerstört die Kürzeste-Pfad-Distanzen, auf
denen die Anchor-Kodierung beruht. In Stufe 3 werden solche Kombinationen vom Generator
(`configs_gen.py:_is_valid_config`) automatisch übersprungen.
