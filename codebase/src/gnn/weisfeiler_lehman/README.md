# 1-Weisfeiler-Lehman distinguishability study

Are the expression graphs we build actually distinguishable from one another?
This module answers that with the classic **1-WL colour-refinement test**
(via PyTorch Geometric's `WLConv`). It loads a dataset's graphs exactly as a
normal pipeline run would (`UnifiedDataLoader`), but instead of training it runs
1-WL and reports which graphs the test can and cannot tell apart.

## What it does

1. Loads every graph of the dataset for the selected graph **mode(s)**.
2. Colours nodes with an initial scheme (default: the semantic operator/operand
   `label`), then iteratively refines colours with `WLConv`. A **single shared**
   `WLConv` per round colours all graphs, so colours are globally comparable.
3. Fingerprints each graph by its per-round colour histograms. Two graphs are
   **1-WL indistinguishable** iff every round's histogram matches — those graphs
   form an *equivalence class*.
4. Writes histograms, an equivalence-class report, and diagrams per mode, plus a
   cross-mode summary.

## Usage

```bash
conda activate pytorch
cd codebase/src/gnn/weisfeiler_lehman

# All three modes (graph, tree, tree-derivative), default dataset, no kappa
python main.py

# Only the graph mode, with kappa (h-function) subgraphs merged in
python main.py --mode graph --add-kappa

# Tree + tree-derivative on a custom dataset, more sample drawings
python main.py --mode tree tree_derivatives --dataset run_key/name --sample-graphs 8

# The 100-graph synthetic dataset (datasets/graphs/synthetic_graphs.json)
python main.py --synthetic
```

### Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode {graph,tree,tree_derivatives} ...` | all three | Graph mode(s); one results subdir each. |
| `--add-kappa` | off | Merge kappa subgraphs from `datasets/kappas/`; subdir gets a `-kappa` suffix. |
| `--synthetic` | off | Load the 100-graph synthetic set; results go to `synthetic-<mode>` subdirs. |
| `--dataset run_key/name` | supervised config | Dataset to load (mirrors a pipeline run). |
| `--coloring {label,node_type,degree,constant}` | `label` | Initial node colouring. `constant` = pure structure. |
| `--directed` | off | Refine on directed edges (default symmetrizes to classic undirected 1-WL). |
| `--edge-direction` | `top_down` | AST edge direction when loading. |
| `--iterations` | `10` | Max refinement rounds (stops early at a stable partition). |
| `--per-graph-histograms` | `50` | Cap on individual histogram PNGs (the CSV always has all graphs). |
| `--sample-graphs` | `6` | Sample graph drawings (one per largest class); `0` disables. |
| `--max-graphs` | none | Cap graphs loaded (quick runs). |

## Output layout

Base-level artifacts are suffixed with a per-dataset `<slug>` (`curated`,
`synthetic`, `curated-kappa`, …) so a curated and a synthetic run can share one
`results/` directory without overwriting each other.

```
results/
  study_table_<slug>.txt         # the box-drawing distinguishability table (also printed)
  study_summary_<slug>.json
  mode_comparison_<slug>.png      # distinguishability rate per mode (the study payoff)
  graph/                         # one subdir per mode: graph | tree | tree-derivative
    summary.json                 #   --add-kappa -> *-kappa; --synthetic -> synthetic-*
    histograms/
      color_heatmap.png          #   graphs x final WL colours
      histograms_matrix.csv      #   every graph's full colour histogram
      per_graph/<id>.png         #   one histogram per graph (capped)
    distinguishability/
      distinguishability_matrix.png   # N x N: dark = indistinguishable pair
      class_sizes.png            #   equivalence-class size distribution
      equivalence_classes.json   #   full class -> members mapping
      collisions.csv             #   only the groups that collide
    samples/<id>.png             #   sample graphs tinted by final WL colour
  tree/ ...
  tree-derivative/ ...
```

## Interpreting the results — a tutorial

This walks through every metric and artifact a run emits and how to read it. The
running example is the synthetic dataset (`python main.py --synthetic`).

### Background: what "1-WL distinguishable" means

The 1-WL test colours every node, then repeatedly recolours each node from
`(own colour, sorted multiset of neighbour colours)`. After each round it takes
the **histogram** of colours per graph. Two graphs are declared **the same**
(indistinguishable) if their histograms agree at *every* round; otherwise the
first round where they differ proves them distinct. This is the exact ceiling of
a standard message-passing GNN: **if 1-WL cannot separate two graphs, neither can
a GCN/GIN/GAT-style GNN.** So a collision here is a hard limit on what any such
model could learn to tell apart — directly relevant to whether the dataset is
learnable.

The initial colour comes from `--coloring` (default `label` = the semantic
operator/operand). `constant` ignores labels and tests **pure structure** — a
useful contrast: if two graphs separate under `label` but collide under
`constant`, only their labels distinguish them, not their shape.

### Step 1 — the summary table (`study_table_<slug>.txt`)

```
            mode            │ distinguishable classes │ colliding graphs │ rate │
├───────────────────────────┼─────────────────────────┼──────────────────┼──────┤
│ synthetic-graph           │ 27 / 100                │ 83               │ 0.27 │
├───────────────────────────┼─────────────────────────┼──────────────────┼──────┤
│ synthetic-tree            │ 24 / 100                │ 87               │ 0.24 │
├───────────────────────────┼─────────────────────────┼──────────────────┼──────┤
│ synthetic-tree-derivative │ 27 / 100                │ 83               │ 0.27 │
```

- **distinguishable classes** `C / N` — the `N` graphs collapse into `C`
  *equivalence classes* (groups 1-WL cannot tell apart internally). `27 / 100`
  means the 100 graphs are really only 27 distinct "shapes" to the test.
- **colliding graphs** — how many of the `N` graphs sit in a class with at least
  one other member (i.e. are *not* unique). `N − colliding` graphs are unique.
  Here `100 − 83 = 17` graphs are 1-WL-unique; the other 83 share their
  fingerprint with someone.
- **rate** = `C / N`, the **distinguishability rate**. `1.0` = every graph is
  unique (ideal); lower = more collisions. `0.27` is low — the generator is
  producing many graphs that are structurally interchangeable to a GNN.
- **comparing modes/rows**: `tree` drops to `0.24` because discarding the
  derivative subgraphs removes structure, so *more* graphs collide. Adding more
  signal (kappa via `--add-kappa`, or the full `graph` mode) should raise the
  rate; if it doesn't, that extra signal is invisible to 1-WL.

### Step 2 — which graphs collide (`distinguishability/`)

- **`collisions.csv`** — one row per colliding class: `class_index, size,
  members`. This is the actionable list: e.g. `0, 18, P_a P_b …` means those 18
  graphs are mutually indistinguishable. Start here to inspect *why* (often
  identical skeletons differing only in numeric constants 1-WL ignores).
- **`equivalence_classes.json`** — the complete partition (every class and its
  members, plus the headline counts), for programmatic analysis.
- **`distinguishability_matrix.png`** — an `N × N` grid, graphs ordered so class
  members are adjacent. **Dark = indistinguishable pair.** Read it as block
  structure: big dark squares on the diagonal are large collision clusters; an
  all-light off-diagonal means classes are cleanly separated. A perfect dataset
  would be dark only on the 1-pixel diagonal.
- **`class_sizes.png`** — bar chart of class sizes, largest first. Orange bars
  (size > 1) are collisions, blue bars (size 1) are unique graphs. A long orange
  head = a few huge interchangeable clusters; a long blue tail = many uniques.

### Step 3 — the histograms (`histograms/`)

- **`per_graph/<id>.png`** — for one graph, a bar chart of how many nodes carry
  each final WL colour (its colour *fingerprint*). **Two graphs in the same
  equivalence class have identical histograms across all rounds**; eyeballing two
  per-graph charts shows you concretely what "the same to 1-WL" looks like.
- **`color_heatmap.png`** — all graphs (rows) × WL colours (columns), cell =
  node count. Rows that look identical are collisions; distinctive bright columns
  are colours unique to a few graphs (structural signatures).
- **`histograms_matrix.csv`** — the same matrix as raw numbers (every graph,
  every colour), for your own plots/stats. Unlike the PNGs it is never capped.

### Step 4 — sample graphs and the cross-mode plot

- **`samples/<id>.png`** — one representative graph per largest class, drawn with
  nodes tinted by final WL colour. Nodes sharing a colour are
  refinement-equivalent; this makes the abstract colouring tangible.
- **`mode_comparison_<slug>.png`** — the study payoff: distinguishability rate
  per mode side by side (the same numbers as the table's `rate` column).

### Drawing a conclusion

Read the **rate** first (how learnable/separable is this dataset at all?), then
open **`collisions.csv`** and the **matrix** to see *which* graphs are the
problem, and finally the **per-graph histograms** to understand *why* they
collide. Re-running with `--coloring constant` vs `label` isolates whether the
distinguishing signal is structural or only in the labels. Because 1-WL bounds
message-passing GNNs, a low rate is a concrete warning that the dataset (or graph
mode) cannot be fully separated by such a model — exactly the question this study
exists to answer.
