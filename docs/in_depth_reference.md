# In-Depth Reference: Augmented Function Graph Loader

This document provides a detailed technical reference for the classes and helper functions defined for loading and merging augmented function graphs in this repository.

---

## 1. Class API Reference

### `KappaEdge`
A helper structure representing a connection edge between the virtual global node and the entry point (root node) of a kappa subgraph.

#### Definition
```python
class KappaEdge:
    def __init__(self, source: str, target: str, type: str):
        self.source = source
        self.target = target
        self.type = type
        self.features = {"weight": 0.0}
```

* **`source`** (`str`): Identifier of the source node (typically the global context node).
* **`target`** (`str`): Identifier of the target node (root of the merged kappa subgraph).
* **`type`** (`str`): Edge type, e.g., `"GlobalToKappa"` or `"KappaToGlobal"`.
* **`features`** (`dict`): Feature dictionary initialized with a default weight of `0.0`.

---

### `AugmentedFunctionGraph(nx.DiGraph)`
Subclass of `networkx.DiGraph` encapsulating virtual node management and collision-free subgraph merging.

#### Methods

* **`HasGlobalNode() -> bool`**
  Checks if a node of type `"global"` or ID `"global"` exists in the graph.
  
* **`GetGlobalNode() -> str`**
  Retrieves the ID of the global node. Raises `KeyError` if no global node exists.
  
* **`CreateVirtualGlobalNode(nodeType: str = "GlobalContext") -> str`**
  Creates a virtual global node in the graph and assigns standard properties matching `ExpressionGraphConverter` requirements. Returns the ID (`"global"`).

* **`MergeDisjointSubgraph(kappa_subgraph: Union[nx.DiGraph, str, dict]) -> str`**
  Merges a disjoint kappa subgraph into the main graph.
  * **Collision Resolution**: Automatically shifts all incoming node IDs using a unique prefix pattern: `kappa_<counter>_<original_id>`.
  * **Normalization**: Maps names, labels, and types into standard formats. Constants have their numeric values parsed and stored. All standard fields (e.g., `belongs_to_f`, `virtual_current_x_val`) are set to default values to align with GNN layout schemas.
  * **Returns**: The new, collision-free ID of the root node of the merged kappa subgraph.
  * **Raises**: `TypeError` on unsupported subgraph types, or `ValueError` on empty subgraphs.

* **`AddEdge(edge: KappaEdge) -> None`**
  Integrates a `KappaEdge` into the graph. Maps properties such as `direction`, `relation_type`, and `edge_type` automatically based on edge direction.

---

## 2. Loader Utility Reference

### `LoadGraphFromLocalStructure`
```python
def LoadGraphFromLocalStructure(folder: Union[Path, str], id: str) -> AugmentedFunctionGraph:
```
* **Implements**: Locates the main graph matching the ID. It scans the given folder for single JSON files, lists, or direct files matching `{id}.json` or `{id}_meta.json`.
* **Representation Compilation**: Supports both GraphML container strings (`graphml_f`, `graphml_derivative1`, `graphml_derivative2`) and pre-parsed node/edge dictionary schemas.

### `LoadAugmentedFunctionGraph`
```python
def LoadAugmentedFunctionGraph(
    graphId: str, graphsFolder: Union[str, Path], kappasFolder: Union[str, Path]
) -> AugmentedFunctionGraph:
```
* **Orchestration Workflow**:
  1. Load mathematical basis graph.
  2. Locate or create a virtual global node.
  3. Scan the `kappasFolder` for any `.json` files.
  4. Parse objects with `id == "kappa"`.
  5. Merge the subgraphs using `MergeDisjointSubgraph`.
  6. Insert bidirectional connections (`GlobalToKappa`, `KappaToGlobal`) from the global node to the merged kappa root, weighting them with the parsed kappa value.
