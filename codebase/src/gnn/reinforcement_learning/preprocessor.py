import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, FrozenSet

import torch
from torch_geometric.data import Data

from gnn.shared.utils.graph_loader import GraphDataLoader
from gnn.reinforcement_learning.feature_layout import (
    NATIVE_GLOBAL_FEATURE_COUNT,
    NATIVE_NODE_FEATURE_COUNT,
)
from gnn.reinforcement_learning.observation_sanitize import finite_float, sanitize_torch_features

# NOTE: `solver` is intentionally NOT a global feature — the chosen solver is the
# network's action/decision, not an observation. Removing it makes the global vector 8-wide.
STATE_GLOBAL_FEATURE_KEYS = (
    "currentX",
    "yTarget",
    "lastStepError",
    "fx",
    "dfx",
    "ddfx",
    "kappa",
    "lastKappa",
)


class Preprocessor:
    def __init__(
        self,
        loader: GraphDataLoader | None = None,
        graphs_dir: str | None = None,
        graph_cache_max: int = 128,
        mode: str = "graph",
        active_features: list[str] | None = None,
        add_kappa: bool = False,
        add_virtual_supernode: bool = False,
    ):
        self.mode = mode
        self.active_features = active_features
        self._graph_cache_max = graph_cache_max
        self._pyg_template_cache: OrderedDict[str, Data] = OrderedDict()

        if loader is not None:
            self.loader = loader
            self.graphs_dir = loader.source_path
            self.converter = loader.converter
        elif graphs_dir is not None:
            self.loader = GraphDataLoader(
                name=Path(graphs_dir).name,
                mode=mode,
                base_dir=graphs_dir,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
            )
            self.graphs_dir = Path(graphs_dir)
            self.converter = self.loader.converter
        else:
            raise ValueError("Either 'loader' or 'graphs_dir' must be provided to Preprocessor.")

        self._known_problem_ids = frozenset(self.loader.list_graph_ids())

    @property
    def padded_node_feature_count(self) -> int:
        structural = len(self.active_features) if self.active_features is not None else NATIVE_NODE_FEATURE_COUNT
        return structural + NATIVE_GLOBAL_FEATURE_COUNT

    @property
    def known_problem_ids(self) -> FrozenSet[str]:
        return self._known_problem_ids

    def _store_template(self, cache_key: str, data: Data) -> None:
        self._pyg_template_cache[cache_key] = data.clone()
        self._pyg_template_cache.move_to_end(cache_key)
        while len(self._pyg_template_cache) > self._graph_cache_max:
            self._pyg_template_cache.popitem(last=False)

    def _graph_template_for_problem_id(self, graph_id: Any) -> Data:
        cache_key = str(graph_id)
        template = self._pyg_template_cache.get(cache_key)
        if template is not None:
            self._pyg_template_cache.move_to_end(cache_key)
            return template.clone()

        data = self.loader.get_graph(graph_id)
        self._store_template(cache_key, data)
        return data

    def process(self, message: Dict[str, Any]):
        graph_id = message.get("id")
        if graph_id is None:
            raise ValueError("Message has no 'id'; cannot load graph.")

        extracted_features = {
            key: finite_float(message.get(key))
            for key in STATE_GLOBAL_FEATURE_KEYS
        }

        data = self._graph_template_for_problem_id(graph_id)

        data.uuid = message.get("uuid")

        if self.active_features is not None and data.x is not None:
            from gnn.shared.utils.graph_utils import slice_active_features
            data.x = slice_active_features(data.x, self.active_features)

        # Broadcast solver-state scalars to all nodes so they participate in
        # message passing — same approach as SL, which encodes all information
        # in node features rather than merging post-pooling.
        feat_list = [extracted_features[key] for key in STATE_GLOBAL_FEATURE_KEYS]
        state_vals = sanitize_torch_features(
            torch.tensor(feat_list, dtype=torch.float).unsqueeze(0).expand(data.x.shape[0], -1)
        )
        data.x = torch.cat([data.x, state_vals], dim=1)

        return data, extracted_features
