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

# NOTE: neither `solver` nor `kappa` is a global scalar feature. `solver` is the
# network's action/decision, not an observation. `kappa` is the live per-step
# h-function index; it now drives graph TOPOLOGY via per-step augmentation (see
# Preprocessor._resolve_kappa), so feeding it as a raw scalar too would be redundant.
# `lastKappa` is kept. The result is a 7-wide global vector — it MUST equal
# NATIVE_GLOBAL_FEATURE_COUNT (asserted below; that constant drives the obs Box width).
STATE_GLOBAL_FEATURE_KEYS = (
    "currentX",
    "yTarget",
    "lastStepError",
    "fx",
    "dfx",
    "ddfx",
    "lastKappa",
)

assert len(STATE_GLOBAL_FEATURE_KEYS) == NATIVE_GLOBAL_FEATURE_COUNT, (
    f"STATE_GLOBAL_FEATURE_KEYS has {len(STATE_GLOBAL_FEATURE_KEYS)} entries but "
    f"NATIVE_GLOBAL_FEATURE_COUNT is {NATIVE_GLOBAL_FEATURE_COUNT}; keep them in sync "
    f"(feature_layout.NATIVE_GLOBAL_FEATURE_COUNT drives the observation Box width)."
)

# The kappa h-function library (datasets/kappas/kappas.json) defines only integer
# indices in [-25, 25] \ {0}. Live kappa values are clamped to this range; 0 means
# "no kappa subgraph" (the library has no 0 entry).
KAPPA_CLIP_ABS = 25


class Preprocessor:
    def __init__(
        self,
        loader: GraphDataLoader | None = None,
        graphs_dir: str | None = None,
        graph_cache_max: int = 128,
        mode: str = "tree_derivatives",
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
        if self.active_features is not None:
            return len(self.active_features)
        return NATIVE_NODE_FEATURE_COUNT

    @property
    def known_problem_ids(self) -> FrozenSet[str]:
        return self._known_problem_ids

    def _store_template(self, cache_key: str, data: Data) -> None:
        self._pyg_template_cache[cache_key] = data.clone()
        self._pyg_template_cache.move_to_end(cache_key)
        while len(self._pyg_template_cache) > self._graph_cache_max:
            self._pyg_template_cache.popitem(last=False)

    def _resolve_kappa(self, message: Dict[str, Any]) -> float:
        """Resolve the live, per-step kappa (h-function index) for graph augmentation.

        The Mathematica state carries an integer ``kappa`` = the optimal h-function
        index at the current iterate x; it changes every step. We clamp it to the
        library bounds ``[-25, 25]`` (kappas.json has no 0 entry, so 0 => no subgraph)
        and return it as a float for the exact-match kappa lookup in the loader.

        A *missing or null* ``kappa`` is treated as a protocol error and raised loudly
        — it must NOT be silently coerced to 0.0 ("no subgraph"), which would conflate a
        real bug with a legitimate zero (cf. memory rl-state-key-silent-fallback and
        observation_sanitize.finite_float). A present-but-zero value is fine.
        """
        raw = message.get("kappa")
        if raw is None:
            raise KeyError(
                "Live state is missing the 'kappa' key required for per-step graph "
                "augmentation. A present-but-zero kappa is legitimate (no subgraph); "
                "an absent/null key indicates a gateway/protocol mismatch — failing "
                "loudly rather than silently degrading to the base graph."
            )
        kappa = int(round(float(raw)))
        kappa = max(-KAPPA_CLIP_ABS, min(KAPPA_CLIP_ABS, kappa))
        return float(kappa)

    def _graph_template_for_problem_id(self, graph_id: Any, kappa_value: float) -> Data:
        # Cache key MUST include kappa: the live kappa changes every step, so a
        # graph_id-only key would freeze the first step's kappa subgraph for the whole
        # episode (a silent correctness bug). Passing an explicit (never-None) kappa_value
        # also keeps the loader on the live-augmentation path instead of the static
        # per-problem kappa_map fallback, so 0 => base graph regardless of the stage flag.
        cache_key = f"{graph_id}|k{kappa_value}"
        template = self._pyg_template_cache.get(cache_key)
        if template is not None:
            self._pyg_template_cache.move_to_end(cache_key)
            return template.clone()

        data = self.loader.get_graph(graph_id, kappa_value=kappa_value)
        self._store_template(cache_key, data)
        return data

    def process(self, message: Dict[str, Any]):
        graph_id = message.get("id")
        if graph_id is None:
            raise ValueError("Message has no 'id'; cannot load graph.")

        kappa_value = self._resolve_kappa(message)

        extracted_features = {
            key: finite_float(message.get(key))
            for key in STATE_GLOBAL_FEATURE_KEYS
        }

        data = self._graph_template_for_problem_id(graph_id, kappa_value)

        feat_list = [extracted_features[key] for key in STATE_GLOBAL_FEATURE_KEYS]
        # Raw (un-normalized) global features; the GlobalEncoder's learnable
        # LayerNorm + Linear handle scaling. No hand-crafted sign-log transform.
        raw_tensor = torch.tensor(feat_list, dtype=torch.float).unsqueeze(0)
        data.global_features = sanitize_torch_features(raw_tensor)

        data.uuid = message.get("uuid")

        if self.active_features is not None and data.x is not None:
            from gnn.shared.utils.graph_utils import slice_active_features
            data.x = slice_active_features(data.x, self.active_features)

        return data, extracted_features
