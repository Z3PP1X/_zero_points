"""Shared two-way learnable feature encoder (categorical embeddings + linear).

A single encoder reused by both the supervised (GraphGym) and RL (PPO) workflows.
It splits the incoming feature columns into:

  * **categorical** features (names present in ``categorical_registry``) -> one
    ``nn.Embedding`` each, and
  * **continuous** features (every other column) -> ``LayerNorm`` -> ``nn.Linear``,

then fuses both via a final ``nn.Linear`` to ``output_dim``.

The categorical/continuous split is resolved BY NAME from ``active_feature_names`` (the
ordered names of the columns actually present in ``x``). This makes the encoder robust to
any active-feature subset/reorder: a categorical column is embedded wherever it sits and
simply skipped when absent — replacing the old hard-coded ``node_type=col0`` / ``label_id=col1``
assumptions that forced a plain-linear fallback whenever a subset was selected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import LeakyReLU


class TwoWayFeatureEncoder(nn.Module):
    """Embed categorical columns + linearly encode continuous columns, then fuse.

    Args:
        active_feature_names: ordered names of the columns present in ``x``.
        output_dim: width of the fused output.
        categorical_registry: ``{feature_name: (vocab_size, embedding_dim)}``.
        activation: applied after fusion (default ``LeakyReLU``).

    ``forward(x)`` returns ``(encoded, node_type_ids)``. ``node_type_ids`` is the integer
    ``node_type`` column (zeros when ``node_type`` is not present, e.g. the edge encoder),
    kept because the RL backbone routes virtual/real nodes by node type.
    """

    def __init__(
        self,
        active_feature_names: list[str],
        output_dim: int,
        categorical_registry: dict[str, tuple[int, int]],
        activation: nn.Module | None = None,
    ):
        super().__init__()
        names = list(active_feature_names)
        self.activation = activation if activation is not None else LeakyReLU()

        # Partition columns by name into categorical (embedded) and continuous (linear).
        self.continuous_cols: list[int] = []
        cat_specs: list[tuple[str, int, int, int]] = []  # (name, col, vocab, emb_dim)
        for col, name in enumerate(names):
            if name in categorical_registry:
                vocab, emb_dim = categorical_registry[name]
                cat_specs.append((name, col, vocab, emb_dim))
            else:
                self.continuous_cols.append(col)

        self.embeddings = nn.ModuleDict()
        # (embedding_key, source_column, vocab_size) in a fixed order for forward concat.
        self._emb_order: list[tuple[str, int, int]] = []
        emb_total = 0
        for name, col, vocab, emb_dim in cat_specs:
            self.embeddings[name] = nn.Embedding(vocab, emb_dim)
            self._emb_order.append((name, col, vocab))
            emb_total += emb_dim
        self._emb_total = emb_total

        num_continuous = len(self.continuous_cols)
        if num_continuous > 0:
            cont_hidden = max(output_dim - emb_total, 4)
            self.cont_norm: nn.Module | None = nn.LayerNorm(num_continuous)
            self.cont_linear: nn.Module | None = nn.Linear(num_continuous, cont_hidden)
        else:
            cont_hidden = 0
            self.cont_norm = None
            self.cont_linear = None

        self.fusion = nn.Linear(cont_hidden + emb_total, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts: list[torch.Tensor] = []
        node_type_ids: torch.Tensor | None = None

        if self.continuous_cols:
            x_cont = x[:, self.continuous_cols]
            parts.append(self.cont_linear(self.cont_norm(x_cont)))

        for name, col, vocab in self._emb_order:
            ids = x[:, col].round().long().clamp(0, vocab - 1)
            parts.append(self.embeddings[name](ids))
            if name == "node_type":
                node_type_ids = ids

        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        encoded = self.activation(self.fusion(fused))

        if node_type_ids is None:
            node_type_ids = x.new_zeros(x.size(0), dtype=torch.long)
        return encoded, node_type_ids
