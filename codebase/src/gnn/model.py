"""
Backward-compatible exports for the SB3 graph backbone.

Implementations live in ``gnn_architectures``; ``TestGraphNetwork`` is the
historical name for the GATv2 stack.
"""

from gnn_architectures import (
    ARCHITECTURE_NAMES,
    GATv2StackNetwork,
    GINStackNetwork,
    GCNStackNetwork,
    SAGEStackNetwork,
    build_gnn,
    maybe_torch_compile,
)

TestGraphNetwork = GATv2StackNetwork

__all__ = [
    "ARCHITECTURE_NAMES",
    "GATv2StackNetwork",
    "GCNStackNetwork",
    "GINStackNetwork",
    "SAGEStackNetwork",
    "TestGraphNetwork",
    "build_gnn",
    "maybe_torch_compile",
]
