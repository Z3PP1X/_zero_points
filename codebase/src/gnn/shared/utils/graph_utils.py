# Backward-compatible re-export shim. All names remain importable from this
# module without changing any call site.
from gnn.shared.utils.graph_vocab import (
    CANONICAL_LABELS, CANONICAL_LABEL_VOCAB,
    ANCHOR_GROUP_FEATURES, ANCHOR_GROUP_BY_LABEL,
    ANCHOR_EXCLUDED_NODE_IDS, NUM_ANCHOR_GROUPS,
    ROOT_COLOR_VOCAB, NUM_ROOT_COLORS,
    HISTOGRAM_GROUP_BY_LABEL, HISTOGRAM_VARIABLE_BIN, HISTOGRAM_CONSTANT_BIN,
    NUM_HISTOGRAM_BINS, HISTOGRAM_FEATURES,
    NUM_CANONICAL_LABELS,
    NODE_TYPE_ONEHOT, NODE_TYPE_ONEHOT_DIM,
    LABEL_ONEHOT_NAMES,
    SUPERNODE_NODE_TYPE, SUPERNODE_NODE_ID,
    NODE_FEATURE_SCHEMA, EDGE_FEATURE_SCHEMA,
    _is_numeric_label, encode_label,
    node_type_onehot, root_color_onehot, label_onehot,
    anchor_group_for_node,
)
from gnn.shared.utils.feature_extraction import (
    _compute_anchor_positional_encoding, _multi_source_bfs,
    _histogram_bin_for_node, _compute_subtree_histograms,
    TopologicalFeatureExtractor,
    inject_virtual_supernode, slice_active_features,
    compute_normalized_dirichlet_energy,
)
from gnn.shared.utils.graph_converter import (
    _find_global_node_id, _mark_function_roots,
    ExpressionGraphData, ExpressionGraphConverter,
    parse_graphml_node_name, _determine_node_type_from_label,
    _parse_constant_value, find_roots, parse_graphml_to_nodes_and_edges,
    create_virtual_global_node,
)
from gnn.shared.utils.kappa_loader import (
    _kappa_graph_cache, _parse_kappa_raw, _normalize_kappa_graph,
    _load_normalized_kappas, _tag_and_connect_kappa,
    KappaEdge, AugmentedFunctionGraph,
    LoadGraphFromLocalStructure, LoadAugmentedFunctionGraph,
    filter_active_kappa,
)
