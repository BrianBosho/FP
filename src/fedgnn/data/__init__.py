"""Data loading, preprocessing, partitioning, and propagation for federated graph learning."""

from src.fedgnn.data.datasets import GraphDataset, load_config
from src.fedgnn.data.data_utils import (
    get_personalized_pagerank_matrix,
    sparse_random_walk_with_restarts,
    sparse_scalar_mul,
    diffusion_kernel,
    chebyshev_expmL_apply,
    get_symmetrically_normalized_adjacency,
    propagate_features_efficient,
    edge_homophily,
    node_homophily,
)
from src.fedgnn.data.propagation import (
    compute_dirichlet_energy,
    get_propagation_matrix,
    monte_carlo_random_walk,
    apply_mask,
    propagate_features,
)
from src.fedgnn.data.loaders import (
    load_dataset,
    load_and_split,
    load_and_split_with_khop,
    load_and_split_with_feature_prop,
)
from src.fedgnn.data.partitioning import (
    label_dirichlet_partition,
    create_subgraph,
    create_k_hop_subgraph,
    get_in_comm_indexes,
    partition_data,
    reset_subgraph_features,
    reset_subgraph_features2,
    prepare_expanded_subgraph_for_propagation,
)
from src.fedgnn.data.positional_encoding import (
    generate_rfp_encoding,
    normalize_features,
)

__all__ = [
    # datasets
    "GraphDataset",
    "load_config",
    # data_utils
    "get_personalized_pagerank_matrix",
    "sparse_random_walk_with_restarts",
    "sparse_scalar_mul",
    "diffusion_kernel",
    "chebyshev_expmL_apply",
    "get_symmetrically_normalized_adjacency",
    "propagate_features_efficient",
    "edge_homophily",
    "node_homophily",
    "compute_dirichlet_energy",
    # loaders
    "load_dataset",
    "load_and_split",
    "load_and_split_with_khop",
    "load_and_split_with_feature_prop",
    # partitioning
    "label_dirichlet_partition",
    "create_subgraph",
    "create_k_hop_subgraph",
    "get_in_comm_indexes",
    "partition_data",
    "reset_subgraph_features",
    "reset_subgraph_features2",
    "prepare_expanded_subgraph_for_propagation",
    # positional_encoding
    "generate_rfp_encoding",
    "normalize_features",
    # propagation
    "get_propagation_matrix",
    "monte_carlo_random_walk",
    "apply_mask",
    "propagate_features",
]
