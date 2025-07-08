from omegaconf import OmegaConf, DictConfig

def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)
import torch_geometric.utils as utils
from torch import Tensor
# from torch_scatter import scatter_add

import torch
import numpy as np

def apply_mask(data, split_data_indexes, subgraph_to_original):
    num_nodes = data.num_nodes
    num_features = data.num_node_features

    # Check if node IDs of data are same as communicate indexes
    # print("Node IDs are the same: ", set(subgraph_to_original.keys()) == set(range(num_nodes)))
    
    # Print the node ids
    indices = list(subgraph_to_original.keys())
    indices.sort()
    # print(f"Sorted Node Ids: {indices}")

    # Print the communicate indexes
    communicate_indexes = list(subgraph_to_original.values())
    communicate_indexes.sort()
    # print(f"Sorted Communicate indexes: {communicate_indexes[:20]}")
    # print (set(indices) == set(range(num_nodes)))

    # Apply your mask logic here using the original indices mapped to the subgraph indices
    mask = torch.zeros(num_nodes, num_features)
    for idx in split_data_indexes:
        mask[idx] = 1

    return mask




def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, num_iterations: int = 50) -> Tensor:
    out = x
    mask = mask.bool()

    if mask is not None:
        out = torch.zeros_like(x)
        out[mask] = x[mask]
    n_nodes = x.size(0)
    adj = get_propagation_matrix(out, edge_index, n_nodes)
    for _ in range(num_iterations):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = x[mask]

    return out

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    # deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)

    deg = torch.zeros(n_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg.index_add_(0, col, edge_weight)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def get_propagation_matrix(x, edge_index, n_nodes):
    # Initialize all edge weights to ones if the graph is unweighted)
    edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

    return adj


def feature_propagation(x: Tensor, edge_index: Tensor, mask: Tensor, num_iterations: int = 10) -> Tensor:
    n_nodes = x.size(0)

    # Create a symmetrically normalized adjacency matrix
    edge_weight = torch.ones(edge_index.size(1))  # Assuming unweighted graph
    edge_index, edge_weight = utils.add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=n_nodes)
    
    # Get Laplacian and normalize
    edge_index, edge_weight = utils.get_laplacian(edge_index, edge_weight, normalization='sym')
    adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)
    
    # Convert to sparse tensor
    adj = adj.to_sparse()

    # Ensure mask is of type bool
    mask = mask.bool()

    # Initialize output with zeros for unknown features, copy known features from x
    out = torch.zeros_like(x)
    if mask is not None:
        out[mask] = x[mask]

    # Propagate features for a number of iterations
    for _ in range(num_iterations):
        out = torch.sparse.mm(adj, out)
        if mask is not None:
            out[mask] = x[mask]  # Reset known features to their original values

    return out



