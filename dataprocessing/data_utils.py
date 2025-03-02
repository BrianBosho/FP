import torch
from torch import Tensor
from torch_geometric.data import Data

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Propagation Functions
def get_propagation_matrix(x: Tensor, edge_index: Tensor, n_nodes: int, device = "cuda") -> Tensor:
    """Get symmetrically normalized adjacency matrix for feature propagation."""
    DEVICE = device
    # Validate input shape
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    
    # Remap indices to [0, n_nodes-1] range
    unique_nodes, remapped_edges = torch.unique(edge_index, return_inverse=True)
    edge_index = remapped_edges.reshape(edge_index.shape)
    
    # Validate remapped indices
    if edge_index.max() >= n_nodes:
        raise ValueError(f"Remapped edge_index still contains indices >= n_nodes ({n_nodes}). Max index: {edge_index.max()}")
    if edge_index.min() < 0:
        raise ValueError(f"edge_index contains negative indices. Min index: {edge_index.min()}")
    
    edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
    
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    
    return torch.sparse_coo_tensor(
        edge_index, edge_weight, 
        size=(n_nodes, n_nodes)
    ).to(DEVICE)

def monte_carlo_random_walk(edge_index, num_nodes, device, walk_length=5, num_walks=10):
    """
    Compute the random walk-based propagation matrix.
    Each node starts multiple random walks and we estimate transition probabilities.
    """
    DEVICE = device
    row, col = edge_index[0], edge_index[1]
    transition_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=DEVICE)

    for v in range(num_nodes):
        for _ in range(num_walks):  # Each node starts multiple random walks
            current_node = v
            for _ in range(walk_length):
                neighbors = col[row == current_node]
                if len(neighbors) == 0:
                    break  # Dead-end, stop walk
                next_node = neighbors[torch.randint(0, len(neighbors), (1,))]
                transition_matrix[v, next_node] += 1

    transition_matrix /= transition_matrix.sum(dim=1, keepdim=True)  # Normalize
    return transition_matrix


def get_symmetrically_normalized_adjacency(edge_index: Tensor, n_nodes: int) -> tuple[Tensor, Tensor]:
    """
    Compute symmetrically normalized adjacency matrix:
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    
    deg = torch.bincount(col, minlength=n_nodes)
    deg_inv_sqrt = deg.float().pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def apply_mask(data: Data, split_index: list, subgraph_to_original: dict) -> Tensor:
    """Create a mask for feature propagation."""
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for idx in split_index:
        mask[idx] = True
    return mask

def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, device, num_iterations: int = 50, mode: str = "propagation") -> Tensor:
    """
    Improved feature propagation with better stability
    """
    DEVICE = device
    x = x.to(DEVICE)
    mask = mask.bool().to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    # Initialize output tensor
    out = torch.zeros_like(x)
    out[mask] = x[mask]

    # Compute propagation matrix once
    n_nodes = x.size(0)
    if mode == "propagation":
        adj = get_propagation_matrix(out, edge_index, n_nodes, device)
    elif mode == "monte_carlo":
        adj = monte_carlo_random_walk(edge_index, n_nodes, device, walk_length=5, num_walks=10)
    # Track previous iteration for convergence
    prev_out = None
    
    for i in range(num_iterations):
        # Diffuse features
        new_out = torch.sparse.mm(adj, out)
        
        # Combine with original features (weighted combination)
        alpha = 1  # Retention rate for original features
        out = alpha * new_out + (1 - alpha) * out
        
        # Reset original known features
        out[mask] = x[mask]
        
        # Check for convergence
        if prev_out is not None and torch.allclose(out, prev_out, rtol=1e-5):
            break
            
        prev_out = out.clone()
    
    return out

# def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, device, num_iterations: int = 5) -> Tensor:
#     """
#     Propagate features through the graph.
#     """
#     DEVICE = device
#     x = x.to(DEVICE)
#     mask = mask.bool().to(DEVICE)
#     edge_index = edge_index.to(DEVICE)

#     if mask is not None:
#         out = torch.zeros_like(x)
#         out[mask] = x[mask]
#     else: 
#         out = x.clone()

#     n_nodes = x.size(0)
    
#     # Debug information
#     # print(f"Feature matrix shape: {x.shape}")
#     # print(f"Edge index shape: {edge_index.shape}")
#     # print(f"Edge index max before remapping: {edge_index.max().item()}")
#     # print(f"Number of nodes: {n_nodes}")
    
#     adj = get_propagation_matrix(out, edge_index, n_nodes, device)
    
#     for _ in range(num_iterations):
#         # Diffuse current features
#         out = torch.sparse.mm(adj, out)
#         # number of nodes with zero features
#         zero_features = (out == 0).sum().item()
#         # print(f"Number of nodes with zero features: {zero_features}")
#         # Reset original known features
#         out[mask] = x[mask]
#     return out
