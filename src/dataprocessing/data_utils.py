import torch
from torch import Tensor
from torch_geometric.data import Data
from src.dataprocessing.propagation_functions import get_personalized_pagerank_matrix, sparse_random_walk_with_restarts, diffusion_kernel, get_symmetrically_normalized_adjacency, propagate_features_efficient

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
    
    edge_index_with_loops, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
    
    edge_index_with_loops = edge_index_with_loops.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    
    return torch.sparse_coo_tensor(
        edge_index_with_loops, edge_weight, 
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

def apply_mask(data: Data, split_index: list, subgraph_to_original: dict) -> Tensor:
    """Create a mask for feature propagation."""
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for idx in split_index:
        mask[idx] = True
    return mask

def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, device, num_iterations: int = 50, mode: str = "adjacency", alpha: float = 0.5) -> Tensor:
    """
    Improved feature propagation with better stability
    
    Args:
        x: Node features
        edge_index: Edge indices
        mask: Boolean mask for known features
        device: Computation device
        num_iterations: Maximum number of iterations
        mode: Type of propagation matrix to use
            - "adjacency": Standard GCN-like propagation
            - "page_rank": PageRank-based propagation
            - "random_walk": Random walk with restarts
            - "diffusion": Heat kernel diffusion
            - "efficient": Efficient propagation (returns directly)
            - "propagation": Custom propagation matrix
        alpha: Weight for diffused features (higher means more weight to diffused features)
            
    Returns:
        Tensor: Propagated features
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

    if mode == "page_rank":
        # PageRank returns SparseTensor, convert to torch.sparse_coo_tensor
        sparse_tensor = get_personalized_pagerank_matrix(edge_index, n_nodes, alpha=alpha)
        row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
        values = sparse_tensor.storage.value()
        indices = torch.stack([row, col], dim=0)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes)).to(DEVICE)
    elif mode == "random_walk":
        # Instead of using sparse_random_walk_with_restarts, use a simpler approach
        # Create a normalized adjacency matrix
        edge_index_with_loops, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
        adj_sparse = torch.sparse_coo_tensor(
            edge_index_with_loops, edge_weight, 
            size=(n_nodes, n_nodes)
        ).to(DEVICE)
        
        # Create identity matrix for teleportation
        indices = torch.arange(n_nodes, device=DEVICE)
        indices = torch.stack([indices, indices], dim=0)
        values = torch.ones(n_nodes, device=DEVICE)
        identity = torch.sparse_coo_tensor(
            indices, values,
            size=(n_nodes, n_nodes)
        ).to(DEVICE)
        
        # Compute random walk with restarts matrix manually
        # RWR = (1-alpha) * adj + alpha * I
        beta = 1 - alpha  # Random walk probability
        adj = beta * adj_sparse + alpha * identity
    elif mode == "diffusion":
        # Diffusion returns SparseTensor, convert to torch.sparse_coo_tensor
        sparse_tensor = diffusion_kernel(edge_index, n_nodes, device)
        row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
        values = sparse_tensor.storage.value()
        indices = torch.stack([row, col], dim=0)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes)).to(DEVICE)
    elif mode == "efficient":
        # This function directly returns propagated features, not a matrix
        return propagate_features_efficient(x, edge_index, mask, device, alpha=alpha, propagation_type="normalized_adjacency")
    elif mode == "adjacency":
        # This returns a tuple of (edge_index, edge_weight), convert to sparse tensor
        edge_index_with_loops, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
        adj = torch.sparse_coo_tensor(
            edge_index_with_loops, edge_weight, 
            size=(n_nodes, n_nodes)
        ).to(DEVICE)
    elif mode == "propagation":
        # This already returns a sparse_coo_tensor
        adj = get_propagation_matrix(out, edge_index, n_nodes, device)
    else:
        raise ValueError(f"Unknown propagation mode: {mode}")

    # Track previous iteration for convergence
    prev_out = None
    
    for i in range(num_iterations):
        # Diffuse features
        new_out = torch.sparse.mm(adj, out)
        
        # Combine with original features (weighted combination)
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
