import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
import torch_scatter
import math
def get_personalized_pagerank_matrix(edge_index: Tensor, num_nodes: int, alpha: float = 0.15, eps: float = 1e-7, max_iter: int = 100):
    """
    Compute the Personalized PageRank matrix efficiently.
    
    Args:
        edge_index: Tensor of shape [2, E] containing the edge indices
        num_nodes: Number of nodes in the graph
        alpha: Teleport probability (restart probability)
        eps: Convergence threshold
        max_iter: Maximum number of iterations
        
    Returns:
        SparseTensor: Sparse PageRank transition matrix
    """
    # Create adjacency matrix
    row, col = edge_index[0], edge_index[1]
    
    # Compute degrees for normalization
    deg = torch_scatter.scatter_add(torch.ones_like(row), col, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg.clamp(min=1.0)
    
    # Initialize PPR matrix as identity
    ppr = torch.eye(num_nodes, device=edge_index.device)
    
    # Initialize random walk matrix: P = D^-1 A
    # This is stored implicitly as (indices, weights)
    
    # Iterative computation of PPR
    prev_ppr = None
    for i in range(max_iter):
        # One step of random walk
        next_ppr = torch.zeros_like(ppr)
        
        # For each node, distribute its current value to neighbors
        for node in range(num_nodes):
            neighbors = col[row == node]
            if len(neighbors) > 0:
                next_ppr[neighbors] += (1 - alpha) * ppr[node] * deg_inv[node]
        
        # Add teleport component
        next_ppr += alpha * torch.eye(num_nodes, device=edge_index.device)
        
        # Check convergence
        if prev_ppr is not None and torch.allclose(next_ppr, prev_ppr, rtol=eps):
            break
            
        ppr = next_ppr
        prev_ppr = ppr.clone()
    
    # Convert to sparse tensor for efficiency
    indices = torch.nonzero(ppr > eps).t()
    values = ppr[indices[0], indices[1]]
    
    return SparseTensor(row=indices[0], col=indices[1], value=values, 
                       sparse_sizes=(num_nodes, num_nodes))


def sparse_random_walk_with_restarts(edge_index: Tensor, num_nodes: int, device: str, 
                                   alpha: float = 0.15, num_iterations: int = 10):
    """
    Efficiently compute random walk with restart transition probabilities.
    Much more scalable than the Monte Carlo approach.
    
    Args:
        edge_index: Edge indices
        num_nodes: Number of nodes in the graph
        device: Device to run computation on
        alpha: Restart probability
        num_iterations: Number of power iterations
        
    Returns:
        SparseTensor: Sparse transition matrix
    """
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(device)
    
    row, col = edge_index[0], edge_index[1]
    
    # Compute out-degrees
    deg = torch_scatter.scatter_add(torch.ones_like(row), row, dim=0, dim_size=num_nodes)
    
    # Create normalized adjacency matrix (sparse)
    edge_weight = 1.0 / deg[row].clamp(min=1.0)  # Normalize by out-degree
    adj = SparseTensor(row=row, col=col, value=edge_weight, 
                      sparse_sizes=(num_nodes, num_nodes))
    
    # Identity matrix for restart
    identity = SparseTensor.eye(num_nodes).to(device)
    
    # Power iteration to compute RWR
    rwr = identity.clone()
    
    # Manual implementation of power iteration without scalar multiplication of SparseTensor
    for _ in range(num_iterations):
        # First compute adj @ rwr
        adj_rwr = adj @ rwr
        
        # Extract components to create new tensors
        row_adj = adj_rwr.storage.row()
        col_adj = adj_rwr.storage.col()
        val_adj = adj_rwr.storage.value() * (1 - alpha)  # Scale values directly
        
        row_id = identity.storage.row()
        col_id = identity.storage.col()
        val_id = identity.storage.value() * alpha  # Scale values directly
        
        # Create new scaled sparse tensors
        term1 = SparseTensor(row=row_adj, col=col_adj, value=val_adj, 
                            sparse_sizes=(num_nodes, num_nodes))
        term2 = SparseTensor(row=row_id, col=col_id, value=val_id, 
                            sparse_sizes=(num_nodes, num_nodes))
        
        # Add the two sparse tensors
        rwr = term1 + term2
    
    return rwr

import math
import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_scatter

def sparse_scalar_mul(sparse: SparseTensor, scalar: float) -> SparseTensor:
    """
    Multiply a SparseTensor by a scalar.
    
    Args:
        sparse: A SparseTensor.
        scalar: A float scalar.
    
    Returns:
        A new SparseTensor with its values multiplied by the scalar.
    """
    # Extract row, col, and values from the sparse tensor
    row = sparse.storage.row()
    col = sparse.storage.col()
    values = sparse.storage.value() * scalar
    return SparseTensor(row=row, col=col, value=values, sparse_sizes=sparse.sparse_sizes())

def diffusion_kernel(edge_index: Tensor, num_nodes: int, device: str, t: float = 1.0):
    """
    Compute the heat kernel/diffusion kernel approximating exp(-tL), where
    L is the normalized Laplacian: L = I - D^(-1/2) A D^(-1/2).
    
    Args:
        edge_index: Tensor of shape [2, E] with edge indices.
        num_nodes: Number of nodes in the graph.
        device: Computation device ("cpu" or "cuda").
        t: Diffusion time parameter.
        
    Returns:
        SparseTensor: A sparse diffusion matrix that can be used as a propagation matrix.
    """
    # Ensure edge_index is on the correct device.
    edge_index = edge_index.to(device)
    
    # Get graph Laplacian components.
    row, col = edge_index[0], edge_index[1]
    
    # Compute node degrees.
    deg = torch_scatter.scatter_add(torch.ones_like(row), row, dim=0, dim_size=num_nodes)
    
    # Create sparse adjacency matrix.
    adj = SparseTensor(row=row, col=col, value=torch.ones_like(row).float(), 
                       sparse_sizes=(num_nodes, num_nodes))
    
    # Diagonal indices for constructing diagonal matrices.
    diag_indices = torch.arange(num_nodes, device=device)
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5).where(deg > 0, torch.zeros_like(deg).float())
    
    # Construct the diagonal matrix D^(-1/2).
    deg_inv_sqrt_mat = SparseTensor(row=diag_indices, col=diag_indices,
                                    value=deg_inv_sqrt,
                                    sparse_sizes=(num_nodes, num_nodes))
    
    # Compute normalized adjacency: D^(-1/2) A D^(-1/2)
    norm_adj = deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    # Negate norm_adj by multiplying its stored values by -1.
    neg_row = norm_adj.storage.row()
    neg_col = norm_adj.storage.col()
    neg_values = norm_adj.storage.value() * (-1)
    neg_norm_adj = SparseTensor(row=neg_row, col=neg_col, value=neg_values,
                                sparse_sizes=(num_nodes, num_nodes))
    
    # Create identity matrix on the proper device.
    identity = SparseTensor.eye(num_nodes).to(device)
    # Compute Laplacian: L = I - norm_adj = I + (-norm_adj)
    laplacian = identity + neg_norm_adj

    # Use a truncated Taylor series to approximate exp(-tL).
    diffusion = SparseTensor.eye(num_nodes).to(device)
    taylor_term = SparseTensor.eye(num_nodes).to(device)
    
    # Compute exp(-tL) ≈ I - tL + (t²/2)L² - (t³/6)L³ + ... (using 5 terms)
    for i in range(1, 6):
        coef = ((-t) ** i) / math.factorial(i)
        taylor_term = taylor_term @ laplacian
        # Use the helper to multiply taylor_term by coef before adding
        diffusion = diffusion + sparse_scalar_mul(taylor_term, coef)
    
    return diffusion

def get_symmetrically_normalized_adjacency(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """
    Compute symmetrically normalized adjacency matrix more efficiently:
    A_norm = D^{-1/2} A D^{-1/2}
    
    Returns both edge_index and edge_weights for sparse representation.
    """
    row, col = edge_index[0], edge_index[1]
    
    # Add self-loops
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loops = self_loops.unsqueeze(0).repeat(2, 1)
    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
    
    # Recompute row, col with self-loops
    row, col = edge_index_with_loops[0], edge_index_with_loops[1]
    
    # Compute degrees
    deg = torch_scatter.scatter_add(torch.ones_like(row), row, dim=0, dim_size=num_nodes)
    
    # Compute D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    # Compute normalized weights
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    return edge_index_with_loops, edge_weight


def propagate_features_efficient(x: Tensor, edge_index: Tensor, mask: Tensor, device: str, 
                               num_iterations: int = 50, alpha: float = 0.5, 
                               propagation_type: str = "normalized_adjacency") -> Tensor:
    """
    Efficient feature propagation with multiple propagation matrix options.
    
    Args:
        x: Node features
        edge_index: Edge indices
        mask: Boolean mask for known features
        device: Computation device
        num_iterations: Maximum number of iterations
        alpha: Teleport/restart probability
        propagation_type: Type of propagation matrix to use
            - "normalized_adjacency": Standard GCN-like propagation
            - "personalized_pagerank": PPR-based propagation
            - "random_walk_restarts": Random walk with restarts
            - "diffusion_kernel": Heat kernel diffusion
            
    Returns:
        Tensor: Propagated features
    """
    x = x.to(device)
    mask = mask.bool().to(device)
    edge_index = edge_index.to(device)
    
    # Initialize output with known features
    out = torch.zeros_like(x)
    out[mask] = x[mask]
    
    num_nodes = x.size(0)
    
    # Compute propagation matrix based on selected type
    if propagation_type == "normalized_adjacency":
        edge_index_norm, edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes)
        adj = SparseTensor(row=edge_index_norm[0], col=edge_index_norm[1], 
                          value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
    
    elif propagation_type == "personalized_pagerank":
        adj = get_personalized_pagerank_matrix(edge_index, num_nodes, alpha=alpha)
    
    elif propagation_type == "random_walk_restarts":
        adj = sparse_random_walk_with_restarts(edge_index, num_nodes, device, alpha=alpha)
    
    elif propagation_type == "diffusion_kernel":
        adj = diffusion_kernel(edge_index, num_nodes, device, t=1.0)
    
    else:
        raise ValueError(f"Unknown propagation type: {propagation_type}")
    
    # Track previous iteration for convergence
    prev_out = None
    
    # Propagation iterations
    for i in range(num_iterations):
        # Diffuse features
        new_out = matmul(adj, out)
        
        # Weighted combination with previous features
        beta = 0.5  # Weight for new features
        out = beta * new_out + (1 - beta) * out
        
        # Reset known features
        out[mask] = x[mask]
        
        # Check for convergence
        if prev_out is not None and torch.allclose(out, prev_out, rtol=1e-5):
            print(f"Converged after {i+1} iterations")
            break
            
        prev_out = out.clone()
    
    return out


# Homophily Measures
def edge_homophily(edge_index: Tensor, labels: Tensor) -> float:
    """
    Calculate edge homophily: fraction of edges connecting same-class nodes.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        labels: Node labels [num_nodes]
        
    Returns:
        float: Edge homophily score between 0 and 1
    """
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).sum().item()
    return same_label / edge_index.size(1)


def node_homophily(edge_index: Tensor, labels: Tensor, num_nodes: int) -> float:
    """
    Calculate node homophily: average ratio of same-label neighbors per node.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        labels: Node labels [num_nodes]
        num_nodes: Number of nodes in the graph
        
    Returns:
        float: Node homophily score between 0 and 1
    """
    src, dst = edge_index[0], edge_index[1]
    
    # For each node, count total neighbors and same-label neighbors
    total_neighbors = torch.zeros(num_nodes, device=edge_index.device)
    same_label_neighbors = torch.zeros(num_nodes, device=edge_index.device)
    
    # Count neighbors efficiently
    total_neighbors.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    
    # Count same-label neighbors
    same_label = (labels[src] == labels[dst]).float()
    same_label_neighbors.scatter_add_(0, src, same_label)
    
    # Calculate ratio for nodes with at least one neighbor
    valid_nodes = total_neighbors > 0
    ratios = same_label_neighbors[valid_nodes] / total_neighbors[valid_nodes]
    
    return ratios.mean().item()