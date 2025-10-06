import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
import torch_scatter
import math
import scipy.special
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


# ============================================================================
# Chebyshev-based Diffusion Kernel Approximations
# ============================================================================

def _normalized_laplacian(edge_index: Tensor, num_nodes: int, device: str) -> SparseTensor:
    """
    Compute the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    
    Args:
        edge_index: Tensor of shape [2, E] with edge indices.
        num_nodes: Number of nodes in the graph.
        device: Computation device ("cpu" or "cuda").
        
    Returns:
        SparseTensor: Normalized Laplacian matrix.
    """
    edge_index = edge_index.to(device)
    row, col = edge_index[0], edge_index[1]

    # Compute node degrees
    deg = torch_scatter.scatter_add(
        torch.ones_like(row, dtype=torch.float32, device=device),
        row, dim=0, dim_size=num_nodes
    )
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0.0

    # Create sparse adjacency matrix
    adj = SparseTensor(
        row=row, col=col, value=torch.ones_like(row, dtype=torch.float32, device=device),
        sparse_sizes=(num_nodes, num_nodes)
    )
    
    # Construct D^{-1/2} A D^{-1/2}
    diag_idx = torch.arange(num_nodes, device=device)
    Dm = SparseTensor(row=diag_idx, col=diag_idx, value=deg_inv_sqrt,
                      sparse_sizes=(num_nodes, num_nodes))
    norm_adj = Dm @ adj @ Dm
    
    # L = I - D^{-1/2} A D^{-1/2}
    # Negate norm_adj by multiplying values by -1
    neg_row = norm_adj.storage.row()
    neg_col = norm_adj.storage.col()
    neg_values = norm_adj.storage.value() * (-1)
    neg_norm_adj = SparseTensor(row=neg_row, col=neg_col, value=neg_values,
                                sparse_sizes=(num_nodes, num_nodes))
    
    I = SparseTensor.eye(num_nodes, device=device)
    L = I + neg_norm_adj  # I - norm_adj = I + (-norm_adj)
    return L


def chebyshev_expmL_apply(
    edge_index: Tensor,
    num_nodes: int,
    X: Tensor,
    t: float = 1.0,
    K: int = 10,
    device: str = "cuda"
) -> Tensor:
    """
    Apply y = exp(-t L) X using a Chebyshev expansion with order K (matrix-free).
    
    This is the RECOMMENDED approach for large graphs as it never materializes
    the full diffusion matrix. Uses only K sparse matrix-vector multiplications.
    
    L is the normalized Laplacian; we rescale to Z = L - I so its spectrum is in [-1, 1].
    The approximation is:
        exp(-t L) = exp(-t) * [ I0(t) T0(Z) + 2 sum_{k=1..K} I_k(t) T_k(Z) ]
    where T_k are Chebyshev polynomials and I_k are modified Bessel functions.
    
    Args:
        edge_index: [2, E] edges (undirected with both directions if needed).
        num_nodes:  Number of nodes.
        X:          [N, F] node feature matrix.
        t:          Diffusion time parameter (typical range: 0.2 - 2.0).
        K:          Chebyshev order (typically 5-10; often K<=5 works well).
        device:     "cpu" or "cuda".
        
    Returns:
        Y ≈ exp(-t L) X  (same shape as X).
    """
    X = X.to(device)
    L = _normalized_laplacian(edge_index, num_nodes, device)
    I = SparseTensor.eye(num_nodes, device=device)
    
    # Z = L - I: Negate I by multiplying values by -1, then add
    neg_I_row = I.storage.row()
    neg_I_col = I.storage.col()
    neg_I_values = I.storage.value() * (-1)
    neg_I = SparseTensor(row=neg_I_row, col=neg_I_col, value=neg_I_values,
                         sparse_sizes=(num_nodes, num_nodes))
    Z = L + neg_I  # L - I = L + (-I), since λ(L)∈[0,2] ⇒ λ(Z)=λ(L)-1∈[-1,1]

    # Precompute Bessel-based Chebyshev coefficients for e^{-tZ}
    # e^{-tL} = e^{-t} * [ I0(t) T0(Z) + 2 sum_{k=1..K} I_k(t) T_k(Z) ]
    # Use scipy.special.iv for modified Bessel function of first kind
    import numpy as np
    
    # Compute Bessel coefficients using scipy (more widely supported than torch.special.iv)
    i0_val = scipy.special.iv(0, t)
    coeff0 = np.exp(-t) * i0_val
    coeff0 = torch.tensor(coeff0, dtype=X.dtype, device=device)

    # Compute coefficients for k=1 to K
    Ik_vals = [scipy.special.iv(k, t) for k in range(1, K+1)]
    coeffs = [torch.tensor(2.0 * np.exp(-t) * Ik_vals[k-1], dtype=X.dtype, device=device) 
              for k in range(1, K+1)]

    # Chebyshev recurrence on features: T0X = X, T1X = Z @ X
    T0X = X
    Y = coeff0 * T0X

    if K == 0:
        return Y

    T1X = Z @ T0X  # SpMM
    Y = Y + coeffs[0] * T1X  # k=1

    Tkm2, Tkm1 = T0X, T1X
    for k in range(2, K+1):
        TkX = 2.0 * (Z @ Tkm1) - Tkm2  # SpMM + axpy
        Y = Y + coeffs[k-1] * TkX
        Tkm2, Tkm1 = Tkm1, TkX

    return Y


def chebyshev_expmL_operator(
    edge_index: Tensor,
    num_nodes: int,
    t: float = 1.0,
    K: int = 10,
    device: str = "cuda"
) -> SparseTensor:
    """
    Build a SparseTensor approximating exp(-t L) via a Chebyshev expansion.
    
    WARNING: This can be memory heavy for larger K as it performs sparse matrix
    multiplication (SpGEMM) K times. Use only for small graphs or small K.
    
    For most cases, prefer chebyshev_expmL_apply() which is matrix-free.
    
    Args:
        edge_index: Tensor of shape [2, E] with edge indices.
        num_nodes: Number of nodes in the graph.
        t: Diffusion time parameter (typical range: 0.2 - 2.0).
        K: Chebyshev order (typically 5-10; keep lower for this version).
        device: Computation device ("cpu" or "cuda").
        
    Returns:
        SparseTensor: Approximate diffusion operator exp(-t L).
    """
    L = _normalized_laplacian(edge_index, num_nodes, device)
    I = SparseTensor.eye(num_nodes, device=device)
    
    # Z = L - I: Negate I by multiplying values by -1, then add
    neg_I_row = I.storage.row()
    neg_I_col = I.storage.col()
    neg_I_values = I.storage.value() * (-1)
    neg_I = SparseTensor(row=neg_I_row, col=neg_I_col, value=neg_I_values,
                         sparse_sizes=(num_nodes, num_nodes))
    Z = L + neg_I  # L - I = L + (-I)

    # Coefficients using scipy for Bessel functions
    import numpy as np
    
    i0_val = scipy.special.iv(0, t)
    c0 = float(np.exp(-t) * i0_val)
    
    Ik_vals = [scipy.special.iv(k, t) for k in range(1, K+1)]
    ck = [float(2.0 * np.exp(-t) * Ik_vals[k-1]) for k in range(1, K+1)]

    # Chebyshev matrices: T0=I, T1=Z, T_{k+1} = 2 Z T_k - T_{k-1}
    T0 = I
    # Use sparse_scalar_mul helper to multiply by scalar
    op = sparse_scalar_mul(T0, c0)

    if K == 0:
        return op

    T1 = Z
    op = op + sparse_scalar_mul(T1, ck[0])

    Tkm2, Tkm1 = T0, T1
    for k in range(2, K+1):
        # Tk = 2 * Z @ Tkm1 - Tkm2
        Tk_temp = Z @ Tkm1
        Tk_scaled = sparse_scalar_mul(Tk_temp, 2.0)
        
        # Subtract Tkm2 by negating and adding
        neg_Tkm2_row = Tkm2.storage.row()
        neg_Tkm2_col = Tkm2.storage.col()
        neg_Tkm2_values = Tkm2.storage.value() * (-1)
        neg_Tkm2 = SparseTensor(row=neg_Tkm2_row, col=neg_Tkm2_col, value=neg_Tkm2_values,
                                sparse_sizes=(num_nodes, num_nodes))
        Tk = Tk_scaled + neg_Tkm2  # 2*Z@Tkm1 - Tkm2 = 2*Z@Tkm1 + (-Tkm2)
        
        op = op + sparse_scalar_mul(Tk, ck[k-1])
        Tkm2, Tkm1 = Tkm1, Tk

    return op


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
                               propagation_type: str = "normalized_adjacency",
                               chebyshev_k: int = 10, diffusion_t: float = 1.0) -> Tensor:
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
            - "diffusion_kernel": Heat kernel diffusion (Taylor approximation)
            - "chebyshev_diffusion": Chebyshev approximation (matrix-free, RECOMMENDED)
            - "chebyshev_diffusion_operator": Chebyshev approximation (builds operator matrix)
        chebyshev_k: Order of Chebyshev polynomial (used for chebyshev_* types)
        diffusion_t: Diffusion time parameter (used for diffusion_kernel and chebyshev_* types)
            
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
        adj = diffusion_kernel(edge_index, num_nodes, device, t=diffusion_t)
    
    elif propagation_type == "chebyshev_diffusion":
        # Build Chebyshev diffusion operator (matrix-free would be too expensive per iteration)
        adj = chebyshev_expmL_operator(edge_index, num_nodes, t=diffusion_t, K=chebyshev_k, device=device)
    
    elif propagation_type == "chebyshev_diffusion_operator":
        # Same as chebyshev_diffusion - both use operator for iterative application
        adj = chebyshev_expmL_operator(edge_index, num_nodes, t=diffusion_t, K=chebyshev_k, device=device)
    
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