import torch
from torch import Tensor
from torch_geometric.data import Data
from src.dataprocessing.propagation_functions import get_personalized_pagerank_matrix, sparse_random_walk_with_restarts, diffusion_kernel, get_symmetrically_normalized_adjacency, chebyshev_expmL_operator

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

def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, device, 
                       num_iterations: int = 50, mode: str = "adjacency", 
                       alpha: float = 0.5, client_id=None, log_file=None,
                       tol: float = 1e-3, config: dict | None = None) -> Tensor:
    """
    Improved feature propagation with logging capabilities.
    
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
            - "diffusion": Heat kernel diffusion (Taylor approximation)
            - "efficient": Efficient propagation (returns directly)
            - "propagation": Custom propagation matrix
            - "chebyshev_diffusion": Chebyshev approximation (matrix-free, RECOMMENDED)
            - "chebyshev_diffusion_operator": Chebyshev approximation (matrix-based)
        alpha: Weight for diffused features (higher means more weight to diffused features)
        client_id: Optional client ID for logging
        log_file: Optional path to JSON log file
        tol: Convergence tolerance
        config: Optional configuration dict; if provided and mode is chebyshev_*,
            reads 'chebyshev_t' and 'chebyshev_k'
    """
    DEVICE = device
    x = x.to(DEVICE)
    mask = mask.bool().to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    # Initialize metrics tracking if logging is enabled
    logging_enabled = log_file is not None and client_id is not None
    
    if logging_enabled:
        import time, json
        import os
        start_time = time.time()
        metrics = {
            "client_id": client_id,
            "nodes_total": x.size(0),
            "nodes_known": mask.sum().item(),
            "nodes_unknown": (x.size(0) - mask.sum()).item(),
            "num_edges": edge_index.size(1),
            "mode": mode,
            "alpha": alpha,
            "deltas": [],
            "iterations": 0,
            "converged": False,
            "runtime": 0,
            "initial_zeros": (x == 0).all(dim=1).sum().item(),
            "final_zeros": 0,
            "energies": []
        }
    
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
        # Use t=1.0 by default (original Taylor series behavior)
        if config is None:
            config = {}
        t_diffusion = config.get("diffusion_t", 0.1)
        sparse_tensor = diffusion_kernel(edge_index, n_nodes, device, t=t_diffusion)
        row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
        values = sparse_tensor.storage.value()
        indices = torch.stack([row, col], dim=0)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes)).to(DEVICE)
    # Remove/disable the 'efficient' shortcut to ensure all modes iterate consistently
    # elif mode == "efficient":
    #     return propagate_features_efficient(x, edge_index, mask, device, alpha=alpha, propagation_type="normalized_adjacency")
    elif mode == "chebyshev_diffusion" or mode == "chebyshev_diffusion_operator":
        # Build Chebyshev diffusion operator once (like diffusion/adja implementations)
        # Use smaller t per-iteration for gentler smoothing
        if config is None:
            config = {}
        t = config.get("chebyshev_t", 1)
        K = config.get("chebyshev_k", 5)
        sparse_tensor = chebyshev_expmL_operator(edge_index, n_nodes, t=t, K=K, device=str(DEVICE))
        row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
        values = sparse_tensor.storage.value()
        indices = torch.stack([row, col], dim=0)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes)).to(DEVICE)
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
    iter_count = 0
    did_converge = False
    
    for i in range(num_iterations):
        iter_count += 1
        # Diffuse features
        new_out = torch.sparse.mm(adj, out)
        
        # Combine with original features (weighted combination)
        out = alpha * new_out + (1 - alpha) * out
        
        # Reset original known features
        out[mask] = x[mask]
        
        # Track metrics if logging is enabled
        if logging_enabled and prev_out is not None:
            delta = torch.norm(out - prev_out).item()
            metrics["deltas"].append(delta)
            energy = compute_dirichlet_energy(out, edge_index)
            metrics["energies"].append(energy)
        
        # Check for convergence
        if prev_out is not None:
            # Early stopping based on absolute L2 delta threshold
            if delta < tol:
                if logging_enabled:
                    metrics["converged"] = True
                did_converge = True
                break
            # Fallback strict check (rarely needed)
            if torch.allclose(out, prev_out, rtol=1e-5):
                if logging_enabled:
                    metrics["converged"] = True
                did_converge = True
                break
            
        prev_out = out.clone()
    
    # Record final metrics if logging is enabled
    if logging_enabled:
        metrics["iterations"] = iter_count
        metrics["runtime"] = time.time() - start_time
        metrics["final_zeros"] = (out == 0).all(dim=1).sum().item()
        
        if len(metrics["deltas"]) > 0:
            metrics["final_delta"] = metrics["deltas"][-1]
        
        # Append client metrics to the experiment JSON file
        with open(log_file, 'r') as f:
            experiment_data = json.load(f)
        
        # Add this client's metrics to the clients array
        experiment_data["clients"].append(metrics)
        
        # Write back the updated experiment data
        with open(log_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
    
    # Console summary for quick visibility
    try:
        # Only print in debug mode - this is controlled by the calling function
        # For now, disable this verbose output by default
        # print(f"feature_propagation: steps={iter_count}, converged={did_converge}, mode={mode}, tol={tol}")
    except Exception:
        pass
    
    return out

def compute_dirichlet_energy(features: torch.Tensor, edge_index: torch.Tensor) -> dict:
    """
    Compute raw and normalized Dirichlet energy of a feature matrix on a graph.

    Returns a dict with:
    - 'raw': unnormalized energy
    - 'per_node': energy normalized by number of nodes
    - 'per_edge': energy normalized by number of edges
    """
    row, col = edge_index[0], edge_index[1]
    diffs = features[row] - features[col]  # shape: [num_edges, feature_dim]
    squared_norms = torch.sum(diffs ** 2, dim=1)  # shape: [num_edges]
    raw_energy = 0.5 * torch.sum(squared_norms).item()

    num_nodes = features.size(0)
    num_edges = edge_index.size(1)

    return {
        "raw": raw_energy,
        "per_node": raw_energy / num_nodes,
        "per_edge": raw_energy / num_edges
    }



