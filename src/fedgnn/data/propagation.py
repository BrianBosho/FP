"""Feature propagation and related utility functions."""

import torch
from torch import Tensor
from torch_geometric.data import Data
from src.fedgnn.data.data_utils import (
    get_personalized_pagerank_matrix,
    sparse_random_walk_with_restarts,
    diffusion_kernel,
    get_symmetrically_normalized_adjacency,
    chebyshev_expmL_apply,
    _normalized_adjacency,
    get_row_normalized_adjacency,
    heat_kernel_exact,
)

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

def monte_carlo_random_walk(edge_index, num_nodes, device, walk_length=5, num_walks=10, max_nodes=5000):
    """
    Compute the random walk-based propagation matrix.
    Each node starts multiple random walks and we estimate transition probabilities.
    """
    if num_nodes > max_nodes:
        raise ValueError(
            f"monte_carlo_random_walk is O(V * num_walks * walk_length) in Python "
            f"and is disabled for num_nodes={num_nodes} > max_nodes={max_nodes}. "
            f"Use sparse_random_walk_with_restarts or chebyshev_diffusion instead."
        )
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

def _frob_dirichlet_residual(X: Tensor, edge_index: Tensor) -> float:
    """Sum of squared edge differences: trace(X^T L X). Decreases toward 0 for Dirichlet minimizers."""
    row, col = edge_index[0], edge_index[1]
    diffs = X[row] - X[col]
    return (diffs ** 2).sum().item()


def _compute_intrinsic_metrics(out: Tensor, X_true: Tensor, unknown: Tensor, edge_index: Tensor) -> dict:
    """MSE, cosine similarity, recovery ratio, and boundary coverage for unknown nodes."""
    import torch.nn.functional as F

    if not unknown.any():
        return {"mse": 0.0, "cosine_sim": 1.0, "recovery_ratio": 1.0, "boundary_coverage": 1.0}

    mse = F.mse_loss(out[unknown], X_true[unknown]).item()
    cosine_sim = F.cosine_similarity(out[unknown], X_true[unknown], dim=1).mean().item()

    # Recovery ratio: how much of the zero-hop MSE is recovered
    mse_zero = F.mse_loss(torch.zeros_like(X_true[unknown]), X_true[unknown]).item()
    recovery_ratio = (mse_zero - mse) / (mse_zero + 1e-12)

    # Boundary coverage: fraction of unknown nodes with >=1 known neighbor
    n_nodes = out.size(0)
    row, col = edge_index[0], edge_index[1]
    has_known_nbr = torch.zeros(n_nodes, dtype=torch.float, device=out.device)
    known_src = mask_from_not_unknown(unknown, row)
    if known_src.any():
        has_known_nbr[col[known_src]] = 1.0
    boundary_coverage = has_known_nbr[unknown].mean().item()

    return {
        "mse": mse,
        "cosine_sim": cosine_sim,
        "recovery_ratio": recovery_ratio,
        "boundary_coverage": boundary_coverage,
    }


def mask_from_not_unknown(unknown: Tensor, row: Tensor) -> Tensor:
    """Return boolean mask over edges where source node is NOT unknown (i.e., is known)."""
    return ~unknown[row]


def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, device,
                       num_iterations: int = 50, mode: str = "adjacency",
                       alpha: float = 0.5, client_id=None, log_file=None,
                       tol: float = 1e-3, config: dict | None = None,
                       init_strategy: str = "zero",
                       intrinsic_eval: bool = False,
                       X_true: Tensor | None = None) -> Tensor | dict:
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
        init_strategy: Initialization strategy for unlabeled nodes
            - "zero": Current behavior (unlabeled nodes start at 0)
            - "mean": Unlabeled nodes initialized to mean of labeled features
            - "neighbor": Unlabeled nodes initialized to average of neighboring labeled features
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
            "residuals": [],
            "norm_drifts": [],
            "variances": [],
            "iterations": 0,
            "converged": False,
            "runtime": 0,
            "initial_zeros": (x == 0).all(dim=1).sum().item(),
            "final_zeros": 0,
            "energies": []
        }
        metrics["init_strategy"] = init_strategy
        log_energy = bool(config.get("log_feature_prop_energy", False))
        metrics["log_energy"] = log_energy
    else:
        log_energy = False

    # Initialize output tensor
    out = torch.zeros_like(x)
    out[mask] = x[mask]

    # Apply initialization strategy for unlabeled nodes
    if init_strategy == "mean" and (~mask).sum() > 0:
        # Compute mean of labeled features
        labeled_features = x[mask]
        mean_features = labeled_features.mean(dim=0)
        out[~mask] = mean_features
        if logging_enabled:
            metrics["initial_zeros"] = (out == 0).all(dim=1).sum().item()
    elif init_strategy == "neighbor" and (~mask).sum() > 0:
        # For each unlabeled node, average features from neighboring labeled nodes
        n_nodes = x.size(0)
        edge_index_with_loops, _ = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
        adj_sparse = torch.sparse_coo_tensor(
            edge_index_with_loops, torch.ones(edge_index_with_loops.size(1), device=DEVICE),
            size=(n_nodes, n_nodes)
        ).to(DEVICE)

        # Count labeled neighbors per node
        labeled_mask = mask.float().unsqueeze(0)  # [1, n_nodes]
        neighbor_labeled_count = torch.sparse.mm(adj_sparse, labeled_mask.t()).squeeze()  # [n_nodes]
        neighbor_labeled_count = neighbor_labeled_count.clamp(min=1)  # avoid div by 0

        # Sum features from labeled neighbors
        labeled_features = x * mask.float().unsqueeze(1)  # zero out unlabeled
        neighbor_sum = torch.sparse.mm(adj_sparse, labeled_features)  # [n_nodes, feat_dim]

        # Average: only for nodes that actually have labeled neighbors
        has_labeled_neighbors = neighbor_labeled_count > 0
        out[has_labeled_neighbors] = neighbor_sum[has_labeled_neighbors] / neighbor_labeled_count[has_labeled_neighbors].unsqueeze(1)

        # Restore labeled nodes to their original features
        out[mask] = x[mask]

        if logging_enabled:
            metrics["initial_zeros"] = (out == 0).all(dim=1).sum().item()

        del adj_sparse, labeled_mask, neighbor_labeled_count, neighbor_sum
        import gc
        gc.collect()
    # else: init_strategy == "zero" → no change needed (out already initialized to zeros)
    out[mask] = x[mask]

    # Compute propagation matrix once
    n_nodes = x.size(0)
    if config is None:
        config = {}

    if (
        mode == "diffusion"
        and n_nodes > 50000
        and bool(config.get("force_chebyshev_for_large_graphs", True))
    ):
        print(
            f"[feature propagation] mode='diffusion' on {n_nodes} nodes uses "
            f"mode='chebyshev_diffusion' because the large-graph first-order "
            f"diffusion fallback is not a valid heat kernel."
        )
        mode = "chebyshev_diffusion"

    # Clear memory before intensive matrix operations
    torch.cuda.empty_cache()

    if mode == "page_rank":
        if n_nodes > 50000:
            raise ValueError(
                f"page_rank mode builds a dense {n_nodes}x{n_nodes} matrix and is "
                f"not suitable for graphs with >50k nodes. Use 'random_walk' or "
                f"'chebyshev_diffusion' instead."
            )
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
        t_diffusion = config.get("diffusion_t", 0.1)

        # Clear memory before diffusion kernel computation (memory-intensive)
        from src.fedgnn.utils.memory import clear_memory_for_diffusion
        clear_memory_for_diffusion()

        sparse_tensor = diffusion_kernel(edge_index, n_nodes, device, t=t_diffusion)
        row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
        values = sparse_tensor.storage.value()
        indices = torch.stack([row, col], dim=0)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes)).to(DEVICE)

        # AGGRESSIVE cleanup: torch_sparse.SparseTensor holds internal CUDA state
        # that standard PyTorch cleanup doesn't release, causing CUDA context corruption
        del sparse_tensor, row, col, values, indices

        # Force multiple rounds of GC to ensure torch_sparse releases resources
        import gc
        for _ in range(3):
            gc.collect()

        # Aggressive CUDA cleanup to ensure torch_sparse releases its CUDA context
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except RuntimeError:
                pass  # CUDA not actually usable

        # Final GC round after CUDA cleanup
        gc.collect()
    # Remove/disable the 'efficient' shortcut to ensure all modes iterate consistently
    # elif mode == "efficient":
    #     return propagate_features_efficient(x, edge_index, mask, device, alpha=alpha, propagation_type="normalized_adjacency")
    elif mode == "chebyshev_diffusion" or mode == "chebyshev_diffusion_operator":
        # chebyshev_t takes priority; fall back to diffusion_t so YAML configs that
        # only set diffusion_t (e.g. scalability configs) are honoured correctly.
        t = config.get("chebyshev_t", config.get("diffusion_t", 1))
        K = config.get("chebyshev_k", 5)

        # Build Z = I - L once here and reuse it across all propagation iterations.
        # Previously chebyshev_expmL_apply rebuilt Z on every call, costing
        # O(E) SparseTensor construction × num_iterations (≈50×) per client.
        from src.fedgnn.utils.memory import clear_memory_for_diffusion
        clear_memory_for_diffusion()
        Z_cached = _normalized_adjacency(edge_index.to(DEVICE), n_nodes, str(DEVICE))

        # adj is not used for chebyshev mode, but must be defined for the code
        # path below (prev_out convergence tracking, etc.)
        adj = None
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
    elif mode == "appnp":
        edge_index_with_loops, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
        adj = torch.sparse_coo_tensor(
            edge_index_with_loops, edge_weight, size=(n_nodes, n_nodes)
        ).to(DEVICE)
        X0 = out.clone()  # Initial features; teleport restores to this at each step
    elif mode == "asymmetric_random_walk":
        edge_index_rw, edge_weight_rw = get_row_normalized_adjacency(edge_index, n_nodes)
        adj = torch.sparse_coo_tensor(
            edge_index_rw, edge_weight_rw, size=(n_nodes, n_nodes)
        ).to(DEVICE)
    elif mode == "heat_kernel_exact":
        import time as _time
        H = heat_kernel_exact(edge_index, n_nodes, str(DEVICE), t=config.get("diffusion_t", 1.0))
        _t0 = _time.perf_counter()
        out = torch.mm(H, out)
        _wall_time = _time.perf_counter() - _t0
        out[mask] = x[mask]
        if intrinsic_eval and X_true is not None:
            X_true = X_true.to(DEVICE)
            _unknown = ~mask
            return {
                "X_imputed": out,
                "n_iters": 1,
                "converged": True,
                "wall_time": _wall_time,
                "residuals": [],
                "intrinsic_metrics": _compute_intrinsic_metrics(out, X_true, _unknown, edge_index),
            }
        return out
    else:
        raise ValueError(f"Unknown propagation mode: {mode}")

    # Initial stats for diagnostics
    initial_norm = torch.norm(out).item() + 1e-12

    # Intrinsic eval setup
    ie_residuals: list[float] = []
    if intrinsic_eval and X_true is not None:
        X_true = X_true.to(DEVICE)
    _unknown = ~mask

    # --- mixed-precision setup ---
    _prop_dtype_str = (config or {}).get("prop_dtype", "float32")
    _cast_dtype = None
    if _prop_dtype_str in ("float16", "fp16"):
        _cast_dtype = torch.float16
    elif _prop_dtype_str in ("bfloat16", "bf16"):
        _cast_dtype = torch.bfloat16

    if _cast_dtype is not None and mode not in ("heat_kernel_exact", "page_rank"):
        try:
            out = out.to(_cast_dtype)
            if adj is not None:
                adj = adj.to(_cast_dtype)
            if mode in ("chebyshev_diffusion", "chebyshev_diffusion_operator") and Z_cached is not None:
                Z_cached = Z_cached.to(_cast_dtype)
            if mode == "appnp":
                X0 = X0.to(_cast_dtype)
            x_anchor = x[mask].to(_cast_dtype)
        except Exception as _e:
            print(f"[propagation] prop_dtype={_prop_dtype_str} cast failed ({_e}); using float32")
            out = out.to(torch.float32)
            _cast_dtype = None
            x_anchor = x[mask]
    else:
        _cast_dtype = None
        x_anchor = x[mask]

    # Track previous iteration for convergence
    prev_out = None
    iter_count = 0
    did_converge = False

    import time as _time
    _loop_start = _time.perf_counter()

    for i in range(num_iterations):
        iter_count += 1
        # Diffuse features
        if mode == "chebyshev_diffusion" or mode == "chebyshev_diffusion_operator":
            new_out = chebyshev_expmL_apply(edge_index, n_nodes, out, t=t, K=K, device=str(DEVICE), Z=Z_cached)
        elif mode == "appnp":
            appnp_alpha_val = config.get("appnp_alpha", 0.1)
            new_out = (1.0 - appnp_alpha_val) * torch.sparse.mm(adj, out) + appnp_alpha_val * X0
        else:
            new_out = torch.sparse.mm(adj, out)

        # Compute Step for Residual diagnostic (existing logging path)
        if logging_enabled:
            step = new_out - out
            residual = torch.norm(step[_unknown]).item() if _unknown.any() else 0.0
            metrics["residuals"].append(residual)

        # Combine with original features — APPNP teleport replaces blending
        if mode == "appnp":
            out = new_out
        else:
            out = alpha * new_out + (1 - alpha) * out

        # Reset original known features
        out[mask] = x_anchor

        # Intrinsic eval: per-iteration Dirichlet residual
        if intrinsic_eval:
            ie_residuals.append(_frob_dirichlet_residual(out, edge_index))

        # Track diagnostics if logging is enabled
        if logging_enabled:
            # Norm Drift
            current_norm = torch.norm(out).item()
            metrics["norm_drifts"].append(current_norm / initial_norm)
            
            # Feature Variance (proxy for over-smoothing)
            # Use variance across nodes, averaged over feature dimensions
            var = torch.var(out, dim=0).mean().item()
            metrics["variances"].append(var)

        # Compute delta for convergence
        _check_interval = max(1, int((config or {}).get("convergence_check_interval", 1)))
        if i % _check_interval == 0:
            if prev_out is not None:
                delta = torch.norm(out.float() - prev_out.float()).item()
                if bool(config.get("feature_prop_relative_tolerance", False)):
                    delta = delta / (torch.norm(prev_out.float()).item() + 1e-12)
                if logging_enabled:
                    metrics["deltas"].append(delta)
                    if log_energy:
                        energy = compute_dirichlet_energy(out, edge_index)
                        metrics["energies"].append(energy)

                # Early stopping based on absolute L2 delta threshold
                if delta < tol:
                    if logging_enabled:
                        metrics["converged"] = True
                    did_converge = True
                    break
                # Fallback strict check (rarely needed)
                if torch.allclose(out.float(), prev_out.float(), rtol=1e-5):
                    if logging_enabled:
                        metrics["converged"] = True
                    did_converge = True
                    break

            prev_out = out.clone()

    _wall_time = _time.perf_counter() - _loop_start

    if _cast_dtype is not None:
        out = out.to(torch.float32)

    # Intrinsic eval return path (parallel to existing logging; does not affect it)
    if intrinsic_eval:
        result = {
            "X_imputed": out,
            "n_iters": iter_count,
            "converged": did_converge,
            "wall_time": _wall_time,
            "residuals": ie_residuals,
        }
        if X_true is not None:
            result["intrinsic_metrics"] = _compute_intrinsic_metrics(out, X_true, _unknown, edge_index)
        if not logging_enabled:
            return result
        # If logging is also enabled, fall through to flush the log then return
        # (set a flag so we return the dict after the log flush)
        _return_intrinsic = result
    else:
        _return_intrinsic = None

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

    if _return_intrinsic is not None:
        return _return_intrinsic

    return out

def propagate_features_multiscale(
    x: Tensor, edge_index: Tensor, mask: Tensor, device,
    scale_iterations: list[int], scale_t: float = 0.5,
    fusion_weights: list[float] | None = None,
    alpha: float = 0.5, config: dict | None = None,
    init_strategy: str = "zero"
) -> Tensor:
    """
    Multi-scale feature propagation via weighted fusion of multiple iteration depths.
    
    Instead of running a single propagation depth, this runs diffusion to multiple
    iteration checkpoints (e.g., 5, 20, 50) and fuses the results with learned/fixed
    weights. This captures both fast local signal (shallow propagation) and deep 
    structural signal (deep propagation) in a single representation.
    
    Args:
        x: Node features
        edge_index: Edge indices
        mask: Boolean mask for known features
        device: Computation device
        scale_iterations: List of iteration depths to run (e.g., [5, 20, 50])
        scale_t: Heat kernel t value for all scales (default 0.5)
        fusion_weights: Per-scale weights (default: uniform)
        alpha: Weight for diffused features in each scale's propagation
        config: Optional configuration dict
        init_strategy: Initialization for unlabeled nodes ("zero", "mean", "neighbor")
    
    Returns:
        Fused feature tensor of same shape as input
    """
    DEVICE = device
    x = x.to(DEVICE)
    mask = mask.bool().to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    
    if config is None:
        config = {}
    
    n_nodes = x.size(0)
    max_iter = max(scale_iterations)
    sorted_scales = sorted(enumerate(scale_iterations), key=lambda s: s[1])
    scale_t = config.get("diffusion_t", scale_t)
    
    if fusion_weights is None:
        fusion_weights = [1.0 / len(scale_iterations)] * len(scale_iterations)
    # Normalize weights
    w_sum = sum(fusion_weights)
    fusion_weights = [w / w_sum for w in fusion_weights]
    
    # Build diffusion matrix once (shared across all scales)
    from src.fedgnn.utils.memory import clear_memory_for_diffusion
    clear_memory_for_diffusion()
    
    sparse_tensor = diffusion_kernel(edge_index, n_nodes, device, t=scale_t)
    row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
    values = sparse_tensor.storage.value()
    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes)).to(DEVICE)
    
    del sparse_tensor, row, col, values, indices
    import gc
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    gc.collect()
    
    # Initialize output and per-scale intermediate states
    out = torch.zeros_like(x)
    out[mask] = x[mask]
    
    # Initialize unlabeled nodes
    if init_strategy == "mean" and (~mask).sum() > 0:
        labeled_features = x[mask]
        mean_features = labeled_features.mean(dim=0)
        out[~mask] = mean_features
    elif init_strategy == "neighbor" and (~mask).sum() > 0:
        edge_index_with_loops, _ = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
        adj_init = torch.sparse_coo_tensor(
            edge_index_with_loops, torch.ones(edge_index_with_loops.size(1), device=DEVICE),
            size=(n_nodes, n_nodes)
        ).to(DEVICE)
        labeled_mask = mask.float().unsqueeze(0)
        neighbor_count = torch.sparse.mm(adj_init, labeled_mask.t()).squeeze().clamp(min=1)
        labeled_x = x * mask.float().unsqueeze(1)
        neighbor_sum = torch.sparse.mm(adj_init, labeled_x)
        has_neighbors = neighbor_count > 0
        out[has_neighbors] = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors].unsqueeze(1)
        out[mask] = x[mask]
        del adj_init, labeled_mask, neighbor_count, neighbor_sum
        gc.collect()
    # else zero init: out already zeros for unlabeled
    out[mask] = x[mask]
    
    # Storage for intermediate states at each scale's checkpoint
    scale_states = [torch.zeros_like(out) for _ in scale_iterations]
    
    initial_norm = torch.norm(out).item() + 1e-12
    
    for i in range(max_iter):
        new_out = torch.sparse.mm(adj, out)
        out = alpha * new_out + (1 - alpha) * out
        out[mask] = x[mask]  # Reset known nodes
        
        # Capture intermediate states at each scale's iteration checkpoint
        for scale_idx, target_iter in sorted_scales:
            if i + 1 == target_iter:
                scale_states[scale_idx] = out.clone()
    
    # Weighted fusion of captured states
    fused = sum(w * state for w, state in zip(fusion_weights, scale_states))
    fused[mask] = x[mask]  # Preserve known features
    
    return fused


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
