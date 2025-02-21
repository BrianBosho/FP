import torch
from torch import Tensor
from torch_geometric.data import Data

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Propagation Functions
def get_propagation_matrix(x: Tensor, edge_index: Tensor, n_nodes: int, device = "cuda") -> Tensor:
    """Get symmetrically normalized adjacency matrix for feature propagation."""
    DEVICE = device
    edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    return torch.sparse_coo_tensor(
        edge_index, edge_weight, 
        size=(n_nodes, n_nodes)
    ).to(DEVICE)

def monte_carlo_random_walk(edge_index, num_nodes, walk_length=5, num_walks=10, device = "cuda"):
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

def propagate_features(x: Tensor, edge_index: Tensor, mask: Tensor, num_iterations: int = 50, device = "cuda") -> Tensor:
    """
    Propagate features through the graph.
    
    Args:
        x: Node feature matrix
        edge_index: Graph connectivity in COO format
        mask: Boolean mask indicating which nodes have known features
        num_iterations: Number of propagation iterations
        
    Returns:
        Propagated feature matrix
    """
    DEVICE = device
    x = x.to(DEVICE)
    mask = mask.bool().to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    if mask is not None:
        out = torch.zeros_like(x)
        out[mask] = x[mask]
    else:
        out = x.clone()

    n_nodes = x.size(0)
    # adj = get_propagation_matrix(out, edge_index, n_nodes)
    adj = monte_carlo_random_walk(edge_index, n_nodes)
    for _ in range(num_iterations):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = x[mask]
    return out
