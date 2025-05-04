import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn.functional as F


def get_symmetrically_normalized_adjacency(edge_index: Tensor, num_nodes: int):
    """
    Returns symmetrically normalized adjacency matrix with self-loops.
    """
    row, col = edge_index
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loops = self_loops.unsqueeze(0).repeat(2, 1)
    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
    row, col = edge_index_with_loops
    deg = torch.bincount(row, minlength=num_nodes)
    deg_inv_sqrt = deg.float().pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index_with_loops, edge_weight


def generate_rfp_encoding(edge_index: Tensor, num_nodes: int,
                          r: int = 8, P: int = 4,
                          normalize: str = "l2",
                          device: str = "cpu") -> Tensor:
    """
    Generate Random Feature Propagation (RFP) positional encodings.

    Args:
        edge_index: Graph edge indices [2, num_edges]
        num_nodes: Number of nodes in the graph
        r: Dimensionality of random features
        P: Number of propagation steps
        normalize: "l2" or "qr"
        device: "cpu" or "cuda"

    Returns:
        Tensor of shape [num_nodes, r * (P + 1)]
    """
    edge_index = edge_index.to(device)
    edge_index_norm, edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes)
    adj = SparseTensor(row=edge_index_norm[0], col=edge_index_norm[1],
                       value=edge_weight, sparse_sizes=(num_nodes, num_nodes)).to(device)

    x = torch.randn(num_nodes, r, device=device)
    rfp_trajectory = [x]

    for _ in range(P):
        x = adj @ x
        if normalize == "l2":
            x = F.normalize(x, p=2, dim=1)
        elif normalize == "qr":
            x, _ = torch.linalg.qr(x)
        rfp_trajectory.append(x)

    return torch.cat(rfp_trajectory, dim=1)
