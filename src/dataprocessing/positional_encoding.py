# rfp_utils.py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

def get_symmetrically_normalized_adjacency(edge_index: Tensor, num_nodes: int):
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

def normalize_features(x: Tensor, method: str = "l2") -> Tensor:
    if method == "l2":
        return F.normalize(x, p=2, dim=1)
    elif method == "qr":
        Q, _ = torch.linalg.qr(x)
        return Q
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def generate_rfp_encoding(edge_index: Tensor, num_nodes: int,
                          r: int = 64, P: int = 16,
                          normalize: str = "qr",
                          device: str = "cpu") -> Tensor:
    edge_index = edge_index.to(device)
    edge_index_norm, edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes)
    adj = SparseTensor(row=edge_index_norm[0], col=edge_index_norm[1],
                       value=edge_weight, sparse_sizes=(num_nodes, num_nodes)).to(device)

    x = torch.randn(num_nodes, r, device=device)
    rfp_trajectory = [x]

    for _ in range(P):
        x = adj @ x
        x = normalize_features(x, method=normalize)
        rfp_trajectory.append(x)

    return torch.cat(rfp_trajectory, dim=1)
