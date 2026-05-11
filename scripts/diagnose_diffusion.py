#!/usr/bin/env python3
"""Diagnostic script to trace NaN origin in diffusion mode."""
import sys
sys.path.insert(0, "/home/bosho/FP")

import torch
import numpy as np
from src.fedgnn.data.datasets import GraphDataset
from src.fedgnn.data.data_utils import diffusion_kernel, get_symmetrically_normalized_adjacency
from src.fedgnn.data.propagation import propagate_features
from src.fedgnn.data.partitioning import partition_data

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Cora
print("\n--- Loading Cora ---")
loader = GraphDataset(device)
data, _ = loader.load_dataset("Cora", device)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.x.shape[1]}")
print(f"Feature range: [{data.x.min():.4f}, {data.x.max():.4f}]")
print(f"Any NaN in features: {torch.isnan(data.x).any().item()}")
print(f"Any Inf in features: {torch.isinf(data.x).any().item()}")

# Check edge_index
print(f"\nEdge index shape: {data.edge_index.shape}")
print(f"Edge index dtype: {data.edge_index.dtype}")
print(f"Max edge index: {data.edge_index.max().item()}, Min: {data.edge_index.min().item()}")

# Test 1: diffusion_kernel directly
print("\n--- Test 1: diffusion_kernel ---")
try:
    sparse_tensor = diffusion_kernel(data.edge_index, data.num_nodes, str(device), t=0.1)
    print("diffusion_kernel completed.")
    values = sparse_tensor.storage.value()
    print(f"Values shape: {values.shape}, dtype: {values.dtype}")
    print(f"Values range: [{values.min():.6f}, {values.max():.6f}]")
    print(f"Any NaN in kernel values: {torch.isnan(values).any().item()}")
    print(f"Any Inf in kernel values: {torch.isinf(values).any().item()}")
    print(f"Number of non-zero entries: {values.numel()}")
    
    # Convert to torch.sparse_coo_tensor and check again
    row, col = sparse_tensor.storage.row(), sparse_tensor.storage.col()
    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, size=(data.num_nodes, data.num_nodes)).to(device)
    adj_dense = adj.to_dense()
    print(f"Any NaN in adj dense: {torch.isnan(adj_dense).any().item()}")
    print(f"Any Inf in adj dense: {torch.isinf(adj_dense).any().item()}")
    print(f"Adj dense row sums (first 10): {adj_dense.sum(dim=1)[:10]}")
    
    # Check if row sums are reasonable
    row_sums = adj_dense.sum(dim=1)
    print(f"Row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    
except Exception as e:
    print(f"ERROR in diffusion_kernel: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Full propagate_features with diffusion
print("\n--- Test 2: propagate_features with diffusion ---")
try:
    # Create a simple mask (all nodes known for simplicity)
    mask = torch.ones(data.num_nodes, dtype=torch.bool)
    out = propagate_features(
        data.x, data.edge_index, mask, device,
        num_iterations=50, mode="diffusion", alpha=0.5,
        config={"diffusion_t": 0.1}
    )
    print("propagate_features completed.")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Any NaN in output: {torch.isnan(out).any().item()}")
    print(f"Any Inf in output: {torch.isinf(out).any().item()}")
except Exception as e:
    print(f"ERROR in propagate_features: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if issue is in partition_data -> create_k_hop_subgraph
print("\n--- Test 3: partition_data with diffusion ---")
try:
    clients_data, _, _ = partition_data(
        data, num_clients=2, beta=10000.0, device=device,
        hop=1, use_feature_prop=True, mode="diffusion",
        config={"diffusion_t": 0.1, "num_iterations": 50, "feature_prop_tolerance": 1e-3}
    )
    print("partition_data completed.")
    for i, client in enumerate(clients_data):
        print(f"Client {i}: nodes={client.num_nodes}, any NaN in x: {torch.isnan(client.x).any().item()}")
except Exception as e:
    print(f"ERROR in partition_data: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Done ---")
