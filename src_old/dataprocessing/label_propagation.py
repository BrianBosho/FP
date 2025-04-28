# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from dataprocessing.propagation_functions import get_symmetrically_normalized_adjacency
# # from dataprocessing.partitioning import partition_data, prepare_expanded_subgraph_for_propagation

# def label_propagation(subgraph: Data, mapping: torch.Tensor, num_classes: int, alpha: float = 0.8, 
#                      max_iter: int = 100, tol: float = 1e-5, device: str = "cuda") -> torch.Tensor:
#     """
#     Performs label propagation on an expanded subgraph to estimate labels for k-hop nodes.
    
#     Args:
#         subgraph (Data): The k-hop expanded subgraph (output of create_k_hop_subgraph or prepare_expanded_subgraph_for_propagation)
#         mapping (torch.Tensor): Indices of original nodes in the subgraph (from k_hop_subgraph)
#         num_classes (int): Number of unique classes in the dataset
#         alpha (float): Propagation strength (0 to 1), controls influence of neighbors vs. initial labels
#         max_iter (int): Maximum number of iterations
#         tol (float): Convergence tolerance (stop if label changes are below this)
#         device (str): Device to run on ("cuda" or "cpu")
    
#     Returns:
#         torch.Tensor: Soft label predictions (num_nodes x num_classes) with probabilities for each class
#     """
#     # Ensure everything is on the correct device
#     DEVICE = torch.device(device)
#     subgraph = subgraph.to(DEVICE)
#     mapping = mapping.to(DEVICE)

#     # Number of nodes in the subgraph
#     num_nodes = subgraph.num_nodes

#     # Create initial label matrix (Y0): one-hot for original nodes, zeros for others
#     Y0 = torch.zeros(num_nodes, num_classes, device=DEVICE)
#     original_labels = subgraph.y[mapping]  # Labels of original nodes
#     Y0[mapping, original_labels] = 1.0  # One-hot encode original labels

#     # Mask for clamping original labels
#     clamp_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
#     clamp_mask[mapping] = True

#     # Get symmetrically normalized adjacency matrix
#     # Handle the case where get_symmetrically_normalized_adjacency returns a tuple
#     adjacency_result = get_symmetrically_normalized_adjacency(subgraph.edge_index, num_nodes)
    
#     # If it's a tuple, assume the first element is the adjacency matrix
#     if isinstance(adjacency_result, tuple):
#         W = adjacency_result[0].to(DEVICE)
#     else:
#         # If it's already a tensor or sparse matrix
#         W = adjacency_result.to(DEVICE)

#     # Initialize current label distribution (Y) as Y0
#     Y = Y0.clone()

#     # Iterative label propagation
#     for iteration in range(max_iter):
#         # Previous Y for convergence check
#         Y_prev = Y.clone()

#         # Propagate: alpha * W * Y + (1 - alpha) * Y0
#         Y = alpha * torch.sparse.mm(W, Y) + (1 - alpha) * Y0

#         # Clamp original labels to their initial values
#         Y[clamp_mask] = Y0[clamp_mask]

#         # Normalize to ensure rows sum to 1 (soft probabilities)
#         Y = F.normalize(Y, p=1, dim=1)

#         # Check convergence
#         diff = torch.norm(Y - Y_prev, p=2)
#         if diff < tol:
#             print(f"Label propagation converged after {iteration + 1} iterations")
#             break

#     # Return soft label predictions
#     return Y

# # # Example usage integrating with your existing functions
# # def partition_data_with_label_propagation(data: Data, num_clients: int, beta: float, hop: int, 
# #                                          num_classes: int, device: str = "cuda", alpha: float = 0.8):
# #     """
# #     Partitions data and applies label propagation to estimate k-hop node labels.
    
# #     Args:
# #         data (Data): Original graph data
# #         num_clients (int): Number of clients
# #         beta (float): Dirichlet concentration parameter
# #         hop (int): Number of hops for subgraph expansion
# #         num_classes (int): Number of classes in the dataset
# #         device (str): Device to run on
# #         alpha (float): Label propagation strength
    
# #     Returns:
# #         list: List of subgraphs with propagated labels
# #     """
# #     # Partition data without feature propagation first
# #     final_subgraphs, initial_subgraphs, split_data_indexes = partition_data(
# #         data, num_clients, beta, device, hop=hop, use_feature_prop=False
# #     )

# #     # Apply label propagation to each subgraph
# #     for i in range(num_clients):
# #         # Get the expanded subgraph and its mapping
# #         expanded_subgraph = final_subgraphs[i]
# #         mapping = expanded_subgraph.mapping  # Stored from create_k_hop_subgraph

# #         # Prepare the subgraph (ensure original labels are set, new nodes are zeroed)
# #         prepared_subgraph = prepare_expanded_subgraph_for_propagation(
# #             initial_subgraphs[i], expanded_subgraph, mapping
# #         )

# #         # Run label propagation
# #         soft_labels = label_propagation(prepared_subgraph, mapping, num_classes, alpha=alpha, device=device)

# #         # Update the subgraph with propagated labels
# #         # Convert soft labels to hard labels (argmax) for y, keep soft labels as an attribute
# #         prepared_subgraph.y = torch.argmax(soft_labels, dim=1)
# #         prepared_subgraph.soft_labels = soft_labels  # Optional: store probabilities

# #         # Replace the original subgraph with the updated one
# #         final_subgraphs[i] = prepared_subgraph

# #     return final_subgraphs, initial_subgraphs, split_data_indexes

# # Example usage:
# # data = Data(x=..., edge_index=..., y=..., train_mask=..., val_mask=..., test_mask=...)
# # num_clients = 10
# # beta = 0.5
# # hop = 2
# # num_classes = 7  # Depends on your dataset
# # subgraphs, initial_subgraphs, indexes = partition_data_with_label_propagation(
# #     data, num_clients, beta, hop, num_classes
# # )

import torch
import torch.nn.functional as F
import numpy as np

def label_propagation(data, max_iterations=30, alpha=0.9):
    """
    Perform label propagation on a graph.
    
    Args:
        data (torch.geometric.data.Data): PyG Data object containing the graph
        max_iterations (int): Maximum number of iterations for label propagation
        alpha (float): Propagation strength (controls balance between original and propagated labels)
    
    Returns:
        torch.Tensor: Propagated label probabilities
    """
    # Ensure data is on CPU for computation
    data = data.cpu()
    
    # Get the adjacency matrix (normalized)
    adj = get_normalized_adjacency(data.edge_index, data.num_nodes)
    
    # Identify labeled and unlabeled nodes
    train_mask = data.train_mask
    labeled_nodes = torch.where(train_mask)[0]
    unlabeled_nodes = torch.where(~train_mask)[0]
    
    # One-hot encode the original labels
    num_classes = data.y.max().item() + 1
    Y = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32)
    Y[labeled_nodes, data.y[labeled_nodes]] = 1.0
    
    # Initialize label probabilities
    Y_propagated = Y.clone()
    
    # Iterative label propagation
    for _ in range(max_iterations):
        # Propagate labels using the adjacency matrix
        Y_prev = Y_propagated.clone()
        
        # Compute propagation: Y = (1-α)AY + αY_original
        Y_propagated = (1 - alpha) * torch.matmul(adj, Y_propagated) + alpha * Y
        
        # Preserve original labeled node labels
        Y_propagated[labeled_nodes] = Y[labeled_nodes]
        
        # Check for convergence
        if torch.allclose(Y_propagated, Y_prev, atol=1e-4):
            break
    
    return Y_propagated

def get_normalized_adjacency(edge_index, num_nodes):
    """
    Create a normalized adjacency matrix using symmetric normalization.
    
    Args:
        edge_index (torch.Tensor): Graph edge indices
        num_nodes (int): Total number of nodes in the graph
    
    Returns:
        torch.Tensor: Normalized adjacency matrix
    """
    # Create sparse adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    
    # Compute degree matrix
    degree = adj.sum(dim=1)
    
    # Symmetric normalization (D^-1/2 A D^-1/2)
    degree_sqrt_inv = torch.pow(degree, -0.5)
    degree_sqrt_inv[torch.isinf(degree_sqrt_inv)] = 0
    
    # Create normalized adjacency matrix
    normalized_adj = degree_sqrt_inv.unsqueeze(1) * adj * degree_sqrt_inv.unsqueeze(0)
    
    return normalized_adj

def apply_label_propagation(data, **kwargs):
    """
    Wrapper function to apply label propagation and update the data object.
    
    Args:
        data (torch.geometric.data.Data): PyG Data object
        **kwargs: Additional arguments for label_propagation function
    
    Returns:
        torch.geometric.data.Data: Updated data object with propagated labels
    """
    # Perform label propagation
    propagated_labels = label_propagation(data, **kwargs)
    
    # Create a new mask for nodes where we want to update labels
    # This could be all non-train nodes or a specific subset
    update_mask = ~data.train_mask
    
    # Create a new y tensor
    new_y = data.y.clone()
    
    # For unlabeled nodes, set the label to the argmax of propagated probabilities
    new_y[update_mask] = propagated_labels[update_mask].argmax(dim=1)
    
    # Create a new data object with updated labels
    updated_data = data.clone()
    updated_data.y = new_y
    
    return updated_data