import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from dataprocessing.data_utils import propagate_features
# from utils import propagate_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_dirichlet_partition(labels: np.ndarray, N: int, K: int, n_parties: int, beta: float) -> list:
    """
    Partition data using Dirichlet distribution for label distribution across clients.
    
    Args:
        labels: Node labels
        N: Total number of nodes
        K: Number of classes
        n_parties: Number of clients
        beta: Dirichlet concentration parameter
    """
    min_size = 0
    min_require_size = 10
    split_data_indexes = []
    np.random.seed(123)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array(
                [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])
    return split_data_indexes

def create_subgraph(data: Data, node_indices: torch.Tensor) -> Data:
    """
    Creates a simple subgraph containing ONLY the specified nodes and their direct connections.
    """
    # Process on CPU first
    node_indices = node_indices.cpu() if isinstance(node_indices, torch.Tensor) else torch.tensor(node_indices, device='cpu')
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device='cpu')
    node_mask[node_indices] = True
    edge_index = data.edge_index.cpu()
    
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    subgraph_node_indices = torch.where(node_mask)[0]
    
    node_map = torch.zeros(data.num_nodes, dtype=torch.long, device='cpu')
    node_map[subgraph_node_indices] = torch.arange(len(subgraph_node_indices))
    edge_index = node_map[edge_index]

    # Create subgraph and move to specified device
    return Data(
        x=data.x.cpu()[subgraph_node_indices].to(DEVICE),
        edge_index=edge_index.to(DEVICE),
        y=data.y.cpu()[subgraph_node_indices].to(DEVICE),
        train_mask=data.train_mask.cpu()[node_mask].to(DEVICE),
        val_mask=data.val_mask.cpu()[node_mask].to(DEVICE),
        test_mask=data.test_mask.cpu()[node_mask].to(DEVICE)
    )

def create_k_hop_subgraph(data: Data, node_indices: torch.Tensor, num_hops: int) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Creates a k-hop subgraph with zeroed features for non-original nodes.
    """
    # Move everything to CPU for processing
    edge_index_cpu = data.edge_index.cpu()
    node_indices_cpu = node_indices.cpu() if isinstance(node_indices, torch.Tensor) else torch.tensor(node_indices)
    
    # Get k-hop subgraph
    subset, edge_index, _, _ = k_hop_subgraph(node_indices_cpu, num_hops, edge_index_cpu, relabel_nodes=False)
    
    # Create node mapping on CPU
    node_map = torch.zeros(data.num_nodes, dtype=torch.long)
    node_map[subset] = torch.arange(len(subset))
    
    # Initialize features of all nodes to zero on CPU
    x = torch.zeros_like(data.x[subset].cpu())
    
    # Create mask for original nodes on CPU
    subset_mask = torch.zeros(len(subset), dtype=torch.bool)
    subset_mask[torch.isin(subset, node_indices_cpu)] = True
    
    # Set features for original nodes
    x[subset_mask] = data.x.cpu()[subset][subset_mask]
    
    # Create subgraph with everything moved to the specified device
    subgraph = Data(
        x=x.to(DEVICE),
        edge_index=edge_index.to(DEVICE),
        y=data.y.cpu()[subset].to(DEVICE),
        train_mask=data.train_mask.cpu()[subset].to(DEVICE),
        val_mask=data.val_mask.cpu()[subset].to(DEVICE),
        test_mask=data.test_mask.cpu()[subset].to(DEVICE)
    )

    return subgraph, node_map.to(DEVICE), subset_mask.to(DEVICE)

def get_in_comm_indexes(edge_index: torch.Tensor, split_data_indexes: list, 
                       num_clients: int, L_hop: int, idx_train: torch.Tensor, 
                       idx_test: torch.Tensor) -> tuple[list, list, list]:
    """Get communication indexes for each client based on hop neighborhood."""
    communicate_indexes = []
    edge_index = edge_index.cpu()

    for i in range(num_clients):
        communicate_index = split_data_indexes[i]

        if L_hop == 0:
            communicate_index, _, _, _ = k_hop_subgraph(
                communicate_index, 0, edge_index, relabel_nodes=True
            )
        else:
            for hop in range(L_hop):
                if hop != L_hop - 1:
                    communicate_index = k_hop_subgraph(
                        communicate_index, 1, edge_index, relabel_nodes=True
                    )[0]
                else:
                    communicate_index, _, _, _ = k_hop_subgraph(
                        communicate_index, 1, edge_index, relabel_nodes=True
                    )

        communicate_index = communicate_index.to(DEVICE)
        communicate_indexes.append(communicate_index)

    return communicate_indexes, [], []

# def partition_data(data: Data, num_clients: int, beta: float, hop: int = 0, 
#                   use_feature_prop: bool = False) -> tuple[list, list, dict]:
#     """
#     Main partitioning function that handles both feature propagation and non-feature propagation cases.
#     """
#     # Move data to CPU for initial processing
#     labels = data.y.cpu().numpy()
#     N = len(labels)
#     K = len(np.unique(labels))

#     # Get initial partition
#     split_data_indexes = label_dirichlet_partition(labels, N, K, num_clients, beta)
    
#     # Create test data
#     test_data = [create_subgraph(data, indices) for indices in split_data_indexes]
    
#     # Track metrics for first client
#     client_metrics = {
#         'Initial number of nodes': len(split_data_indexes[0]),
#         'K-hop value': hop
#     }
    
#     # Get communication indexes if using hops
#     # if hop > 0:
#     #     communicate_indexes, _, _ = get_in_comm_indexes(
#     #         data.edge_index, 
#     #         split_data_indexes, 
#     #         num_clients, 
#     #         hop, 
#     #         data.train_mask, 
#     #         data.test_mask
#     #     )
#     #     client_metrics['Number of nodes after subgraph expansion'] = len(communicate_indexes[0])
#     # else:
#     #     communicate_indexes = split_data_indexes
#     communicate_indexes = split_data_indexes
       

#     # Create client data
#     clients_data = []
#     for i in range(num_clients):
#         # Always create k-hop subgraph first
#         subgraph, node_map, original_nodes_mask = create_k_hop_subgraph(data, communicate_indexes[i], hop)
#         #
        
#         if i == 0:  # Track metrics for first client before FP
#             zero_vectors_before = (subgraph.x == 0).all(dim=1).sum().item()
            
#             total_nodes = subgraph.num_nodes
#             client_metrics.update({
#                 'Number of nodes after subgraph expansion': subgraph.num_nodes,
#                 'Zero feature vectors before FP': zero_vectors_before,
#                 'Non-zero feature vectors before FP': subgraph.num_nodes - zero_vectors_before,
#                 'Percentage of zero vectors before FP': f"{(zero_vectors_before/total_nodes)*100:.2f}%"
#             })
        
#         if use_feature_prop:
#             # Apply feature propagation
#             zero_vector_mask = (subgraph.x == 0).all(dim=1)
#             non_zero_vector_mask = ~zero_vector_mask
#             subgraph.x = propagate_features(subgraph.x, subgraph.edge_index, non_zero_vector_mask)
            
#             if i == 0:  # Track metrics for first client after FP
#                 zero_vectors_after = (subgraph.x == 0).all(dim=1).sum().item()
#                 client_metrics.update({
#                     'Number of nodes after feature propagation': subgraph.num_nodes,
#                     'Zero feature vectors after FP': zero_vectors_after,
#                     'Non-zero feature vectors after FP': subgraph.num_nodes - zero_vectors_after,
#                     'Percentage of zero vectors after FP': f"{(zero_vectors_after/subgraph.num_nodes)*100:.2f}%"
#                 })
#         else:
#             if i == 0:  # Track metrics for first client (no FP)
#                 client_metrics.update({
#                     'Number of nodes after feature propagation': subgraph.num_nodes,
#                     'Zero feature vectors': zero_vectors_before,
#                     'Non-zero feature vectors': subgraph.num_nodes - zero_vectors_before,
#                     'Percentage of zero vectors': f"{(zero_vectors_before/total_nodes)*100:.2f}%"
#                 })
                
#         clients_data.append(subgraph)

#     return clients_data, test_data, client_metrics, split_data_indexes

def partition_data(data: Data, num_clients: int, beta: float, hop: int = 0, 
                  use_feature_prop: bool = False) -> tuple[list, list, list]:
    """
    Main partitioning function that handles both feature propagation and non-feature propagation cases.
    """
    # Move data to CPU for initial processing
    labels = data.y.cpu().numpy()
    N = len(labels)
    K = len(np.unique(labels))

    # Get initial partition
    split_data_indexes = label_dirichlet_partition(labels, N, K, num_clients, beta)
    
    # Create test data
    test_data = [create_subgraph(data, indices) for indices in split_data_indexes]
    
    # Create client data
    clients_data = []
    for i in range(num_clients):
        # Create k-hop subgraph
        subgraph, node_map, original_nodes_mask = create_k_hop_subgraph(data, split_data_indexes[i], hop)
        
        if use_feature_prop:
            # Apply feature propagation
            zero_vector_mask = (subgraph.x == 0).all(dim=1)
            non_zero_vector_mask = ~zero_vector_mask
            subgraph.x = propagate_features(subgraph.x, subgraph.edge_index, non_zero_vector_mask)
                
        clients_data.append(subgraph)

    return clients_data, test_data, split_data_indexes

def prepare_expanded_subgraph_for_propagation(original_subgraph: Data, expanded_subgraph: Data, original_indices: torch.Tensor):
    """
    Prepares expanded subgraph for feature propagation by:
    - Zeroing features of new nodes (non-original nodes)
    - Setting appropriate masks (only original nodes used for training)
    - Maintaining original features and labels for initial nodes
    """
    # Determine device from original subgraph
    device = original_subgraph.x.device
    
    # Get the mapping of original nodes in the expanded graph
    original_nodes_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    
    # The k_hop_subgraph function returns nodes in order where original nodes come first
    # This is guaranteed by the relabel_nodes=True parameter
    original_nodes_mask[:len(original_indices)] = True
    
    # Print some verification info
    print(f"Original nodes: {len(original_indices)}")
    print(f"Expanded nodes: {expanded_subgraph.num_nodes}")
    print(f"Original nodes in expanded graph: {original_nodes_mask.sum().item()}")
    
    # Create new feature matrix (all zeros initially)
    new_x = torch.zeros_like(expanded_subgraph.x, device=device)
    # Copy original features for original nodes
    new_x[original_nodes_mask] = original_subgraph.x
    
    # Create new masks (only original nodes are used for training)
    new_train_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    new_val_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    new_test_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    
    # Copy original masks for original nodes
    new_train_mask[original_nodes_mask] = original_subgraph.train_mask
    new_val_mask[original_nodes_mask] = original_subgraph.val_mask
    new_test_mask[original_nodes_mask] = original_subgraph.test_mask
    
    # Create new labels (zeros for new nodes)
    new_y = torch.zeros(expanded_subgraph.num_nodes, dtype=expanded_subgraph.y.dtype, device=device)
    new_y[original_nodes_mask] = original_subgraph.y
    
    # Create new Data object
    prepared_subgraph = Data(
        x=new_x,
        edge_index=expanded_subgraph.edge_index.to(device),  # Ensure edge_index is also on correct device
        y=new_y,
        train_mask=new_train_mask,
        val_mask=new_val_mask,
        test_mask=new_test_mask
    )
    
    return prepared_subgraph


