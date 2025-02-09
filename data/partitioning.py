import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from utils import propagate_features

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
    
    Args:
        data: Input graph data
        node_indices: Indices of nodes to include in subgraph
    """
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_mask[node_indices] = True
    
    if data.edge_index is None:
        raise ValueError("data.edge_index is None")

    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    edge_index = data.edge_index[:, edge_mask]
    subgraph_node_indices = torch.where(node_mask)[0]
    node_map = torch.zeros(data.num_nodes, dtype=torch.long)
    node_map[subgraph_node_indices] = torch.arange(len(subgraph_node_indices))
    edge_index = node_map[edge_index]

    return Data(
        x=data.x[subgraph_node_indices],
        edge_index=edge_index,
        y=data.y[subgraph_node_indices],
        train_mask=data.train_mask[node_mask],
        val_mask=data.val_mask[node_mask],
        test_mask=data.test_mask[node_mask]
    )

def create_k_hop_subgraph(data: Data, node_indices: torch.Tensor, num_hops: int) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Creates a k-hop subgraph with zeroed features for non-original nodes.
    
    Args:
        data: Input graph data
        node_indices: Indices of nodes to include in subgraph
        num_hops: Number of hops to include in neighborhood
    
    Returns:
        subgraph: The created subgraph
        node_map: Mapping from original to new node indices
        subset_mask: Mask indicating original nodes
    """
    subset, edge_index, _, _ = k_hop_subgraph(node_indices, num_hops, data.edge_index, relabel_nodes=True)
    node_map = torch.zeros(data.num_nodes, dtype=torch.long)
    node_map[subset] = torch.arange(len(subset))
    
    # Initialize features
    x = torch.zeros_like(data.x[subset])
    node_indices_tensor = torch.tensor(node_indices) if not isinstance(node_indices, torch.Tensor) else node_indices
    
    # Set masks
    subset_mask = torch.zeros(len(subset), dtype=torch.bool)
    subset_mask[torch.isin(subset, node_indices_tensor)] = True
    x[subset_mask] = data.x[subset][subset_mask]
    
    # Create masks for original indices
    train_mask = torch.zeros(len(subset), dtype=torch.bool)
    val_mask = torch.zeros(len(subset), dtype=torch.bool)
    test_mask = torch.zeros(len(subset), dtype=torch.bool)
    
    original_indices_mask = torch.isin(subset, node_indices_tensor)
    train_mask[original_indices_mask] = data.train_mask[subset][original_indices_mask]
    val_mask[original_indices_mask] = data.val_mask[subset][original_indices_mask]
    test_mask[original_indices_mask] = data.test_mask[subset][original_indices_mask]
    
    subgraph = Data(
        x=x,
        edge_index=edge_index,
        y=data.y[subset],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return subgraph, node_map, subset_mask

def get_in_comm_indexes(edge_index: torch.Tensor, split_data_indexes: list, 
                       num_clients: int, L_hop: int, idx_train: torch.Tensor, 
                       idx_test: torch.Tensor) -> tuple[list, list, list]:
    """Get communication indexes for each client based on hop neighborhood."""
    communicate_indexes = []

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

def partition_data(data: Data, num_clients: int, beta: float, hop: int = 0, 
                  use_feature_prop: bool = False) -> tuple[list, list]:
    """
    Main partitioning function that handles both feature propagation and non-feature propagation cases.
    
    Args:
        data: PyG Data object
        num_clients: Number of clients
        beta: Dirichlet concentration parameter
        hop: Number of hops for neighborhood
        use_feature_prop: Whether to use feature propagation
    """
    labels = data.y.numpy()
    N = len(labels)
    K = len(np.unique(labels))

    # Get initial partition
    split_data_indexes = label_dirichlet_partition(labels, N, K, num_clients, beta)
    
    # Create test data
    test_data = [create_subgraph(data, indices) for indices in split_data_indexes]
    
    # Get communication indexes if using hops
    if hop > 0:
        communicate_indexes, _, _ = get_in_comm_indexes(
            data.edge_index, 
            split_data_indexes, 
            num_clients, 
            hop, 
            data.train_mask, 
            data.test_mask
        )
    else:
        communicate_indexes = split_data_indexes

    # Create client data
    clients_data = []
    for i in range(num_clients):
        if use_feature_prop:
            subgraph = create_subgraph(data, communicate_indexes[i])
            # Apply feature propagation
            zero_vector_mask = (subgraph.x == 0).all(dim=1)
            non_zero_vector_mask = ~zero_vector_mask
            subgraph.x = propagate_features(subgraph.x, subgraph.edge_index, non_zero_vector_mask)
        else:
            subgraph, _, _ = create_k_hop_subgraph(data, communicate_indexes[i], hop)
        clients_data.append(subgraph)

    return clients_data, test_data
