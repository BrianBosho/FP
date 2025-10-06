import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from src.dataprocessing.data_utils import propagate_features, compute_dirichlet_energy
from src.dataprocessing.positional_encoding import generate_rfp_encoding
import torch.nn.functional as F
# from utils import propagate_features

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def create_subgraph(data: Data, node_indices: torch.Tensor, device = "cuda") -> Data:
    """
    Creates a simple subgraph containing ONLY the specified nodes and their direct connections.
    """
    DEVICE = device
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

def create_k_hop_subgraph(data: Data, node_indices: torch.Tensor, num_hops: int, device = "cuda", full_data: bool = False, fulltraining_flag: bool = True) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Creates a k-hop subgraph with zeroed features for non-original nodes.
    """
    DEVICE = device
    # Move everything to CPU for processing
    edge_index_cpu = data.edge_index.cpu()
    node_indices_cpu = node_indices.cpu() if isinstance(node_indices, torch.Tensor) else torch.tensor(node_indices)
    
    # Get k-hop subgraph
    subset, edge_index, mapping, _ = k_hop_subgraph(node_indices_cpu, num_hops, edge_index_cpu, relabel_nodes=True)
    
    # Create node mapping on CPU
    node_map = torch.zeros(data.num_nodes, dtype=torch.long)
    node_map[subset] = torch.arange(len(subset))
    
    # Skip feature initialization here since it will be done in reset_subgraph_features2
    
    # Create a basic subgraph with all data
    subgraph = Data(
        x=data.x.cpu()[subset].to(DEVICE),  # We'll reset these features in reset_subgraph_features2
        edge_index=edge_index.to(DEVICE),
        y=data.y.cpu()[subset].to(DEVICE),
        train_mask=data.train_mask.cpu()[subset].to(DEVICE),
        val_mask=data.val_mask.cpu()[subset].to(DEVICE),
        test_mask=data.test_mask.cpu()[subset].to(DEVICE)
    )

    # Reset features and masks based on original nodes
    subgraph = reset_subgraph_features2(subgraph, mapping, full_data, fulltraining_flag)

    return subgraph, node_map.to(DEVICE), mapping.to(DEVICE)

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


def partition_data(data: Data, num_clients: int, beta: float, device, hop: int = 0, 
                   use_feature_prop: bool = False, full_data: bool = False, fulltraining_flag: bool = False, 
                   mode: str = "propagation", use_pe: bool = True, pe_r: int = 64, pe_P: int = 16, 
                   config: dict = None) -> tuple[list, list, list]:
    """
    Main partitioning function that handles both feature propagation and positional encoding.
    
    Args:
        data: PyG Data object containing the graph
        num_clients: Number of clients for partitioning
        beta: Dirichlet concentration parameter
        device: Device to run computations on
        hop: Number of hops for subgraph expansion
        use_feature_prop: Whether to use feature propagation
        full_data: If True, use all node features in k-hop neighborhood
        fulltraining_flag: If True, use all masks from k-hop subgraph
        mode: Feature propagation mode
        use_pe: Whether to use positional encoding (default: True)
        pe_r: Dimensionality of random features for positional encoding (default: 64)
        pe_P: Number of propagation steps for positional encoding (default: 16)
        config: Configuration dictionary from YAML file (optional)
    """
    import time, os, json
    DEVICE = device
    
    # Update parameters from config if provided (coalesce None to defaults)
    if config is not None:
        use_pe = config.get("use_pe", use_pe)
        pe_r = config.get("pe_r", pe_r)
        pe_P = config.get("pe_P", pe_P)
        normalize = config.get("normalize", "qr")
        num_iterations = config.get("num_iterations", 50)
        fp_tolerance = config.get("feature_prop_tolerance", 1e-3)

        # If keys exist but are None (e.g., from wandb config), fallback to defaults
        if normalize is None:
            normalize = "qr"
        if num_iterations is None:
            num_iterations = 50
        if fp_tolerance is None:
            fp_tolerance = 1e-3
    else:
        normalize = "qr"
        num_iterations = 50
        fp_tolerance = 1e-3
    
    labels = data.y.cpu().numpy()
    N = len(labels)
    K = len(np.unique(labels))
    initial_graph_de = compute_dirichlet_energy(data.x, data.edge_index)

    split_data_indexes = label_dirichlet_partition(labels, N, K, num_clients, beta)
    initial_subgraphs = [create_subgraph(data, indices, device) for indices in split_data_indexes]

    clients_data = []
    if hop > 0:
        for i in range(num_clients):
            subgraph, node_map, mapping = create_k_hop_subgraph(data, split_data_indexes[i], hop, device, full_data, fulltraining_flag)
            clients_data.append(subgraph)
    else:
        clients_data = initial_subgraphs    

    use_pe = use_feature_prop and use_pe

    # Setup logging if needed
    if use_feature_prop:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_id = f"prop_exp_{timestamp}_{mode}_beta_{beta}_hop_{hop}"
        
        # Determine logs directory from config or use default
        if config is not None and "results_dir" in config:
            # Use results directory from config if available
            results_dir = config.get("results_dir")
            logs_dir = os.path.join(results_dir, "propagation_stats")
        else:
            # Use default logs directory
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "propagation_stats")
        
        os.makedirs(logs_dir, exist_ok=True)
        json_file = os.path.join(logs_dir, f"{experiment_id}.json")
        experiment_data = {
            "experiment_id": experiment_id,
            "propagation_mode": mode,
            "num_clients": num_clients,
            "beta": beta,
            "hop": hop,
            "initial_energy": initial_graph_de,
            "use_pe": use_pe,
            "pe_r": pe_r if use_pe else None,
            "pe_P": pe_P if use_pe else None,
            "normalize": normalize if use_pe else None,
            "clients": []
        }
        with open(json_file, 'w') as f:
            json.dump(experiment_data, f)

    final_subgraphs = []
    for i in range(num_clients):
        # Determine original node mask
        if hasattr(clients_data[i], 'mapping'):
            original_nodes_mask = torch.zeros(clients_data[i].num_nodes, dtype=torch.bool, device=DEVICE)
            original_nodes_mask[clients_data[i].mapping] = True
        else:
            zero_vector_mask = (clients_data[i].x == 0).all(dim=1)
            original_nodes_mask = ~zero_vector_mask

        # Step 1: Feature Propagation comes first
        if use_feature_prop:
            # Decide FP device (default CPU; overridable by config)
            fp_device = torch.device(
                config.get("feature_prop_device", "cuda") if config is not None else "cuda"
            )

            # Move necessary tensors for FP to fp_device
            x_fp = clients_data[i].x.to(fp_device)
            edge_index_fp = clients_data[i].edge_index.to(fp_device)
            original_nodes_mask_fp = original_nodes_mask.to(fp_device)

            # Run FP on fp_device
            x_fp = propagate_features(
                x_fp,
                edge_index_fp,
                original_nodes_mask_fp,
                fp_device,
                num_iterations=num_iterations,
                mode=mode,
                client_id=i,
                log_file=json_file,
                tol=fp_tolerance
            )

            # Move features back to original DEVICE for training/PE
            clients_data[i].x = x_fp.to(DEVICE)

        # Step 2: Then Positional Encoding (if requested)
        if use_pe:
            clients_data[i].original_feature_dim = clients_data[i].x.size(1)
            rfp = generate_rfp_encoding(
                edge_index=clients_data[i].edge_index,
                num_nodes=clients_data[i].num_nodes,
                r=pe_r, 
                P=pe_P,
                normalize=normalize,
                device=DEVICE
            )
            orig_features = F.normalize(clients_data[i].x, p=2, dim=1)
            rfp_norm = F.normalize(rfp, p=2, dim=1) * 0.5
            clients_data[i].x = torch.cat([orig_features, rfp_norm], dim=1)

            if hasattr(clients_data[i], 'num_features'):
                clients_data[i].num_features = clients_data[i].x.size(1)

        final_subgraphs.append(clients_data[i])

    if use_feature_prop:
        print(f"Feature propagation logs saved to: {json_file}")

    return final_subgraphs, initial_subgraphs, split_data_indexes



# def partition_data(data: Data, num_clients: int, beta: float, device, hop: int = 0, 
#                   use_feature_prop: bool = False, full_data: bool = False, fulltraining_flag: bool = False, 
#                   mode: str = "propagation", use_pe: bool = False, pe_r: int = 16, pe_P: int = 4) -> tuple[list, list, list]:
#     """
#     Main partitioning function that handles both feature propagation and non-feature propagation cases.
    
#     Args:
#         data: PyG Data object containing the graph
#         num_clients: Number of clients for partitioning
#         beta: Dirichlet concentration parameter
#         device: Device to run computations on
#         hop: Number of hops for subgraph expansion
#         use_feature_prop: Whether to use feature propagation
#         full_data: If True, use all node features in k-hop neighborhood
#         fulltraining_flag: If True, use all masks from k-hop subgraph
#         mode: Feature propagation mode
#         use_pe: Whether to use positional encoding before feature propagation (default: False)
#         pe_r: Dimensionality of random features for positional encoding
#         pe_P: Number of propagation steps for positional encoding
#     """
#     # Move data to CPU for initial processing
#     DEVICE = device
#     labels = data.y.cpu().numpy()
#     N = len(labels)
#     K = len(np.unique(labels))
#     initial_graph_de = compute_dirichlet_energy(data.x, data.edge_index)


#     # Get initial partition
#     split_data_indexes = label_dirichlet_partition(labels, N, K, num_clients, beta)
    
#     # Create test data
#     initial_subgraphs = [create_subgraph(data, indices, device) for indices in split_data_indexes]

#     if hop > 0:
#         # get k-hop subgraph for each client
#         clients_data = []
#         for i in range(num_clients):
#             subgraph, node_map, mapping = create_k_hop_subgraph(data, split_data_indexes[i], hop, device, full_data, fulltraining_flag)
#             clients_data.append(subgraph)
        
#         # clients_data = k_hop_subgraphs
#     else:
#         clients_data = initial_subgraphs    
   
#     if use_feature_prop:
#         # Create a timestamp-based experiment ID
#         import time, os, json
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         experiment_id = f"prop_exp_{timestamp}_{mode}_beta_{beta}_hop_{hop}"
        
#         # Create logs directory if it doesn't exist
#         logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "propagation_stats")
#         os.makedirs(logs_dir, exist_ok=True)
        
#         # Create experiment JSON file with metadata
#         json_file = os.path.join(logs_dir, f"{experiment_id}.json")
#         experiment_data = {
#             "experiment_id": experiment_id,
#             "propagation_mode": mode,
#             "num_clients": num_clients,
#             "beta": beta,
#             "hop": hop,
#             "initial_energy": initial_graph_de,
#             "use_pe": use_pe,
#             "pe_r": pe_r if use_pe else None,
#             "pe_P": pe_P if use_pe else None,
#             "clients": []
#         }
#         with open(json_file, 'w') as f:
#             json.dump(experiment_data, f)
        
#         # Apply feature propagation to each client's subgraph
#         final_subgraphs = []
#         for i in range(num_clients):
#             # Get the original node mapping if available
#             original_nodes_mask = None
#             if hasattr(clients_data[i], 'mapping'):
#                 # Create mask based on original node mapping
#                 original_nodes_mask = torch.zeros(clients_data[i].num_nodes, dtype=torch.bool, device=DEVICE)
#                 original_nodes_mask[clients_data[i].mapping] = True
#             else:
#                 # Fallback to checking for zero vectors
#                 zero_vector_mask = (clients_data[i].x == 0).all(dim=1)
#                 original_nodes_mask = ~zero_vector_mask
            
#             # Step 1: First generate and add positional encodings if requested
#             if use_pe:
#                 # Store original dimension before concatenation
#                 clients_data[i].original_feature_dim = clients_data[i].x.size(1)
                
#                 # Generate RFP encoding with L2 normalization
#                 rfp = generate_rfp_encoding(
#                     edge_index=clients_data[i].edge_index,
#                     num_nodes=clients_data[i].num_nodes,
#                     r=pe_r, 
#                     P=pe_P,
#                     normalize="qr",  # Use QR for better orthogonality
#                     device=DEVICE
#                 )
                
#                 # Normalize both features before concatenation
#                 orig_features = F.normalize(clients_data[i].x, p=2, dim=1)
#                 rfp_norm = F.normalize(rfp, p=2, dim=1)
                
#                 # Concatenate normalized features
#                 clients_data[i].x = torch.cat([orig_features, rfp_norm], dim=1)
                
#                 # Update num_features if it exists
#                 if hasattr(clients_data[i], 'num_features'):
#                     clients_data[i].num_features = clients_data[i].x.size(1)
            
#             # Step 2: Then propagate features using original node mask
#             clients_data[i].x = propagate_features(
#                 clients_data[i].x, 
#                 clients_data[i].edge_index, 
#                 original_nodes_mask,  # Use original node mask for propagation
#                 DEVICE, 
#                 mode=mode,
#                 client_id=i,
#                 log_file=json_file
#             )
            
#             final_subgraphs.append(clients_data[i])
        
#         print(f"Feature propagation logs saved to: {json_file}")
#     else:
#         # Even if we're not using feature propagation, we can still generate PE if requested
#         if use_pe:
#             for i in range(num_clients):
#                 # Generate RFP encoding
#                 rfp = generate_rfp_encoding(
#                     edge_index=clients_data[i].edge_index,
#                     num_nodes=clients_data[i].num_nodes,
#                     r=pe_r, 
#                     P=pe_P,
#                     normalize="qr",  # Use QR for better orthogonality
#                     device=DEVICE
#                 )
                
#                 # Store original feature dimension before concatenation
#                 clients_data[i].original_feature_dim = clients_data[i].x.size(1)
                
#                 # Normalize both features before concatenation
#                 orig_features = F.normalize(clients_data[i].x, p=2, dim=1)
#                 rfp_norm = F.normalize(rfp, p=2, dim=1)
                
#                 # Concatenate normalized features
#                 clients_data[i].x = torch.cat([orig_features, rfp_norm], dim=1)
                
#                 # Update num_features if it exists
#                 if hasattr(clients_data[i], 'num_features'):
#                     clients_data[i].num_features = clients_data[i].x.size(1)
        
#         final_subgraphs = clients_data

#     return final_subgraphs, initial_subgraphs, split_data_indexes

def reset_subgraph_features(subset_data: Data, mapping: torch.Tensor) -> Data:

    """
    Reset features of non-original nodes to zero while maintaining graph structure and masks.
    
    Args:
        subset_data (Data): PyG Data object containing the subgraph
        mapping (torch.Tensor): Tensor containing indices of original nodes in the subgraph
    
    Returns:
        Data: New PyG Data object with reset features for non-original nodes
    """
    # Create mask for original nodes in the subgraph
    subset_mask = torch.zeros(subset_data.num_nodes, dtype=torch.bool)
    subset_mask[mapping] = True
    
    # Initialize tensors with zeros, maintaining the original size
    reset_x = torch.zeros_like(subset_data.x)
    reset_y = torch.zeros_like(subset_data.y)
    
    # Copy only the original nodes' data, leaving others as zeros
    reset_x[subset_mask] = subset_data.x[subset_mask]
    reset_y[subset_mask] = subset_data.y[subset_mask]
    
    # Clone original masks
    reset_train_mask = subset_data.train_mask.clone()
    reset_val_mask = subset_data.val_mask.clone()
    reset_test_mask = subset_data.test_mask.clone()
    
    # Create new Data object
    reset_data = Data(
        x=reset_x,
        y=reset_y,
        train_mask=reset_train_mask,
        val_mask=reset_val_mask,
        test_mask=reset_test_mask,
        edge_index=subset_data.edge_index,
        mapping=mapping
    )
    
    return reset_data


def reset_subgraph_features2(subset_data: Data, mapping: torch.Tensor, full_data: bool = True, fulltraining_flag: bool = True) -> Data:
    """
    Reset features of non-original nodes to zero while maintaining graph structure and masks.
    Only original nodes specified in the mapping will be used for train/val/test splits.
    
    Args:
        subset_data (Data): PyG Data object containing the subgraph
        mapping (torch.Tensor): Tensor containing indices of original nodes in the subgraph
        full_data (bool, optional): If True, keep all node features. If False, zero out non-original nodes. Default is False.
        fulltraining_flag (bool, optional): If True, use all masks from k-hop subgraph. If False, restrict to original nodes. Default is False.
    
    Returns:
        Data: New PyG Data object with reset features for non-original nodes
    """
    # Create mask for original nodes in the subgraph
    subset_mask = torch.zeros(subset_data.num_nodes, dtype=torch.bool)
    subset_mask[mapping] = True
    
    if full_data:
        # Keep all features as they are
        reset_x = subset_data.x.clone()
        reset_y = subset_data.y.clone()
    else:
        reset_y = subset_data.y.clone()
        # Initialize with zeros and only copy original nodes' data
        reset_x = torch.zeros_like(subset_data.x)
       # reset_y = torch.zeros_like(subset_data.y)
        reset_x[subset_mask] = subset_data.x[subset_mask]
        #reset_y[subset_mask] = subset_data.y[subset_mask]
    
    if fulltraining_flag:
        # Use all masks from the expanded subgraph
        reset_train_mask = subset_data.train_mask.clone()
        reset_val_mask = subset_data.val_mask.clone()
        reset_test_mask = subset_data.test_mask.clone()
    else:
        # Create new masks based on the mapping (original nodes only)
        reset_train_mask = torch.zeros(subset_data.num_nodes, dtype=torch.bool)
        reset_val_mask = torch.zeros(subset_data.num_nodes, dtype=torch.bool)
        reset_test_mask = torch.zeros(subset_data.num_nodes, dtype=torch.bool)

        # Set the masks for original nodes only
        for orig_idx in mapping:
            if subset_data.train_mask[orig_idx]:
                reset_train_mask[orig_idx] = True
            if subset_data.val_mask[orig_idx]:
                reset_val_mask[orig_idx] = True
            if subset_data.test_mask[orig_idx]:
                reset_test_mask[orig_idx] = True
    
    # Create new Data object
    reset_data = Data(
        x=reset_x,
        y=reset_y,
        train_mask=reset_train_mask,
        val_mask=reset_val_mask,
        test_mask=reset_test_mask,
        edge_index=subset_data.edge_index,
        mapping=mapping
    )
    
    return reset_data
def prepare_expanded_subgraph_for_propagation(original_subgraph: Data, expanded_subgraph: Data, mapping: torch.Tensor):
    """
    Prepares expanded subgraph for feature propagation by:
    - Zeroing features of new nodes (non-original nodes)
    - Setting appropriate masks based on original node mappings
    - Maintaining original features and labels for initial nodes
    
    Args:
        original_subgraph: The initial subgraph containing only the original nodes
        expanded_subgraph: The k-hop expanded subgraph
        mapping: Tensor mapping original node indices to their positions in expanded subgraph
                (returned by k_hop_subgraph)
    """
    # Determine device from original subgraph
    device = original_subgraph.x.device
    
    # Create new feature matrix (all zeros initially)
    new_x = torch.zeros_like(expanded_subgraph.x, device=device)
    
    # Create new masks (all False initially)
    new_train_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    new_val_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    new_test_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    
    # Create original nodes mask
    original_nodes_mask = torch.zeros(expanded_subgraph.num_nodes, dtype=torch.bool, device=device)
    original_nodes_mask[mapping] = True
    
    # Copy features and labels only for original nodes using the mapping
    new_x[mapping] = original_subgraph.x
    
    # Create new labels (zeros for new nodes)
    new_y = torch.zeros(expanded_subgraph.num_nodes, dtype=expanded_subgraph.y.dtype, device=device)
    new_y[mapping] = original_subgraph.y
    
    # Copy masks only for original nodes using the mapping
    new_train_mask[mapping] = original_subgraph.train_mask
    new_val_mask[mapping] = original_subgraph.val_mask
    new_test_mask[mapping] = original_subgraph.test_mask
    
    # Create new Data object
    prepared_subgraph = Data(
        x=new_x,
        edge_index=expanded_subgraph.edge_index.to(device),
        y=new_y,
        train_mask=new_train_mask,
        val_mask=new_val_mask,
        test_mask=new_test_mask,
        original_nodes_mask=original_nodes_mask  # Adding this for potential future use
    )
    
    return prepared_subgraph