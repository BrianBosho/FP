from src.dataprocessing.datasets import GraphDataset
from src.dataprocessing.partitioning import partition_data
from typing import Tuple, List, Optional
from src.dataprocessing.positional_encoding import generate_rfp_encoding
import torch
import torch.nn.functional as F

# Note: Datasets are now loaded from the root-level 'datasets' folder
# instead of being stored in the src directory. This change is implemented
# in the GraphDataset class which manages dataset paths.

def load_dataset(name: str, device):
    """
    Regime 1: Load any supported dataset without partitioning.
    """
    dataset_loader = GraphDataset(device)
    return dataset_loader.load_dataset(name, device)

def load_and_split(name: str, device, num_clients: int = 10, beta: float = 0.5, config: dict = None):
    """
    Regime 2: Load dataset and split into n subgraphs.
    
    Args:
        name: Dataset name
        device: Device to run computations on
        num_clients: Number of clients
        beta: Dirichlet concentration parameter
        config: Configuration dictionary from YAML file (optional)
    """
    data, dataset = load_dataset(name, device)
    clients_data, test_data, split_data_indexes = partition_data(
        data, 
        num_clients, 
        beta, 
        device, 
        hop=0,
        config=config
    )
    return data, dataset, clients_data, test_data

def load_and_split_with_khop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, fulltraining_flag: bool = False, imputation_method: str = "zero", propagation_mode: str = "propagation", config: dict = None):
    """
    Regime 3: Load dataset, split into n subgraphs, and include k-hop neighbors.
    
    Args:
        name: Dataset name
        device: Device to run computations on
        num_clients: Number of clients
        beta: Dirichlet concentration parameter
        hop: Number of hops for neighbor expansion
        fulltraining_flag: Whether to use full training data from k-hop subgraph
        imputation_method: Method for feature imputation (zero, full, propagation, etc.)
        propagation_mode: Type of propagation to use (page_rank, random_walk, etc.)
        config: Configuration dictionary from YAML file (optional)
    """
    # Process different imputation methods
    if imputation_method == "zero":
        use_feature_prop = False
        full_data = False

    elif imputation_method == "full":
        use_feature_prop = False
        full_data = True
    elif imputation_method == "page_rank":
        use_feature_prop = True
        full_data = False
        propagation_mode = "page_rank"
    elif imputation_method == "random_walk":
        use_feature_prop = True
        full_data = False
        propagation_mode = "random_walk"
    elif imputation_method == "diffusion" or imputation_method == "difussion":
        use_feature_prop = True
        full_data = False
        propagation_mode = "diffusion"
    elif imputation_method == "efficient":
        use_feature_prop = True
        full_data = False
        propagation_mode = "efficient"
    elif imputation_method == "adjacency":
        use_feature_prop = True
        full_data = False
        propagation_mode = "adjacency"
    elif imputation_method == "propagation":
        use_feature_prop = True
        full_data = False
        propagation_mode = "propagation"
    elif imputation_method == "chebyshev_diffusion":
        use_feature_prop = True
        full_data = False
        propagation_mode = "chebyshev_diffusion"
    elif imputation_method == "chebyshev_diffusion_operator":
        use_feature_prop = True
        full_data = False
        propagation_mode = "chebyshev_diffusion_operator"
    else:
        # Default case to handle any unrecognized imputation method
        use_feature_prop = False
        full_data = False
        print(f"Warning: Unrecognized imputation method '{imputation_method}'. Using default (zero imputation).")

    data, dataset = load_dataset(name, device)
    
    # Get positional encoding flag from config if available
    use_pe = True
    if config is not None:
        use_pe = config.get("use_pe", use_pe)
    
    use_pe = use_feature_prop and use_pe
    
    # Pass config to partition_data
    clients_data, test_data, _ = partition_data(
        data, 
        num_clients, 
        beta, 
        device, 
        hop=hop, 
        use_feature_prop=use_feature_prop, 
        full_data=full_data, 
        fulltraining_flag=fulltraining_flag, 
        mode=propagation_mode,
        config=config
    )
    
    # If using PE, apply it to the global data for consistency in evaluation
    if use_pe:
        # Get PE parameters from config if available
        pe_r = 64
        pe_P = 16
        normalize = "qr"
        
        if config is not None:
            pe_r = config.get("pe_r", pe_r)
            pe_P = config.get("pe_P", pe_P)
            normalize = config.get("normalize", normalize)
        
        rfp = generate_rfp_encoding(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            r=pe_r,
            P=pe_P,
            normalize=normalize,
            device=device
        )
        orig_features = F.normalize(data.x.to(device), p=2, dim=1)
        rfp_norm = F.normalize(rfp, p=2, dim=1) * 0.5  # use same rfp_alpha as clients
        data.x = torch.cat([orig_features, rfp_norm], dim=1)
    
    return data, dataset, clients_data, test_data

def load_and_split_with_feature_prop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, use_feature_prop: bool = True, full_data: bool = False, fulltraining_flag: bool = False, config: dict = None):
    """
    Regime 4: Load dataset, split into n subgraphs, include k-hop neighbors, and propagate features.
    
    Args:
        name: Dataset name
        device: Device to run computations on
        num_clients: Number of clients
        beta: Dirichlet concentration parameter
        hop: Number of hops for neighbor expansion
        use_feature_prop: Whether to use feature propagation
        full_data: Whether to use all node features
        fulltraining_flag: Whether to use full training data from k-hop subgraph
        config: Configuration dictionary from YAML file (optional)
    """
    data, dataset = load_dataset(name, device)
    clients_data, test_data, _ = partition_data(
        data, 
        num_clients, 
        beta, 
        device, 
        hop=hop, 
        use_feature_prop=use_feature_prop, 
        full_data=full_data, 
        fulltraining_flag=fulltraining_flag,
        config=config,
        num_iterations=num_iterations
    )
    return data, dataset, clients_data, test_data