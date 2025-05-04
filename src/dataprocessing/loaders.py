from dataprocessing.datasets import GraphDataset
from dataprocessing.partitioning import partition_data
from typing import Tuple, List, Optional
from dataprocessing.positional_encoding import generate_rfp_encoding
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

def load_and_split(name: str, device, num_clients: int = 10, beta: float = 0.5):
    """
    Regime 2: Load dataset and split into n subgraphs.
    """
    data, dataset = load_dataset(name, device)
    clients_data, test_data,  split_data_indexes = partition_data(data, num_clients, beta, device, hop=0)
    return data, dataset, clients_data, test_data 

def load_and_split_with_khop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, fulltraining_flag: bool = False, imputation_method: str = "zero", propagation_mode: str = "propagation"):
    """
    Regime 3: Load dataset, split into n subgraphs, and include k-hop neighbors.
    """
    # lets use different imputation methods
    # 1. zero imputation
    # 2. feature propagation
    # 3. full data 

    if imputation_method == "zero":
        use_feature_prop = False
        full_data = False

    elif imputation_method == "full":
        use_feature_prop = False
        full_data = True
        # add monte carlo mode
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
    else:
        # Default case to handle any unrecognized imputation method
        use_feature_prop = False
        full_data = False
        print(f"Warning: Unrecognized imputation method '{imputation_method}'. Using default (zero imputation).")

    data, dataset = load_dataset(name, device)
    use_pe = True
    use_pe = use_feature_prop and use_pe
    # Inject RFP positional coding into global data.x (for global evaluation)
  
    clients_data, test_data,  _ = partition_data(data, num_clients, beta, device, hop=hop, use_feature_prop=use_feature_prop, full_data=full_data, fulltraining_flag=fulltraining_flag, mode=propagation_mode)
    
    if use_pe:
        rfp = generate_rfp_encoding(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            r=64,           # match what clients use (e.g., pe_r)
            P=16,           # match what clients use (e.g., pe_P)
            normalize="qr",
            device=device
        )
        orig_features = F.normalize(data.x.to(device), p=2, dim=1)
        rfp_norm = F.normalize(rfp, p=2, dim=1) * 0.5  # use same rfp_alpha as clients
        data.x = torch.cat([orig_features, rfp_norm], dim=1)
    
    return data, dataset, clients_data, test_data

def load_and_split_with_feature_prop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, use_feature_prop: bool = True, full_data: bool = False, fulltraining_flag: bool = False):
    """
    Regime 4: Load dataset, split into n subgraphs, include k-hop neighbors, and propagate features.
    """
    data, dataset = load_dataset(name, device)
    clients_data, test_data,  _ = partition_data(data, num_clients, beta, device, hop=hop, use_feature_prop=use_feature_prop, full_data=full_data, fulltraining_flag=fulltraining_flag)
    return data, dataset, clients_data, test_data