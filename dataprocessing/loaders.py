from dataprocessing.datasets import GraphDataset
from dataprocessing.partitioning import partition_data
from typing import Tuple, List, Optional

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
    clients_data, test_data,  _ = partition_data(data, num_clients, beta, device, hop=hop, use_feature_prop=use_feature_prop, full_data=full_data, fulltraining_flag=fulltraining_flag, mode=propagation_mode)
    return data, dataset, clients_data, test_data

def load_and_split_with_feature_prop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, use_feature_prop: bool = True, full_data: bool = False, fulltraining_flag: bool = False):
    """
    Regime 4: Load dataset, split into n subgraphs, include k-hop neighbors, and propagate features.
    """
    data, dataset = load_dataset(name, device)
    clients_data, test_data,  _ = partition_data(data, num_clients, beta, device, hop=hop, use_feature_prop=use_feature_prop, full_data=full_data, fulltraining_flag=fulltraining_flag)
    return data, dataset, clients_data, test_data