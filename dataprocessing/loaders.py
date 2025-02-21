from dataprocessing.datasets import GraphDataset
from dataprocessing.partitioning import partition_data
from typing import Tuple, List, Optional

def load_dataset(name: str):
    """
    Regime 1: Load any supported dataset without partitioning.
    """
    dataset_loader = GraphDataset()
    return dataset_loader.load_dataset(name)

def load_and_split(name: str, device, num_clients: int = 10, beta: float = 0.5):
    """
    Regime 2: Load dataset and split into n subgraphs.
    """
    data, dataset = load_dataset(name)
    clients_data, test_data,  split_data_indexes = partition_data(data, num_clients, beta, device, hop=0)
    return data, dataset, clients_data, test_data 

def load_and_split_with_khop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, use_feature_prop: bool = False):
    """
    Regime 3: Load dataset, split into n subgraphs, and include k-hop neighbors.
    """
    data, dataset = load_dataset(name)
    clients_data, test_data,  _ = partition_data(data, num_clients, beta, device, hop=hop, use_feature_prop=False)
    return data, dataset, clients_data, test_data

def load_and_split_with_feature_prop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, use_feature_prop: bool = True):
    """
    Regime 4: Load dataset, split into n subgraphs, include k-hop neighbors, and propagate features.
    """
    data, dataset = load_dataset(name)
    clients_data, test_data,  _ = partition_data(data, num_clients, beta, device, hop=hop, use_feature_prop=True)
    return data, dataset, clients_data, test_data