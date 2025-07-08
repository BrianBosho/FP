from torch_geometric.datasets import Planetoid, FacebookPagePage
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as transforms
from torch_geometric.utils import to_undirected, add_remaining_self_loops
import torch
import numpy as np
from typing import Tuple, Optional, Union, Literal
import torch_geometric
import os
import yaml
from pathlib import Path


def load_config():
    # Get the project root directory (parent of conf/)
    project_root = Path(__file__).parent.parent
    # config_path = project_root / "conf" / "base.yaml"
    config_path = "/home/brian_bosho/FP/FP/federated-gnn/conf/base.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Make all paths absolute by joining with project root
    for category in ['planetoid', 'ogbn']:
        for dataset, path in config['paths'][category].items():
            config['paths'][category][dataset] = str(project_root / path)
    
    config['paths']['datasets_dir'] = str(project_root / config['paths']['datasets_dir'])
    return config

class GraphDataset:
    """Base class for handling different graph datasets with consistent structure."""
    
    SUPPORTED_DATASETS = {
        "planetoid": ["Cora", "Citeseer", "Pubmed"],
        "facebook": ["FacebookPagePage"],
        "ogb": ["ogbn-arxiv", "ogbn-products"]
    }

    def __init__(self, device = "cuda"):
        self.data = None
        self.dataset = None
        self.config = load_config()
        self.device = device

        
        # Create dataset directories if they don't exist
        for path in [
            self.config['paths']['datasets_dir'],
            self.config['paths']['planetoid']['cora'],
            self.config['paths']['planetoid']['citeseer'],
            self.config['paths']['planetoid']['pubmed'],
            self.config['paths']['ogbn']['arxiv']
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_available_datasets():
        """Returns all available dataset names."""
        return {
            name 
            for category in GraphDataset.SUPPORTED_DATASETS.values() 
            for name in category
        }

    def _get_dataset_path(self, name: str) -> str:
        """Get the appropriate path for each dataset."""
        name_lower = name.lower()
        if name in self.SUPPORTED_DATASETS["planetoid"]:
            return self.config['paths']['planetoid'][name_lower]
        elif name in self.SUPPORTED_DATASETS["facebook"]:
            return os.path.join(self.config['paths']['datasets_dir'], 'facebook')
        elif name in self.SUPPORTED_DATASETS["ogb"]:
            return self.config['paths']['ogbn']['arxiv']
        else:
            raise ValueError(f"Dataset {name} not supported")

    def load_dataset(self, name: str, device) -> Tuple[torch_geometric.data.Data, object]:
        """
        Load any supported dataset with consistent structure.
        
        Args:
            name: Name of the dataset to load
            
        Returns:
            data: Processed PyG Data object
            dataset: Raw dataset object
        """
        name_lower = name.lower()
        
        if name in self.SUPPORTED_DATASETS["planetoid"]:
            return self._load_planetoid(name, device)
        elif name in self.SUPPORTED_DATASETS["facebook"]:
            return self._load_facebook(device)
        elif name in self.SUPPORTED_DATASETS["ogb"]:
            return self._load_ogb(name, device)
        else:
            raise ValueError(f"Dataset {name} not supported. Available datasets: {self.get_available_datasets()}")

    def _load_planetoid(self, name: str, device):
        """Load and process Planetoid dataset."""
        DEVICE = device
        dataset_path = self._get_dataset_path(name)
        dataset = Planetoid(root=dataset_path, name=name)
        data = dataset[0].to(DEVICE)
        return data, dataset

    def _load_facebook(self, device = "cuda"):
        """Load and process Facebook dataset."""
        DEVICE = device
        dataset_path = self._get_dataset_path("FacebookPagePage")
        dataset = FacebookPagePage(root=dataset_path)
        data = dataset[0]
        # Set standard masks
        data.train_mask = range(18000)
        data.val_mask = range(18001, 20000)
        data.test_mask = range(20001, 22470)
        return data.to(DEVICE), dataset

    def _load_ogb(self, name: str, device = "cuda"):
        """Load and process OGB dataset."""
        DEVICE = device
        dataset_path = self._get_dataset_path(name)
        dataset = PygNodePropPredDataset(name=name, root=dataset_path)
        data = dataset[0]
        
        # Process split indices
        split_idx = dataset.get_idx_split()
        num_nodes = data.num_nodes
        
        # Create standard masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[split_idx["train"]] = True
        val_mask[split_idx["valid"]] = True
        test_mask[split_idx["test"]] = True
        
        # Add standard attributes
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # Process edges
        if hasattr(data, 'edge_index'):
            data.edge_index = to_undirected(data.edge_index)
            data.edge_index, _ = add_remaining_self_loops(
                data.edge_index, 
                num_nodes=data.x.shape[0]
            )
        
        # Standardize label format
        if hasattr(data, 'y') and len(data.y.shape) > 1 and data.y.shape[1] == 1:
            data.y = data.y.squeeze(1)
            
        return data.to(DEVICE), dataset