"""Dataset loading, preprocessing, partitioning, and propagation utilities."""

from torch_geometric.datasets import Planetoid, FacebookPagePage, Amazon, WebKB
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


try:
    # Preferred when `src/` is importable as a package
    from src.fedgnn.utils.project_paths import find_repo_root as _find_repo_root
except ImportError:
    def _find_repo_root(start: Path) -> Path:
        """
        Backward compatibility: if running from within `src/` and `src.*` imports fail,
        find repo root by searching for `conf/base.yaml` up the directory tree.
        """
        start = start.resolve()
        for p in [start, *start.parents]:
            if (p / "conf" / "base.yaml").exists():
                return p
        # Fallback: src/fedgnn/data/datasets.py -> src/fedgnn/data -> src/fedgnn -> src -> repo root
        return start.parents[3]


def load_config():
    # Repo root contains `conf/base.yaml`
    project_root = _find_repo_root(Path(__file__).resolve())
    config_path = project_root / "conf" / "base.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Make all paths absolute by joining with project root
    for category in ['planetoid', 'ogbn', 'amazon', 'webkb']:
        for dataset, path in config['paths'][category].items():
            config['paths'][category][dataset] = str(project_root / path)

    config['paths']['datasets_dir'] = str(project_root / config['paths']['datasets_dir'])
    return config

class GraphDataset:
    """Base class for handling different graph datasets with consistent structure."""

    SUPPORTED_DATASETS = {
        "planetoid": ["Cora", "Citeseer", "Pubmed"],
        "facebook": ["FacebookPagePage"],
        "ogb": ["ogbn-arxiv", "ogbn-products"],
        "amazon": ["Computers", "Photo"],
        "webkb": ["Texas", "Wisconsin"],
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
            self.config['paths']['ogbn']['arxiv'],
            self.config['paths']['amazon']['computers'],
            self.config['paths']['amazon']['photo'],
            self.config['paths']['webkb']['texas'],
            self.config['paths']['webkb']['wisconsin'],
        ]:
            resolved = Path(path).resolve()
            if not resolved.exists():
                resolved.mkdir(parents=True, exist_ok=True)

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
            path = self.config['paths']['planetoid'][name_lower]
        elif name in self.SUPPORTED_DATASETS["facebook"]:
            path = os.path.join(self.config['paths']['datasets_dir'], 'facebook')
        elif name in self.SUPPORTED_DATASETS["ogb"]:
            path = self.config['paths']['ogbn']['arxiv']
        elif name in self.SUPPORTED_DATASETS["amazon"]:
            if name == "Computers":
                path = self.config['paths']['amazon']['computers']
            else:
                path = self.config['paths']['amazon']['photo']
        elif name in self.SUPPORTED_DATASETS["webkb"]:
            path = self.config['paths']['webkb'][name_lower]
        else:
            raise ValueError(f"Dataset {name} not supported")
        return str(Path(path).resolve())

    def load_dataset(self, name: str, device, config: Optional[dict] = None) -> Tuple[torch_geometric.data.Data, object]:
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
        elif name in self.SUPPORTED_DATASETS["amazon"]:
            return self._load_amazon(name, device)
        elif name in self.SUPPORTED_DATASETS["webkb"]:
            return self._load_webkb(name, device, config=config)
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

    def _load_amazon(self, name: str, device = "cuda"):
        """Load and process Amazon Computers/Photo datasets.
        Ensures masks exist, edges are undirected with self-loops, and labels are 1-D.
        """
        DEVICE = device
        dataset_path = self._get_dataset_path(name)

        # Prefer adding masks via AddTrainValTestMask if available; fallback to RandomNodeSplit
        try:
            add_mask_transform = transforms.AddTrainValTestMask(split='train_rest', num_val=500, num_test=1000)
        except AttributeError:
            add_mask_transform = transforms.RandomNodeSplit(split='train_rest', num_val=500, num_test=1000)

        dataset = Amazon(root=dataset_path, name=name)
        data = dataset[0]

        # Apply mask transform on the in-memory data object if masks missing
        if not (hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')):
            data = add_mask_transform(data)

        # Ensure boolean masks
        if hasattr(data, 'train_mask') and data.train_mask.dtype != torch.bool:
            data.train_mask = data.train_mask.bool()
        if hasattr(data, 'val_mask') and data.val_mask.dtype != torch.bool:
            data.val_mask = data.val_mask.bool()
        if hasattr(data, 'test_mask') and data.test_mask.dtype != torch.bool:
            data.test_mask = data.test_mask.bool()

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

    def _select_or_create_node_split(self, data, split_seed: int):
        """Ensure WebKB-style data has 1-D train/val/test masks.

        PyG WebKB usually ships 10 Geom-GCN splits as [num_nodes, 10] masks. The
        training code expects 1-D masks, so choose one split deterministically
        from the experiment seed. If masks are absent, create a deterministic
        60/20/20 node split.
        """
        if (
            hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')
            and data.train_mask is not None
        ):
            if getattr(data.train_mask, "dim", lambda: 1)() == 2:
                num_splits = int(data.train_mask.size(1))
                split_idx = int(split_seed) % max(1, num_splits)
                data.train_mask = data.train_mask[:, split_idx].bool()
                data.val_mask = data.val_mask[:, split_idx].bool()
                data.test_mask = data.test_mask[:, split_idx].bool()
            else:
                data.train_mask = data.train_mask.bool()
                data.val_mask = data.val_mask.bool()
                data.test_mask = data.test_mask.bool()
            return data

        rng = np.random.default_rng(int(split_seed))
        perm = rng.permutation(int(data.num_nodes))
        n_train = int(0.6 * len(perm))
        n_val = int(0.2 * len(perm))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        return data

    def _load_webkb(self, name: str, device = "cuda", config: Optional[dict] = None):
        """Load Texas/Wisconsin heterophilic WebKB datasets."""
        DEVICE = device
        dataset_path = self._get_dataset_path(name)
        dataset = WebKB(root=dataset_path, name=name)
        data = dataset[0]

        split_seed = 0
        if config is not None and hasattr(config, "get"):
            configured_seed = config.get("experiment_seed")
            if configured_seed is not None:
                split_seed = int(configured_seed)
            elif config.get("webkb_split_index") is not None:
                split_seed = int(config.get("webkb_split_index"))

        data = self._select_or_create_node_split(data, split_seed)

        if hasattr(data, 'edge_index'):
            data.edge_index = to_undirected(data.edge_index)
            data.edge_index, _ = add_remaining_self_loops(
                data.edge_index,
                num_nodes=data.x.shape[0]
            )

        if hasattr(data, 'y') and len(data.y.shape) > 1 and data.y.shape[1] == 1:
            data.y = data.y.squeeze(1)

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
