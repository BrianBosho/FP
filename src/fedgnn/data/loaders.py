"""Data loading utilities for federated graph learning."""

from src.fedgnn.data.datasets import GraphDataset
from src.fedgnn.data.partitioning import partition_data
from src.fedgnn.data.shard_cache import (
    build_cache_payload,
    get_cache_dir,
    load_shard_cache,
    shard_cache_enabled,
    write_shard_cache,
)
from typing import Tuple, List, Optional
from src.fedgnn.data.positional_encoding import generate_rfp_encoding
import time
import torch
import torch.nn.functional as F

# Note: Datasets are now loaded from the root-level 'datasets' folder
# instead of being stored in the src directory. This change is implemented
# in the GraphDataset class which manages dataset paths.

try:
    from omegaconf import ListConfig
    _SEQUENCE_TYPES = (list, tuple, ListConfig)
except Exception:
    _SEQUENCE_TYPES = (list, tuple)


def _as_bool(value) -> bool:
    """Coerce scalar or one-item config sequences into a boolean."""
    if isinstance(value, _SEQUENCE_TYPES):
        value = value[0] if value else False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "on"}
    return bool(value)

def load_dataset(name: str, device, config: dict = None):
    """
    Regime 1: Load any supported dataset without partitioning.
    """
    dataset_loader = GraphDataset(device)
    return dataset_loader.load_dataset(name, device, config=config)

def load_and_split(
    name: str,
    device,
    num_clients: int = 10,
    beta: float = 0.5,
    config: dict = None,
    timing_sink: dict | None = None,
):
    """
    Regime 2: Load dataset and split into n subgraphs.

    Args:
        name: Dataset name
        device: Device to run computations on
        num_clients: Number of clients
        beta: Dirichlet concentration parameter
        config: Configuration dictionary from YAML file (optional)
    """
    t0 = time.perf_counter()
    data, dataset = load_dataset(name, device, config=config)
    if timing_sink is not None:
        timing_sink["dataset_load_s"] = timing_sink.get("dataset_load_s", 0.0) + time.perf_counter() - t0
    cache_payload = build_cache_payload(name, "zero_hop", num_clients, beta, 0, False, config)
    cache_dir = get_cache_dir(name, "zero_hop", num_clients, beta, 0, False, config)
    if shard_cache_enabled(config):
        t1 = time.perf_counter()
        refs = load_shard_cache(cache_dir, cache_payload)
        dt = time.perf_counter() - t1
        if refs is not None:
            if timing_sink is not None:
                timing_sink["shard_cache_hit_s"] = timing_sink.get("shard_cache_hit_s", 0.0) + dt
            print(f"✓ Loaded {len(refs)} client shards from cache: {cache_dir}")
            return data, dataset, refs, refs

    t2 = time.perf_counter()
    clients_data, test_data, split_data_indexes = partition_data(
        data,
        num_clients,
        beta,
        device,
        hop=0,
        config=config,
        timing_sink=timing_sink,
    )
    if timing_sink is not None:
        timing_sink["partition_s"] = timing_sink.get("partition_s", 0.0) + time.perf_counter() - t2
    if shard_cache_enabled(config):
        t3 = time.perf_counter()
        refs = write_shard_cache(cache_dir, clients_data, cache_payload)
        if timing_sink is not None:
            timing_sink["shard_cache_write_s"] = timing_sink.get("shard_cache_write_s", 0.0) + time.perf_counter() - t3
        print(f"✓ Wrote {len(refs)} client shards to cache: {cache_dir}")
        return data, dataset, refs, refs
    return data, dataset, clients_data, test_data

def load_and_split_with_khop(name: str, device, num_clients: int = 10, beta: float = 0.5, hop: int = 2, fulltraining_flag: bool = False, imputation_method: str = "zero", propagation_mode: str = "propagation", config: dict = None, timing_sink: dict | None = None):
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
    elif imputation_method == "chebyshev_diffusion" or imputation_method == "chebyshev-diffusion":
        use_feature_prop = True
        full_data = False
        propagation_mode = "chebyshev_diffusion"
    elif imputation_method == "chebyshev_diffusion_operator" or imputation_method == "chebyshev-diffusion-operator":
        use_feature_prop = True
        full_data = False
        propagation_mode = "chebyshev_diffusion_operator"
    elif imputation_method == "appnp":
        use_feature_prop = True
        full_data = False
        propagation_mode = "appnp"
    elif imputation_method == "asymmetric_random_walk":
        use_feature_prop = True
        full_data = False
        propagation_mode = "asymmetric_random_walk"
    elif imputation_method == "heat_kernel_exact":
        use_feature_prop = True
        full_data = False
        propagation_mode = "heat_kernel_exact"
    else:
        # Default case to handle any unrecognized imputation method
        use_feature_prop = False
        full_data = False
        print(f"Warning: Unrecognized imputation method '{imputation_method}'. Using default (zero imputation).")

    t_ds = time.perf_counter()
    data, dataset = load_dataset(name, device, config=config)
    if timing_sink is not None:
        timing_sink["dataset_load_s"] = timing_sink.get("dataset_load_s", 0.0) + time.perf_counter() - t_ds

    cache_payload = build_cache_payload(
        name,
        imputation_method,
        num_clients,
        beta,
        hop,
        fulltraining_flag,
        config,
    )
    cache_dir = get_cache_dir(
        name,
        imputation_method,
        num_clients,
        beta,
        hop,
        fulltraining_flag,
        config,
    )
    if shard_cache_enabled(config):
        t_sh = time.perf_counter()
        refs = load_shard_cache(cache_dir, cache_payload)
        dt_sh = time.perf_counter() - t_sh
        if refs is not None:
            if timing_sink is not None:
                timing_sink["shard_cache_hit_s"] = timing_sink.get("shard_cache_hit_s", 0.0) + dt_sh
            print(f"✓ Loaded {len(refs)} client shards from cache: {cache_dir}")
            return data, dataset, refs, refs

    # Get positional encoding flag from config if available
    use_pe = True
    if config is not None:
        use_pe = config.get("use_pe", use_pe)

    use_pe = use_feature_prop and _as_bool(use_pe)

    # Pass config to partition_data
    t_pt = time.perf_counter()
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
        config=config,
        timing_sink=timing_sink,
    )
    if timing_sink is not None:
        timing_sink["partition_s"] = timing_sink.get("partition_s", 0.0) + time.perf_counter() - t_pt

    if shard_cache_enabled(config):
        t_wr = time.perf_counter()
        refs = write_shard_cache(cache_dir, clients_data, cache_payload)
        if timing_sink is not None:
            timing_sink["shard_cache_write_s"] = timing_sink.get("shard_cache_write_s", 0.0) + time.perf_counter() - t_wr
        print(f"✓ Wrote {len(refs)} client shards to cache: {cache_dir}")
        return data, dataset, refs, refs

    # If using PE, apply it to the global data for consistency in evaluation
    if use_pe:
        # Get PE parameters from config if available
        pe_r = 64
        pe_P = 16
        normalize = "qr"
        rfp_qr_max_nodes = 50000

        if config is not None:
            pe_r = config.get("pe_r", pe_r)
            pe_P = config.get("pe_P", pe_P)
            normalize = config.get("normalize", normalize)
            rfp_qr_max_nodes = config.get("rfp_qr_max_nodes", rfp_qr_max_nodes)

        pe_seed = None
        if config is not None and config.get("experiment_seed") is not None:
            pe_seed = int(config.get("experiment_seed"))

        t_pe = time.perf_counter()
        data.original_feature_dim = data.x.size(1)
        rfp = generate_rfp_encoding(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            r=pe_r,
            P=pe_P,
            normalize=normalize,
            device=device,
            seed=pe_seed,
            qr_max_nodes=rfp_qr_max_nodes,
        )
        orig_features = F.normalize(data.x.to(device), p=2, dim=1)
        rfp_norm = F.normalize(rfp, p=2, dim=1) * 0.5  # use same rfp_alpha as clients
        data.x = torch.cat([orig_features, rfp_norm], dim=1)
        if timing_sink is not None:
            timing_sink["positional_encoding_global_s"] = (
                timing_sink.get("positional_encoding_global_s", 0.0) + time.perf_counter() - t_pe
            )

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
    data, dataset = load_dataset(name, device, config=config)
    clients_data, test_data, _ = partition_data(
        data,
        num_clients,
        beta,
        device,
        hop=hop,
        use_feature_prop=use_feature_prop,
        full_data=full_data,
        fulltraining_flag=fulltraining_flag,
        config=config
    )
    return data, dataset, clients_data, test_data
