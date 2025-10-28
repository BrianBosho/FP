"""
Communication Cost Analysis for Federated GNN Methods

This script calculates and compares communication costs for:
- FedGCN (1-hop expansion)
- FedGAT Matrix variant
- FedGAT Vector variant

Across various datasets and partition schemes.
"""

import numpy as np
import torch
import pandas as pd
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataprocessing.loaders import load_and_split_with_khop
from omegaconf import OmegaConf


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def node_degrees(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """
    Undirected degree from PyG edge_index (counts both ends).
    edge_index: shape [2, E]
    """
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.index_add_(0, edge_index[0], ones)
    deg.index_add_(0, edge_index[1], ones)
    return deg.numpy()


def sci_notation(n: Union[int, float]) -> str:
    """Return number n in scientific notation with 3 significant figures."""
    return f"{n:.3e}"


# ============================================================================
# FedGCN COST CALCULATION
# ============================================================================

def fedgcn_scalars(total_nodes: int, 
                   new_nodes: int, 
                   feature_dim: int, 
                   deduped: bool = False) -> Dict[str, Union[int, str]]:
    """
    Compute FedGCN communication scalars for 1-hop pretraining.

    Args:
        total_nodes (int): N, original graph node count.
        new_nodes (int): sum of new nodes added across clients
                         (no de-dupe = per-client count).
        feature_dim (int): feature dimension d.
        deduped (bool): if True, assumes new_nodes are globally unique (download only).
                        if False, matches paper's per-client count (upload + download).

    Returns:
        dict with 'upload', 'download', 'total' scalars (ints).
    """
    if deduped:
        # Only download of new raw features
        upload = 0
        download = new_nodes * feature_dim
    else:
        # Paper-faithful 1-hop formula
        upload = new_nodes * feature_dim
        download = total_nodes * feature_dim
    total = upload + download
    return {
        "upload": upload,
        "download": download,
        "total": total,
        "upload_exp": sci_notation(upload),
        "download_exp": sci_notation(download),
        "total_exp": sci_notation(total),
    }


# ============================================================================
# FedGAT COST CALCULATION
# ============================================================================

def fedgat_matrix_scalars(d: int,
                          degrees: np.ndarray,
                          include_upload: bool = False,
                          n_nodes_upload: Optional[int] = None,
                          include_linear_terms: bool = False) -> int:
    """
    Matrix FedGAT (paper's original) scalar count.

    Per node i: 4 * d * deg(i)^2  [+ (2*d+2)*deg(i) if include_linear_terms]
    Sum over nodes; optionally add initial upload Nd.

    Args:
      d: feature dim
      degrees: array of node degrees for this (sub)graph
      include_upload: if True, add Nd for this (sub)graph
      n_nodes_upload: overrides N used in upload term (default = len(degrees))
      include_linear_terms: adds smaller linear-in-degree terms

    Returns:
      scalar count (int)
    """
    # degrees may be halved (non-integer). Keep float!
    deg = np.asarray(degrees, dtype=np.float64)
    total = 4.0 * float(d) * float(np.sum(deg * deg, dtype=np.float64))
    if include_linear_terms:
        sdeg = float(np.sum(deg, dtype=np.float64))
        total += 2.0 * float(d) * sdeg + 2.0 * sdeg
    if include_upload:
        N = int(n_nodes_upload) if n_nodes_upload is not None else int(len(deg))
        total += float(N) * float(d)
    return int(round(total))


def fedgat_vector_scalars(d: int,
                          degrees: np.ndarray,
                          include_upload: bool = False,
                          n_nodes_upload: Optional[int] = None,
                          mats_per_node: int = 3,
                          include_other_terms: bool = False) -> int:
    """
    Vector FedGAT (appendix) scalar count (dominant term).

    Per node i (dominant): mats_per_node * (2*deg(i)) * d
    Sum over nodes; optionally add initial upload Nd.
    include_other_terms adds a tiny O(sum deg) correction.

    Args mirror matrix version; mats_per_node defaults to 3 (M1,M2,K1).

    Returns:
      scalar count (int)
    """
    deg = np.asarray(degrees, dtype=np.float64)
    total = float(mats_per_node) * float(d) * float(np.sum(2.0 * deg, dtype=np.float64))
    if include_other_terms:
        total += 2.0 * float(np.sum(2.0 * deg, dtype=np.float64))
    if include_upload:
        N = int(n_nodes_upload) if n_nodes_upload is not None else int(len(deg))
        total += float(N) * float(d)
    return int(round(total))


def fedgat_matrix_for_clients(d: int,
                              client_degrees: List[np.ndarray],
                              include_upload: bool = False,
                              include_linear_terms: bool = False) -> Dict[str, Union[int, np.ndarray]]:
    """
    Compute per-client + total for Matrix FedGAT.
    Upload Nd is per-client (uses each client's N).
    """
    per = []
    for deg in client_degrees:
        per.append(fedgat_matrix_scalars(d, deg,
                                         include_upload=include_upload,
                                         n_nodes_upload=len(deg),
                                         include_linear_terms=include_linear_terms))
    return {"S_clients": np.array(per, dtype=np.int64),
            "S_total": int(np.sum(per, dtype=np.int64))}


def fedgat_vector_for_clients(d: int,
                              client_degrees: List[np.ndarray],
                              include_upload: bool = False,
                              mats_per_node: int = 3,
                              include_other_terms: bool = False) -> Dict[str, Union[int, np.ndarray]]:
    """
    Compute per-client + total for Vector FedGAT.
    Upload Nd is per-client (uses each client's N).
    """
    per = []
    for deg in client_degrees:
        per.append(fedgat_vector_scalars(d, deg,
                                         include_upload=include_upload,
                                         n_nodes_upload=len(deg),
                                         mats_per_node=mats_per_node,
                                         include_other_terms=include_other_terms))
    return {"S_clients": np.array(per, dtype=np.int64),
            "S_total": int(np.sum(per, dtype=np.int64))}


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

def analyze_partition(dataset_name: str,
                      num_clients: int,
                      beta: float,
                      hop: int,
                      device: torch.device,
                      config: OmegaConf) -> Dict[str, any]:
    """
    Analyze a single partition scheme for a dataset.
    
    Returns dict with:
        - dataset_name, num_clients, beta, hop
        - total_nodes, total_edges, feature_dim
        - total_initial_nodes, total_expanded_nodes, total_new_nodes
        - fedgcn_cost, fedgat_matrix_cost, fedgat_vector_cost
        - client_degrees (list of degree arrays)
    """
    print(f"\nAnalyzing {dataset_name} | clients={num_clients}, beta={beta}, hop={hop}")
    
    # Load data with k-hop expansion
    data, dataset, clients_data, test_data = load_and_split_with_khop(
        dataset_name, device, num_clients, beta, config=config, hop=hop
    )
    
    # Basic graph stats
    total_nodes = data.num_nodes
    total_edges = data.edge_index.shape[1]
    feature_dim = data.x.shape[1]
    
    # Client statistics
    total_initial_nodes = 0
    total_expanded_nodes = 0
    client_degrees = []
    
    for client in clients_data:
        initial_nodes = len(client.mapping)
        expanded_nodes = client.num_nodes
        
        total_initial_nodes += initial_nodes
        total_expanded_nodes += expanded_nodes
        
        # Calculate degrees (divide by 2 since edge_index counts both directions)
        deg = node_degrees(client.edge_index, client.num_nodes)
        deg = deg / 2
        client_degrees.append(deg)
    
    total_new_nodes = total_expanded_nodes - total_initial_nodes
    
    # Calculate communication costs
    
    # FedGCN cost
    fedgcn = fedgcn_scalars(total_nodes, total_new_nodes, feature_dim, deduped=False)
    
    # FedGAT Matrix cost
    fedgat_matrix = fedgat_matrix_for_clients(feature_dim, client_degrees, include_upload=True)
    
    # FedGAT Vector cost
    fedgat_vector = fedgat_vector_for_clients(feature_dim, client_degrees, include_upload=True)
    
    return {
        "dataset_name": dataset_name,
        "num_clients": num_clients,
        "beta": beta,
        "hop": hop,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "feature_dim": feature_dim,
        "total_initial_nodes": total_initial_nodes,
        "total_expanded_nodes": total_expanded_nodes,
        "total_new_nodes": total_new_nodes,
        "expansion_ratio": total_expanded_nodes / total_initial_nodes,
        "fedgcn_total": fedgcn["total"],
        "fedgcn_upload": fedgcn["upload"],
        "fedgcn_download": fedgcn["download"],
        "fedgat_matrix_total": fedgat_matrix["S_total"],
        "fedgat_vector_total": fedgat_vector["S_total"],
        "client_degrees": client_degrees,  # For detailed analysis if needed
    }


def run_communication_cost_analysis(
    datasets: List[str] = ["Cora", "Citeseer", "Pubmed"],
    client_counts: List[int] = [5, 10, 20],
    betas: List[float] = [0.5, 1.0, 10.0],
    hops: List[int] = [1],
    device: str = "cpu",
    config_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Run comprehensive communication cost analysis across multiple configurations.
    
    Args:
        datasets: List of dataset names
        client_counts: List of client counts to test
        betas: List of beta values (Dirichlet concentration parameter)
        hops: List of k-hop expansion values
        device: Device to use ("cpu" or "cuda")
        config_path: Path to config file (defaults to conf/base.yaml)
    
    Returns:
        DataFrame with all results
    """
    # Setup
    device = torch.device(device)
    
    if config_path is None:
        config_path = project_root / "conf" / "base.yaml"
    config = OmegaConf.load(str(config_path))
    
    # Collect results
    results = []
    
    total_runs = len(datasets) * len(client_counts) * len(betas) * len(hops)
    run_count = 0
    
    for dataset_name in datasets:
        for num_clients in client_counts:
            for beta in betas:
                for hop in hops:
                    run_count += 1
                    print(f"\n{'='*80}")
                    print(f"Run {run_count}/{total_runs}")
                    print(f"{'='*80}")
                    
                    try:
                        result = analyze_partition(
                            dataset_name, num_clients, beta, hop, device, config
                        )
                        results.append(result)
                        
                        # Print summary
                        print(f"  Total nodes: {result['total_nodes']}")
                        print(f"  Feature dim: {result['feature_dim']}")
                        print(f"  Expansion ratio: {result['expansion_ratio']:.2f}x")
                        print(f"  FedGCN cost: {sci_notation(result['fedgcn_total'])}")
                        print(f"  FedGAT Matrix: {sci_notation(result['fedgat_matrix_total'])}")
                        print(f"  FedGAT Vector: {sci_notation(result['fedgat_vector_total'])}")
                        
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add scientific notation columns
    if not df.empty:
        df["fedgcn_total_exp"] = df["fedgcn_total"].apply(sci_notation)
        df["fedgat_matrix_total_exp"] = df["fedgat_matrix_total"].apply(sci_notation)
        df["fedgat_vector_total_exp"] = df["fedgat_vector_total"].apply(sci_notation)
        
        # Add ratios
        df["fedgat_matrix_vs_fedgcn"] = df["fedgat_matrix_total"] / df["fedgcn_total"]
        df["fedgat_vector_vs_fedgcn"] = df["fedgat_vector_total"] / df["fedgcn_total"]
        df["fedgat_vector_vs_matrix"] = df["fedgat_vector_total"] / df["fedgat_matrix_total"]
    
    return df


# ============================================================================
# REPORTING
# ============================================================================

def print_summary_report(df: pd.DataFrame):
    """Print a formatted summary report."""
    print("\n" + "="*80)
    print("COMMUNICATION COST ANALYSIS SUMMARY")
    print("="*80)
    
    # Group by dataset
    for dataset in df["dataset_name"].unique():
        df_dataset = df[df["dataset_name"] == dataset]
        
        print(f"\n{dataset}")
        print("-"*80)
        
        # Show key columns
        display_cols = [
            "num_clients", "beta", "hop",
            "total_nodes", "feature_dim", "expansion_ratio",
            "fedgcn_total_exp", "fedgat_matrix_total_exp", "fedgat_vector_total_exp",
            "fedgat_matrix_vs_fedgcn", "fedgat_vector_vs_fedgcn"
        ]
        
        print(df_dataset[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    print("\nFedGAT Matrix vs FedGCN ratio:")
    print(f"  Mean: {df['fedgat_matrix_vs_fedgcn'].mean():.2f}x")
    print(f"  Min:  {df['fedgat_matrix_vs_fedgcn'].min():.2f}x")
    print(f"  Max:  {df['fedgat_matrix_vs_fedgcn'].max():.2f}x")
    
    print("\nFedGAT Vector vs FedGCN ratio:")
    print(f"  Mean: {df['fedgat_vector_vs_fedgcn'].mean():.2f}x")
    print(f"  Min:  {df['fedgat_vector_vs_fedgcn'].min():.2f}x")
    print(f"  Max:  {df['fedgat_vector_vs_fedgcn'].max():.2f}x")
    
    print("\nFedGAT Vector vs Matrix ratio:")
    print(f"  Mean: {df['fedgat_vector_vs_matrix'].mean():.2f}x")
    print(f"  Min:  {df['fedgat_vector_vs_matrix'].min():.2f}x")
    print(f"  Max:  {df['fedgat_vector_vs_matrix'].max():.2f}x")


def save_results(df: pd.DataFrame, output_path: Optional[str] = None):
    """Save results to CSV."""
    if output_path is None:
        output_path = project_root / "results" / "communication_cost_analysis.csv"
    
    # Drop client_degrees column (not CSV-serializable)
    df_save = df.drop(columns=["client_degrees"], errors="ignore")
    
    df_save.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete analysis with default parameters."""
    
    # Configuration
    datasets = ["Cora", "Citeseer", "Pubmed"]
    client_counts = [5, 10, 20]
    betas = [0.5, 1.0, 10.0]
    hops = [1]
    device = "cpu"
    
    print("="*80)
    print("FEDERATED GNN COMMUNICATION COST ANALYSIS")
    print("="*80)
    print(f"\nDatasets: {datasets}")
    print(f"Client counts: {client_counts}")
    print(f"Beta values: {betas}")
    print(f"K-hop values: {hops}")
    print(f"Device: {device}")
    
    # Run analysis
    df = run_communication_cost_analysis(
        datasets=datasets,
        client_counts=client_counts,
        betas=betas,
        hops=hops,
        device=device
    )
    
    # Print report
    print_summary_report(df)
    
    # Save results
    save_results(df)
    
    return df


if __name__ == "__main__":
    df_results = main()
