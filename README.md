# Federated GNN Learning

A framework for experimenting with Federated Learning on Graph Neural Networks.

## Overview

This project implements a federated learning system for Graph Neural Networks (GNNs) that enables training across multiple decentralized clients without sharing raw data. The framework supports various GNN architectures and data distribution strategies.

## Key Features

- Federated training with configurable number of clients
- Support for GCN and GAT architectures
- Multiple data loading options including k-hop neighborhood approaches
- Non-IID data partitioning with Dirichlet distribution
- Comprehensive evaluation metrics
- Ray-based parallel client processing

## Quick Start

### Testing the Framework

To test that the framework is working properly:

```bash
python src/test.py
```

This runs a lightweight version of the experiments with minimal computation to verify everything works correctly.

### Running Experiments

To run a federated learning experiment:

```bash
# Import the main experiment function
from src.run import main_experiment, load_configuration

# Load configuration
clients_num, beta, cfg = load_configuration("conf/base.yaml")

# Run experiment with specific parameters
results_data, result_text = main_experiment(
    clients_num=10,        # Number of clients
    beta=0.5,              # Dirichlet concentration parameter (lower = more non-IID)
    data_loading_option="zero_hop",  # Data loading strategy
    model_type="GCN",      # Model architecture (GCN or GAT)
    cfg=cfg,               # Configuration parameters
    dataset_name="Cora",   # Dataset to use
    hop=1,                 # Number of hops for k-hop strategies
    fulltraining_flag=False # Whether to use full training
)

# Print results
print(result_text)
```

## Data Loading Options

- `zero_hop`: Basic data splitting without neighborhood information
- `page_rank`, `random_walk`, `diffusion`, `efficient`, `adjacency`, `propagation`, `zero`, `full`: Various k-hop neighborhood strategies

## Project Structure

- `src/`: Source code
  - `run.py`: Main experiment execution code
  - `test.py`: Test framework for verifying code integrity
  - `client.py`: Implementation of federated clients
  - `server.py`: Implementation of federated server
  - `gnn_models.py`: GNN model implementations
  - `core/`: Core functionality
  - `dataprocessing/`: Data processing utilities
  - `run_utils.py`: Utilities for experiment execution
- `conf/`: Configuration files
  - `base.yaml`: Default configuration

## Requirements

- PyTorch
- Ray
- PyTorch Geometric
- CUDA-compatible GPU (optional but recommended)

## Example Results

The framework outputs comprehensive results including:
- Global model performance
- Average client performance
- Performance distribution across clients
- Standard deviation across experiment runs 