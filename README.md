# Federated Graph Neural Networks (Federated-GNN)

This repository contains code for running Federated Learning experiments with Graph Neural Networks (GNNs). It allows you to train GNN models in a federated setting across multiple clients and evaluate their performance.

## Overview

Federated Learning enables training machine learning models across multiple decentralized clients without sharing raw data. This implementation focuses on federated training of Graph Neural Networks on graph-structured data.

Features:
- Train GNN models in a federated setting
- Support for various GNN architectures (GCN, GAT)
- Different data loading and partitioning strategies
- Control over client heterogeneity (via Dirichlet distribution parameter beta)
- Comprehensive experiment configuration and results logging

## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- PyTorch Geometric
- Ray (for distributed computing)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/federated-gnn.git
cd federated-gnn

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

The repository provides organized configuration files for different datasets:

### Dataset-Specific Configurations

Run experiments with pre-configured settings for each dataset:

```bash
# Cora dataset experiments
python -m src.experiments.run_experiments --config conf/cora_config.yaml

# Citeseer dataset experiments  
python -m src.experiments.run_experiments --config conf/citeseer_config.yaml

# Pubmed dataset experiments
python -m src.experiments.run_experiments --config conf/pubmed_config.yaml

# ogbn-arxiv dataset experiments
python -m src.experiments.run_experiments --config conf/ogbn-arxiv_config.yaml
```

### Configuration Structure

The `conf/` folder contains clean, organized configuration files:

```
conf/
├── base.yaml              # Base configuration with common settings
├── cora_config.yaml       # Cora dataset configuration
├── citeseer_config.yaml   # Citeseer dataset configuration  
├── pubmed_config.yaml     # Pubmed dataset configuration
└── ogbn-arxiv_config.yaml # ogbn-arxiv dataset configuration
```

Each dataset config includes optimized settings for that specific dataset:

```yaml
# Example: cora_config.yaml
num_clients: [10]
num_rounds: 10
epochs: 3
beta: [1, 10, 10000]
lr: 0.5
optimizer: SGD
datasets: [Cora]
data_loading: [full, adjacency, zero_hop]
models: [GCN, GAT]
use_wandb: false  # Wandb logging control
wandb_project: "FGL3-Cora"
```

## Configuration Parameters

Key parameters for experiment configuration:

| Parameter | Description |
|-----------|-------------|
| `num_clients` | Number of clients in federated learning |
| `num_rounds` | Number of communication rounds |
| `epochs` | Number of local training epochs per round |
| `beta` | Dirichlet distribution parameter to control data heterogeneity |
| `lr` | Learning rate |
| `datasets` | Dataset(s) to use (Cora, Citeseer, Pubmed, etc.) |
| `data_loading` | Data loading strategy (full, adjacency, zero_hop, etc.) |
| `models` | GNN models to use (GCN, GAT) |
| `results_dir` | Directory to save experiment results |
| `save_results` | Whether to save detailed results |
| `hop` | Number of hops for graph propagation |
| `fulltraining_flag` | Whether to use full training flag |

### Wandb Integration

The experiments support configurable wandb logging through YAML configuration files:

```yaml
# Wandb configuration
use_wandb: false          # Set to true to enable wandb logging
wandb_project: "FGL3"     # Project name for wandb
wandb_entity: null        # Set to your wandb entity/team name (optional)
wandb_mode: "online"      # "online", "offline", or "disabled"
```

**Usage Examples:**

- **Disable wandb completely** (faster local testing):
  ```yaml
  use_wandb: false
  ```

- **Enable wandb with custom project**:
  ```yaml
  use_wandb: true
  wandb_project: "my-federated-experiments"
  wandb_mode: "online"
  ```

- **Offline mode** (for air-gapped environments):
  ```yaml
  use_wandb: true
  wandb_mode: "offline"
  ```

## Finding and Analyzing Results

Experiment results are saved in the specified `results_dir` with the following structure:

```
results_dir/
├── dataset_data-loading_model_beta{value}_clients{count}/
│   ├── results_*.json          # Detailed experiment results in JSON format
│   ├── results_*.txt           # Human-readable experiment output
│   └── training_*.csv          # Training logs with client-wise metrics
└── summary_results_*.json      # Summary of all experiments in JSON format
└── summary_results_*.txt       # Human-readable summary of all experiments
```

For ablation studies, a summary file is also created in the parent directory that contains aggregated results from all experiment combinations.

The key metrics reported are:
- Average Global Result: Test accuracy of the global model
- Average Client Result: Average test accuracy across all clients

## Analyzing Training Logs

The repository includes utilities for analyzing the training logs:

```python
from training_logs_analysis import parse_client_csv

# Parse the training logs
results = parse_client_csv('path/to/training_logs.csv')

# Access various dataframes with training metrics
loss_df = results['loss_df']  # Loss values indexed by (round, epoch)
acc_df = results['acc_df']    # Accuracy values indexed by (round, epoch)
```

## Citation

If you use this code in your research, please cite:

```
@article{fedgnn2025,
  title={Federated Learning with Graph Neural Networks},
  author={Brian Bosho},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 