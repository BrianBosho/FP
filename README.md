# Federated Graph Neural Networks (Federated-GNN)

This repository contains code for running Federated Learning experiments with Graph Neural Networks (GNNs). It allows you to train GNN models in a federated setting across multiple clients and evaluate their performance.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run experiment with Cora dataset
python -m src.experiments.run_experiments --config conf/cora_config.yaml

# 3. View results in results/cora/ directory
```

That's it! The experiment will train a federated GNN model and save results automatically.

## Overview

Federated Learning enables training machine learning models across multiple decentralized clients without sharing raw data. This implementation focuses on federated training of Graph Neural Networks on graph-structured data.

Features:
- Train GNN models in a federated setting
- Support for various GNN architectures (GCN, GAT)
- Different data loading and partitioning strategies
- Control over client heterogeneity (via Dirichlet distribution parameter beta)
- Comprehensive experiment configuration and results logging
- Configurable Weights & Biases integration
- Debug mode for detailed logging

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

### Secure WandB Configuration

To prevent exposing your WandB API key, use environment variables. We recommend using a `.env` file (which is gitignored).

1. Create a `.env` file in the `federated-gnn/` directory (copy from template):
   ```bash
   cp env.template .env
   ```

2. Edit `.env` and add your API key:
   ```bash
   WANDB_API_KEY=your_actual_api_key_here
   WANDB_PROJECT=FGL4
   ```

The system will automatically load these variables when running experiments.

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

### Debug Mode

Control output verbosity with the debug flag in your config:

**Clean Output (default):**
```yaml
debug: false  # Minimal output, production-ready
```

**Verbose Output (for debugging):**
```yaml
debug: true   # Detailed logs, model info, training details
```

When `debug: false`, you get clean output (~60% reduction) showing only essential information. When `debug: true`, you see comprehensive debugging details including client initialization, model parameters, training metrics, and more.

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

## Running Multiple Experiments

This section explains how to run multiple experiments without crashes or conflicts.

### Problem Overview

When running multiple federated learning experiments in parallel, you may encounter:
- **GPU memory conflicts**: Multiple processes trying to use the same GPU memory
- **Ray port conflicts**: Ray framework uses port 6379 by default, causing conflicts
- **Process interference**: Multiple experiments modifying shared resources

### Solutions

We provide 4 different approaches, choose based on your needs:

#### Solution 1: Sequential Execution (Safest, Simplest)

**Best for**: Running experiments overnight without worrying about conflicts.

```bash
# Make executable
chmod +x scripts/run_sequential_experiments.sh

# Run all experiments one after another
./scripts/run_sequential_experiments.sh
```

**Pros:**
- No conflicts possible
- Simple to understand
- Uses full GPU per experiment
- Easy to debug

**Cons:**
- Slower (no parallelism)
- GPU may be underutilized

**How it works:**
Runs each experiment to completion before starting the next one.

#### Solution 2: Parallel with Custom Ray Ports (Recommended)

**Best for**: Running 2-3 experiments in parallel on a single GPU with 49GB memory.

```bash
# Make executable
chmod +x scripts/run_parallel_with_ports.sh

# Edit MAX_PARALLEL in the script if needed (default: 2)
nano scripts/run_parallel_with_ports.sh  # Change MAX_PARALLEL=2 to desired value

# Run experiments in parallel
./scripts/run_parallel_with_ports.sh
```

**Pros:**
- Faster than sequential (2-3x speedup)
- Safe (unique Ray ports prevent conflicts)
- Automatic queueing
- Good GPU utilization

**Cons:**
- Each experiment gets less GPU memory
- Requires code modification (already done)

**How it works:**
- Assigns unique Ray port to each experiment (6379, 6380, 6381, etc.)
- Runs MAX_PARALLEL experiments simultaneously
- Automatically queues remaining experiments

**GPU Memory Estimation:**
Based on your nvidia-smi output:
- Each experiment uses ~570-635 MiB per client + ~500 MiB base
- With 10 clients: ~6-6.5 GB per experiment
- Your 49GB GPU can handle ~7 experiments simultaneously
- **Recommended MAX_PARALLEL: 2-3** (to leave headroom)

#### Solution 3: Queue-based Execution

**Best for**: Running many experiments with controlled parallelism.

```bash
# Make executable
chmod +x scripts/run_queued_experiments.sh

# Edit MAX_PARALLEL if needed
nano scripts/run_queued_experiments.sh

# Run with queue management
./scripts/run_queued_experiments.sh
```

**Pros:**
- Handles large experiment lists
- Controlled parallelism
- Automatic queueing

**Cons:**
- More complex setup

#### Solution 4: Parallel Execution (Advanced)

**Best for**: Maximum parallelism when you have multiple GPUs or very careful resource management.

```bash
# Make executable
chmod +x scripts/run_parallel_experiments.sh

# Run experiments in parallel
./scripts/run_parallel_experiments.sh
```

**Pros:**
- Maximum parallelism
- Full GPU utilization

**Cons:**
- Higher risk of conflicts
- Requires careful monitoring

## Memory Optimization

### Observed Problem
```
Total GPU Usage: 37GB / 48GB
- Driver (python): 9GB
- 3 active training clients: 3 × 6.8GB = 20.4GB  
- 7 idle clients: 7 × 1-2GB = 7-14GB  ← PROBLEM: Should be 0GB!
```

### Root Causes Identified

1. **Ray GPU Reservation (`@ray.remote(num_gpus=1/10)`)**  
   **Problem**: Ray pre-allocates GPU memory for each actor, even when idle  
   **Solution**: Remove GPU reservation  
   ```python
   # Before
   @ray.remote(num_gpus=1/10)
   class FLClient:
   
   # After  
   @ray.remote  # No GPU reservation
   class FLClient:
   ```

2. **Feature Propagation on GPU**  
   **Problem**: Preprocessing on CUDA uses 9GB of driver memory  
   **Solution**: Move to CPU  
   ```yaml
   # base.yaml
   feature_prop_device: "cpu"  # Changed from "cuda"
   ```

3. **Parameter Transfer Keeping GPU References**  
   **Problem**: When returning model parameters, GPU tensors were kept alive  
   **Solution**: Explicitly copy to CPU and detach  
   ```python
   params_cpu = tuple(p.detach().cpu() for p in self.model.parameters())
   return {'params': params_cpu}
   ```

### Implemented Optimizations

- Removed GPU reservation from FLClient
- CPU feature propagation in base.yaml
- Explicit CPU parameter return in client.py
- Explicit cleanup after aggregation in server.py

### Expected Memory Footprint (After Optimizations)

**Scenario: 4 clients, all training in parallel**
```
Driver: ~500MB-1GB (preprocessing on CPU, minimal GPU footprint)
Active clients: 4 × 6.8GB = 27.2GB (during training)
Idle clients: 0GB (on CPU, no GPU reservation)
Ray overhead: ~1-2GB
```

## Model Configuration System

### What We've Built

A flexible, configuration-driven system for managing GNN model architectures across different datasets.

### Key Features

- **Unified Model Architecture**: Single models (GCN, GAT) that adapt through configuration
- **Configurable Hyperparameters**: hidden_dim, num_layers, dropout, normalization, num_heads
- **Three-Level Configuration**: Global defaults, model-specific defaults, experiment-specific overrides
- **Smart Configuration Loading**: Automatic detection of dataset-specific variants

### Configuration Structure

```yaml
model_architecture:
  # Global defaults
  default:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"
    
  # Model-specific settings
  GCN:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"
    
  GAT:
    hidden_dim: 8
    num_layers: 2
    dropout: 0.6
    num_heads: 8
```

### Available Hyperparameters

#### Common Parameters (All Models)
- **hidden_dim**: Hidden layer dimensions (8-512 depending on dataset size)
- **num_layers**: Number of GNN layers (2-5)
- **dropout**: Dropout rate (0.0-1.0)
- **normalization**: "batch", "layer", "group", or "none"

#### GAT-Specific Parameters
- **num_heads**: Number of attention heads (4, 8, 16)

### How to Override Settings

#### Option 1: Override in Dataset Config File
```yaml
# cora_config.yaml
model_architecture:
  GCN:
    hidden_dim: 32
    num_layers: 3
    dropout: 0.6
    normalization: "batch"
```

#### Option 2: Modify Base Config
Edit `conf/base.yaml` to change defaults for all experiments.

### Recommended Hyperparameters by Dataset

#### Small Datasets (Cora, Citeseer)
- hidden_dim: 16-32
- num_layers: 2
- dropout: 0.5-0.6
- normalization: none

#### Medium Datasets (Pubmed)
- hidden_dim: 16-32
- num_layers: 2
- dropout: 0.5

#### Large Datasets (ogbn-arxiv)
- hidden_dim: 256
- num_layers: 3
- dropout: 0.3-0.5
- normalization: batch

#### Amazon Co-Purchase (Computers, Photos)
- hidden_dim: 64-128
- num_layers: 2-3
- dropout: 0.4-0.5

### Normalization Types

| Type | Best For | Pros | Cons |
|------|----------|------|------|
| none | 2-layer models, small graphs | Simple, fast | May not scale |
| batch | 3+ layers, large batches | Stable training | Needs sufficient batch size |
| layer | Variable batch sizes | Works with any batch size | Slightly slower |
| group | Medium networks | Good balance | Need to set groups |

## Quick Start: Running Multiple Experiments

### TL;DR - Just Run This

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Test single experiment first
./scripts/experiment_utils.sh test

# Run multiple experiments in parallel (RECOMMENDED)
./scripts/run_parallel_with_ports.sh

# Monitor progress
./scripts/experiment_utils.sh status
```

### Commands Cheat Sheet

```bash
# Run experiments (choose one)
./scripts/run_parallel_with_ports.sh     # Parallel (2-3 at once) - BEST
./scripts/run_sequential_experiments.sh   # One at a time - SAFEST

# Monitor
./scripts/experiment_utils.sh status      # Check what's running
./scripts/experiment_utils.sh monitor     # Live GPU view
watch -n 2 nvidia-smi                     # Alternative GPU monitor

# Manage
./scripts/experiment_utils.sh stop        # Stop everything
./scripts/experiment_utils.sh clean       # Clean up after stopping
./scripts/experiment_utils.sh results     # Show results summary

# Debug
./scripts/experiment_utils.sh logs        # List log files
tail -f logs/parallel_*/cora*.log         # Follow specific log
```

### Before You Start

1. **Check GPU is available:**
   ```bash
   nvidia-smi
   ```

2. **Make sure no other experiments are running:**
   ```bash
   ./scripts/experiment_utils.sh status
   ```

3. **If processes are stuck, clean up:**
   ```bash
   ./scripts/experiment_utils.sh stop
   ./scripts/experiment_utils.sh clean
   ```

### Customizing What Runs

Edit the experiment list in your chosen script:

```bash
# Edit parallel runner
nano scripts/run_parallel_with_ports.sh

# Find this section and modify:
EXPERIMENTS=(
    "conf/ogbn-arxiv_config.yaml"
    "conf/photos_config.yaml"
    # Add or remove configs here
)
```

### When Things Go Wrong

```bash
# Kill everything
pkill -f run_experiments
ray stop

# Check what's still running
ps aux | grep python

# Force clean GPU
./scripts/experiment_utils.sh clean

# Check GPU memory is freed
nvidia-smi
```

### Understanding Your GPU

Based on your nvidia-smi output:
- **Total GPU Memory:** 49,152 MiB (49 GB)
- **Per Experiment:** ~6-7 GB
- **Safe Parallel Limit:** 2-3 experiments 