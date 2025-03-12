# Running Federated Learning Experiments

This guide explains how to run federated learning experiments using the enhanced framework via the `run_experiments_enhanced.py` script.

## Overview

The experiment runner script automatically:

1. Creates multiple experiment configurations based on your settings
2. Runs them sequentially or in parallel
3. Collects and combines results for analysis
4. Generates detailed logs and summary files

## Setting Up an Experiment Configuration

Experiments are defined using a YAML configuration file. Here's an example:

```yaml
# Experiment Configuration for Enhanced Federated Learning

# Datasets to run experiments on
datasets:
  - "Cora"
  - "Citeseer"
  - "Pubmed"

# Data loading strategies
data_loading_options:
  - "zero_hop"
  - "one_hop"

# Number of clients to use
num_clients_options:
  - 3
  - 5

# Beta parameter for Dirichlet distribution (controls non-IID degree)
beta_options:
  - 0.5  # More non-IID
  - 1.0  # Less non-IID

# Number of communication rounds
rounds_options:
  - 10
  - 20

# Number of local epochs for each client per round
epochs_options:
  default: 1
  ogbn-arxiv: 2
  ogbn-products: 1

# Batch sizes for different datasets
batch_size_options:
  default: 1024
  ogbn-products: 512

# Use GPU if available
use_cuda: false

# Use memory efficient operations
memory_efficient: true

# Number of times to repeat each experiment
repeats: 1
```

Place this file in the `FP/experiments/` directory.

## Running Experiments

### Prerequisites

Make sure your Python environment has all the required dependencies:
- PyTorch
- PyTorch Geometric
- Ray
- pandas
- PyYAML

### Basic Usage

```bash
cd /home/brian_bosho/FP
python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml
```

### Run in Parallel Mode

For faster execution, run experiments in parallel:

```bash
python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml --parallel
```

### Specify a Custom Results Directory

```bash
python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml --results_dir ./my_results
```

### Limit Parallel Workers

When running in parallel mode, you can limit the number of concurrent experiments:

```bash
python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml --parallel --max_workers 4
```

## Understanding the Configuration File

### Basic Parameters

- `datasets`: List of datasets to run experiments on
- `data_loading_options`: Methods for loading and partitioning data
- `num_clients_options`: Number of federated clients to test
- `beta_options`: Dirichlet distribution parameter values (smaller = more non-IID)
- `rounds_options`: Number of communication rounds to perform

### Advanced Parameters

- `epochs_options`: Number of local training epochs per round (can be dataset-specific)
- `batch_size_options`: Mini-batch sizes for different datasets
- `use_cuda`: Whether to use GPU acceleration
- `memory_efficient`: Whether to use memory optimizations
- `repeats`: Number of times to repeat each experiment (for statistical significance)

## Experiment Output

When you run the experiments, the following directories and files are created:

```
results/experiments_YYYYMMDD_HHMMSS/
├── configs/                # Individual configuration files for each experiment
├── logs/                   # Detailed logs for each experiment
├── experiment_commands.json # List of all commands executed
├── execution_summary.json  # Summary of execution results
├── combined_results.csv    # Combined results in CSV format
└── combined_results.xlsx   # Combined results in Excel format (if available)
```

## Analyzing Results

The most important output file is `combined_results.csv`, which contains:

- Configuration parameters for each experiment
- Training and testing metrics for each round
- Final performance metrics

You can load this file into any data analysis tool (Excel, Python with pandas, R, etc.) for further analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the results
results = pd.read_csv("results/experiments_YYYYMMDD_HHMMSS/combined_results.csv")

# Group by dataset and compute average final accuracy
summary = results.groupby(['dataset', 'data_loading', 'num_clients', 'beta']).agg({
    'global_test_acc': ['mean', 'std']
})

# Plot results
plt.figure(figsize=(12, 8))
# ... your visualization code ...
plt.show()
```

## Example Configurations

### Small-Scale Experiment (Quick Testing)

```yaml
datasets: ["Cora"]
data_loading_options: ["zero_hop"]
num_clients_options: [3]
beta_options: [0.5]
rounds_options: [5]
epochs_options:
  default: 1
use_cuda: false
memory_efficient: true
repeats: 1
```

### Comparing Data Loading Methods

```yaml
datasets: ["Cora", "Citeseer", "Pubmed"]
data_loading_options: ["zero_hop", "one_hop", "two_hop"]
num_clients_options: [5]
beta_options: [0.5]
rounds_options: [20]
epochs_options:
  default: 1
use_cuda: false
memory_efficient: true
repeats: 3
```

### Evaluating Non-IID Impact

```yaml
datasets: ["Cora"]
data_loading_options: ["zero_hop"]
num_clients_options: [5]
beta_options: [0.1, 0.3, 0.5, 0.7, 1.0]  # From more to less non-IID
rounds_options: [20]
epochs_options:
  default: 1
use_cuda: false
memory_efficient: true
repeats: 3
```

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors when running large datasets:

1. Reduce the number of clients (`num_clients_options`)
2. Reduce batch sizes (`batch_size_options`)
3. Run experiments sequentially instead of in parallel
4. For OGBN-Products, make sure `memory_efficient` is set to `true`

### Execution Errors

If an experiment fails:

1. Check the experiment logs in the `logs/` directory
2. Look for error messages in the `execution_summary.json` file
3. Try running the individual experiment command directly

## Tips for Large-Scale Experiments

1. Start with small tests before running large experiment batches
2. Use the `--parallel` mode with a reasonable `--max_workers` value
3. For large datasets, run them separately from small datasets
4. Consider running overnight for exhaustive parameter sweeps 