# Run Experiments Documentation

## Overview

`run_experiments.py` is a script for running federated learning experiments with graph neural networks. It provides functionality to run either a single experiment with specific parameters or to run multiple experiments with different combinations of parameters.

## Dependencies

The script relies on the following packages:
- os, sys, argparse
- torch
- ray
- numpy
- datetime
- logging
- json
- pandas

It also imports functions from `run.py`:
- `load_configuration`
- `main_experiment`
- `load_and_split_with_khop`
- `load_and_split_with_feature_prop`

## Command Line Arguments

```
usage: run_experiments.py [-h] [--mode {single,all}] [--dataset DATASET]
                         [--model {GCN,GAT}]
                         [--data_loading {zero_hop,zero,full,random_walk,diffusion,efficient,adjacency,propagation}]
                         [--num_clients NUM_CLIENTS] [--hop HOP]
                         [--config_path CONFIG_PATH] [--output_dir OUTPUT_DIR]
                         [--repetitions REPETITIONS]
```

### Options

- `--mode`: Run mode, either "single" (one experiment) or "all" (all combinations). Default: "all"
- `--dataset`: Dataset name. Default: "Cora". Available options: "Cora", "Citeseer"
- `--model`: Model type. Default: "GCN". Available options: "GCN", "GAT"
- `--data_loading`: Data loading method. Available options:
  - "zero_hop"
  - "zero"
  - "full"
  - "random_walk"
  - "diffusion"
  - "efficient"
  - "adjacency"
  - "propagation"
- `--num_clients`: Number of clients. Default: Value from configuration file
- `--hop`: Number of hops for k-hop methods. Default: 1
- `--config_path`: Path to configuration file. Default: "conf/base.yaml"
- `--output_dir`: Output directory for results. Default: "results"
- `--repetitions`: Number of repetitions for each experiment. Default: 5

## Running the Script

### Running a Single Experiment

```bash
python run_experiments.py --mode single --dataset Cora --model GCN --data_loading zero_hop --num_clients 5 --hop 1
```

### Running All Experiment Combinations

```bash
python run_experiments.py --mode all --output_dir results --config_path conf/my_config.yaml
```

## Configuration File

The script uses a YAML configuration file (default: "conf/base.yaml") that includes:
- `num_clients`: Number of federated learning clients
- `beta`: Parameter for data distribution
- `fulltraining_flag`: Whether to use full training or not

You can specify a different configuration file using the `--config_path` argument.

## Output

Results are saved in structured directories with timestamps. For each experiment, the following files are generated:

1. Text file: Plain text output for backward compatibility
2. JSON file: Structured experiment results
3. CSV files:
   - rounds_*.csv: Data for each round
   - summary_*.csv: Summary statistics
   - results_arrays_*.csv: Global and client results arrays

When running all experiments, a summary file is also created that lists all experiments that were run and their status.

## Directory Structure

Results are organized in the following structure:
```
output_dir/
└── results_TIMESTAMP/
    ├── DATASET_DATA-LOADING_MODEL/
    │   ├── results_*.txt
    │   ├── results_*.json
    │   ├── rounds_*.csv
    │   ├── summary_*.csv
    │   └── results_arrays_*.csv
    └── summary.txt
```

## Logging

The script logs information to both a file and standard output. Log files are saved in the "logs" directory with a timestamp in the filename.

## Examples

### Basic usage (all experiments)

```bash
python run_experiments.py
```

### Single experiment with specific parameters

```bash
python run_experiments.py --mode single --dataset Citeseer --model GCN --data_loading random_walk --hop 2
```

### Custom configuration and output directory

```bash
python run_experiments.py --config_path conf/custom.yaml --output_dir my_results
```

## Data Loading Methods

The script supports various data loading methods for federated learning:

- `zero_hop`: Clients only receive their own data without any neighbor information
- `zero`: Similar to zero_hop but with a different implementation
- `full`: Clients receive the full dataset
- `random_walk`: Clients receive data based on random walks from their own nodes
- `diffusion`: Data is diffused across the graph to clients
- `efficient`: An efficient implementation of data loading
- `adjacency`: Clients receive data based on adjacency relationships
- `propagation`: Feature propagation-based data loading

The `hop` parameter controls how many hops of neighbor information to include for relevant methods. 