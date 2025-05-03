# Federated-GNN Source Code

This directory contains the source code for the Federated Graph Neural Networks project.

## Quick Start

### Simple Experiment

Run a single experiment with a specific configuration:

```bash
python run_simple.py --config simple_test.yaml
```

### Ablation Study

Run multiple experiments varying parameters:

```bash
python run_experiments.py --config multi_ablation_test.yaml
```

## Key Files

- `run.py`: Core implementation of federated training
- `run_simple.py`: Script for running single experiments
- `run_experiments.py`: Script for running ablation studies with multiple parameters
- `client.py`: Implementation of federated clients
- `server.py`: Implementation of federated server
- `models.py`: GNN model implementations
- `training_logs_analysis.py`: Utilities for analyzing training logs

## Configuration Examples

Example YAML files:
- `simple_config_example.yaml`: Example configuration for simple experiments
- `experiment_config_example.yaml`: Example configuration for ablation studies

## Results

Experiment results are saved in the `results/` directory:

```
results/
├── experiment_name/
│   ├── results_*.json          # Detailed experiment results
│   ├── results_*.txt           # Human-readable output
│   └── training_*.csv          # Training logs
└── summary_results_*.txt       # Summary of all experiments
```

## Analyzing Results

Use the analysis utilities to parse training logs:

```python
from training_logs_analysis import parse_client_csv
results = parse_client_csv('results/experiment_name/training_logs.csv')
```

You can also load the JSON results for further analysis:

```python
import json
with open('results/experiment_name/results.json', 'r') as f:
    results = json.load(f)
``` 