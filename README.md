# Federated Learning Project (FP)

This repository contains code for running federated learning experiments with graph neural networks.

## Directory Structure

- `src/`: Source code files
  - `dataprocessing/`: Data processing modules
  - `result_processing/`: Result processing utilities
- `scripts/`: Utility scripts for running experiments
- `tests/`: Test files
- `config/`: Configuration files
- `docs/`: Documentation
- `notebooks/`: Jupyter notebooks for analysis
- `results/`: Experiment results
- `results_archive/`: Archive of all previous results
- `logs/`: Log files

## Running Experiments

To run experiments, use the `run_experiments.py` script in the `scripts` directory:

```bash
# Run a single experiment
python scripts/run_experiments.py --mode single --dataset Cora --model GCN --data_loading zero_hop

# Run all experiment combinations
python scripts/run_experiments.py --mode all
```

For more details on the available options, see the documentation in `docs/run_experiments_documentation.md`.

## Configuration

Configuration files are stored in the `config/` directory. The default configuration file is `base.yaml`.

## Repository Management

The repository includes scripts for organizing and cleaning up files:

```bash
# Move all result folders to results_archive
./scripts/organize_results.sh

# Clean up duplicated files and directories
./scripts/cleanup_duplicates.sh

# Remove unnecessary log files (nohup.out)
./scripts/cleanup.sh
```

You can also run the setup script which provides interactive options for all of these tasks:

```bash
./setup.sh
```

## Requirements

See `requirements.txt` for the required Python packages. 