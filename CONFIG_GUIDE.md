# Configuration Guide

## Overview
The `conf/` folder has been cleaned and organized to provide a simple, maintainable structure.

## Current Structure

```
conf/
├── base.yaml              # Base configuration with common settings
├── cora_config.yaml       # Cora dataset configuration
├── citeseer_config.yaml   # Citeseer dataset configuration  
├── pubmed_config.yaml     # Pubmed dataset configuration
└── ogbn-arxiv_config.yaml # ogbn-arxiv dataset configuration
```

## Usage

### Run experiments with specific datasets:

```bash
# Cora experiments
python -m src.experiments.run_experiments --config conf/cora_config.yaml

# Citeseer experiments  
python -m src.experiments.run_experiments --config conf/citeseer_config.yaml

# Pubmed experiments
python -m src.experiments.run_experiments --config conf/pubmed_config.yaml

# ogbn-arxiv experiments
python -m src.experiments.run_experiments --config conf/ogbn-arxiv_config.yaml
```

## Configuration Features

Each dataset config includes:

### Core Settings
- `num_clients`: Number of federated clients
- `num_rounds`: Communication rounds
- `epochs`: Local training epochs per round
- `beta`: Dirichlet distribution parameter (controls data heterogeneity)
- `lr`: Learning rate
- `optimizer`: SGD or Adam
- `datasets`: Target dataset name

### Data Loading Options
- `data_loading`: `full`, `adjacency`, `zero_hop`
- `models`: `GCN`, `GAT`
- `use_pe`: Positional encoding options

### Wandb Integration
- `use_wandb`: Enable/disable wandb logging
- `wandb_project`: Project name
- `wandb_entity`: Team/entity name (optional)
- `wandb_mode`: `online`, `offline`, or `disabled`

### Performance Settings
- `use_amp`: Mixed precision training
- `max_concurrent_clients`: Limit parallel clients
- `batch_size`: Batch size for training

## Archived Files

All old configuration files have been moved to `archive/conf/`:
- `archive/conf/ablation/` - Old ablation configs (25+ files)
- `archive/conf/baseline/` - Baseline configurations
- `archive/conf/review/` - Review configurations
- `archive/conf/test/` - Test configurations
- `archive/conf/sweep*.yaml` - Sweep configurations

## Benefits

1. **Simplicity**: Only 5 config files instead of 50+
2. **Clarity**: One config per dataset
3. **Maintainability**: Easy to modify and understand
4. **Consistency**: All configs follow the same structure
5. **Wandb Control**: Easy to enable/disable logging per dataset

## Customization

To modify settings for a specific dataset:
1. Edit the corresponding config file (e.g., `cora_config.yaml`)
2. Adjust parameters as needed
3. Run experiments with the updated config

To create a new dataset config:
1. Copy an existing config file
2. Modify the `datasets` field and other dataset-specific settings
3. Save with a descriptive name (e.g., `new_dataset_config.yaml`)
