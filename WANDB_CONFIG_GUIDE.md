# Wandb Configuration Guide

## Overview
The federated GNN experiments now support configurable wandb logging through YAML configuration files.

## Configuration Options

Add these settings to your YAML config files:

```yaml
# Wandb configuration
use_wandb: false          # Set to true to enable wandb logging
wandb_project: "FGL3"     # Project name for wandb
wandb_entity: null        # Set to your wandb entity/team name (optional)
wandb_mode: "online"      # "online", "offline", or "disabled"
```

## Usage Examples

### 1. Disable Wandb Completely
```yaml
use_wandb: false
```
- No wandb logging will occur
- Faster execution (no wandb overhead)
- Perfect for local testing

### 2. Enable Wandb with Custom Project
```yaml
use_wandb: true
wandb_project: "my-federated-experiments"
wandb_entity: "my-team"  # Optional
wandb_mode: "online"
```

### 3. Offline Mode (for air-gapped environments)
```yaml
use_wandb: true
wandb_project: "FGL3"
wandb_mode: "offline"
```
- Logs to local files
- Can sync later with `wandb sync`

## Current Configuration

Your `ogbn-arxiv_config.yaml` is currently set to:
```yaml
use_wandb: false  # Wandb logging disabled
wandb_project: "FGL3"
wandb_entity: null
wandb_mode: "online"
```

## Running Experiments

The command remains the same:
```bash
python -m src.experiments.run_experiments --config conf/ablation/ogbn-arxiv_config.yaml
```

- If `use_wandb: false`: No wandb logging occurs
- If `use_wandb: true`: Wandb logging occurs with configured settings

## Backward Compatibility

- If wandb settings are not specified, defaults to enabled for backward compatibility
- Existing configs without wandb settings will continue to work

## Benefits

1. **Performance**: Disable wandb for faster local testing
2. **Flexibility**: Change project names without code changes
3. **Organization**: Different projects for different experiment types
4. **Offline Support**: Work in environments without internet access
