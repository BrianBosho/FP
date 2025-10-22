# Model Architecture Configuration Guide

This guide explains how to configure model architectures using YAML config files.

## Quick Start

All model architecture settings are defined in `base.yaml` under the `model_architecture` section. You can override these in dataset-specific config files.

## Configuration Structure

```yaml
model_architecture:
  # Global defaults applied to all models
  default:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"  # Options: "batch", "layer", "group", "none"
    
  # Model-specific settings (override defaults)
  GCN:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"
    
  GCN_arxiv:  # Special config for ogbn-arxiv dataset
    hidden_dim: 256
    num_layers: 3
    dropout: 0.5
    normalization: "batch"
    
  GAT:
    hidden_dim: 8
    num_layers: 2
    dropout: 0.6
    num_heads: 8  # GAT-specific parameter
```

## Available Hyperparameters

### Common Parameters (All Models)

- **hidden_dim** (int): Hidden layer dimensions
  - Small datasets (Cora, Citeseer): 8-32
  - Medium datasets (Pubmed): 16-64
  - Large datasets (ogbn-arxiv): 128-512

- **num_layers** (int): Number of GNN layers (excluding input layer)
  - Shallow networks: 2 layers (good for small graphs)
  - Deep networks: 3-5 layers (better for large graphs, may need normalization)

- **dropout** (float): Dropout rate (0.0 - 1.0)
  - Low dropout: 0.3-0.5 (larger datasets, less overfitting risk)
  - High dropout: 0.5-0.7 (smaller datasets, more overfitting risk)

- **normalization** (string): Type of normalization layer
  - `"none"`: No normalization (default for 2-layer models)
  - `"batch"`: Batch Normalization (recommended for 3+ layers, large datasets)
  - `"layer"`: Layer Normalization (good for variable batch sizes)
  - `"group"`: Group Normalization (compromise between batch and layer)

### GAT-Specific Parameters

- **num_heads** (int): Number of attention heads
  - Common values: 4, 8, 16
  - More heads = more expressive but slower training

## How to Override Settings

### Option 1: Override in Dataset Config File

In your dataset-specific config (e.g., `cora_config.yaml`):

```yaml
# cora_config.yaml
datasets:
   - Cora

# Override model architecture for this experiment
model_architecture:
  GCN:
    hidden_dim: 32          # Change from default 16
    num_layers: 3           # Change from default 2
    dropout: 0.6            # Change from default 0.5
    normalization: "batch"  # Add batch normalization
```

### Option 2: Modify Base Config

Edit `conf/base.yaml` to change defaults for all experiments:

```yaml
# base.yaml
model_architecture:
  default:
    hidden_dim: 32  # New default for all models
```

## Examples

### Example 1: Deeper GCN for Cora

```yaml
# cora_config.yaml
model_architecture:
  GCN:
    hidden_dim: 32
    num_layers: 4
    dropout: 0.5
    normalization: "batch"  # Important for deeper networks!
```

### Example 2: Experiment with Different GAT Configurations

```yaml
# citeseer_config.yaml
model_architecture:
  GAT:
    hidden_dim: 16
    num_layers: 3
    dropout: 0.7
    num_heads: 4
    normalization: "layer"
```

### Example 3: Custom ogbn-arxiv Settings

```yaml
# ogbn-arxiv_config.yaml
model_architecture:
  GCN_arxiv:
    hidden_dim: 512      # Larger hidden dimension
    num_layers: 4        # Deeper network
    dropout: 0.3         # Lower dropout for large dataset
    normalization: "batch"
    use_unified_model: true  # Use standard GCN instead of GCN_arxiv class
```

## Using Unified GCN for All Datasets

By default, we now use a **unified GCN model** that can adapt to any dataset through configuration. This means:

- ✅ **No need for separate GCN_arxiv class** - just configure GCN appropriately
- ✅ **Same codebase for all datasets** - easier to maintain
- ✅ **Flexible experimentation** - change architecture without code changes

To use the unified model for ogbn-arxiv:

```yaml
# ogbn-arxiv_config.yaml
model_architecture:
  GCN_arxiv:
    hidden_dim: 256
    num_layers: 3
    dropout: 0.5
    normalization: "batch"
    use_unified_model: true  # This is the default
```

To force using the old GCN_arxiv class (if needed):

```yaml
model_architecture:
  GCN_arxiv:
    use_unified_model: false  # Use separate GCN_arxiv class
```

## Training Hyperparameters

You can also override training settings:

```yaml
training:
  lr: 0.01              # Learning rate
  optimizer: "Adam"     # "Adam" or "SGD"
  weight_decay: 0.0005  # L2 regularization
  epochs: 10            # Epochs per round
  patience: 20          # Early stopping patience
```

## Tips for Hyperparameter Tuning

1. **Start with defaults** - They're based on published papers
2. **Change one thing at a time** - Easier to understand impact
3. **Use batch normalization for deep networks** (3+ layers)
4. **Higher dropout for smaller datasets** (Cora, Citeseer)
5. **Lower dropout for larger datasets** (ogbn-arxiv, ogbn-products)
6. **Monitor validation accuracy** - Stop if overfitting

## Configuration Priority (Lowest to Highest)

1. Hard-coded defaults in `get_model_config()`
2. `model_architecture.default` in `base.yaml`
3. `model_architecture.{ModelType}` in `base.yaml`
4. `model_architecture.{ModelType}` in dataset config (e.g., `cora_config.yaml`)
5. Dataset-specific variants (e.g., `GCN_arxiv` for ogbn-arxiv automatically)

## Troubleshooting

**Q: My changes aren't taking effect**
- Check if the config file is being loaded correctly
- Verify the YAML syntax (indentation matters!)
- Look for the model configuration printout in the logs

**Q: Out of memory errors**
- Reduce `hidden_dim`
- Reduce `num_layers`
- Enable `use_amp: true` in base.yaml for mixed precision

**Q: Model underfitting**
- Increase `hidden_dim`
- Increase `num_layers` (and enable normalization)
- Reduce `dropout`
- Increase learning rate

**Q: Model overfitting**
- Increase `dropout`
- Reduce `hidden_dim`
- Add/change `normalization`
- Reduce learning rate
