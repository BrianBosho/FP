# Quick Reference: Model Hyperparameters by Dataset

## Recommended Starting Points (Based on Literature)

### Small Datasets (Cora, Citeseer)
| Parameter | Recommended | Range | Notes |
|-----------|------------|-------|-------|
| hidden_dim | 16-32 | 8-64 | Start with 16 |
| num_layers | 2 | 2-3 | Deeper may overfit |
| dropout | 0.5-0.6 | 0.3-0.7 | Higher for small datasets |
| normalization | none | none, batch | Usually not needed |
| lr | 0.01 (Adam) | 0.001-0.1 | 0.5 for SGD |
| epochs | 3-10 | 3-20 | Use early stopping |

**Example Config:**
```yaml
model_architecture:
  GCN:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"
```

### Medium Datasets (Pubmed)
| Parameter | Recommended | Range | Notes |
|-----------|------------|-------|-------|
| hidden_dim | 16-32 | 16-128 | Can go wider |
| num_layers | 2 | 2-3 | |
| dropout | 0.5 | 0.3-0.6 | |
| normalization | none | none, batch | Optional |
| lr | 0.01 (Adam) | 0.001-0.1 | |
| epochs | 5-10 | 3-20 | |

**Example Config:**
```yaml
model_architecture:
  GAT:  # Or GCN
    hidden_dim: 8
    num_layers: 2
    dropout: 0.6
    num_heads: 8  # GAT only
    normalization: "none"
```

### Large Datasets (ogbn-arxiv, ogbn-products)
| Parameter | Recommended | Range | Notes |
|-----------|------------|-------|-------|
| hidden_dim | 256 | 128-512 | Much wider needed |
| num_layers | 3 | 3-5 | Deeper networks work |
| dropout | 0.3-0.5 | 0.3-0.6 | Lower dropout |
| normalization | batch | batch, layer | Essential for 3+ layers! |
| lr | 0.001 (Adam) | 0.0001-0.01 | Lower for stability |
| epochs | 3-5 | 3-10 | More data = fewer epochs |

**Example Config:**
```yaml
model_architecture:
  GCN_arxiv:  # Auto-applies to GCN on ogbn-arxiv
    hidden_dim: 256
    num_layers: 3
    dropout: 0.5
    normalization: "batch"
    use_unified_model: true  # Use standard GCN class
```

### Amazon Co-Purchase (Computers, Photos)
| Parameter | Recommended | Range | Notes |
|-----------|------------|-------|-------|
| hidden_dim | 64-128 | 32-256 | Mid-range |
| num_layers | 2-3 | 2-4 | |
| dropout | 0.4-0.5 | 0.3-0.6 | |
| normalization | none/batch | none, batch | Optional |
| lr | 0.01 (Adam) | 0.001-0.1 | |
| epochs | 5-10 | 3-20 | |

## Normalization Types - When to Use

| Type | Best For | Pros | Cons |
|------|----------|------|------|
| **none** | 2-layer models, small graphs | Simple, fast | May not scale to deep networks |
| **batch** | 3+ layers, large batches | Stable training, proven | Needs sufficient batch size |
| **layer** | Variable batch sizes | Works with any batch size | Slightly slower |
| **group** | Medium networks | Good balance | Need to set groups (default: 8) |

## GAT-Specific: Number of Heads

| Heads | Best For | Memory | Training Time |
|-------|----------|--------|---------------|
| 1 | Testing, baselines | Low | Fast |
| 4 | Small datasets | Medium | Medium |
| 8 | Most datasets (default) | High | Slow |
| 16 | Very large, complex graphs | Very High | Very Slow |

**Rule of thumb**: More heads = more capacity but slower. Start with 8.

## Common Configuration Patterns

### Pattern 1: Baseline (Fast, Simple)
```yaml
model_architecture:
  GCN:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"
training:
  lr: 0.01
  optimizer: "Adam"
  epochs: 5
```

### Pattern 2: Deep Network (More Capacity)
```yaml
model_architecture:
  GCN:
    hidden_dim: 32
    num_layers: 4
    dropout: 0.5
    normalization: "batch"  # Required!
training:
  lr: 0.01
  optimizer: "Adam"
  epochs: 10
```

### Pattern 3: Wide Network (More Capacity, Shallow)
```yaml
model_architecture:
  GCN:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.5
    normalization: "none"
training:
  lr: 0.005  # Lower LR for larger model
  optimizer: "Adam"
  epochs: 10
```

### Pattern 4: Regularized (Prevent Overfitting)
```yaml
model_architecture:
  GCN:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.7  # High dropout
    normalization: "batch"
training:
  lr: 0.01
  optimizer: "Adam"
  weight_decay: 0.001  # L2 regularization
  epochs: 20
  patience: 10  # Early stopping
```

## How to Choose Hyperparameters

### 1. Start with Literature Values
Use the defaults in `base.yaml` - they're from published papers.

### 2. Adjust Based on Dataset Size
- **Small** (<10K nodes): hidden_dim=16, num_layers=2, dropout=0.5-0.6
- **Medium** (10K-100K): hidden_dim=32-64, num_layers=2-3, dropout=0.4-0.5
- **Large** (>100K): hidden_dim=128-256, num_layers=3-4, dropout=0.3-0.5

### 3. Add Normalization for Deep Networks
If `num_layers >= 3`, use `normalization: "batch"`

### 4. Tune Learning Rate
- Adam: Start with 0.01, try 0.001 and 0.1
- SGD: Start with 0.5, try 0.1 and 1.0

### 5. Monitor Validation Performance
- **Overfitting** (train >> val): Increase dropout, add regularization
- **Underfitting** (train ≈ val, both low): Increase capacity (hidden_dim or num_layers)

## Quick Experiments to Run

### Experiment Set 1: Depth
```yaml
# Shallow: num_layers=2, normalization="none"
# Medium: num_layers=3, normalization="batch"
# Deep: num_layers=4, normalization="batch"
```

### Experiment Set 2: Width
```yaml
# Narrow: hidden_dim=8
# Medium: hidden_dim=16
# Wide: hidden_dim=32
# Very Wide: hidden_dim=64
```

### Experiment Set 3: Regularization
```yaml
# Low: dropout=0.3
# Medium: dropout=0.5
# High: dropout=0.7
```

### Experiment Set 4: Normalization
```yaml
# None: normalization="none"
# Batch: normalization="batch"
# Layer: normalization="layer"
```

## Copy-Paste Configs for Quick Testing

### Config A: Baseline GCN
```yaml
model_architecture:
  GCN:
    hidden_dim: 16
    num_layers: 2
    dropout: 0.5
    normalization: "none"
```

### Config B: Deep GCN
```yaml
model_architecture:
  GCN:
    hidden_dim: 32
    num_layers: 4
    dropout: 0.5
    normalization: "batch"
```

### Config C: Wide GCN
```yaml
model_architecture:
  GCN:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.5
    normalization: "none"
```

### Config D: Regularized GAT
```yaml
model_architecture:
  GAT:
    hidden_dim: 8
    num_layers: 2
    dropout: 0.7
    num_heads: 8
    normalization: "none"
```

### Config E: Large-Scale GCN (ogbn-arxiv)
```yaml
model_architecture:
  GCN_arxiv:
    hidden_dim: 256
    num_layers: 3
    dropout: 0.5
    normalization: "batch"
    use_unified_model: true
```
