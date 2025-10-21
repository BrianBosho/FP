# Memory Optimization - Final Implementation

## Observed Problem (from nvidia-smi)
```
Total GPU Usage: 37GB / 48GB
- Driver (python): 9GB
- 3 active training clients: 3 × 6.8GB = 20.4GB  
- 7 idle clients: 7 × 1-2GB = 7-14GB  ← PROBLEM: Should be 0GB!
- Other Ray processes: ~2-3GB
```

## Root Causes Identified

### 1. Ray GPU Reservation (`@ray.remote(num_gpus=1/10)`)
**Problem**: Ray pre-allocates GPU memory for each actor, even when idle on CPU
- Each client actor reserved ~1-2GB permanently
- 10 clients × 1.5GB = 15GB wasted on idle actors

**Solution**: Remove GPU reservation
```python
# Before
@ray.remote(num_gpus=1/10)
class FLClient:

# After  
@ray.remote  # No GPU reservation
class FLClient:
```

### 2. Feature Propagation on GPU
**Problem**: Preprocessing on CUDA uses 9GB of driver memory
- `feature_prop_device: "cuda"` in base.yaml
- All diffusion/propagation happens on GPU during data loading

**Solution**: Move to CPU
```yaml
# base.yaml
feature_prop_device: "cpu"  # Changed from "cuda"
```

### 3. Parameter Transfer Keeping GPU References
**Problem**: When returning model parameters, GPU tensors were kept alive
```python
return {'params': tuple(self.model.parameters())}  # GPU refs!
```

**Solution**: Explicitly copy to CPU and detach
```python
params_cpu = tuple(p.detach().cpu() for p in self.model.parameters())
return {'params': params_cpu}
```

## Implemented Optimizations

### File: `src/client.py`

1. **Removed GPU Reservation**
   ```python
   @ray.remote  # No num_gpus argument
   class FLClient:
   ```

2. **CPU/GPU Swapping** (already implemented)
   - Clients keep data/model on CPU by default
   - Move to GPU only during train/eval/test
   - Return to CPU immediately after

3. **Explicit CPU Parameter Return**
   ```python
   def get_params(self):
       self._move_to_device(self.cpu_device)
       params_cpu = tuple(p.detach().cpu() for p in self.model.parameters())
       buffers_cpu = tuple(b.detach().cpu() for b in self.model.buffers())
       return {'params': params_cpu, 'buffers': buffers_cpu}
   ```

### File: `src/server.py`

4. **Explicit Cleanup After Aggregation**
   ```python
   params_dict = ray.get(t)
   # ... use params_dict ...
   del params_dict  # Explicitly free memory
   ```

### File: `conf/base.yaml`

5. **CPU Feature Propagation**
   ```yaml
   feature_prop_device: "cpu"  # Saves ~9GB GPU during preprocessing
   ```

## Expected Memory Footprint (After Optimizations)

### Scenario: 4 clients, all training in parallel
```
Driver: ~500MB-1GB (preprocessing on CPU, minimal GPU footprint)
Active clients: 4 × 6.8GB = 27.2GB (during training)
Idle clients: 0GB (on CPU, no GPU reservation)
Ray overhead: ~1-2GB
═══════════════════════════════════
Total: ~29-31GB (down from 37GB)
```

### Scenario: 4 clients, 2 training in parallel (max_concurrent_clients=2)
```
Driver: ~500MB-1GB
Active clients: 2 × 6.8GB = 13.6GB
Idle clients: 0GB
Ray overhead: ~1-2GB
═══════════════════════════════════
Total: ~15-17GB (down from 37GB)
```

## Configuration Recommendations

### For ogbn-arxiv (Large Dataset)
```yaml
# ogbn-arxiv_config.yaml
num_clients: [4]
max_concurrent_clients: 2  # Train 2 at a time for safety
```

### For Medium Datasets (Pubmed with diffusion)
```yaml
num_clients: [10]
max_concurrent_clients: 3  # Train 3 at a time
```

### For Small Datasets (Cora, CiteSeer)
```yaml
num_clients: [10]
max_concurrent_clients: 5  # Or 0 for all parallel
```

## Per-Client Memory Breakdown

### During Training (Active Client)
```
Model parameters: ~50MB (GCN with 256 hidden)
Graph data (features): ~200-500MB (depending on feature dim)
Graph edges: ~100-200MB
Intermediate activations: ~5-6GB (during forward/backward pass)
─────────────────────────────────
Total: ~6.8GB
```

### When Idle (After moving to CPU)
```
Ray actor overhead: ~10-50MB
Python process: ~50-100MB
─────────────────────────────────
Total: ~0MB GPU (everything on CPU)
```

## Further Optimizations (If Still Needed)

### 1. Reduce Client Count
```yaml
num_clients: [2]  # Fewer clients = less total memory
```

### 2. Enable Mini-Batch Training
```yaml
use_minibatch: true
batch_size: 1024
num_neighbors: [10, 10]
```
This reduces per-client active memory from 6.8GB to ~2-3GB

### 3. Gradient Checkpointing (Advanced)
Modify models to use gradient checkpointing - trades compute for memory

### 4. Reduce Feature Dimensions
```yaml
pe_r: 32  # Reduce from 64
pe_P: 8   # Reduce from 16
```

### 5. Reduce Model Size
```python
# In models.py for GCN_arxiv
hidden_dim=128  # Reduce from 256
```

## Monitoring

### Watch GPU Memory in Real-Time
```bash
watch -n 1 nvidia-smi
```

### Expected Behavior After Optimizations
- Driver process should use <1GB GPU memory
- Only `max_concurrent_clients` should show ~6-7GB at any time
- Idle clients should show 0MB or <100MB
- Total should stay under 20GB with `max_concurrent_clients=2`

## Verification Commands

```bash
# Check config
cat conf/base.yaml | grep feature_prop_device
cat conf/ogbn-arxiv_config.yaml | grep max_concurrent_clients

# Run experiment
python -m src.experiments.run_experiments --config conf/ogbn-arxiv_config.yaml

# Monitor in separate terminal
watch -n 1 nvidia-smi
```

## Summary of Changes

| Optimization | File | Memory Saved | Trade-off |
|-------------|------|--------------|-----------|
| Remove GPU reservation | `src/client.py` | ~15GB (1.5GB × 10 clients) | None |
| CPU preprocessing | `conf/base.yaml` | ~8GB (driver) | Slower preprocessing |
| CPU parameter return | `src/client.py` | ~2-3GB (refs) | None |
| Explicit cleanup | `src/server.py` | ~1-2GB (refs) | None |
| **Total Savings** | | **~26-28GB** | Minimal |

With these optimizations, you should be able to run 4 clients with only 2 training in parallel (~15-17GB total), or even all 4 in parallel if you reduce to 3 clients (~20-22GB total).
